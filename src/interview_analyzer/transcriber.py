"""Local, free transcription via faster-whisper, with speaker labeling so
the transcript distinguishes you from the interviewer.

Speaker labeling has two mechanisms, tried in this order:

  1. **Channel separation** (the default, used whenever a recording has
     the microphone on its own channel -- see recorder.py's module
     docstring): the mic ("You") and system-audio loopback
     ("Interviewer") channels are transcribed independently and merged by
     timestamp. Which channel a word came from *is* who said it, not a
     guess -- far more reliable than acoustic diarization, and needs no
     extra setup (no Hugging Face account/token, no model download).
  2. **Acoustic diarization** via pyannote.audio (`_diarize_and_label`),
     used only as a fallback for mono recordings (e.g. the microphone
     wasn't available, so there's nothing to separate by channel) -- see
     docs/diarization_setup.md for the one-time setup it needs.
"""
from __future__ import annotations

import logging
import pathlib
import threading
from typing import Callable, Optional

from .config_loader import Config

logger = logging.getLogger(__name__)


class TranscriptionCancelled(Exception):
    """Raised when `cancel_event` is set mid-transcription. Whatever
    segments were already decoded are discarded -- there's no meaningful
    "resume from here" for a batch Whisper run, so a cancelled transcript
    is treated as fully cancelled, not partial."""


def _channel_count(audio_path: pathlib.Path) -> int:
    """Cheap channel-count probe (no full audio decode) via PyAV -- the
    same library faster-whisper itself uses internally to decode audio, so
    this works for whatever format is actually on disk (WAV before
    compression, opus/mp3 after -- see compress.py, which preserves the
    channel count). Returns 1 (safe/conservative default) if the file
    can't be probed for any reason."""
    import av

    try:
        with av.open(str(audio_path)) as container:
            return container.streams.audio[0].channels
    except Exception:  # noqa: BLE001
        return 1


def transcribe(
    audio_path: pathlib.Path,
    cfg: Config,
    on_progress: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """Returns a speaker-labeled transcript as plain text, e.g.:

    [Interviewer] Tell me about a time you dealt with conflict.
    [You] So, um, there was this one time...

    If given, `on_progress(fraction)` is called with a real 0.0-1.0 estimate
    as transcription proceeds -- faster-whisper yields segments lazily as it
    works through the audio, so each segment's end timestamp against the
    total duration is an honest progress signal, not a fake animation.

    If given, `cancel_event` is checked between segments; when set, raises
    `TranscriptionCancelled` instead of returning (checked between segments
    because that's the only natural interruption point faster-whisper's
    synchronous, single-call API offers -- there's no mid-segment hook).
    """
    from faster_whisper import WhisperModel

    tcfg = cfg.transcription
    model = WhisperModel(
        tcfg.get("whisper_model", "small"),
        device=tcfg.get("device", "cpu"),
        compute_type="int8" if tcfg.get("device", "cpu") == "cpu" else "float16",
    )

    language = _resolve_whisper_language(tcfg.get("language", "auto"))
    vad_filter = tcfg.get("vad_filter", True)
    transcribe_kwargs = dict(
        beam_size=5,
        language=language,
        # vad_filter segments the audio on actual speech first, instead of
        # decoding fixed ~30s windows regardless of content -- without it, a
        # short utterance right after a long silence can land inside a
        # mostly-silent window and get silently dropped (reproduced on a
        # real recording: speech ~55s in, after a long "waiting for
        # participants" pause, was missing from the transcript until this
        # was enabled). speech_pad_ms keeps a little silence on each side of
        # a detected speech span so words right at the boundary aren't clipped.
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=300) if vad_filter else None,
        # Off by default: with it on, a long silent stretch can make Whisper
        # keep "conditioning" on the last thing it transcribed and either
        # hallucinate or under-transcribe the next real utterance instead of
        # treating it fresh. That's part of the same 55s-dropped-speech bug.
        condition_on_previous_text=False,
        no_speech_threshold=0.4,
    )

    if tcfg.get("diarization", True) and _channel_count(audio_path) >= 2:
        labeled = _transcribe_dual_channel(model, audio_path, transcribe_kwargs, on_progress, cancel_event)
    else:
        segment_stream, info = model.transcribe(str(audio_path), **transcribe_kwargs)
        total_duration = info.duration or 0
        segments = []
        for seg in segment_stream:
            if cancel_event is not None and cancel_event.is_set():
                raise TranscriptionCancelled()
            segments.append(seg)
            if on_progress is not None and total_duration > 0:
                on_progress(min(seg.end / total_duration, 1.0))

        if tcfg.get("diarization", True):
            labeled = _diarize_and_label(audio_path, segments)
        else:
            labeled = [(seg.start, seg.end, "Speaker", seg.text.strip()) for seg in segments]

    if tcfg.get("language", "auto") == "hinglish":
        labeled = [(s, e, spk, _to_latin_if_available(text)) for s, e, spk, text in labeled]

    lines = [f"[{spk}] {text}" for _, _, spk, text in labeled]
    return "\n".join(lines)


def _transcribe_dual_channel(
    model,
    audio_path: pathlib.Path,
    transcribe_kwargs: dict,
    on_progress: Optional[Callable[[float], None]],
    cancel_event: Optional[threading.Event],
) -> list[tuple[float, float, str, str]]:
    """Transcribes the microphone ("You") and system-audio loopback
    ("Interviewer") channels independently -- see recorder.py's module
    docstring for why they're on separate channels -- then merges the two
    segment lists by start time into one chronological transcript.

    Progress is split evenly across the two passes (0-50% for "You", 50-
    100% for "Interviewer") since both channels cover the same total
    duration.

    Note: a laptop microphone often picks up *some* of the other side's
    audio through the speakers too (unless you're on headphones), so the
    "You" channel can occasionally pick up a stray word or two of the
    interviewer's -- there's no acoustic echo cancellation here to prevent
    that, just channel separation for the dominant source on each side.
    """
    from faster_whisper.audio import decode_audio

    sampling_rate = model.feature_extractor.sampling_rate
    you_audio, interviewer_audio = decode_audio(
        str(audio_path), sampling_rate=sampling_rate, split_stereo=True
    )

    labeled: list[tuple[float, float, str, str]] = []

    def _run_channel(audio_array, speaker_label: str, progress_offset: float, progress_share: float) -> None:
        segment_stream, info = model.transcribe(audio_array, **transcribe_kwargs)
        total_duration = info.duration or 0
        for seg in segment_stream:
            if cancel_event is not None and cancel_event.is_set():
                raise TranscriptionCancelled()
            text = seg.text.strip()
            if text:
                labeled.append((seg.start, seg.end, speaker_label, text))
            if on_progress is not None and total_duration > 0:
                on_progress(progress_offset + progress_share * min(seg.end / total_duration, 1.0))

    _run_channel(you_audio, "You", 0.0, 0.5)
    _run_channel(interviewer_audio, "Interviewer", 0.5, 0.5)

    labeled.sort(key=lambda item: item[0])
    return labeled


def _resolve_whisper_language(language_setting: str) -> Optional[str]:
    """Maps this app's `transcription.language` setting to the language code
    Whisper expects. "auto" (or anything unrecognized) leaves it as None so
    Whisper auto-detects, same as before this setting existed. "hinglish"
    isn't a real Whisper language code -- Whisper has no dedicated
    code-switched Hindi/English mode -- so it's pinned to "hi", which in
    practice transcribes embedded English words inline rather than forcing
    everything into Hindi; the output is then optionally romanized in
    `transcribe()` above. See docs/language_support.md."""
    if language_setting in (None, "", "auto"):
        return None
    if language_setting == "hinglish":
        return "hi"
    return language_setting


def _to_latin_if_available(text: str) -> str:
    """Best-effort transliteration of Devanagari text to the Latin
    alphabet, for the optional "hinglish" language pack -- so the transcript
    reads as Romanized Hinglish instead of mixed Devanagari/Latin script.
    Falls back to the untouched text if the optional `indic-transliteration`
    package isn't installed (same optional-dependency pattern as
    diarization's pyannote.audio -- see docs/language_support.md)."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        logger.warning(
            "indic-transliteration isn't installed; leaving hinglish transcript in its "
            "original script instead of romanizing it. `pip install indic-transliteration` to enable it."
        )
        return text
    try:
        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception:  # noqa: BLE001
        return text


def _diarize_and_label(audio_path: pathlib.Path, segments) -> list[tuple[float, float, str, str]]:
    """Best-effort diarization using pyannote.audio. Falls back to a single
    generic speaker label if pyannote isn't installed/configured (it needs a
    free Hugging Face token to download pretrained models on first run) —
    the pipeline still works, just without speaker separation.
    """
    try:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        diarization = pipeline(str(audio_path))

        speaker_turns = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        labeled = []
        for seg in segments:
            mid = (seg.start + seg.end) / 2
            speaker = next(
                (spk for (s, e, spk) in speaker_turns if s <= mid <= e), "Speaker"
            )
            labeled.append((seg.start, seg.end, speaker, seg.text.strip()))
        return labeled
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Diarization unavailable (%s). Falling back to unlabeled transcript. "
            "See docs/diarization_setup.md to enable speaker separation.", e
        )
        return [(seg.start, seg.end, "Speaker", seg.text.strip()) for seg in segments]
