"""Transcription, via either a local model (faster-whisper, free, fully
private -- the default) or Groq's hosted Whisper API (`transcription.engine:
"groq"`, opt-in -- much faster on a CPU-only machine, but the audio leaves
this machine; see docs/using_cloud_apis.md). Both paths produce the same
speaker-labeled plain-text format.

Speaker labeling has two mechanisms, tried in this order, for either engine:

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
     docs/diarization_setup.md for the one-time setup it needs. Runs
     locally regardless of which transcription engine produced the text.
"""
from __future__ import annotations

import difflib
import logging
import os
import pathlib
import tempfile
import threading
from types import SimpleNamespace
from typing import Callable, Optional

import requests

from . import api_keys
from .config_loader import Config

logger = logging.getLogger(__name__)


class TranscriptionCancelled(Exception):
    """Raised when `cancel_event` is set mid-transcription. Whatever
    segments were already decoded are discarded -- there's no meaningful
    "resume from here" for a batch Whisper run, so a cancelled transcript
    is treated as fully cancelled, not partial."""


def _looks_like_mic_bleed(you_text: str, interviewer_text: str) -> bool:
    """True if `you_text` looks like it's substantially made of an echo of
    `interviewer_text` picked up by the microphone through the speakers,
    rather than something the user actually said. Reproduced on a real
    interview: without headphones, a laptop mic can pick up nearly an
    entire interviewer sentence verbatim -- well beyond the "stray word or
    two" this app's docs describe as an expected minor caveat, and
    disruptive enough to duplicate a whole line under the wrong speaker.

    Measured as *coverage* (how much of `you_text` is found, in matching
    blocks, within `interviewer_text`) rather than difflib's own overall
    similarity ratio -- coverage is robust to `interviewer_text` being a
    concatenation of several nearby segments with extra content around the
    matching part (see _filter_mic_bleed, which spans real bleed across a
    segment boundary this way), where a plain ratio would be pulled down
    by the length mismatch even for a full duplication.

    A length threshold avoids false-positiving on short replies that
    legitimately share a few common words with the question."""
    if len(you_text) < 20:
        return False
    matcher = difflib.SequenceMatcher(None, you_text.lower(), interviewer_text.lower())
    matched_chars = sum(block.size for block in matcher.get_matching_blocks())
    return (matched_chars / len(you_text)) >= 0.7


def _filter_mic_bleed(labeled: list[tuple[float, float, str, str]]) -> list[tuple[float, float, str, str]]:
    """Drops "You" segments that look like mic bleed of temporally-nearby
    "Interviewer" segments (see _looks_like_mic_bleed) -- used by both
    dual-channel transcription paths (local and Groq) right after merging,
    since bleed is a property of the recording itself, not of which engine
    transcribed it. Only ever drops "You" segments, never "Interviewer"
    ones: bleed only runs mic-picks-up-speaker-output, not the reverse (the
    interviewer's side has no way to hear -- let alone transcribe -- your
    microphone).

    Nearby Interviewer segments are concatenated into one block before
    comparing, not checked one at a time -- real bleed of one continuous
    interviewer sentence can land split across two of their segments (the
    two channels' VAD boundaries don't line up), so no single Interviewer
    segment alone would look similar enough on its own."""
    interviewer_segments = [(s, e, t) for s, e, spk, t in labeled if spk == "Interviewer"]
    filtered = []
    for s, e, spk, t in labeled:
        if spk == "You":
            nearby_text = " ".join(
                it for is_, ie, it in interviewer_segments
                if s - 2.0 <= ie and is_ <= e + 2.0
            )
            if nearby_text and _looks_like_mic_bleed(t, nearby_text):
                continue
        filtered.append((s, e, spk, t))
    return filtered


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


def load_whisper_model(cfg: Config):
    """Loads a WhisperModel per `cfg.transcription`'s settings. Split out
    from transcribe() so a caller that needs to transcribe many separate
    audio files against the same settings (see live_transcribe.py, which
    transcribes one interview's audio in periodic segments during
    recording rather than one whole-file call at the end) can load the
    model once and reuse it, instead of paying WhisperModel's real loading
    cost on every call.

    Falls back to an offline-only load (`local_files_only=True`) if the
    normal load fails -- WhisperModel always checks Hugging Face Hub for
    updates before using a model by default, even one that's already fully
    downloaded and cached locally, so a transient network issue could
    otherwise break transcription for a model that doesn't need any
    network access at all (observed directly: a real Hub outage broke
    loading an already-cached model). If the model genuinely isn't cached
    yet (e.g. first run), the offline retry fails too and the original
    (informative, network-related) error is what gets raised.
    """
    from faster_whisper import WhisperModel

    tcfg = cfg.transcription
    model_name = tcfg.get("whisper_model", "small")
    device = tcfg.get("device", "cpu")
    compute_type = "int8" if device == "cpu" else "float16"
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:  # noqa: BLE001
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type, local_files_only=True)
        except Exception:
            raise e from None
        logger.info("Loaded %s from local cache after a network issue reaching Hugging Face Hub.", model_name)
        return model


def transcribe(
    audio_path: pathlib.Path,
    cfg: Config,
    on_progress: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    model=None,
) -> str:
    """Returns a speaker-labeled transcript as plain text, e.g.:

    [Interviewer] Tell me about a time you dealt with conflict.
    [You] So, um, there was this one time...

    If given, `on_progress(fraction)` is called with a real 0.0-1.0 estimate
    as transcription proceeds.

    If given, `cancel_event` is checked between segments (local engine) or
    before each upload (Groq engine); when set, raises
    `TranscriptionCancelled` instead of returning.

    If given, `model` is used instead of loading a new one from `cfg` --
    see load_whisper_model()'s docstring for why a caller might already
    have one loaded. Ignored when `transcription.engine` is "groq" -- that
    path has no local model to reuse.
    """
    tcfg = cfg.transcription
    if tcfg.get("engine", "faster-whisper") == "groq":
        labeled = _transcribe_via_groq(audio_path, cfg, on_progress, cancel_event)
    else:
        labeled = _transcribe_local(audio_path, cfg, on_progress, cancel_event, model)

    if tcfg.get("language", "auto") == "hinglish":
        labeled = [(s, e, spk, _to_latin_if_available(text)) for s, e, spk, text in labeled]

    lines = [f"[{spk}] {text}" for _, _, spk, text in labeled]
    return "\n".join(lines)


def _transcribe_local(
    audio_path: pathlib.Path,
    cfg: Config,
    on_progress: Optional[Callable[[float], None]],
    cancel_event: Optional[threading.Event],
    model,
) -> list[tuple[float, float, str, str]]:
    """The default, local, fully-private transcription path via
    faster-whisper. See transcribe()'s docstring for the shared contract;
    this returns (start, end, speaker, text) tuples rather than the final
    joined string, since transcribe() applies hinglish transliteration and
    joins for both this path and _transcribe_via_groq identically."""
    tcfg = cfg.transcription
    if model is None:
        model = load_whisper_model(cfg)

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
        # A short style/context hint biases decoding towards this kind of
        # audio -- real, informal spoken conversation with filler words --
        # rather than Whisper's tendency to "clean up" or drop disfluent
        # speech into a more formal-sounding (and less accurate) sentence.
        # Configurable since it's most useful for accented/non-US-English
        # speech (see config.yaml's comment and docs/language_support.md);
        # set to "" to disable.
        initial_prompt=tcfg.get("initial_prompt") or None,
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

    return labeled


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
    audio through the speakers too (unless you're on headphones) -- there's
    no acoustic echo cancellation here, just channel separation for the
    dominant source on each side. `_filter_mic_bleed` below catches the
    disruptive case (a whole sentence duplicated under the wrong speaker);
    a stray word or two blending into a real answer can still slip through
    undetected.
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

    labeled = _filter_mic_bleed(labeled)
    labeled.sort(key=lambda item: item[0])
    return labeled


def _transcribe_via_groq(
    audio_path: pathlib.Path,
    cfg: Config,
    on_progress: Optional[Callable[[float], None]],
    cancel_event: Optional[threading.Event],
) -> list[tuple[float, float, str, str]]:
    """Transcribes via Groq's hosted Whisper API instead of a local model --
    much faster on a CPU-only machine (Groq's free tier processes audio at
    up to ~228x real-time), at the real cost of the audio leaving this
    machine and going to Groq's servers. See docs/using_cloud_apis.md.

    Progress reporting is coarser than the local path: Groq's transcription
    endpoint is a single blocking HTTP call with no streaming, so there's
    no per-segment signal to report mid-call -- on_progress only fires
    between channels (dual-channel) or once at the end (mono), rather than
    continuously. Acceptable given how fast the call itself is.
    """
    env_var = cfg.analysis.get("cloud_api_key_env_var", "INTERVIEW_ANALYZER_API_KEY")
    api_key = os.environ.get(env_var) or api_keys.load_key("groq")
    if not api_key:
        raise RuntimeError(
            f"No Groq API key found. Get a free one (no credit card) at "
            f"https://console.groq.com/keys, then set it in the Settings tab's "
            f"\"Cloud API key\" section, or set the {env_var} environment variable."
        )

    tcfg = cfg.transcription
    groq_model = tcfg.get("groq_whisper_model", "whisper-large-v3-turbo")
    language = _resolve_whisper_language(tcfg.get("language", "auto"))
    prompt = tcfg.get("initial_prompt") or None

    if tcfg.get("diarization", True) and _channel_count(audio_path) >= 2:
        return _groq_transcribe_dual_channel(
            audio_path, groq_model, language, prompt, api_key, on_progress, cancel_event
        )

    segments = _groq_transcribe_file(audio_path, groq_model, language, prompt, api_key, cancel_event)
    if on_progress is not None:
        on_progress(1.0)
    if tcfg.get("diarization", True):
        # _diarize_and_label expects faster-whisper Segment-like objects
        # (attribute access, not dict keys) -- a thin adapter so pyannote
        # diarization works identically regardless of which engine
        # produced the underlying segments
        fake_segments = [SimpleNamespace(start=s["start"], end=s["end"], text=s["text"]) for s in segments]
        return _diarize_and_label(audio_path, fake_segments)
    return [(s["start"], s["end"], "Speaker", s["text"].strip()) for s in segments]


def _groq_transcribe_file(
    audio_path: pathlib.Path,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    api_key: str,
    cancel_event: Optional[threading.Event],
) -> list[dict]:
    """One blocking call to Groq's /audio/transcriptions endpoint, returning
    its segment list (each with "start"/"end"/"text") -- the Groq
    equivalent of faster-whisper's segment stream, just materialized all
    at once rather than yielded lazily (Groq's API isn't a streaming
    endpoint)."""
    if cancel_event is not None and cancel_event.is_set():
        raise TranscriptionCancelled()

    data = {"model": model, "response_format": "verbose_json", "timestamp_granularities[]": "segment"}
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    with open(audio_path, "rb") as f:
        resp = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (pathlib.Path(audio_path).name, f)},
            data=data,
            timeout=300,
        )
    resp.raise_for_status()
    return resp.json().get("segments", [])


def _groq_transcribe_dual_channel(
    audio_path: pathlib.Path,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    api_key: str,
    on_progress: Optional[Callable[[float], None]],
    cancel_event: Optional[threading.Event],
) -> list[tuple[float, float, str, str]]:
    """Groq equivalent of _transcribe_dual_channel above -- same left=mic
    ("You") / right=loopback ("Interviewer") convention, same merge-by-
    start-time logic. Groq's API has no concept of multi-channel input, so
    each channel is decoded and written to its own temp mono WAV file for
    a separate upload, then the temp files are cleaned up immediately
    after."""
    from faster_whisper.audio import decode_audio

    import soundfile as sf

    you_audio, interviewer_audio = decode_audio(str(audio_path), sampling_rate=16000, split_stereo=True)

    labeled: list[tuple[float, float, str, str]] = []

    def _run_channel(audio_array, speaker_label: str, progress_offset: float, progress_share: float) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise TranscriptionCancelled()
        fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="groq_channel_")
        os.close(fd)
        tmp_path = pathlib.Path(tmp_name)
        try:
            sf.write(str(tmp_path), audio_array, 16000, subtype="PCM_16")
            segments = _groq_transcribe_file(tmp_path, model, language, prompt, api_key, cancel_event)
        finally:
            tmp_path.unlink(missing_ok=True)
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                labeled.append((seg["start"], seg["end"], speaker_label, text))
        if on_progress is not None:
            on_progress(progress_offset + progress_share)

    _run_channel(you_audio, "You", 0.0, 0.5)
    _run_channel(interviewer_audio, "Interviewer", 0.5, 0.5)

    labeled = _filter_mic_bleed(labeled)
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
