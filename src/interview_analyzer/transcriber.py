"""Local, free transcription via faster-whisper, with optional speaker
diarization so the transcript distinguishes you from the interviewer.
"""
from __future__ import annotations

import logging
import pathlib

from .config_loader import Config

logger = logging.getLogger(__name__)


def transcribe(audio_path: pathlib.Path, cfg: Config) -> str:
    """Returns a speaker-labeled transcript as plain text, e.g.:

    [Interviewer] Tell me about a time you dealt with conflict.
    [You] So, um, there was this one time...
    """
    from faster_whisper import WhisperModel

    tcfg = cfg.transcription
    model = WhisperModel(
        tcfg.get("whisper_model", "small"),
        device=tcfg.get("device", "cpu"),
        compute_type="int8" if tcfg.get("device", "cpu") == "cpu" else "float16",
    )

    segments, _info = model.transcribe(str(audio_path), beam_size=5)
    segments = list(segments)

    if tcfg.get("diarization", True):
        labeled = _diarize_and_label(audio_path, segments)
    else:
        labeled = [(seg.start, seg.end, "Speaker", seg.text.strip()) for seg in segments]

    lines = [f"[{spk}] {text}" for _, _, spk, text in labeled]
    return "\n".join(lines)


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
