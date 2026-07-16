"""Compresses the raw WAV capture down to a small speech-quality file.

Keeps storage minimal during the (short, configurable) retention window.
Requires ffmpeg on PATH (free, widely available). Falls back to leaving
the WAV in place if ffmpeg isn't found, with a warning.
"""
from __future__ import annotations

import logging
import pathlib
import shutil
import subprocess

logger = logging.getLogger(__name__)


def compress_audio(
    wav_path: pathlib.Path, bitrate_kbps: int = 64, fmt: str = "opus"
) -> pathlib.Path:
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found on PATH; keeping uncompressed WAV: %s", wav_path)
        return wav_path

    out_path = wav_path.with_suffix(f".{fmt}")
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-b:a", f"{bitrate_kbps}k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("ffmpeg compression failed, keeping WAV: %s", result.stderr)
        return wav_path

    wav_path.unlink(missing_ok=True)
    logger.info("Compressed %s -> %s", wav_path, out_path)
    return out_path
