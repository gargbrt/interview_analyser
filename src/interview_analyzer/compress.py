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
    try:
        # stdin=DEVNULL and an explicit timeout are both real, reproduced-
        # elsewhere protections, not defensive-for-its-own-sake: a
        # subprocess that inherits the parent's stdin can hang waiting for
        # input that will never arrive in a non-interactive context (e.g.
        # a CI runner's shell), and any external process can simply take
        # too long or never exit on unusual/malformed input. Neither
        # failure mode should be able to block this call forever.
        result = subprocess.run(
            cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120,
        )
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg compression timed out after 120s, keeping WAV: %s", wav_path)
        return wav_path
    if result.returncode != 0:
        logger.warning("ffmpeg compression failed, keeping WAV: %s", result.stderr)
        return wav_path

    wav_path.unlink(missing_ok=True)
    logger.info("Compressed %s -> %s", wav_path, out_path)
    return out_path
