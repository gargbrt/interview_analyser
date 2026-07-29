"""Tests for compress.py's compress_audio -- shrinks the raw WAV capture to
opus/mp3 for the retention window. No dedicated tests existed for this
before a real incident: a subprocess.run() call here with no timeout and no
explicit stdin redirect could hang indefinitely (reproduced as a real CI
hang on macOS, where ffmpeg is preinstalled -- a background thread ran
this against a malformed/fake WAV file after its mocks had already been
torn down, and the resulting real ffmpeg subprocess call never returned).
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from interview_analyzer.compress import compress_audio


def test_keeps_the_wav_when_ffmpeg_is_not_on_path(tmp_path):
    wav_path = tmp_path / "call.wav"
    wav_path.write_bytes(b"fake wav data")

    with patch("interview_analyzer.compress.shutil.which", return_value=None):
        result = compress_audio(wav_path)

    assert result == wav_path
    assert wav_path.exists()


def test_returns_the_compressed_path_and_deletes_the_wav_on_success(tmp_path):
    wav_path = tmp_path / "call.wav"
    wav_path.write_bytes(b"fake wav data")

    with patch("interview_analyzer.compress.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("interview_analyzer.compress.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        # ffmpeg would normally create this file itself -- simulate that
        # side effect so the real wav_path.unlink() has something to do
        out_path = wav_path.with_suffix(".opus")
        out_path.write_bytes(b"fake compressed audio")

        result = compress_audio(wav_path)

    assert result == out_path
    assert not wav_path.exists()
    assert out_path.exists()


def test_keeps_the_wav_when_ffmpeg_exits_non_zero(tmp_path):
    wav_path = tmp_path / "call.wav"
    wav_path.write_bytes(b"fake wav data")

    with patch("interview_analyzer.compress.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("interview_analyzer.compress.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="decode error")

        result = compress_audio(wav_path)

    assert result == wav_path
    assert wav_path.exists()


def test_keeps_the_wav_and_does_not_raise_when_ffmpeg_hangs_past_the_timeout(tmp_path):
    """Regression coverage for the real incident this fix addresses: a
    hung ffmpeg subprocess must be treated the same as any other failure
    (fall back to the uncompressed WAV) instead of blocking the caller
    forever or propagating TimeoutExpired uncaught."""
    wav_path = tmp_path / "call.wav"
    wav_path.write_bytes(b"fake wav data")

    with patch("interview_analyzer.compress.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("interview_analyzer.compress.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=120)

        result = compress_audio(wav_path)

    assert result == wav_path
    assert wav_path.exists()


def test_subprocess_call_redirects_stdin_and_sets_a_bounded_timeout(tmp_path):
    """A subprocess that inherits the parent's stdin can hang waiting for
    input that will never arrive in a non-interactive context (this is
    exactly what happened in CI) -- guard that both protections stay in
    place rather than only testing the observable fallback behavior."""
    wav_path = tmp_path / "call.wav"
    wav_path.write_bytes(b"fake wav data")

    with patch("interview_analyzer.compress.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("interview_analyzer.compress.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="")

        compress_audio(wav_path)

    _args, kwargs = mock_run.call_args
    assert kwargs.get("stdin") == subprocess.DEVNULL
    assert kwargs.get("timeout") == 120
