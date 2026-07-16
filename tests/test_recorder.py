"""Tests for SystemAudioRecorder's pause/resume/stop behavior.

Real WASAPI loopback capture needs actual Windows audio hardware, so the
`pyaudiowpatch` PyAudio object is faked here -- everything downstream of
"a stream that yields frames" (pause/resume gating, WAV writing, stop/
cleanup) is real code under test.
"""
from __future__ import annotations

import wave
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from interview_analyzer.recorder import SystemAudioRecorder

FRAME = b"\x00\x01" * 1024  # 1024 samples of 16-bit mono silence-ish data


class _FakeStream:
    """Never yields real frames: tests drive audio via `_handle_frame()`
    directly for deterministic assertions, rather than racing against
    SystemAudioRecorder's real background capture thread. The one read
    attempt the background thread makes fails immediately and the thread
    exits (see `_record_loop`'s `except` clause), so it never competes
    with the test's own writes.
    """

    def read(self, n, exception_on_overflow=False):
        raise RuntimeError("fake stream produces no frames; tests drive frames directly")

    def stop_stream(self):
        pass

    def close(self):
        pass


class _FakePyAudio:
    paInt16 = 8
    paWASAPI = 13

    def __init__(self):
        pass

    def get_host_api_info_by_type(self, _api):
        return {"defaultOutputDevice": 0}

    def get_device_info_by_index(self, _idx):
        return {
            "name": "Speakers",
            "isLoopbackDevice": True,
            "maxInputChannels": 1,
            "defaultSampleRate": 16000.0,
            "index": 0,
        }

    def get_loopback_device_info_generator(self):
        return iter([])

    def get_sample_size(self, _fmt):
        return 2

    def open(self, **kwargs):
        return _FakeStream()

    def terminate(self):
        pass


@contextmanager
def _fake_recorder():
    fake_module = MagicMock()
    fake_module.PyAudio = _FakePyAudio
    fake_module.paInt16 = _FakePyAudio.paInt16
    fake_module.paWASAPI = _FakePyAudio.paWASAPI
    with patch("interview_analyzer.recorder.pyaudio", fake_module):
        yield SystemAudioRecorder(sample_rate=16000, channels=1)


def test_handle_frame_writes_when_not_paused(tmp_path):
    with _fake_recorder() as rec:
        out_path = tmp_path / "call.wav"
        rec.start(out_path)

        assert rec.is_paused is False
        rec._handle_frame(FRAME)
        rec._handle_frame(FRAME)
        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnframes() == 2048  # 2 frames of 1024 samples each


def test_pause_discards_frames_until_resumed(tmp_path):
    with _fake_recorder() as rec:
        out_path = tmp_path / "call.wav"
        rec.start(out_path)

        rec._handle_frame(FRAME)  # written (1024 samples)

        rec.pause()
        assert rec.is_paused is True
        rec._handle_frame(FRAME)  # discarded
        rec._handle_frame(FRAME)  # discarded

        rec.resume()
        assert rec.is_paused is False
        rec._handle_frame(FRAME)  # written (1024 samples)

        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        # only the 2 un-paused frames should have made it to disk
        assert wf.getnframes() == 2048


def test_pause_and_resume_are_idempotent(tmp_path):
    with _fake_recorder() as rec:
        rec.start(tmp_path / "call.wav")

        rec.pause()
        rec.pause()  # calling twice shouldn't toggle back off
        assert rec.is_paused is True

        rec.resume()
        rec.resume()  # calling twice shouldn't error
        assert rec.is_paused is False

        rec.stop()


def test_stop_works_while_paused(tmp_path):
    with _fake_recorder() as rec:
        out_path = tmp_path / "call.wav"
        rec.start(out_path)

        rec._handle_frame(FRAME)
        rec.pause()

        result_path = rec.stop()

    assert result_path == out_path
    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnframes() == 1024
