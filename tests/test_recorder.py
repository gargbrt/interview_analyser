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

import numpy as np

from interview_analyzer.recorder import SystemAudioRecorder, _mix_pcm16

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
def _fake_recorder(mic_available: bool = False, mic_open_fails: bool = False, include_microphone: bool = True):
    """`mic_available=False` (the default) matches the plain _FakePyAudio
    behavior every existing pause/resume/elapsed test relies on -- no
    get_default_input_device_info() means _get_microphone_device() raises
    AttributeError internally and gracefully skips the mic. The mic-
    specific tests below opt into mic_available=True explicitly."""
    class _ParamFakePyAudio(_FakePyAudio):
        def get_default_input_device_info(self):
            if not mic_available:
                raise OSError("no default input device")
            return {
                "name": "Microphone", "isLoopbackDevice": False,
                "maxInputChannels": 1, "defaultSampleRate": 16000.0, "index": 1,
            }

        def open(self, **kwargs):
            if mic_open_fails and kwargs.get("input_device_index") == 1:
                raise OSError("Access is denied (no microphone permission)")
            return _FakeStream()

    fake_module = MagicMock()
    fake_module.PyAudio = _ParamFakePyAudio
    fake_module.paInt16 = _FakePyAudio.paInt16
    fake_module.paWASAPI = _FakePyAudio.paWASAPI
    with patch("interview_analyzer.recorder.pyaudio", fake_module):
        yield SystemAudioRecorder(sample_rate=16000, channels=1, include_microphone=include_microphone)


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


def test_elapsed_seconds_tracks_written_frames_not_paused_time(tmp_path):
    with _fake_recorder() as rec:
        rec.start(tmp_path / "call.wav")
        assert rec.elapsed_seconds == 0.0

        rec._handle_frame(FRAME)  # 1024 samples @ 16000Hz = 0.064s
        assert rec.elapsed_seconds == 1024 / 16000

        rec.pause()
        rec._handle_frame(FRAME)  # discarded -- must not advance the clock
        rec._handle_frame(FRAME)
        assert rec.elapsed_seconds == 1024 / 16000

        rec.resume()
        rec._handle_frame(FRAME)
        assert rec.elapsed_seconds == 2048 / 16000

        rec.stop()


def test_elapsed_seconds_is_zero_before_start(tmp_path):
    with _fake_recorder() as rec:
        assert rec.elapsed_seconds == 0.0


# -- microphone capture (mixed into the loopback recording) ----------------

def test_is_capturing_microphone_true_when_available(tmp_path):
    with _fake_recorder(mic_available=True) as rec:
        assert rec.is_capturing_microphone is False  # not until start()
        rec.start(tmp_path / "call.wav")
        assert rec.is_capturing_microphone is True
        rec.stop()


def test_is_capturing_microphone_false_when_disabled_in_config(tmp_path):
    with _fake_recorder(mic_available=True, include_microphone=False) as rec:
        rec.start(tmp_path / "call.wav")
        assert rec.is_capturing_microphone is False
        rec.stop()


def test_is_capturing_microphone_false_when_no_default_microphone(tmp_path):
    with _fake_recorder(mic_available=False) as rec:
        rec.start(tmp_path / "call.wav")
        assert rec.is_capturing_microphone is False
        rec.stop()


def test_is_capturing_microphone_false_when_opening_it_fails(tmp_path):
    """e.g. Windows blocking microphone access for desktop apps -- must
    fall back to system-audio-only, not crash the whole recording."""
    with _fake_recorder(mic_available=True, mic_open_fails=True) as rec:
        rec.start(tmp_path / "call.wav")
        assert rec.is_capturing_microphone is False
        rec.stop()


def test_mix_pcm16_adds_samples_and_clips_on_overflow():
    a = np.array([100, -100, 32000], dtype=np.int16).tobytes()
    b = np.array([50, -50, 32000], dtype=np.int16).tobytes()

    mixed = np.frombuffer(_mix_pcm16(a, b), dtype=np.int16)

    assert mixed[0] == 150
    assert mixed[1] == -150
    assert mixed[2] == 32767  # 64000 clipped to int16 max, not wrapped around


def test_mix_pcm16_pads_shorter_chunk_with_silence():
    a = np.array([10, 20, 30], dtype=np.int16).tobytes()
    b = np.array([5], dtype=np.int16).tobytes()  # a partial/short read

    mixed = np.frombuffer(_mix_pcm16(a, b), dtype=np.int16)

    assert list(mixed) == [15, 20, 30]
