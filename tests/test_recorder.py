"""Tests for SystemAudioRecorder's pause/resume/stop behavior, and the
mic/loopback channel-separation logic used for speaker labeling (see
transcriber.py).

Real WASAPI loopback capture needs actual Windows audio hardware, so the
`pyaudiowpatch` PyAudio object is faked here -- everything downstream of
"a stream that yields frames" (pause/resume gating, WAV writing, channel
separation, stop/cleanup) is real code under test.
"""
from __future__ import annotations

import time
import wave
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np

from interview_analyzer.recorder import (
    SystemAudioRecorder,
    _MacAudioRecorder,
    _WindowsAudioRecorder,
    _downmix_to_mono,
    _interleave_stereo,
)

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


class _FakeReadableStream:
    """Unlike _FakeStream, actually yields real (fake) data -- used to
    exercise _record_loop's real channel-separation logic end-to-end,
    rather than bypassing it via direct _handle_frame() calls."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = iter(chunks)

    def read(self, n, exception_on_overflow=False):
        try:
            return next(self._chunks)
        except StopIteration:
            raise RuntimeError("fake stream exhausted -- stops _record_loop cleanly")

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
        # Constructs _WindowsAudioRecorder directly rather than going
        # through the SystemAudioRecorder() factory -- these tests are
        # specifically exercising the Windows/pyaudiowpatch-backed
        # implementation (see module docstring), which must stay testable
        # regardless of which OS actually runs the suite (CI now also runs
        # on macos-latest, where the factory would otherwise dispatch to
        # _MacAudioRecorder instead and these tests would test the wrong
        # backend entirely).
        yield _WindowsAudioRecorder(sample_rate=16000, channels=1, include_microphone=include_microphone)


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


def test_frames_written_and_actual_sample_rate_are_public_and_match_internal_state(tmp_path):
    """live_transcribe.py reads these two properties from another thread
    while a recording is in progress -- regression guard that they exist,
    are public, and track the same values elapsed_seconds is derived from."""
    with _fake_recorder() as rec:
        rec.start(tmp_path / "call.wav")
        assert rec.frames_written == 0
        assert rec.actual_sample_rate == 16000

        rec._handle_frame(FRAME)
        assert rec.frames_written == 1024
        assert rec.frames_written == rec._frames_written

        rec.stop()


def test_wav_file_is_opened_unbuffered_so_a_concurrent_reader_sees_writes_immediately(tmp_path):
    """Regression guard for a real bug found building live transcription:
    a buffered writer can leave a concurrent reader seeing stale/empty
    data for small writes (Python's `wave` module only patches the header
    in its internal buffer, which isn't visible to another file handle
    until that buffer is flushed). live_transcribe.py depends on every
    write being immediately visible on disk."""
    import wave as wave_module

    wav_path = tmp_path / "call.wav"
    with _fake_recorder() as rec:
        rec.start(wav_path)
        rec._handle_frame(FRAME)

        # a completely separate read handle, opened while the recorder's
        # writer is still open -- must see the frame just written, with no
        # explicit flush() call from the test
        with wave_module.open(str(wav_path), "rb") as r:
            assert r.getnframes() == 1024

        rec.stop()


# -- microphone capture (its own channel, separate from loopback) ----------

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


def test_output_is_mono_when_no_microphone(tmp_path):
    with _fake_recorder(mic_available=False) as rec:
        out_path = tmp_path / "call.wav"
        rec.start(out_path)
        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnchannels() == 1


def test_output_is_stereo_when_microphone_available(tmp_path):
    with _fake_recorder(mic_available=True) as rec:
        out_path = tmp_path / "call.wav"
        rec.start(out_path)
        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnchannels() == 2


# -- _downmix_to_mono / _interleave_stereo (pure functions) ----------------

def test_downmix_to_mono_is_a_no_op_when_already_mono():
    data = np.array([10, 20, 30], dtype=np.int16).tobytes()
    assert _downmix_to_mono(data, 1) == data


def test_downmix_to_mono_averages_stereo_channels():
    # interleaved L,R,L,R: (10,20) -> 15, (30,40) -> 35
    data = np.array([10, 20, 30, 40], dtype=np.int16).tobytes()
    result = np.frombuffer(_downmix_to_mono(data, 2), dtype=np.int16)
    assert list(result) == [15, 35]


def test_downmix_to_mono_drops_a_short_trailing_partial_frame():
    # 5 samples at "2 channels" -- the last sample has no pair, must not crash
    data = np.array([10, 20, 30, 40, 999], dtype=np.int16).tobytes()
    result = np.frombuffer(_downmix_to_mono(data, 2), dtype=np.int16)
    assert list(result) == [15, 35]


def test_interleave_stereo_puts_left_and_right_in_order():
    left = np.array([1, 2, 3], dtype=np.int16).tobytes()
    right = np.array([10, 20, 30], dtype=np.int16).tobytes()
    result = np.frombuffer(_interleave_stereo(left, right), dtype=np.int16)
    assert list(result) == [1, 10, 2, 20, 3, 30]


def test_interleave_stereo_pads_the_shorter_chunk_with_silence():
    left = np.array([1, 2, 3], dtype=np.int16).tobytes()
    right = np.array([10], dtype=np.int16).tobytes()  # a partial/short read
    result = np.frombuffer(_interleave_stereo(left, right), dtype=np.int16)
    assert list(result) == [1, 10, 2, 0, 3, 0]


# -- end-to-end channel separation via the real _record_loop ---------------
# Unlike the tests above (which drive _handle_frame() directly for
# deterministic pause/resume assertions), these exercise the actual
# background-thread read/downmix/interleave path, to verify the WAV file
# that comes out the other end genuinely has mic on the left channel and
# loopback on the right -- the property transcriber.py's speaker labeling
# depends on.

def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_record_loop_separates_mic_and_loopback_onto_left_and_right_channels(tmp_path):
    loopback_chunk = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()
    mic_chunk = np.array([10, 20, 30, 40], dtype=np.int16).tobytes()

    class _ParamFakePyAudio(_FakePyAudio):
        def get_default_input_device_info(self):
            return {
                "name": "Microphone", "isLoopbackDevice": False,
                "maxInputChannels": 1, "defaultSampleRate": 16000.0, "index": 1,
            }

        def open(self, **kwargs):
            if kwargs.get("input_device_index") == 1:
                return _FakeReadableStream([mic_chunk])
            return _FakeReadableStream([loopback_chunk])

    fake_module = MagicMock()
    fake_module.PyAudio = _ParamFakePyAudio
    fake_module.paInt16 = _FakePyAudio.paInt16
    fake_module.paWASAPI = _FakePyAudio.paWASAPI

    out_path = tmp_path / "call.wav"
    with patch("interview_analyzer.recorder.pyaudio", fake_module):
        rec = _WindowsAudioRecorder(sample_rate=16000, channels=1, include_microphone=True)
        rec.start(out_path)
        assert _wait_until(lambda: rec._frames_written > 0), "background thread never wrote a frame"
        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnchannels() == 2
        arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).reshape(-1, 2)
    assert list(arr[:, 0]) == [10, 20, 30, 40]    # left = mic = "you"
    assert list(arr[:, 1]) == [100, 200, 300, 400]  # right = loopback = "them"


def test_record_loop_downmixes_a_stereo_loopback_device_before_placing_it_on_its_channel(tmp_path):
    """Real hardware sometimes reports a native-stereo loopback device
    (e.g. stereo speakers) -- that must be downmixed to mono before being
    placed on the (single) loopback output channel, not passed through
    raw (which would silently corrupt the interleaving)."""
    # interleaved stereo loopback: (100,200)->150, (300,400)->350
    loopback_chunk = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()
    # the mic is opened at the SAME channel count as the loopback device
    # (2, here) -- see _open_microphone_stream -- so this is also 2
    # interleaved stereo frames: (10,10)->10, (20,20)->20
    mic_chunk = np.array([10, 10, 20, 20], dtype=np.int16).tobytes()

    class _ParamFakePyAudio(_FakePyAudio):
        def get_device_info_by_index(self, _idx):
            return {
                "name": "Speakers", "isLoopbackDevice": True,
                "maxInputChannels": 2, "defaultSampleRate": 16000.0, "index": 0,
            }

        def get_default_input_device_info(self):
            return {
                "name": "Microphone", "isLoopbackDevice": False,
                "maxInputChannels": 1, "defaultSampleRate": 16000.0, "index": 1,
            }

        def open(self, **kwargs):
            if kwargs.get("input_device_index") == 1:
                return _FakeReadableStream([mic_chunk])
            return _FakeReadableStream([loopback_chunk])

    fake_module = MagicMock()
    fake_module.PyAudio = _ParamFakePyAudio
    fake_module.paInt16 = _FakePyAudio.paInt16
    fake_module.paWASAPI = _FakePyAudio.paWASAPI

    out_path = tmp_path / "call.wav"
    with patch("interview_analyzer.recorder.pyaudio", fake_module):
        rec = _WindowsAudioRecorder(sample_rate=16000, channels=1, include_microphone=True)
        rec.start(out_path)
        assert _wait_until(lambda: rec._frames_written > 0), "background thread never wrote a frame"
        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).reshape(-1, 2)
    assert list(arr[:, 0]) == [10, 20]
    assert list(arr[:, 1]) == [150, 350]


def test_record_loop_fills_silence_on_mic_channel_if_mic_drops_mid_recording(tmp_path):
    """The WAV header is already committed to stereo once a mic opens
    successfully -- if it then fails mid-recording, subsequent frames must
    keep the file's channel layout intact (silence on the mic side) rather
    than corrupt it by writing mono frames into a stereo file."""
    loopback_chunks = [
        np.array([100, 200], dtype=np.int16).tobytes(),
        np.array([300, 400], dtype=np.int16).tobytes(),
    ]

    class _FailingMicStream:
        def read(self, n, exception_on_overflow=False):
            raise OSError("microphone disconnected")

        def stop_stream(self):
            pass

        def close(self):
            pass

    class _ParamFakePyAudio(_FakePyAudio):
        def get_default_input_device_info(self):
            return {
                "name": "Microphone", "isLoopbackDevice": False,
                "maxInputChannels": 1, "defaultSampleRate": 16000.0, "index": 1,
            }

        def open(self, **kwargs):
            if kwargs.get("input_device_index") == 1:
                return _FailingMicStream()
            return _FakeReadableStream(loopback_chunks)

    fake_module = MagicMock()
    fake_module.PyAudio = _ParamFakePyAudio
    fake_module.paInt16 = _FakePyAudio.paInt16
    fake_module.paWASAPI = _FakePyAudio.paWASAPI

    out_path = tmp_path / "call.wav"
    with patch("interview_analyzer.recorder.pyaudio", fake_module):
        rec = _WindowsAudioRecorder(sample_rate=16000, channels=1, include_microphone=True)
        rec.start(out_path)
        assert _wait_until(lambda: rec._frames_written >= 4), "background thread never wrote both frames"
        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnchannels() == 2  # stays stereo despite the mic dropping
        arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).reshape(-1, 2)
    assert list(arr[:, 0]) == [0, 0, 0, 0]  # silence on the mic channel throughout
    assert list(arr[:, 1]) == [100, 200, 300, 400]


# -- SystemAudioRecorder() platform dispatch --------------------------------
# Windows behavior (the only platform this has ever run on for real, and
# what every test above exercises via the real pyaudiowpatch-backed class)
# must never change based on macOS support existing -- these confirm the
# factory function routes to the right backend and nothing else.

class TestPlatformDispatch:
    def test_returns_windows_backend_on_win32(self):
        with patch("interview_analyzer.recorder.sys.platform", "win32"), \
             patch("interview_analyzer.recorder.pyaudio", MagicMock()):
            rec = SystemAudioRecorder()
        assert isinstance(rec, _WindowsAudioRecorder)

    def test_returns_mac_backend_on_darwin(self):
        with patch("interview_analyzer.recorder.sys.platform", "darwin"), \
             patch("interview_analyzer.recorder.sd", MagicMock()):
            rec = SystemAudioRecorder()
        assert isinstance(rec, _MacAudioRecorder)

    def test_raises_a_clear_error_on_an_unsupported_platform(self):
        with patch("interview_analyzer.recorder.sys.platform", "linux"):
            try:
                SystemAudioRecorder()
                assert False, "expected RuntimeError"
            except RuntimeError as e:
                assert "linux" in str(e)


# -- macOS backend (_MacAudioRecorder) --------------------------------------
# Real capture needs a real Mac with a virtual loopback device installed
# (see docs/macos_setup.md), so `sounddevice` is faked here the same way
# `pyaudiowpatch` is faked above for Windows -- everything downstream of "a
# stream that yields frames" is real code under test. This is the one area
# of this port that most needs real-hardware verification on an actual Mac
# (see PR/README notes) -- these tests cover the logic, not real audio I/O.

class _FakeSdStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def read(self, n):
        raise RuntimeError("fake stream produces no frames; tests drive frames directly")


class _FakeReadableSdStream:
    def __init__(self, chunks: list, **kwargs):
        self._chunks = iter(chunks)
        self.kwargs = kwargs

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def read(self, n):
        try:
            chunk = next(self._chunks)
        except StopIteration:
            raise RuntimeError("fake stream exhausted -- stops _record_loop cleanly")
        return np.frombuffer(chunk, dtype=np.int16), False


class _FakeSoundDevice:
    """Mimics the bits of the `sounddevice` module _MacAudioRecorder uses."""

    def __init__(self, devices=None, default_input_index=0, mic_open_fails=False, readable_chunks=None):
        self._devices = devices if devices is not None else [
            {"name": "BlackHole 2ch", "max_input_channels": 1, "max_output_channels": 2, "default_samplerate": 16000.0},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000.0},
        ]
        self.default = MagicMock()
        self.default.device = [default_input_index, 1]
        self._mic_open_fails = mic_open_fails
        # {device_index: [chunks]} -- if given, InputStream returns a
        # _FakeReadableSdStream that actually yields data for that device
        self._readable_chunks = readable_chunks or {}
        self.InputStream = self._make_input_stream

    def query_devices(self, index=None):
        if index is not None:
            return self._devices[index]
        return self._devices

    def _make_input_stream(self, **kwargs):
        if self._mic_open_fails and kwargs.get("device") == self.default.device[0]:
            raise OSError("Access denied (no microphone permission)")
        device_index = kwargs.get("device")
        if device_index in self._readable_chunks:
            return _FakeReadableSdStream(self._readable_chunks[device_index], **kwargs)
        return _FakeSdStream(**kwargs)


@contextmanager
def _fake_mac_recorder(mic_available: bool = False, mic_open_fails: bool = False, include_microphone: bool = True):
    devices = [
        {"name": "BlackHole 2ch", "max_input_channels": 1, "max_output_channels": 2, "default_samplerate": 16000.0},
    ]
    default_input_index = -1
    if mic_available:
        devices.append(
            {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000.0}
        )
        default_input_index = 1

    fake_sd = _FakeSoundDevice(devices=devices, default_input_index=default_input_index, mic_open_fails=mic_open_fails)
    with patch("interview_analyzer.recorder.sd", fake_sd):
        yield _MacAudioRecorder(sample_rate=16000, channels=1, include_microphone=include_microphone)


def test_mac_recorder_writes_frames_when_not_paused(tmp_path):
    with _fake_mac_recorder() as rec:
        out_path = tmp_path / "call.wav"
        rec.start(out_path)
        rec._handle_frame(FRAME)
        rec._handle_frame(FRAME)
        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnframes() == 2048


def test_mac_recorder_pause_resume(tmp_path):
    with _fake_mac_recorder() as rec:
        rec.start(tmp_path / "call.wav")
        rec.pause()
        assert rec.is_paused is True
        rec.resume()
        assert rec.is_paused is False
        rec.stop()


def test_mac_recorder_elapsed_seconds_tracks_written_frames(tmp_path):
    with _fake_mac_recorder() as rec:
        rec.start(tmp_path / "call.wav")
        rec._handle_frame(FRAME)
        assert rec.elapsed_seconds == 1024 / 16000
        rec.stop()


def test_mac_recorder_frames_written_and_actual_sample_rate_are_public(tmp_path):
    """See the matching Windows test -- live_transcribe.py needs these on
    both platforms."""
    with _fake_mac_recorder() as rec:
        rec.start(tmp_path / "call.wav")
        assert rec.frames_written == 0
        assert rec.actual_sample_rate == 16000

        rec._handle_frame(FRAME)
        assert rec.frames_written == 1024
        assert rec.frames_written == rec._frames_written

        rec.stop()


def test_mac_recorder_wav_file_is_opened_unbuffered(tmp_path):
    """See the matching Windows test's docstring -- same fix, same reason,
    applied to both recorder backends identically."""
    import wave as wave_module

    wav_path = tmp_path / "call.wav"
    with _fake_mac_recorder() as rec:
        rec.start(wav_path)
        rec._handle_frame(FRAME)

        with wave_module.open(str(wav_path), "rb") as r:
            assert r.getnframes() == 1024

        rec.stop()


def test_mac_recorder_raises_clear_error_when_no_loopback_device_found(tmp_path):
    fake_sd = _FakeSoundDevice(devices=[
        {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000.0},
    ])
    with patch("interview_analyzer.recorder.sd", fake_sd):
        rec = _MacAudioRecorder()
        try:
            rec.start(tmp_path / "call.wav")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "BlackHole" in str(e)
            assert "macos_setup.md" in str(e)


def test_mac_recorder_is_capturing_microphone_true_when_available(tmp_path):
    with _fake_mac_recorder(mic_available=True) as rec:
        assert rec.is_capturing_microphone is False
        rec.start(tmp_path / "call.wav")
        assert rec.is_capturing_microphone is True
        rec.stop()


def test_mac_recorder_is_capturing_microphone_false_when_disabled(tmp_path):
    with _fake_mac_recorder(mic_available=True, include_microphone=False) as rec:
        rec.start(tmp_path / "call.wav")
        assert rec.is_capturing_microphone is False
        rec.stop()


def test_mac_recorder_is_capturing_microphone_false_when_no_default_mic(tmp_path):
    with _fake_mac_recorder(mic_available=False) as rec:
        rec.start(tmp_path / "call.wav")
        assert rec.is_capturing_microphone is False
        rec.stop()


def test_mac_recorder_is_capturing_microphone_false_when_opening_it_fails(tmp_path):
    """e.g. macOS blocking microphone access in System Settings -- must
    fall back to system-audio-only, not crash the whole recording."""
    with _fake_mac_recorder(mic_available=True, mic_open_fails=True) as rec:
        rec.start(tmp_path / "call.wav")
        assert rec.is_capturing_microphone is False
        rec.stop()


def test_mac_recorder_requires_sounddevice_installed():
    with patch("interview_analyzer.recorder.sd", None):
        try:
            _MacAudioRecorder()
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "sounddevice" in str(e)


def test_mac_recorder_output_is_mono_when_no_microphone(tmp_path):
    with _fake_mac_recorder(mic_available=False) as rec:
        out_path = tmp_path / "call.wav"
        rec.start(out_path)
        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnchannels() == 1


def test_mac_recorder_output_is_stereo_when_microphone_available(tmp_path):
    with _fake_mac_recorder(mic_available=True) as rec:
        out_path = tmp_path / "call.wav"
        rec.start(out_path)
        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnchannels() == 2


def test_mac_recorder_separates_mic_and_loopback_onto_left_and_right_channels(tmp_path):
    loopback_chunk = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()
    mic_chunk = np.array([10, 20, 30, 40], dtype=np.int16).tobytes()

    fake_sd = _FakeSoundDevice(
        devices=[
            {"name": "BlackHole 2ch", "max_input_channels": 1, "max_output_channels": 2, "default_samplerate": 16000.0},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000.0},
        ],
        default_input_index=1,
        readable_chunks={0: [loopback_chunk], 1: [mic_chunk]},
    )

    out_path = tmp_path / "call.wav"
    with patch("interview_analyzer.recorder.sd", fake_sd):
        rec = _MacAudioRecorder(sample_rate=16000, channels=1, include_microphone=True)
        rec.start(out_path)
        assert _wait_until(lambda: rec._frames_written > 0), "background thread never wrote a frame"
        rec.stop()

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnchannels() == 2
        arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).reshape(-1, 2)
    assert list(arr[:, 0]) == [10, 20, 30, 40]
    assert list(arr[:, 1]) == [100, 200, 300, 400]
