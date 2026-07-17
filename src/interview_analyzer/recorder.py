"""System-audio recorder.

Platform-specific capture backend, chosen automatically by `SystemAudioRecorder()`:

  - **Windows**: WASAPI loopback capture via `pyaudiowpatch` -- captures the
    default output device (what you HEAR) directly, no per-app integration
    or extra setup needed.
  - **macOS**: macOS has no built-in loopback API, so this captures from a
    virtual audio device (e.g. BlackHole) via `sounddevice` -- the user
    installs that driver once and routes system output through it. See
    docs/macos_setup.md for the one-time setup this requires.

Both backends also open the default microphone (unless disabled) and mix
it in, so your own voice is in the transcript too -- loopback/virtual-
device capture alone only ever gets the other side of the conversation.

`_WindowsAudioRecorder` and `_MacAudioRecorder` are two independent
implementations (not a shared base class) so that porting/changing the
macOS backend can never accidentally change Windows behavior -- the
Windows class here is unchanged from before this file supported macOS.
"""
from __future__ import annotations

import logging
import pathlib
import sys
import threading
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# pyaudiowpatch is Windows-only; imported lazily so the module can still be
# imported (e.g. for tests) on non-Windows platforms.
try:
    import pyaudiowpatch as pyaudio
except ImportError:  # pragma: no cover
    pyaudio = None

# sounddevice is cross-platform but only actually used on macOS here (see
# SystemAudioRecorder() below) -- also imported lazily so this module still
# imports fine on Windows without it installed.
try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None


def _mix_pcm16(a: bytes, b: bytes) -> bytes:
    """Mix two 16-bit PCM chunks by sample-wise addition, clipped to avoid
    wraparound distortion. If the chunks differ in length (one device
    returned a partial read), the shorter one is zero-padded first."""
    arr_a = np.frombuffer(a, dtype=np.int16)
    arr_b = np.frombuffer(b, dtype=np.int16)
    if len(arr_a) != len(arr_b):
        n = max(len(arr_a), len(arr_b))
        arr_a = np.pad(arr_a, (0, n - len(arr_a)))
        arr_b = np.pad(arr_b, (0, n - len(arr_b)))
    mixed = arr_a.astype(np.int32) + arr_b.astype(np.int32)
    mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
    return mixed.tobytes()


def SystemAudioRecorder(sample_rate: int = 16000, channels: int = 1, include_microphone: bool = True):
    """Returns the platform-appropriate recorder backend. Both backends
    expose the identical interface (start/pause/resume/stop/elapsed_seconds/
    is_paused/is_capturing_microphone), so callers (watcher.py) never need
    to branch on platform themselves -- this is the only place that does.

    A function rather than a class so the two backends can be completely
    independent implementations (see module docstring) while callers still
    just do `SystemAudioRecorder(...)` like a constructor.
    """
    if sys.platform == "win32":
        return _WindowsAudioRecorder(sample_rate, channels, include_microphone)
    if sys.platform == "darwin":
        return _MacAudioRecorder(sample_rate, channels, include_microphone)
    raise RuntimeError(
        f"Unsupported platform for audio recording: {sys.platform!r}. "
        "Interview Analyzer supports Windows and macOS."
    )


class _WindowsAudioRecorder:
    """Records WASAPI loopback audio (+ the microphone, mixed in, unless
    disabled) to a WAV file in a background thread.

    A WAV is used as the intermediate capture format for reliability; the
    orchestrator compresses it to opus/mp3 immediately after stopping (see
    `compress.py`) so the on-disk footprint stays small during the retention
    window.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1, include_microphone: bool = True):
        if pyaudio is None:
            raise RuntimeError(
                "pyaudiowpatch is required for recording on Windows. "
                "Install with `pip install pyaudiowpatch`."
            )
        self.sample_rate = sample_rate
        self.channels = channels
        self.include_microphone = include_microphone
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._mic_stream = None
        self._wav_file: Optional[wave.Wave_write] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._out_path: Optional[pathlib.Path] = None
        # the device's actual capture rate/channels can differ from the
        # requested sample_rate/channels above; elapsed_seconds needs the
        # real ones, set once start() picks a device
        self._actual_sample_rate = sample_rate
        self._actual_channels = channels
        self._frames_written = 0  # only counts frames actually written (i.e. not while paused)

    def _get_loopback_device(self):
        wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = self._pa.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )
        if not default_speakers.get("isLoopbackDevice", False):
            for device in self._pa.get_loopback_device_info_generator():
                if default_speakers["name"] in device["name"]:
                    return device
        return default_speakers

    def _get_microphone_device(self):
        """Return the default microphone (a real input device, not a
        loopback/"stereo mix" device), or None if none is available."""
        try:
            info = self._pa.get_default_input_device_info()
        except Exception as e:  # noqa: BLE001
            logger.info("No default microphone found (%s); recording system audio only.", e)
            return None
        if info.get("isLoopbackDevice"):
            return None
        return info

    def start(self, out_path: pathlib.Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._out_path = out_path
        device = self._get_loopback_device()

        self._actual_channels = int(device["maxInputChannels"]) or self.channels
        self._actual_sample_rate = int(device["defaultSampleRate"])
        self._frames_written = 0

        self._wav_file = wave.open(str(out_path), "wb")
        self._wav_file.setnchannels(self._actual_channels)
        self._wav_file.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        self._wav_file.setframerate(self._actual_sample_rate)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._actual_channels,
            rate=self._actual_sample_rate,
            input=True,
            input_device_index=device["index"],
        )

        self._mic_stream = None
        if self.include_microphone:
            self._open_microphone_stream()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info("Recording started -> %s", out_path)

    def _open_microphone_stream(self) -> None:
        mic_device = self._get_microphone_device()
        if mic_device is None:
            return
        try:
            # request the SAME rate/channels as the loopback stream rather
            # than the mic's own native format -- WASAPI's shared-mode
            # engine resamples transparently, and it's what lets the two
            # streams be mixed sample-for-sample below
            self._mic_stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._actual_channels,
                rate=self._actual_sample_rate,
                input=True,
                input_device_index=mic_device["index"],
            )
            logger.info("Also capturing microphone (%s), mixed into the recording.", mic_device.get("name"))
        except Exception as e:  # noqa: BLE001
            self._mic_stream = None
            logger.warning(
                "Couldn't open the microphone (%s) -- recording system audio only, so "
                "your own voice will likely be missing from the transcript. This is "
                "often a Windows privacy setting: Settings > Privacy & security > "
                "Microphone > \"Let desktop apps access your microphone\" must be on.",
                e,
            )

    def _record_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                # Always drain the stream, even while paused, so the input
                # buffer doesn't back up and overflow -- paused audio is
                # simply discarded rather than written to the WAV file.
                data = self._stream.read(1024, exception_on_overflow=False)
            except Exception as e:  # noqa: BLE001
                logger.warning("Recording read error: %s", e)
                break

            if self._mic_stream is not None:
                try:
                    mic_data = self._mic_stream.read(1024, exception_on_overflow=False)
                    data = _mix_pcm16(data, mic_data)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Microphone read error; continuing with system audio only: %s", e)
                    self._mic_stream = None

            self._handle_frame(data)

    def _handle_frame(self, data: bytes) -> None:
        """Write one chunk of captured audio, unless paused. Split out from
        `_record_loop` so pause/resume behavior is directly unit-testable
        without needing a real background thread/audio device."""
        if not self._pause_event.is_set():
            self._wav_file.writeframes(data)
            self._frames_written += len(data) // (2 * max(self._actual_channels, 1))

    @property
    def elapsed_seconds(self) -> float:
        """Duration of audio actually captured so far (excludes time spent
        paused), derived from frames written rather than a separate wall
        clock so it can never drift from what's actually on disk."""
        if not self._actual_sample_rate:
            return 0.0
        return self._frames_written / self._actual_sample_rate

    @property
    def is_capturing_microphone(self) -> bool:
        """True once recording has actually started with a working
        microphone stream (False before start(), or if none was available/
        openable and it fell back to system-audio-only)."""
        return self._mic_stream is not None

    def pause(self) -> None:
        """Stop writing captured audio to disk until `resume()` is called.

        The underlying stream(s) keep running (so we don't have to reopen
        the device on resume); frames read while paused are simply
        discarded.
        """
        if not self._pause_event.is_set():
            self._pause_event.set()
            logger.info("Recording paused -> %s", self._out_path)

    def resume(self) -> None:
        if self._pause_event.is_set():
            self._pause_event.clear()
            logger.info("Recording resumed -> %s", self._out_path)

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def stop(self) -> Optional[pathlib.Path]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._mic_stream:
            self._mic_stream.stop_stream()
            self._mic_stream.close()
        if self._wav_file:
            self._wav_file.close()
        logger.info("Recording stopped -> %s", self._out_path)
        return self._out_path

    def __del__(self):
        try:
            self._pa.terminate()
        except Exception:  # noqa: BLE001
            pass


class _MacAudioRecorder:
    """Records from a virtual loopback audio device (e.g. BlackHole) via
    `sounddevice`, plus the microphone (mixed in, unless disabled), to a
    WAV file in a background thread.

    macOS has no equivalent of Windows' WASAPI loopback mode -- there's no
    way to capture "whatever's currently playing" without the user routing
    system audio through a virtual device first. `start()` raises a clear,
    actionable RuntimeError if no such device is found rather than silently
    recording nothing. See docs/macos_setup.md for the one-time setup.
    """

    # Free/common virtual-loopback driver names to look for, in query
    # order -- BlackHole is the standard free choice; the others are
    # included since some users may already have one installed for other
    # reasons.
    _LOOPBACK_NAME_HINTS = ("BlackHole", "Loopback", "Soundflower")

    def __init__(self, sample_rate: int = 16000, channels: int = 1, include_microphone: bool = True):
        if sd is None:
            raise RuntimeError(
                "sounddevice is required for recording on macOS. "
                "Install with `pip install sounddevice`."
            )
        self.sample_rate = sample_rate
        self.channels = channels
        self.include_microphone = include_microphone
        self._stream = None
        self._mic_stream = None
        self._wav_file: Optional[wave.Wave_write] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._out_path: Optional[pathlib.Path] = None
        self._actual_sample_rate = sample_rate
        self._actual_channels = channels
        self._frames_written = 0

    def _find_loopback_device(self) -> Optional[dict]:
        for index, device in enumerate(sd.query_devices()):
            if device.get("max_input_channels", 0) <= 0:
                continue
            name = device.get("name", "")
            if any(hint in name for hint in self._LOOPBACK_NAME_HINTS):
                return {**device, "index": index}
        return None

    def _get_microphone_device(self) -> Optional[dict]:
        try:
            index = sd.default.device[0]
            if index is None or index < 0:
                return None
            device = sd.query_devices(index)
        except Exception as e:  # noqa: BLE001
            logger.info("No default microphone found (%s); recording system audio only.", e)
            return None
        if device.get("max_input_channels", 0) <= 0:
            return None
        return {**device, "index": index}

    def start(self, out_path: pathlib.Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._out_path = out_path

        device = self._find_loopback_device()
        if device is None:
            raise RuntimeError(
                "No virtual loopback audio device found (e.g. BlackHole). macOS has no "
                "built-in way to capture system audio -- install a free virtual audio "
                "driver and set it as your audio output (or in a Multi-Output Device) "
                "first. See docs/macos_setup.md for one-time setup steps."
            )

        self._actual_channels = min(int(device.get("max_input_channels") or self.channels), 2) or self.channels
        self._actual_sample_rate = int(device.get("default_samplerate") or self.sample_rate)
        self._frames_written = 0

        self._wav_file = wave.open(str(out_path), "wb")
        self._wav_file.setnchannels(self._actual_channels)
        self._wav_file.setsampwidth(2)  # 16-bit PCM
        self._wav_file.setframerate(self._actual_sample_rate)

        self._stream = sd.InputStream(
            device=device["index"],
            channels=self._actual_channels,
            samplerate=self._actual_sample_rate,
            dtype="int16",
            blocksize=1024,
        )
        self._stream.start()

        self._mic_stream = None
        if self.include_microphone:
            self._open_microphone_stream()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info("Recording started -> %s", out_path)

    def _open_microphone_stream(self) -> None:
        mic_device = self._get_microphone_device()
        if mic_device is None:
            return
        try:
            # request the SAME rate/channels as the loopback stream so the
            # two streams can be mixed sample-for-sample below -- mirrors
            # the Windows backend's approach
            self._mic_stream = sd.InputStream(
                device=mic_device["index"],
                channels=self._actual_channels,
                samplerate=self._actual_sample_rate,
                dtype="int16",
                blocksize=1024,
            )
            self._mic_stream.start()
            logger.info("Also capturing microphone (%s), mixed into the recording.", mic_device.get("name"))
        except Exception as e:  # noqa: BLE001
            self._mic_stream = None
            logger.warning(
                "Couldn't open the microphone (%s) -- recording system audio only, so "
                "your own voice will likely be missing from the transcript. Check "
                "System Settings > Privacy & Security > Microphone for this app.",
                e,
            )

    def _record_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                # Always drain the stream, even while paused, so the input
                # buffer doesn't back up -- paused audio is simply
                # discarded rather than written to the WAV file.
                data, _overflowed = self._stream.read(1024)
                data = data.tobytes()
            except Exception as e:  # noqa: BLE001
                logger.warning("Recording read error: %s", e)
                break

            if self._mic_stream is not None:
                try:
                    mic_data, _overflowed = self._mic_stream.read(1024)
                    data = _mix_pcm16(data, mic_data.tobytes())
                except Exception as e:  # noqa: BLE001
                    logger.warning("Microphone read error; continuing with system audio only: %s", e)
                    self._mic_stream = None

            self._handle_frame(data)

    def _handle_frame(self, data: bytes) -> None:
        if not self._pause_event.is_set():
            self._wav_file.writeframes(data)
            self._frames_written += len(data) // (2 * max(self._actual_channels, 1))

    @property
    def elapsed_seconds(self) -> float:
        if not self._actual_sample_rate:
            return 0.0
        return self._frames_written / self._actual_sample_rate

    @property
    def is_capturing_microphone(self) -> bool:
        return self._mic_stream is not None

    def pause(self) -> None:
        if not self._pause_event.is_set():
            self._pause_event.set()
            logger.info("Recording paused -> %s", self._out_path)

    def resume(self) -> None:
        if self._pause_event.is_set():
            self._pause_event.clear()
            logger.info("Recording resumed -> %s", self._out_path)

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def stop(self) -> Optional[pathlib.Path]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._mic_stream:
            self._mic_stream.stop()
            self._mic_stream.close()
        if self._wav_file:
            self._wav_file.close()
        logger.info("Recording stopped -> %s", self._out_path)
        return self._out_path
