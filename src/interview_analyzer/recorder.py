"""System-audio recorder.

Platform-specific capture backend, chosen automatically by `SystemAudioRecorder()`:

  - **Windows**: WASAPI loopback capture via `pyaudiowpatch` -- captures the
    default output device (what you HEAR) directly, no per-app integration
    or extra setup needed.
  - **macOS**: macOS has no built-in loopback API, so this captures from a
    virtual audio device (e.g. BlackHole) via `sounddevice` -- the user
    installs that driver once and routes system output through it. See
    docs/macos_setup.md for the one-time setup this requires.

Both backends also open the default microphone (unless disabled). When it's
available, the microphone and system-audio (loopback) streams are kept on
**separate channels** of the output WAV -- left = you (microphone), right =
the other side of the call (loopback) -- rather than mixed into one, so
transcriber.py can label each speaker deterministically from which channel
a segment came from. This is far more reliable than acoustic diarization
(pyannote), and needs no extra setup (no Hugging Face token, no model
download). Falls back to mono (loopback only) when no microphone is
available, in which case transcriber.py falls back to pyannote if
configured.

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


def _downmix_to_mono(data: bytes, channels: int) -> bytes:
    """Downmixes interleaved 16-bit PCM audio with the given channel count
    to mono by averaging across channels -- a no-op when already mono.
    Used to normalize each source (mic, loopback) to one channel each
    *before* placing them on separate output channels, regardless of a
    device's native channel count (e.g. stereo speakers)."""
    if channels <= 1:
        return data
    arr = np.frombuffer(data, dtype=np.int16)
    usable = (len(arr) // channels) * channels  # drop a short trailing partial frame, if any
    arr = arr[:usable].astype(np.int32).reshape(-1, channels)
    mono = arr.mean(axis=1).astype(np.int16)
    return mono.tobytes()


def _interleave_stereo(left: bytes, right: bytes) -> bytes:
    """Interleaves two mono 16-bit PCM chunks into one stereo (L,R,L,R,...)
    chunk -- left=you (microphone), right=them (system audio) by
    convention in this app (see transcriber.py). If the chunks differ in
    length (one stream returned a partial read), the shorter one is
    zero-padded first."""
    arr_l = np.frombuffer(left, dtype=np.int16)
    arr_r = np.frombuffer(right, dtype=np.int16)
    if len(arr_l) != len(arr_r):
        n = max(len(arr_l), len(arr_r))
        arr_l = np.pad(arr_l, (0, n - len(arr_l)))
        arr_r = np.pad(arr_r, (0, n - len(arr_r)))
    stereo = np.empty(len(arr_l) * 2, dtype=np.int16)
    stereo[0::2] = arr_l
    stereo[1::2] = arr_r
    return stereo.tobytes()


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
        self._out_file = None  # the raw file object wave.Wave_file wraps -- see start()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._out_path: Optional[pathlib.Path] = None
        # the loopback device's actual capture rate/channels can differ
        # from the requested sample_rate/channels above; elapsed_seconds
        # needs the real rate, set once start() picks a device. Both mic
        # and loopback are downmixed to mono at this channel count before
        # being placed on separate output channels (see _record_loop).
        self._actual_sample_rate = sample_rate
        self._actual_channels = channels
        # the OUTPUT wav's channel count -- 1 (mono, loopback only) or 2
        # (stereo: left=mic/you, right=loopback/them) -- decided in start()
        # once we know whether a microphone actually opened
        self._output_channels = channels
        self._frames_written = 0  # only counts frames actually written (i.e. not while paused)
        self._recording_failed = False

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

        # speaker separation (see module docstring) needs the mic on its
        # own channel -- only possible once we know whether one actually
        # opened successfully, so this must come after _open_microphone_stream
        self._output_channels = 2 if self._mic_stream is not None else 1

        # Unbuffered (buffering=0), rather than wave.open(str(out_path))'s
        # default buffered file -- with small buffered writes (our chunks
        # are ~2-4KB), a separate reader can see stale/empty data for a
        # long time (up to the ~8KB default buffer filling), which
        # live_transcribe.py -- reading this same file from another thread
        # while it's still being written -- depends on not happening.
        # Negligible cost here: this app's audio bitrate is a few tens of
        # KB/s, far below what unbuffered writes matter for.
        self._out_file = open(str(out_path), "wb", buffering=0)
        self._wav_file = wave.open(self._out_file, "wb")
        self._wav_file.setnchannels(self._output_channels)
        self._wav_file.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        self._wav_file.setframerate(self._actual_sample_rate)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info("Recording started -> %s", out_path)

    def _open_microphone_stream(self) -> None:
        mic_device = self._get_microphone_device()
        if mic_device is None:
            return
        try:
            # request the SAME rate/channels as the loopback stream --
            # WASAPI's shared-mode engine resamples transparently, and it's
            # what lets the two streams be downmixed/interleaved
            # sample-for-sample below
            self._mic_stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._actual_channels,
                rate=self._actual_sample_rate,
                input=True,
                input_device_index=mic_device["index"],
            )
            logger.info(
                "Also capturing microphone (%s) on its own channel, separate from "
                "system audio, so the transcript can label who said what.",
                mic_device.get("name"),
            )
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
                # A dead loopback stream is fatal to this recording (unlike
                # a dropped mic, there's nothing to fall back to) -- flag it
                # so the watcher notices and finalizes the interview with
                # whatever was captured so far, instead of it silently
                # sitting "recording" forever (see recording_failed).
                self._recording_failed = True
                break
            loopback_mono = _downmix_to_mono(data, self._actual_channels)

            if self._output_channels == 1:
                self._handle_frame(loopback_mono)
                continue

            # stereo output: left = mic ("you"), right = loopback ("them")
            mic_mono = None
            if self._mic_stream is not None:
                try:
                    mic_data = self._mic_stream.read(1024, exception_on_overflow=False)
                    mic_mono = _downmix_to_mono(mic_data, self._actual_channels)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Microphone read error; continuing with silence on your channel: %s", e)
                    self._mic_stream = None
            if mic_mono is None:
                # the mic dropped mid-recording (or was never opened this
                # tick) -- the WAV header is already committed to stereo
                # for this whole file, so keep writing silence on the mic
                # channel rather than collapsing to mono, which would
                # corrupt the frame layout for everything already written
                mic_mono = b"\x00\x00" * (len(loopback_mono) // 2)

            self._handle_frame(_interleave_stereo(mic_mono, loopback_mono))

    def _handle_frame(self, data: bytes) -> None:
        """Write one chunk of captured audio, unless paused. Split out from
        `_record_loop` so pause/resume behavior is directly unit-testable
        without needing a real background thread/audio device."""
        if not self._pause_event.is_set():
            self._wav_file.writeframes(data)
            self._frames_written += len(data) // (2 * max(self._output_channels, 1))

    @property
    def elapsed_seconds(self) -> float:
        """Duration of audio actually captured so far (excludes time spent
        paused), derived from frames written rather than a separate wall
        clock so it can never drift from what's actually on disk."""
        if not self._actual_sample_rate:
            return 0.0
        return self._frames_written / self._actual_sample_rate

    @property
    def frames_written(self) -> int:
        """How many frames have actually been written to the WAV file on
        disk so far (excludes paused time -- same counter elapsed_seconds
        is derived from). Used by live_transcribe.py to know how much of
        the file is safe to read without racing the writer: `writeframes()`
        happens-before this counter increments (see _handle_frame), so any
        value read here is guaranteed to already be flushed to the wave
        writer, whose header-patching keeps the on-disk file's declared
        length in sync with it on every write."""
        return self._frames_written

    @property
    def actual_sample_rate(self) -> int:
        """The real capture sample rate in use (the loopback device's
        native rate, which can differ from the sample_rate requested at
        construction -- see start()). 0 before start() is called."""
        return self._actual_sample_rate

    @property
    def is_capturing_microphone(self) -> bool:
        """True once recording has actually started with a working
        microphone stream (False before start(), or if none was available/
        openable and it fell back to system-audio-only)."""
        return self._mic_stream is not None

    @property
    def recording_failed(self) -> bool:
        """True if the recording loop exited because of a real capture
        error (e.g. the loopback device was reset, disconnected, or a
        driver crash disrupted the audio session) rather than a normal
        stop() call. Callers (see watcher.py's _tick) use this to finalize
        the interview immediately instead of waiting forever for
        meeting-absence detection, which might never come if whatever
        killed the stream didn't also affect the meeting app itself."""
        return self._recording_failed

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
        if self._out_file:
            # wave.Wave_write.close() only closes the underlying file
            # itself if IT opened it -- since start() passes an
            # already-open file object (see start()'s comment), that
            # object must be closed here too
            self._out_file.close()
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
        self._out_file = None  # the raw file object wave.Wave_file wraps -- see start()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._out_path: Optional[pathlib.Path] = None
        self._actual_sample_rate = sample_rate
        self._actual_channels = channels
        # the OUTPUT wav's channel count -- 1 (mono, loopback only) or 2
        # (stereo: left=mic/you, right=loopback/them) -- decided in start()
        # once we know whether a microphone actually opened
        self._output_channels = channels
        self._frames_written = 0
        self._recording_failed = False

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

        # speaker separation (see module docstring) needs the mic on its
        # own channel -- only possible once we know whether one actually
        # opened successfully, so this must come after _open_microphone_stream
        self._output_channels = 2 if self._mic_stream is not None else 1

        # see the matching comment in _WindowsAudioRecorder.start() -- kept
        # unbuffered so live_transcribe.py can reliably read this file
        # from another thread while it's still being written.
        self._out_file = open(str(out_path), "wb", buffering=0)
        self._wav_file = wave.open(self._out_file, "wb")
        self._wav_file.setnchannels(self._output_channels)
        self._wav_file.setsampwidth(2)  # 16-bit PCM
        self._wav_file.setframerate(self._actual_sample_rate)

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
            # two streams can be downmixed/interleaved sample-for-sample
            # below -- mirrors the Windows backend's approach
            self._mic_stream = sd.InputStream(
                device=mic_device["index"],
                channels=self._actual_channels,
                samplerate=self._actual_sample_rate,
                dtype="int16",
                blocksize=1024,
            )
            self._mic_stream.start()
            logger.info(
                "Also capturing microphone (%s) on its own channel, separate from "
                "system audio, so the transcript can label who said what.",
                mic_device.get("name"),
            )
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
                # see the matching comment in _WindowsAudioRecorder._record_loop
                self._recording_failed = True
                break
            loopback_mono = _downmix_to_mono(data, self._actual_channels)

            if self._output_channels == 1:
                self._handle_frame(loopback_mono)
                continue

            # stereo output: left = mic ("you"), right = loopback ("them")
            mic_mono = None
            if self._mic_stream is not None:
                try:
                    mic_data, _overflowed = self._mic_stream.read(1024)
                    mic_mono = _downmix_to_mono(mic_data.tobytes(), self._actual_channels)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Microphone read error; continuing with silence on your channel: %s", e)
                    self._mic_stream = None
            if mic_mono is None:
                # the mic dropped mid-recording (or was never opened this
                # tick) -- the WAV header is already committed to stereo
                # for this whole file, so keep writing silence on the mic
                # channel rather than collapsing to mono, which would
                # corrupt the frame layout for everything already written
                mic_mono = b"\x00\x00" * (len(loopback_mono) // 2)

            self._handle_frame(_interleave_stereo(mic_mono, loopback_mono))

    def _handle_frame(self, data: bytes) -> None:
        if not self._pause_event.is_set():
            self._wav_file.writeframes(data)
            self._frames_written += len(data) // (2 * max(self._output_channels, 1))

    @property
    def elapsed_seconds(self) -> float:
        if not self._actual_sample_rate:
            return 0.0
        return self._frames_written / self._actual_sample_rate

    @property
    def frames_written(self) -> int:
        """See _WindowsAudioRecorder.frames_written's docstring -- same
        contract, used by live_transcribe.py to know how much of the WAV
        file on disk is safe to read without racing the writer."""
        return self._frames_written

    @property
    def actual_sample_rate(self) -> int:
        return self._actual_sample_rate

    @property
    def is_capturing_microphone(self) -> bool:
        return self._mic_stream is not None

    @property
    def recording_failed(self) -> bool:
        """See the matching property on _WindowsAudioRecorder."""
        return self._recording_failed

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
        if self._out_file:
            # see the matching comment in _WindowsAudioRecorder.stop()
            self._out_file.close()
        logger.info("Recording stopped -> %s", self._out_path)
        return self._out_path
