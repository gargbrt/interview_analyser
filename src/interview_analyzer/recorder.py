"""System-audio recorder.

Uses WASAPI loopback capture (via pyaudiowpatch) so it works identically
regardless of which conferencing app or browser is producing the audio —
no per-app integration needed. Captures the default output device (what
you HEAR -- the interviewer's side) and, unless disabled, also opens the
default microphone (what YOU say) and mixes the two into one track --
without the microphone, the recording only contains the other side of the
conversation, since loopback capture alone never includes your own voice.
"""
from __future__ import annotations

import logging
import pathlib
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


class SystemAudioRecorder:
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
                "pyaudiowpatch is required for recording and is Windows-only. "
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
