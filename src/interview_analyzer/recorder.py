"""System-audio recorder.

Uses WASAPI loopback capture (via pyaudiowpatch) so it works identically
regardless of which conferencing app or browser is producing the audio —
no per-app integration needed. Captures the default output device
(what you hear) plus, optionally, the default microphone, mixed to one
track, encoded compactly since we only need speech-quality audio for
transcription (not archival quality).
"""
from __future__ import annotations

import logging
import pathlib
import threading
import wave
from typing import Optional

logger = logging.getLogger(__name__)

# pyaudiowpatch is Windows-only; imported lazily so the module can still be
# imported (e.g. for tests) on non-Windows platforms.
try:
    import pyaudiowpatch as pyaudio
except ImportError:  # pragma: no cover
    pyaudio = None


class SystemAudioRecorder:
    """Records WASAPI loopback audio to a WAV file in a background thread.

    A WAV is used as the intermediate capture format for reliability; the
    orchestrator compresses it to opus/mp3 immediately after stopping (see
    `compress.py`) so the on-disk footprint stays small during the retention
    window.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        if pyaudio is None:
            raise RuntimeError(
                "pyaudiowpatch is required for recording and is Windows-only. "
                "Install with `pip install pyaudiowpatch`."
            )
        self.sample_rate = sample_rate
        self.channels = channels
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._wav_file: Optional[wave.Wave_write] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._out_path: Optional[pathlib.Path] = None

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

    def start(self, out_path: pathlib.Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._out_path = out_path
        device = self._get_loopback_device()

        self._wav_file = wave.open(str(out_path), "wb")
        self._wav_file.setnchannels(int(device["maxInputChannels"]) or self.channels)
        self._wav_file.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        self._wav_file.setframerate(int(device["defaultSampleRate"]))

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=int(device["maxInputChannels"]),
            rate=int(device["defaultSampleRate"]),
            input=True,
            input_device_index=device["index"],
        )

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info("Recording started -> %s", out_path)

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
            self._handle_frame(data)

    def _handle_frame(self, data: bytes) -> None:
        """Write one chunk of captured audio, unless paused. Split out from
        `_record_loop` so pause/resume behavior is directly unit-testable
        without needing a real background thread/audio device."""
        if not self._pause_event.is_set():
            self._wav_file.writeframes(data)

    def pause(self) -> None:
        """Stop writing captured audio to disk until `resume()` is called.

        The underlying stream keeps running (so we don't have to reopen the
        device on resume); frames read while paused are simply discarded.
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
        if self._wav_file:
            self._wav_file.close()
        logger.info("Recording stopped -> %s", self._out_path)
        return self._out_path

    def __del__(self):
        try:
            self._pa.terminate()
        except Exception:  # noqa: BLE001
            pass
