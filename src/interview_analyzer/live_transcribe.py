"""Transcribes an in-progress recording incrementally, in the background,
*while the call is still being recorded* -- instead of waiting until the
call ends to start transcribing the whole thing, which is what made
processing feel slow for long interviews.

How it stays safe for the live recording (the hard constraint this module
is built around: recording must never be affected by this):

  - `recorder.py`'s capture loop is never touched. This module only ever
    *reads* the WAV file the recorder is writing (via a second, independent
    file handle) and reads `recorder.frames_written` (a plain int
    attribute) -- both non-mutating. Verified empirically that Python's
    `wave` module keeps the on-disk header correct after every
    `writeframes()` call (not just at close()), so a concurrent read handle
    always sees a valid, complete WAV up to whatever's been written so far.
  - A stress test of truly concurrent read-while-write did surface an
    occasional (~0.5% in an adversarial tight loop) transient EOFError from
    the reader -- never any error on the writer's side. This module treats
    any read/transcribe failure as "retry next cycle" (see
    `_process_range`), which self-heals in real usage since our polling
    cadence is minutes apart, not milliseconds.
  - Segment boundaries are driven by `recorder.frames_written` (audio
    actually captured, i.e. pause-aware) rather than wall-clock time, so a
    long pause just delays the next segment rather than producing a
    segment full of silence or corrupting anything -- see
    `_maybe_process_next_segment`.
  - Any failure anywhere in this module (model load, segment read,
    transcription) is caught, logged, and marks the worker `failed` --
    callers (watcher.py) fall back to the existing whole-file
    `transcriber.transcribe()` call in that case, so a bug or edge case
    here degrades to today's known-correct behavior rather than producing
    a broken transcript.
"""
from __future__ import annotations

import logging
import pathlib
import tempfile
import threading
import time
import wave
from typing import Optional

from .config_loader import Config
from .transcriber import load_whisper_model, transcribe

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_SECONDS = 300  # 5 minutes
DEFAULT_POLL_SECONDS = 20


def _extract_segment(source_path: pathlib.Path, start_frame: int, end_frame: int) -> pathlib.Path:
    """Reads frames [start_frame, end_frame) out of `source_path` (opened
    read-only, independent of whatever handle the recorder is still
    writing through) into a new standalone temp WAV file with the same
    format, and returns its path. Raises on any read failure (e.g. a rare
    transient race with the writer -- see module docstring) rather than
    returning a partial/corrupt file; callers should retry rather than
    treat that as fatal.
    """
    with wave.open(str(source_path), "rb") as src:
        n_channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        src.setpos(start_frame)
        data = src.readframes(end_frame - start_frame)

    fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="live_segment_")
    import os

    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    with wave.open(str(tmp_path), "wb") as dst:
        dst.setnchannels(n_channels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)
        dst.writeframes(data)
    return tmp_path


class LiveTranscriptionWorker:
    """Periodically transcribes newly-recorded audio in the background
    while a call is still being recorded, so that by the time the call
    ends, most of the transcript is already done -- only the final
    (usually short) partial segment is left to transcribe, via `finish()`.

    Purely a speed optimization: `finish()` returns None (instead of a
    transcript) on any failure, timeout, or if it's disabled, and callers
    are expected to fall back to the normal whole-file
    `transcriber.transcribe()` call in that case -- this worker never being
    the sole path to a correct transcript is what makes it safe to add.
    """

    def __init__(
        self,
        cfg: Config,
        wav_path: pathlib.Path,
        recorder,
        segment_seconds: Optional[float] = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ):
        self.cfg = cfg
        self._wav_path = wav_path
        self._recorder = recorder
        tcfg = cfg.transcription
        self._segment_seconds = segment_seconds or tcfg.get("live_segment_seconds", DEFAULT_SEGMENT_SECONDS)
        self._poll_seconds = poll_seconds
        self._transcript_parts: list[str] = []
        self._last_frame = 0
        self._failed = False
        self._model = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        # True only while finish() is processing the final segment -- there's
        # no "next cycle" to fall back on at that point (see _process_range),
        # so a read failure there must be treated as a hard failure rather
        # than something to silently retry later.
        self._finishing = False

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def _ensure_model_loaded(self) -> bool:
        if self._model is not None:
            return True
        try:
            self._model = load_whisper_model(self.cfg)
            return True
        except Exception:  # noqa: BLE001
            logger.warning("Live transcription: couldn't load the Whisper model; will fall back "
                            "to whole-file transcription after the call ends.", exc_info=True)
            self._failed = True
            return False

    def _run(self) -> None:
        # loads the model once, up front, in this background thread --
        # doubles as a "pre-warm" so it's already loaded (not a several-
        # second cold start) by the time finish()'s final segment runs
        if not self._ensure_model_loaded():
            return
        while not self._failed and not self._stop_event.wait(self._poll_seconds):
            self._maybe_process_next_segment()

    def _segment_frames(self) -> int:
        rate = self._recorder.actual_sample_rate or 0
        return int(self._segment_seconds * rate)

    def _maybe_process_next_segment(self) -> None:
        segment_frames = self._segment_frames()
        if segment_frames <= 0:
            return
        target = self._recorder.frames_written
        if target - self._last_frame < segment_frames:
            return
        self._process_range(self._last_frame, target)

    def _extract_segment_with_retries(
        self, start_frame: int, end_frame: int, attempts: int = 3, delay: float = 0.5,
    ) -> Optional[pathlib.Path]:
        """A handful of quick retries absorbs the occasional transient read
        race with the writer observed empirically (concurrent read-while-
        write against a `wave` file very occasionally raises EOFError --
        see the module docstring) -- real writes happen far more than
        `delay` apart, so a retry a moment later reliably sees a
        consistent file. Returns None only if every attempt failed."""
        for attempt in range(attempts):
            try:
                return _extract_segment(self._wav_path, start_frame, end_frame)
            except Exception:  # noqa: BLE001
                if attempt == attempts - 1:
                    return None
                time.sleep(delay)
        return None

    def _process_range(self, start_frame: int, end_frame: int) -> None:
        if end_frame <= start_frame or self._failed:
            return
        if self._model is None and not self._ensure_model_loaded():
            return
        # the final segment (see finish()) gets a bigger retry budget --
        # there's no later cycle to fall back on for it, unlike a periodic
        # mid-recording segment, so it's worth trying harder before giving
        # up and forcing the (always-safe, just slower) whole-file fallback
        if self._finishing:
            segment_path = self._extract_segment_with_retries(start_frame, end_frame, attempts=6, delay=1.0)
        else:
            segment_path = self._extract_segment_with_retries(start_frame, end_frame)
        if segment_path is None:
            # a persistent race with the writer, or the file briefly not
            # being in the expected state -- don't advance _last_frame, so
            # the next periodic cycle naturally retries this same range
            # plus whatever's accumulated since. finish() (no next cycle
            # to fall back on) instead treats this as a hard failure --
            # see its docstring.
            logger.info(
                "Live transcription: couldn't read segment [%s:%s) yet; will retry.",
                start_frame, end_frame,
            )
            if self._finishing:
                self._failed = True
            return
        try:
            text = transcribe(segment_path, self.cfg, model=self._model)
            if text.strip():
                self._transcript_parts.append(text)
            self._last_frame = end_frame
        except Exception:  # noqa: BLE001
            logger.warning(
                "Live transcription failed on segment [%s:%s); falling back to whole-file "
                "transcription after the call ends.", start_frame, end_frame, exc_info=True,
            )
            self._failed = True
        finally:
            segment_path.unlink(missing_ok=True)

    def finish(self, timeout: float = 120) -> Optional[str]:
        """Stops the periodic background loop, transcribes whatever's left
        (from the last processed frame up to the recorder's final frame
        count), and returns the full assembled transcript -- or None if
        live transcription failed/timed out/never got a model loaded, in
        which case the caller should fall back to a normal whole-file
        `transcribe()` call. Safe to call even if `start()` was never
        called (finish immediately with whatever's accumulated, i.e.
        nothing)."""
        self._stop_event.set()
        if self._started:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Live transcription worker didn't stop within %ss; falling back "
                                "to whole-file transcription.", timeout)
                return None
        if self._failed:
            return None
        self._finishing = True
        final_frames = self._recorder.frames_written
        self._process_range(self._last_frame, final_frames)
        if self._failed:
            return None
        # Hard safety net for "no missing bits": _process_range is only
        # ever supposed to advance _last_frame exactly up to the frame
        # count it was asked to cover, so this should always hold by
        # construction -- but given the whole point of this feature is
        # never silently shipping an incomplete transcript, don't trust
        # that reasoning alone. If any frame of the recording somehow
        # isn't accounted for, fail closed into the whole-file fallback
        # rather than return a transcript that might be missing audio.
        if self._last_frame != final_frames:
            logger.warning(
                "Live transcription coverage mismatch (processed up to frame %s, recording "
                "has %s) -- falling back to whole-file transcription to be safe.",
                self._last_frame, final_frames,
            )
            return None
        return "\n".join(part for part in self._transcript_parts if part.strip())
