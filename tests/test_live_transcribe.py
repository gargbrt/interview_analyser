"""Tests for live_transcribe.py: transcribing an in-progress recording in
periodic background segments, so most of the transcript is already done by
the time the call ends. Real WAV files are used for `_extract_segment`
(cheap, and it's the one part that touches real file I/O against a
still-recording file -- worth exercising for real rather than mocking wave
itself). `transcribe()`/`load_whisper_model()` are mocked throughout, same
"real faster-whisper is a manual-verification boundary" pattern as
test_transcriber.py.
"""
from __future__ import annotations

import time
import wave
from unittest.mock import MagicMock, patch

import numpy as np

from interview_analyzer.config_loader import Config
from interview_analyzer.live_transcribe import LiveTranscriptionWorker, _extract_segment


def _write_wav(path, n_frames: int, channels: int = 1, sample_rate: int = 16000) -> None:
    """Writes a mono/stereo WAV where sample values count up from 0, so an
    extracted range's content can be verified against exactly what should
    have been in it."""
    arr = np.arange(n_frames * channels, dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(arr.tobytes())


class _FakeRecorder:
    """Stands in for SystemAudioRecorder -- LiveTranscriptionWorker only
    ever reads these two attributes."""

    def __init__(self, actual_sample_rate: int = 16000):
        self.actual_sample_rate = actual_sample_rate
        self.frames_written = 0


def _config(**overrides) -> Config:
    transcription = {"whisper_model": "tiny", "device": "cpu", "live_segment_seconds": 10}
    transcription.update(overrides)
    return Config(raw={"transcription": transcription})


class TestExtractSegment:
    def test_extracts_exactly_the_requested_frame_range(self, tmp_path):
        wav_path = tmp_path / "call.wav"
        _write_wav(wav_path, n_frames=1000)

        segment_path = _extract_segment(wav_path, start_frame=100, end_frame=300)
        try:
            with wave.open(str(segment_path), "rb") as r:
                assert r.getnframes() == 200
                data = np.frombuffer(r.readframes(200), dtype=np.int16)
            assert data[0] == 100
            assert data[-1] == 299
        finally:
            segment_path.unlink(missing_ok=True)

    def test_preserves_channel_count_and_format(self, tmp_path):
        wav_path = tmp_path / "call.wav"
        _write_wav(wav_path, n_frames=500, channels=2)

        segment_path = _extract_segment(wav_path, start_frame=0, end_frame=100)
        try:
            with wave.open(str(segment_path), "rb") as r:
                assert r.getnchannels() == 2
                assert r.getsampwidth() == 2
                assert r.getframerate() == 16000
        finally:
            segment_path.unlink(missing_ok=True)

    def test_can_read_while_a_real_writer_still_has_the_file_open(self, tmp_path):
        """The core safety assumption this whole module rests on -- see the
        module docstring's empirical notes. Deliberately mirrors
        recorder.py's actual write setup (an unbuffered file object handed
        to wave.open), which is what makes this reliable -- a *buffered*
        writer was measured to leave a concurrent reader seeing zero bytes
        for small writes (recorder.py used to do this before it was
        changed for exactly this reason). LiveTranscriptionWorker still
        retries a couple of times regardless (see
        _extract_segment_with_retries) as a backstop for the rarer,
        genuinely-transient race the module docstring describes."""
        wav_path = tmp_path / "call.wav"
        w = wave.open(open(str(wav_path), "wb", buffering=0), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.arange(1000, dtype=np.int16).tobytes())

        segment_path = _extract_segment(wav_path, start_frame=0, end_frame=500)
        try:
            with wave.open(str(segment_path), "rb") as r:
                assert r.getnframes() == 500
        finally:
            segment_path.unlink(missing_ok=True)
            w.close()


class TestLiveTranscriptionWorkerSegmentProcessing:
    """Drives _maybe_process_next_segment/_process_range directly rather
    than through the real background thread's poll loop, for deterministic
    timing -- same approach test_recorder.py uses for _handle_frame."""

    def _worker(self, tmp_path, recorder=None, **cfg_overrides):
        wav_path = tmp_path / "call.wav"
        _write_wav(wav_path, n_frames=16000 * 30)  # 30s of audio on disk
        recorder = recorder or _FakeRecorder()
        worker = LiveTranscriptionWorker(_config(**cfg_overrides), wav_path, recorder)
        return worker, recorder

    def test_does_nothing_until_a_full_segment_has_accumulated(self, tmp_path):
        worker, recorder = self._worker(tmp_path)  # live_segment_seconds=10 -> 160000 frames
        worker._model = MagicMock()
        recorder.frames_written = 16000 * 5  # only 5s so far

        with patch("interview_analyzer.live_transcribe.transcribe") as mock_transcribe:
            worker._maybe_process_next_segment()

        mock_transcribe.assert_not_called()
        assert worker._last_frame == 0

    def test_processes_a_segment_once_the_threshold_is_crossed(self, tmp_path):
        worker, recorder = self._worker(tmp_path)
        worker._model = MagicMock()
        recorder.frames_written = 16000 * 12  # past the 10s threshold

        with patch("interview_analyzer.live_transcribe.transcribe", return_value="[You] hello") as mock_transcribe:
            worker._maybe_process_next_segment()

        mock_transcribe.assert_called_once()
        assert worker._last_frame == 16000 * 12
        assert worker._transcript_parts == ["[You] hello"]

    def test_pause_simply_delays_the_next_segment_since_frames_written_does_not_advance(self, tmp_path):
        """No pause-specific code exists in this module by design -- frame
        counts (not wall-clock time) drive segmenting, and frames_written
        already excludes paused time (see recorder.py). This test is the
        regression guard for that design holding."""
        worker, recorder = self._worker(tmp_path)
        worker._model = MagicMock()
        recorder.frames_written = 16000 * 8  # short of the 10s threshold, e.g. paused here

        with patch("interview_analyzer.live_transcribe.transcribe") as mock_transcribe:
            for _ in range(5):  # simulate several poll cycles while "paused"
                worker._maybe_process_next_segment()

        mock_transcribe.assert_not_called()

    def test_empty_or_silent_segment_is_not_appended_to_the_transcript(self, tmp_path):
        worker, recorder = self._worker(tmp_path)
        worker._model = MagicMock()
        recorder.frames_written = 16000 * 12

        with patch("interview_analyzer.live_transcribe.transcribe", return_value=""):
            worker._maybe_process_next_segment()

        assert worker._transcript_parts == []
        assert worker._last_frame == 16000 * 12  # still advances -- it just had nothing to say

    def test_extraction_failure_retries_a_few_times_then_gives_up_for_this_cycle(self, tmp_path):
        worker, recorder = self._worker(tmp_path)
        worker._model = MagicMock()
        recorder.frames_written = 16000 * 12

        with patch("interview_analyzer.live_transcribe._extract_segment", side_effect=OSError("busy")) as mock_extract, \
             patch("interview_analyzer.live_transcribe.time.sleep"):
            worker._maybe_process_next_segment()

        assert mock_extract.call_count == 3  # the configured retry budget
        assert worker._last_frame == 0
        assert worker._failed is False  # a read race is not treated as fatal mid-recording

    def test_extraction_succeeds_on_a_later_retry_attempt(self, tmp_path):
        worker, recorder = self._worker(tmp_path)
        worker._model = MagicMock()
        recorder.frames_written = 16000 * 12
        real_segment_path = _extract_segment(worker._wav_path, 0, 100)  # a valid, real temp segment

        with patch(
            "interview_analyzer.live_transcribe._extract_segment",
            side_effect=[OSError("busy"), real_segment_path],
        ), patch("interview_analyzer.live_transcribe.time.sleep"), \
             patch("interview_analyzer.live_transcribe.transcribe", return_value="[You] hi") as mock_transcribe:
            worker._maybe_process_next_segment()

        mock_transcribe.assert_called_once()
        assert worker._last_frame == 16000 * 12

    def test_transcription_failure_marks_the_worker_failed(self, tmp_path):
        worker, recorder = self._worker(tmp_path)
        worker._model = MagicMock()
        recorder.frames_written = 16000 * 12

        with patch("interview_analyzer.live_transcribe.transcribe", side_effect=RuntimeError("boom")):
            worker._maybe_process_next_segment()

        assert worker._failed is True

    def test_model_load_failure_marks_the_worker_failed_without_raising(self, tmp_path):
        worker, recorder = self._worker(tmp_path)
        recorder.frames_written = 16000 * 12

        with patch("interview_analyzer.live_transcribe.load_whisper_model", side_effect=RuntimeError("no model")):
            worker._maybe_process_next_segment()

        assert worker._failed is True


class TestLiveTranscriptionWorkerFinish:
    def _worker(self, tmp_path, total_seconds=30):
        wav_path = tmp_path / "call.wav"
        _write_wav(wav_path, n_frames=16000 * total_seconds)
        recorder = _FakeRecorder()
        recorder.frames_written = 16000 * total_seconds
        worker = LiveTranscriptionWorker(_config(), wav_path, recorder)
        return worker, recorder

    def test_finish_without_start_transcribes_the_whole_call_in_one_go(self, tmp_path):
        """start() was never called (e.g. live transcription only ever
        matters for calls long enough to cross one segment) -- finish()
        must still work, treating the whole file as one segment."""
        worker, recorder = self._worker(tmp_path)

        with patch("interview_analyzer.live_transcribe.load_whisper_model", return_value=MagicMock()), \
             patch("interview_analyzer.live_transcribe.transcribe", return_value="[You] the whole thing") as mock_transcribe:
            result = worker.finish(timeout=5)

        assert result == "[You] the whole thing"
        mock_transcribe.assert_called_once()

    def test_finish_after_some_live_segments_only_transcribes_the_remainder(self, tmp_path):
        worker, recorder = self._worker(tmp_path, total_seconds=30)
        worker._model = MagicMock()
        # simulate one segment already processed live, covering the first 20s
        worker._last_frame = 16000 * 20
        worker._transcript_parts = ["[Interviewer] first part"]

        with patch("interview_analyzer.live_transcribe.transcribe", return_value="[You] final bit") as mock_transcribe:
            result = worker.finish(timeout=5)

        # only the remaining 10s (frame 320000 to 480000) gets transcribed
        mock_transcribe.assert_called_once()
        assert result == "[Interviewer] first part\n[You] final bit"

    def test_finish_returns_none_if_any_segment_ever_failed(self, tmp_path):
        worker, recorder = self._worker(tmp_path)
        worker._model = MagicMock()
        worker._failed = True

        result = worker.finish(timeout=5)

        assert result is None

    def test_finish_returns_none_if_the_final_segment_cannot_be_read(self, tmp_path):
        """Unlike a periodic mid-recording read failure (which just waits
        for the next cycle), a failure on the final catch-up segment has
        no later cycle to fall back on -- it must count as a hard failure
        rather than silently dropping the tail of the transcript."""
        worker, recorder = self._worker(tmp_path)
        worker._model = MagicMock()
        worker._transcript_parts = ["[Interviewer] first part"]
        worker._last_frame = 16000 * 20

        with patch("interview_analyzer.live_transcribe._extract_segment", side_effect=OSError("busy")), \
             patch("interview_analyzer.live_transcribe.time.sleep"):
            result = worker.finish(timeout=5)

        assert result is None
        assert worker._failed is True

    def test_finish_is_safe_to_call_twice(self, tmp_path):
        worker, recorder = self._worker(tmp_path)
        with patch("interview_analyzer.live_transcribe.load_whisper_model", return_value=MagicMock()), \
             patch("interview_analyzer.live_transcribe.transcribe", return_value="[You] hi"):
            first = worker.finish(timeout=5)
            second = worker.finish(timeout=5)

        assert first == "[You] hi"
        assert second == "[You] hi"  # nothing new to transcribe the second time


class TestLiveTranscriptionWorkerEndToEnd:
    """Exercises the real background thread (start() + the poll loop),
    unlike the tests above which call the processing methods directly."""

    def test_a_running_worker_picks_up_a_segment_without_manual_driving(self, tmp_path):
        wav_path = tmp_path / "call.wav"
        _write_wav(wav_path, n_frames=16000 * 30)
        recorder = _FakeRecorder()
        recorder.frames_written = 16000 * 12  # already past the 10s threshold when it starts polling

        worker = LiveTranscriptionWorker(_config(), wav_path, recorder, poll_seconds=0.05)
        with patch("interview_analyzer.live_transcribe.load_whisper_model", return_value=MagicMock()), \
             patch("interview_analyzer.live_transcribe.transcribe", return_value="[You] hi"):
            worker.start()
            deadline = time.monotonic() + 3
            while worker._last_frame == 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            result = worker.finish(timeout=5)

        assert result == "[You] hi"

    def test_recorder_is_never_mutated_by_the_worker(self, tmp_path):
        """Regression guard for the hard constraint this feature was built
        under: live transcription must be read-only with respect to the
        recorder object."""
        wav_path = tmp_path / "call.wav"
        _write_wav(wav_path, n_frames=16000 * 15)
        recorder = _FakeRecorder()
        recorder.frames_written = 16000 * 15
        recorder_snapshot = dict(recorder.__dict__)

        worker = LiveTranscriptionWorker(_config(), wav_path, recorder, poll_seconds=0.05)
        with patch("interview_analyzer.live_transcribe.load_whisper_model", return_value=MagicMock()), \
             patch("interview_analyzer.live_transcribe.transcribe", return_value="[You] hi"):
            worker.start()
            time.sleep(0.2)
            worker.finish(timeout=5)

        assert recorder.__dict__ == recorder_snapshot
