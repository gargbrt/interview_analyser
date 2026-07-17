"""Tests for how MeetingWatcher wires live_transcribe.py's
LiveTranscriptionWorker into the record -> transcribe -> analyze -> report
pipeline: starting it alongside a recording, handing it off to the
background processing thread at stop time (without blocking
`_stop_and_process`'s own quick return), and using its result -- or
falling back to normal whole-file transcription -- in
`_run_analysis_pipeline`.

LiveTranscriptionWorker itself is mocked throughout (its own behavior is
covered in test_live_transcribe.py); these tests are about the
orchestration around it.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from interview_analyzer.config_loader import Config
from interview_analyzer.watcher import MeetingWatcher


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _test_config(tmp_path, live_during_recording: bool = True) -> Config:
    return Config(raw={
        "retention_days": 3,
        "poll_interval_seconds": 0.01,
        "start_debounce_polls": 1,
        "stop_debounce_polls": 1,
        "watched_processes": {
            "desktop_apps": ["Zoom.exe"], "browser_tab_keywords": [], "browser_processes": [],
        },
        "audio": {"sample_rate": 16000, "channels": 1, "bitrate_kbps": 64,
                   "format": "opus", "raw_dir": str(tmp_path / "audio")},
        "transcription": {"engine": "faster-whisper", "whisper_model": "tiny", "device": "cpu",
                           "diarization": False, "live_during_recording": live_during_recording},
        "analysis": {"engine": "ollama", "llm_model": "llama3.1:8b",
                     "ollama_host": "http://localhost:11434"},
        "storage": {"db_path": str(tmp_path / "interviews.db")},
        "output": {"output_dir": str(tmp_path / "output"), "reports_subdir": "reports",
                    "trends_filename": "trends.md"},
    })


FAKE_ANALYSIS = {"qa_pairs": [], "session_summary": {}}


class TestStartRecordingCreatesLiveWorker:
    def test_creates_and_starts_a_live_worker_when_enabled(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)

        with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
             patch("interview_analyzer.watcher.ask_consent", return_value=True), \
             patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
             patch("interview_analyzer.watcher.RecordingControlPanel"), \
             patch("interview_analyzer.watcher.LiveTranscriptionWorker") as MockWorker:
            watcher._tick()

        MockWorker.assert_called_once()
        called_cfg, called_wav_path, called_recorder = MockWorker.call_args.args
        assert called_cfg is watcher.cfg
        assert called_wav_path.suffix == ".wav"
        assert called_recorder is MockRecorder.return_value
        MockWorker.return_value.start.assert_called_once()
        assert watcher._live_worker is MockWorker.return_value

    def test_skips_the_live_worker_when_disabled_in_config(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path, live_during_recording=False), user_id=1)

        with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
             patch("interview_analyzer.watcher.ask_consent", return_value=True), \
             patch("interview_analyzer.watcher.SystemAudioRecorder"), \
             patch("interview_analyzer.watcher.RecordingControlPanel"), \
             patch("interview_analyzer.watcher.LiveTranscriptionWorker") as MockWorker:
            watcher._tick()

        MockWorker.assert_not_called()
        assert watcher._live_worker is None

    def test_skips_the_live_worker_when_transcription_engine_is_not_local(self, tmp_path):
        """Groq's hosted Whisper is already fast enough at the end of a
        call that incremental during-the-call transcription has nothing
        useful to save -- and would mean repeatedly uploading small
        segments to a free tier with its own rate limits for no benefit."""
        watcher = MeetingWatcher(_test_config(tmp_path, live_during_recording=True), user_id=1)
        watcher.cfg.raw["transcription"]["engine"] = "groq"

        with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
             patch("interview_analyzer.watcher.ask_consent", return_value=True), \
             patch("interview_analyzer.watcher.SystemAudioRecorder"), \
             patch("interview_analyzer.watcher.RecordingControlPanel"), \
             patch("interview_analyzer.watcher.LiveTranscriptionWorker") as MockWorker:
            watcher._tick()

        MockWorker.assert_not_called()
        assert watcher._live_worker is None

    def test_recording_still_starts_even_if_the_live_worker_fails_to_construct(self, tmp_path):
        """A bug or edge case in live transcription must never prevent the
        actual recording from starting -- this is the hard safety
        requirement the whole feature was built around."""
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)

        with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
             patch("interview_analyzer.watcher.ask_consent", return_value=True), \
             patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
             patch("interview_analyzer.watcher.RecordingControlPanel"), \
             patch("interview_analyzer.watcher.LiveTranscriptionWorker", side_effect=RuntimeError("boom")):
            watcher._tick()

        assert watcher._current_interview_id is not None
        MockRecorder.return_value.start.assert_called_once()
        assert watcher._live_worker is None


class TestStopAndProcessUsesLiveTranscript:
    def _start_a_recording(self, watcher, tmp_path, MockRecorder):
        fake_wav = tmp_path / "audio" / "fake.wav"
        fake_wav.parent.mkdir(parents=True, exist_ok=True)
        fake_wav.write_bytes(b"RIFF....WAVEfake")
        MockRecorder.return_value.stop.return_value = fake_wav
        watcher._tick()  # detected -> consent -> recording starts

    def test_uses_the_live_transcript_and_never_calls_whole_file_transcribe(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        fake_opus = tmp_path / "audio" / "fake.opus"

        with patch("interview_analyzer.watcher.detect_active_meeting",
                    side_effect=[("Zoom", True), None]), \
             patch("interview_analyzer.watcher.ask_consent", return_value=True), \
             patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
             patch("interview_analyzer.watcher.RecordingControlPanel"), \
             patch("interview_analyzer.watcher.LiveTranscriptionWorker") as MockWorker, \
             patch("interview_analyzer.watcher.compress_audio") as mock_compress, \
             patch("interview_analyzer.watcher.transcribe") as mock_transcribe, \
             patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):
            fake_opus.parent.mkdir(parents=True, exist_ok=True)
            fake_opus.write_bytes(b"fake")
            mock_compress.return_value = fake_opus
            MockWorker.return_value.finish.return_value = "[You] a live-transcribed call"

            self._start_a_recording(watcher, tmp_path, MockRecorder)
            interview_id = watcher._current_interview_id
            watcher._tick()  # meeting gone -> stop -> hand off to background processing

            assert _wait_until(lambda: not watcher.status["processing_jobs"])

        mock_transcribe.assert_not_called()
        record = watcher.db.get(interview_id)
        assert record.transcript == "[You] a live-transcribed call"

    def test_falls_back_to_whole_file_transcription_when_the_live_worker_has_nothing(self, tmp_path):
        """finish() returning None covers: disabled, failed, or timed out
        -- any of those must transparently fall back to today's original
        whole-file transcribe() call."""
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        fake_opus = tmp_path / "audio" / "fake.opus"

        with patch("interview_analyzer.watcher.detect_active_meeting",
                    side_effect=[("Zoom", True), None]), \
             patch("interview_analyzer.watcher.ask_consent", return_value=True), \
             patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
             patch("interview_analyzer.watcher.RecordingControlPanel"), \
             patch("interview_analyzer.watcher.LiveTranscriptionWorker") as MockWorker, \
             patch("interview_analyzer.watcher.compress_audio") as mock_compress, \
             patch("interview_analyzer.watcher.transcribe", return_value="[You] fell back") as mock_transcribe, \
             patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):
            fake_opus.parent.mkdir(parents=True, exist_ok=True)
            fake_opus.write_bytes(b"fake")
            mock_compress.return_value = fake_opus
            MockWorker.return_value.finish.return_value = None

            self._start_a_recording(watcher, tmp_path, MockRecorder)
            interview_id = watcher._current_interview_id
            watcher._tick()

            assert _wait_until(lambda: not watcher.status["processing_jobs"])

        mock_transcribe.assert_called_once()
        record = watcher.db.get(interview_id)
        assert record.transcript == "[You] fell back"

    def test_stop_and_process_returns_without_waiting_for_the_live_worker_to_finish(self, tmp_path):
        """finish() runs on the background processing thread, not
        synchronously inside _stop_and_process -- otherwise a slow final
        segment would block the watcher's polling loop (and a brand new
        recording starting), which _stop_and_process's own docstring rules
        out."""
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        finish_was_blocked = MagicMock()

        def _slow_finish():
            finish_was_blocked()
            return "[You] done"

        with patch("interview_analyzer.watcher.detect_active_meeting",
                    side_effect=[("Zoom", True), None]), \
             patch("interview_analyzer.watcher.ask_consent", return_value=True), \
             patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
             patch("interview_analyzer.watcher.RecordingControlPanel"), \
             patch("interview_analyzer.watcher.LiveTranscriptionWorker") as MockWorker, \
             patch("interview_analyzer.watcher.compress_audio") as mock_compress, \
             patch("interview_analyzer.watcher.transcribe"), \
             patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):
            fake_opus = tmp_path / "audio" / "fake.opus"
            fake_opus.parent.mkdir(parents=True, exist_ok=True)
            fake_opus.write_bytes(b"fake")
            mock_compress.return_value = fake_opus
            MockWorker.return_value.finish.side_effect = _slow_finish

            self._start_a_recording(watcher, tmp_path, MockRecorder)
            tick_start = time.monotonic()
            watcher._tick()  # must return almost immediately
            tick_duration = time.monotonic() - tick_start

            assert _wait_until(lambda: not watcher.status["processing_jobs"])

        assert tick_duration < 1.0
        finish_was_blocked.assert_called_once()  # it did run -- just not inline
