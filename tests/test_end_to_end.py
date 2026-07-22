"""End-to-end test of the full pipeline: consent -> record -> stop ->
compress -> transcribe -> analyze -> report -> trend update -> cleanup.

Real audio capture (WASAPI loopback) and a real local Whisper/Ollama model
can't run in this environment (no Windows hardware, no mic, no GPU/Ollama
server here) -- those three boundaries are mocked. Everything else
(the DB, the consent gate, the file lifecycle, report generation, retention
cleanup, user scoping) runs for real, so this verifies the orchestration
logic end-to-end even though real-world audio capture still needs to be
verified by hand on a real Windows machine with a live call.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from unittest.mock import patch

from interview_analyzer.config_loader import Config
from interview_analyzer.db import InterviewDB
from interview_analyzer.infographic import infographic_path
from interview_analyzer.watcher import MeetingWatcher


def _wait_until(predicate, timeout=5.0):
    """Background processing now runs on its own thread (see watcher.py's
    _process_in_background), so tests must wait for a job to actually
    finish -- while any mocks it depends on are still active -- rather
    than asserting immediately after the tick that kicked it off returns."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _test_config(tmp_path) -> Config:
    return Config(raw={
        "retention_days": 3,
        "poll_interval_seconds": 0.01,
        "start_debounce_polls": 1,
        "stop_debounce_polls": 1,
        "watched_processes": {
            "desktop_apps": ["Zoom.exe"],
            "browser_tab_keywords": [],
            "browser_processes": [],
        },
        "audio": {
            "sample_rate": 16000, "channels": 1, "bitrate_kbps": 64,
            "format": "opus", "raw_dir": str(tmp_path / "audio"),
        },
        # live_during_recording is off here since these tests mock the
        # recorder itself -- live_transcribe.py needs a real (or at least
        # int-valued) frames_written/actual_sample_rate to run against; its
        # own behavior is covered separately in test_live_transcribe.py and
        # test_watcher_live_transcription.py
        "transcription": {"engine": "faster-whisper", "whisper_model": "tiny",
                           "device": "cpu", "diarization": False, "live_during_recording": False},
        "analysis": {"engine": "ollama", "llm_model": "llama3.1:8b",
                     "ollama_host": "http://localhost:11434"},
        "storage": {"db_path": str(tmp_path / "interviews.db")},
        "output": {"output_dir": str(tmp_path / "output"), "reports_subdir": "reports",
                    "trends_filename": "trends.md"},
    })


FAKE_TRANSCRIPT = (
    "[Interviewer] Tell me about a time you disagreed with a teammate.\n"
    "[You] Um, so, there was this one time, it was kind of a whole thing, "
    "and I guess we just eventually figured it out."
)

FAKE_ANALYSIS = {
    "qa_pairs": [{
        "question": "Tell me about a time you disagreed with a teammate.",
        "answer_summary": "Vague, no clear structure or resolution described.",
        "issues": [
            {"category": "structure", "detail": "No STAR structure."},
            {"category": "specificity", "detail": "No concrete outcome given."},
        ],
        "suggested_improvement": "State the situation, your specific action, and the measurable result.",
    }],
    "session_summary": {
        "top_strengths": ["Comfortable, conversational tone"],
        "top_issues": ["Lacks structure", "Lacks specificity/outcomes"],
        "one_thing_to_practice_next": "Rehearse 3 behavioral stories in STAR format.",
    },
}


def test_full_pipeline_end_to_end_with_consent_granted(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)

    # simulate: a fake WAV exists once "recording" stops, consent is granted,
    # transcription/analysis are mocked at their external boundaries only
    with patch("interview_analyzer.watcher.detect_active_meeting", side_effect=[("Zoom", True), None]), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True), \
         patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
         patch("interview_analyzer.watcher.RecordingControlPanel") as MockPanel, \
         patch("interview_analyzer.watcher.compress_audio") as mock_compress, \
         patch("interview_analyzer.watcher.transcribe", return_value=FAKE_TRANSCRIPT), \
         patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):

        fake_wav = tmp_path / "audio" / "fake.wav"
        fake_wav.parent.mkdir(parents=True, exist_ok=True)
        fake_wav.write_bytes(b"RIFF....WAVEfake")
        MockRecorder.return_value.stop.return_value = fake_wav

        fake_opus = tmp_path / "audio" / "fake.opus"
        fake_opus.write_bytes(b"fake compressed audio")
        mock_compress.return_value = fake_opus

        # tick 1: meeting detected -> consent asked -> recording "starts"
        watcher._tick()
        assert watcher._current_interview_id is not None
        interview_id = watcher._current_interview_id
        MockPanel.assert_called_once()  # pause/resume/stop control shown while recording

        # tick 2: meeting gone -> stop, hand off to a background thread for
        # transcribe/analyze/report/cleanup-eligible
        watcher._tick()
        MockPanel.return_value.close.assert_called_once()  # control panel torn down with the recording

        assert _wait_until(lambda: not watcher.status["processing_jobs"]), \
            "background processing never finished"

    # --- assertions on end state ---
    record = watcher.db.get(interview_id)
    assert record.ended_at is not None
    assert record.transcript == FAKE_TRANSCRIPT
    assert record.analysis["session_summary"]["top_issues"] == [
        "Lacks structure", "Lacks specificity/outcomes"
    ]
    assert record.report_path is not None

    report_path = tmp_path.joinpath(record.report_path) if not record.report_path.startswith("/") \
        else __import__("pathlib").Path(record.report_path)
    assert report_path.exists()
    content = report_path.read_text()
    assert "Tell me about a time you disagreed" in content
    assert "Rehearse 3 behavioral stories" in content

    # the HTML infographic is generated alongside the markdown report
    infographic = infographic_path(record, cfg)
    assert infographic.exists()
    infographic_content = infographic.read_text(encoding="utf-8")
    assert "Tell me about a time you disagreed" in infographic_content
    assert "Rehearse 3 behavioral stories" in infographic_content

    trends_path = tmp_path / "output" / "trends_user1.md"
    assert trends_path.exists()
    assert "Lacks structure" in trends_path.read_text()

    trends_infographic_path_ = tmp_path / "output" / "trends_user1_infographic.html"
    assert trends_infographic_path_.exists()
    assert "Lacks structure" in trends_infographic_path_.read_text(encoding="utf-8")


def test_full_pipeline_consent_declined_skips_recording(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
         patch("interview_analyzer.watcher.ask_consent", return_value=False) as mock_consent, \
         patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder:

        watcher._tick()
        # second tick while still "in the same call" should NOT re-prompt
        watcher._tick()

    assert watcher._current_interview_id is None
    MockRecorder.return_value.start.assert_not_called()
    assert mock_consent.call_count == 1  # only asked once, not once per poll
    assert watcher.db.list_all() == []  # nothing was ever recorded


def test_manual_stop_via_control_panel_ends_call_early_and_runs_pipeline(tmp_path):
    """The pause/resume/stop control panel's Stop button should end the
    recording (and run the full transcribe/analyze/report pipeline)
    immediately, without waiting for the meeting app to close."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True), \
         patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
         patch("interview_analyzer.watcher.RecordingControlPanel") as MockPanel, \
         patch("interview_analyzer.watcher.compress_audio") as mock_compress, \
         patch("interview_analyzer.watcher.transcribe", return_value=FAKE_TRANSCRIPT), \
         patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):

        fake_wav = tmp_path / "audio" / "fake.wav"
        fake_wav.parent.mkdir(parents=True, exist_ok=True)
        fake_wav.write_bytes(b"RIFF....WAVEfake")
        MockRecorder.return_value.stop.return_value = fake_wav

        fake_opus = tmp_path / "audio" / "fake.opus"
        fake_opus.write_bytes(b"fake compressed audio")
        mock_compress.return_value = fake_opus

        # tick 1: meeting detected -> consent asked -> recording "starts",
        # control panel constructed with pause/resume/stop callbacks
        watcher._tick()
        assert watcher._current_interview_id is not None
        interview_id = watcher._current_interview_id

        panel_kwargs = MockPanel.call_args.kwargs
        panel_kwargs["on_pause"]()
        MockRecorder.return_value.pause.assert_called_once()
        panel_kwargs["on_resume"]()
        MockRecorder.return_value.resume.assert_called_once()

        # user clicks Stop on the control panel mid-call (app is still
        # "running" -- this must not wait for detect_active_meeting to go
        # False)
        panel_kwargs["on_stop"]()
        assert watcher._manual_stop_requested.is_set()

        # next poll tick should notice the manual-stop flag and process
        # immediately, even though the meeting app is still detected
        watcher._tick()
        MockPanel.return_value.close.assert_called_once()

        assert _wait_until(lambda: not watcher.status["processing_jobs"]), \
            "background processing never finished"

    assert watcher._current_interview_id is None
    record = watcher.db.get(interview_id)
    assert record.ended_at is not None
    assert record.transcript == FAKE_TRANSCRIPT
    assert record.report_path is not None


def test_recording_failure_finalizes_the_interview_even_though_the_meeting_is_still_detected(tmp_path):
    """Regression coverage for a real bug: a live ~1-hour recording ended
    abruptly (the capture stream started raising errors -- e.g. a device
    reset/disconnect, or an unrelated driver crash disrupting the audio
    session) while the meeting app itself kept running, so
    detect_active_meeting() never went false and the interview sat
    "recording" forever with no transcript/analysis/report and no way to
    tell why. The watcher now polls the recorder's own recording_failed
    flag and finalizes immediately, regardless of whether the meeting
    still looks active."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True), \
         patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
         patch("interview_analyzer.watcher.RecordingControlPanel") as MockPanel, \
         patch("interview_analyzer.watcher.compress_audio") as mock_compress, \
         patch("interview_analyzer.watcher.transcribe", return_value=FAKE_TRANSCRIPT), \
         patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):

        fake_wav = tmp_path / "audio" / "fake.wav"
        fake_wav.parent.mkdir(parents=True, exist_ok=True)
        fake_wav.write_bytes(b"RIFF....WAVEfake")
        MockRecorder.return_value.stop.return_value = fake_wav
        MockRecorder.return_value.recording_failed = False

        fake_opus = tmp_path / "audio" / "fake.opus"
        fake_opus.write_bytes(b"fake compressed audio")
        mock_compress.return_value = fake_opus

        # tick 1: meeting detected -> consent asked -> recording "starts"
        watcher._tick()
        assert watcher._current_interview_id is not None
        interview_id = watcher._current_interview_id

        # the capture stream dies mid-recording -- detect_active_meeting is
        # still mocked to return a match, simulating that whatever killed
        # the stream didn't also close the meeting app
        MockRecorder.return_value.recording_failed = True

        # tick 2: must finalize because of the failure flag, not because
        # the meeting ended (it's still "detected" as active)
        watcher._tick()
        MockPanel.return_value.close.assert_called_once()

        assert _wait_until(lambda: not watcher.status["processing_jobs"]), \
            "background processing never finished"

    assert watcher._current_interview_id is None
    record = watcher.db.get(interview_id)
    assert record.ended_at is not None
    assert record.transcript == FAKE_TRANSCRIPT
    assert record.report_path is not None


def test_retention_cleanup_deletes_audio_but_keeps_analysis(tmp_path):
    cfg = _test_config(tmp_path)
    db = InterviewDB(tmp_path / "interviews.db")

    audio_path = tmp_path / "audio" / "old_interview.opus"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"old audio")

    iid = db.start_interview("Teams", str(audio_path), retention_days=0, user_id=1)
    db.save_transcript(iid, FAKE_TRANSCRIPT)
    db.save_analysis(iid, FAKE_ANALYSIS)

    # backdate expiry to simulate the retention window having passed
    past = (dt.datetime.now() - dt.timedelta(days=1)).isoformat()
    db._conn.execute("UPDATE interviews SET audio_expires_at = ? WHERE id = ?", (past, iid))
    db._conn.commit()

    from interview_analyzer.cleanup import run_cleanup
    run_cleanup(db)

    record = db.get(iid)
    assert record.audio_deleted is True
    assert not audio_path.exists()
    # transcript + analysis survive audio deletion -- this is what powers
    # cross-interview trend tracking without needing to keep any audio
    assert record.transcript == FAKE_TRANSCRIPT
    assert record.analysis == FAKE_ANALYSIS
