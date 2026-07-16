"""Tests for MeetingWatcher's status snapshot and on_state_change
notifications, which the tray icon and dashboard rely on to stay in sync
without polling internal state directly."""
from __future__ import annotations

from unittest.mock import patch

from interview_analyzer.config_loader import Config
from interview_analyzer.watcher import MeetingWatcher


def _test_config(tmp_path) -> Config:
    return Config(raw={
        "retention_days": 3,
        "poll_interval_seconds": 0.01,
        "start_debounce_polls": 1,
        "watched_processes": {
            "desktop_apps": ["Zoom.exe"], "browser_tab_keywords": [], "browser_processes": [],
        },
        "audio": {"sample_rate": 16000, "channels": 1, "bitrate_kbps": 64,
                   "format": "opus", "raw_dir": str(tmp_path / "audio")},
        "transcription": {"engine": "faster-whisper", "whisper_model": "tiny",
                           "device": "cpu", "diarization": False},
        "analysis": {"engine": "ollama", "llm_model": "llama3.1:8b",
                     "ollama_host": "http://localhost:11434"},
        "storage": {"db_path": str(tmp_path / "interviews.db")},
        "output": {"output_dir": str(tmp_path / "output"), "reports_subdir": "reports",
                    "trends_filename": "trends.md"},
    })


def test_status_is_idle_before_any_recording(tmp_path):
    watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
    assert watcher.status == {"state": "idle"}


def test_status_reports_recording_then_paused_then_idle(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    notifications = []
    watcher.set_on_state_change(lambda: notifications.append(watcher.status.copy()))

    with patch("interview_analyzer.watcher.detect_active_meeting", side_effect=["Zoom", "Zoom", None]), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True), \
         patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
         patch("interview_analyzer.watcher.RecordingControlPanel"), \
         patch("interview_analyzer.watcher.compress_audio") as mock_compress, \
         patch("interview_analyzer.watcher.transcribe", return_value="transcript"), \
         patch("interview_analyzer.watcher.analyze_transcript", return_value={"qa_pairs": [], "session_summary": {}}):

        recorder_mock = MockRecorder.return_value
        recorder_mock.is_paused = False
        # a real SystemAudioRecorder flips is_paused as a side effect of
        # pause()/resume() -- mirror that so notify() ordering is real
        recorder_mock.pause.side_effect = lambda: setattr(recorder_mock, "is_paused", True)
        recorder_mock.resume.side_effect = lambda: setattr(recorder_mock, "is_paused", False)

        fake_wav = tmp_path / "audio" / "fake.wav"
        fake_wav.parent.mkdir(parents=True, exist_ok=True)
        fake_wav.write_bytes(b"RIFF....WAVEfake")
        recorder_mock.stop.return_value = fake_wav
        fake_opus = tmp_path / "audio" / "fake.opus"
        fake_opus.write_bytes(b"fake")
        mock_compress.return_value = fake_opus

        watcher._tick()  # detected -> recording starts
        assert watcher.status["state"] == "recording"
        assert watcher.status["app_name"] == "Zoom"

        watcher.pause_recording()
        recorder_mock.pause.assert_called_once()
        assert watcher.status["state"] == "paused"

        watcher.resume_recording()
        recorder_mock.resume.assert_called_once()
        assert watcher.status["state"] == "recording"

        watcher._tick()  # still detected, no-op transition
        watcher._tick()  # meeting gone -> stop and process
        assert watcher.status == {"state": "idle"}

    # every transition (start, pause, resume, stop) fired a notification
    states = [n["state"] for n in notifications]
    assert states == ["recording", "paused", "recording", "idle", "idle"]


def test_request_stop_recording_sets_flag_without_touching_recorder(tmp_path):
    watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
    assert not watcher._manual_stop_requested.is_set()
    watcher.request_stop_recording()
    assert watcher._manual_stop_requested.is_set()


def test_pause_resume_are_no_ops_when_idle(tmp_path):
    watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
    # no recorder exists yet -- must not raise
    watcher.pause_recording()
    watcher.resume_recording()
    assert watcher.status == {"state": "idle"}


def test_shutdown_stops_run_forever_loop(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=None), \
         patch.object(watcher, "_tick", wraps=watcher._tick) as mock_tick:
        def _shutdown_after_first_tick(*a, **kw):
            watcher.shutdown()
        mock_tick.side_effect = _shutdown_after_first_tick
        watcher.run_forever()  # must return, not loop forever

    assert mock_tick.call_count == 1
