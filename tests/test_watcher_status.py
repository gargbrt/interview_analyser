"""Tests for MeetingWatcher's status snapshot and on_state_change
notifications, which the tray icon and dashboard rely on to stay in sync
without polling internal state directly."""
from __future__ import annotations

import time
from unittest.mock import patch

from interview_analyzer.config_loader import Config
from interview_analyzer.watcher import MeetingWatcher


def _test_config(tmp_path) -> Config:
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
        "transcription": {"engine": "faster-whisper", "whisper_model": "tiny",
                           "device": "cpu", "diarization": False},
        "analysis": {"engine": "ollama", "llm_model": "llama3.1:8b",
                     "ollama_host": "http://localhost:11434"},
        "storage": {"db_path": str(tmp_path / "interviews.db")},
        "output": {"output_dir": str(tmp_path / "output"), "reports_subdir": "reports",
                    "trends_filename": "trends.md"},
    })


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_status_is_idle_before_any_recording(tmp_path):
    watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
    assert watcher.status == {"state": "idle", "processing_jobs": {}}


def test_status_reports_recording_then_paused_then_idle(tmp_path):
    """Processing now runs on its own background thread (started by
    _stop_and_process), so the mocks must stay active until that thread
    actually finishes -- the test waits for processing_jobs to empty out
    *inside* the `with patch(...)` block, not after leaving it."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    notifications = []
    watcher.set_on_state_change(lambda: notifications.append(watcher.status.copy()))

    with patch("interview_analyzer.watcher.detect_active_meeting", side_effect=[("Zoom", True), ("Zoom", True), None]), \
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
        watcher._tick()  # meeting gone -> stop, hand off to a background thread
        assert watcher.status["state"] == "idle"  # recording itself stops immediately

        # background processing must finish while the mocks are still
        # active, or it'd fall through to the real transcribe/analyze
        assert _wait_until(lambda: not watcher.status["processing_jobs"]), \
            "background processing never finished"

    assert watcher.status == {"state": "idle", "processing_jobs": {}}

    # every transition (start, pause, resume, stop, each processing stage,
    # done) fired a notification, in order
    states = [n["state"] for n in notifications]
    assert states == [
        "recording", "paused", "recording",  # start, pause, resume
        "idle",  # recording stops immediately -- before background processing begins
        "idle", "idle", "idle", "idle",  # compress/transcribe/analyze/report -- job updates, state stays idle
        "idle",  # job cleared
    ]
    stages = [n["processing_jobs"].get(1, {}).get("stage") for n in notifications if n["processing_jobs"]]
    assert stages == ["compressing", "transcribing", "analyzing", "generating_report"]


def test_post_call_cooldown_suppresses_immediate_reprompt_for_same_app(tmp_path):
    """A just-ended call's window/tab can linger and keep matching the
    detection keywords (e.g. a Meet tab still open on the post-call
    screen); the cooldown stops that from immediately re-triggering
    consent for a call that's actually already over."""
    cfg = _test_config(tmp_path)
    cfg.raw["post_call_cooldown_seconds"] = 60
    watcher = MeetingWatcher(cfg, user_id=1)
    watcher._last_ended_app = "Zoom"
    watcher._last_ended_at = time.time()  # ended just now

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True) as mock_consent:
        watcher._tick()

    mock_consent.assert_not_called()
    assert watcher._current_interview_id is None


def test_post_call_cooldown_expires_and_allows_reprompt(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.raw["post_call_cooldown_seconds"] = 5
    watcher = MeetingWatcher(cfg, user_id=1)
    watcher._last_ended_app = "Zoom"
    watcher._last_ended_at = time.time() - 10  # cooldown already elapsed

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True) as mock_consent, \
         patch("interview_analyzer.watcher.SystemAudioRecorder"), \
         patch("interview_analyzer.watcher.RecordingControlPanel"):
        watcher._tick()

    mock_consent.assert_called_once()
    assert watcher._current_interview_id is not None


def test_post_call_cooldown_is_per_app_not_global(tmp_path):
    """Ending a Zoom call shouldn't suppress a prompt for a different app
    (e.g. Teams) detected right after."""
    cfg = _test_config(tmp_path)
    cfg.raw["post_call_cooldown_seconds"] = 60
    cfg.raw["watched_processes"]["desktop_apps"] = ["Zoom.exe", "Teams.exe"]
    watcher = MeetingWatcher(cfg, user_id=1)
    watcher._last_ended_app = "Zoom"
    watcher._last_ended_at = time.time()

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Teams", True)), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True) as mock_consent, \
         patch("interview_analyzer.watcher.SystemAudioRecorder"), \
         patch("interview_analyzer.watcher.RecordingControlPanel"):
        watcher._tick()

    mock_consent.assert_called_once()


def test_browser_tab_match_needs_more_sustained_presence_than_a_desktop_app(tmp_path):
    """A browser tab title match (e.g. Meet's pre-join lobby screen) is a
    much weaker signal than a real app process being open, so it must not
    trigger consent after the same (short) desktop-app debounce."""
    cfg = _test_config(tmp_path)
    cfg.raw["start_debounce_polls"] = 2
    cfg.raw["browser_start_debounce_polls"] = 4
    watcher = MeetingWatcher(cfg, user_id=1)

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Meet", False)), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True) as mock_consent:
        watcher._tick()  # poll 1
        watcher._tick()  # poll 2 -- would already trigger a desktop-app match
        mock_consent.assert_not_called()
        watcher._tick()  # poll 3
        mock_consent.assert_not_called()
        watcher._tick()  # poll 4 -- now the browser threshold is reached

    mock_consent.assert_called_once()


def test_desktop_app_match_uses_the_shorter_debounce(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.raw["start_debounce_polls"] = 2
    cfg.raw["browser_start_debounce_polls"] = 4
    watcher = MeetingWatcher(cfg, user_id=1)

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True) as mock_consent, \
         patch("interview_analyzer.watcher.SystemAudioRecorder"), \
         patch("interview_analyzer.watcher.RecordingControlPanel"):
        watcher._tick()  # poll 1
        mock_consent.assert_not_called()
        watcher._tick()  # poll 2 -- reaches the desktop-app threshold

    mock_consent.assert_called_once()


def test_declined_flag_blocks_reprompt_before_cooldown_expires(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.raw["declined_cooldown_seconds"] = 300
    watcher = MeetingWatcher(cfg, user_id=1)
    watcher._declined_this_session["Zoom"] = time.time()  # declined a moment ago

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True) as mock_consent:
        watcher._tick()

    mock_consent.assert_not_called()


def test_declined_flag_expires_and_allows_reprompt_for_a_genuinely_new_call(tmp_path):
    """This is the fix for "declined once, then never asked again even for
    a real new call" -- a decline must not block forever, since flaky
    browser-tab detection can otherwise never report the call as truly over."""
    cfg = _test_config(tmp_path)
    cfg.raw["declined_cooldown_seconds"] = 5
    watcher = MeetingWatcher(cfg, user_id=1)
    watcher._declined_this_session["Zoom"] = time.time() - 10  # declined well past the cooldown

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=("Zoom", True)), \
         patch("interview_analyzer.watcher.ask_consent", return_value=True) as mock_consent, \
         patch("interview_analyzer.watcher.SystemAudioRecorder"), \
         patch("interview_analyzer.watcher.RecordingControlPanel"):
        watcher._tick()

    mock_consent.assert_called_once()
    assert "Zoom" not in watcher._declined_this_session  # the stale entry is cleaned up


def test_request_stop_recording_sets_flag_without_touching_recorder(tmp_path):
    watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
    assert not watcher._manual_stop_requested.is_set()
    watcher.request_stop_recording()
    assert watcher._manual_stop_requested.is_set()


def test_request_start_recording_raises_if_already_recording(tmp_path):
    watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
    watcher._current_interview_id = 42  # simulate an active recording

    try:
        watcher.request_start_recording("Meet")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_request_start_recording_starts_on_the_next_tick_without_detection_or_consent(tmp_path):
    """The manual fallback for when automatic detection misses a real
    call -- clicking "Start recording" is the consent, so ask_consent must
    not even be called, and it shouldn't wait on detect_active_meeting."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)

    with patch("interview_analyzer.watcher.detect_active_meeting", return_value=None) as mock_detect, \
         patch("interview_analyzer.watcher.ask_consent") as mock_consent, \
         patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
         patch("interview_analyzer.watcher.RecordingControlPanel"):
        MockRecorder.return_value.is_paused = False
        watcher.request_start_recording("Meet")
        assert watcher.status["state"] == "idle"  # not yet -- happens on the next tick

        watcher._tick()

    mock_consent.assert_not_called()
    mock_detect.assert_not_called()  # the manual-start branch returns before checking detection
    MockRecorder.return_value.start.assert_called_once()
    assert watcher.status["state"] == "recording"
    assert watcher.status["app_name"] == "Meet"


def test_request_start_recording_clears_a_stale_decline_for_that_app(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    watcher._declined_this_session["Meet"] = time.time()

    with patch("interview_analyzer.watcher.SystemAudioRecorder") as MockRecorder, \
         patch("interview_analyzer.watcher.RecordingControlPanel"):
        MockRecorder.return_value.is_paused = False
        watcher.request_start_recording("Meet")
        watcher._tick()

    assert "Meet" not in watcher._declined_this_session
    assert watcher.status["state"] == "recording"


def test_request_start_recording_trims_whitespace_and_defaults_label(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)

    with patch("interview_analyzer.watcher.SystemAudioRecorder"), \
         patch("interview_analyzer.watcher.RecordingControlPanel"):
        watcher.request_start_recording("  ")  # blank/whitespace-only label
        watcher._tick()

    assert watcher.status["app_name"] == "Manual"


def test_pause_resume_are_no_ops_when_idle(tmp_path):
    watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
    # no recorder exists yet -- must not raise
    watcher.pause_recording()
    watcher.resume_recording()
    assert watcher.status == {"state": "idle", "processing_jobs": {}}


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
