"""Tests for the tray icon's pure status-text formatting, plus the menu's
logout item (constructing a pystray.Icon and inspecting its Menu works
headlessly -- it's only `.run()`'s actual system-tray rendering that needs
a real display, covered by manual verification like the other real-
hardware/real-display boundaries noted in test_end_to_end.py).

Recording/paused and background processing are independent (a new call can
record while an earlier one is still being transcribed/analyzed), reflected
in status as a top-level `state` (idle/recording/paused) plus an always-
present `processing_jobs` dict, rather than "processing" being a `state`."""
from __future__ import annotations

from unittest.mock import MagicMock

from interview_analyzer.tray import TrayIcon, _visual_state, status_text


def test_idle_status_text():
    assert status_text({"state": "idle", "processing_jobs": {}}) == "Idle — watching for calls"


def test_recording_status_text_includes_app_name():
    assert status_text({"state": "recording", "app_name": "Zoom", "processing_jobs": {}}) == "Recording — Zoom"


def test_paused_status_text_includes_app_name():
    assert status_text({"state": "paused", "app_name": "Teams", "processing_jobs": {}}) == "Paused — Teams"


def test_recording_status_text_falls_back_when_app_name_missing():
    assert status_text({"state": "recording", "processing_jobs": {}}) == "Recording — call"


def test_processing_status_text_while_idle_shows_just_the_job():
    """No live recording, one background job -- no need to also say "Idle"."""
    status = {"state": "idle", "processing_jobs": {1: {"stage": "transcribing", "progress": None}}}
    assert status_text(status) == "Transcribing…"


def test_processing_status_text_combines_with_active_recording():
    """A new call recording while an earlier interview is still processing
    -- both should be visible, since they're now independent."""
    status = {
        "state": "recording", "app_name": "Zoom",
        "processing_jobs": {1: {"stage": "analyzing", "progress": None}},
    }
    assert status_text(status) == "Recording — Zoom · Analyzing…"


def test_processing_status_text_shows_stage_labels():
    for stage, label in [
        ("compressing", "Compressing"), ("transcribing", "Transcribing"),
        ("analyzing", "Analyzing"), ("generating_report", "Generating report"),
    ]:
        status = {"state": "idle", "processing_jobs": {1: {"stage": stage, "progress": None}}}
        assert status_text(status) == f"{label}…"


def test_processing_status_text_falls_back_for_unknown_stage():
    status = {"state": "idle", "processing_jobs": {1: {"stage": "mystery", "progress": None}}}
    assert status_text(status) == "Processing…"


def test_processing_status_text_includes_percentage_when_available():
    def job(progress):
        return {"state": "idle", "processing_jobs": {1: {"stage": "transcribing", "progress": progress}}}

    assert status_text(job(0.42)) == "Transcribing… 42%"
    assert status_text(job(0.0)) == "Transcribing… 0%"
    assert status_text(job(1.0)) == "Transcribing… 100%"


def test_processing_status_text_omits_percentage_when_not_available():
    """Analyzing/generating_report have no partial-progress signal -- the
    percentage must not show a stale/misleading number for them."""
    status = {"state": "idle", "processing_jobs": {1: {"stage": "analyzing", "progress": None}}}
    assert status_text(status) == "Analyzing…"


def test_processing_status_text_summarizes_multiple_concurrent_jobs():
    status = {
        "state": "idle",
        "processing_jobs": {
            1: {"stage": "transcribing", "progress": 0.5},
            2: {"stage": "analyzing", "progress": None},
        },
    }
    assert status_text(status) == "2 interviews processing"


def test_visual_state_prioritizes_recording_over_background_processing():
    status = {
        "state": "recording", "app_name": "Zoom",
        "processing_jobs": {1: {"stage": "transcribing", "progress": 0.5}},
    }
    assert _visual_state(status) == "recording"


def test_visual_state_is_processing_when_idle_with_jobs():
    status = {"state": "idle", "processing_jobs": {1: {"stage": "transcribing", "progress": 0.5}}}
    assert _visual_state(status) == "processing"


def test_visual_state_is_idle_with_no_jobs():
    assert _visual_state({"state": "idle", "processing_jobs": {}}) == "idle"


def test_visual_state_is_paused_regardless_of_jobs():
    status = {
        "state": "paused", "app_name": "Zoom",
        "processing_jobs": {1: {"stage": "analyzing", "progress": None}},
    }
    assert _visual_state(status) == "paused"


def _make_watcher(state="idle"):
    watcher = MagicMock()
    watcher.status = {"state": state, "processing_jobs": {}}
    return watcher


def _logout_item(tray: TrayIcon):
    # iterating a pystray.Menu only yields currently-visible items -- .items
    # holds the raw list regardless of visibility, needed to assert a hidden
    # item is actually hidden rather than just absent from the iteration
    return next(item for item in tray._build_menu().items if str(item.text) == "Log out")


class TestLogoutMenuItem:
    def test_hidden_when_no_logout_callback_given(self):
        tray = TrayIcon(_make_watcher(), open_dashboard=lambda: None, on_quit=lambda: None)
        assert _logout_item(tray).visible is False

    def test_visible_while_idle_when_a_logout_callback_is_given(self):
        tray = TrayIcon(_make_watcher("idle"), open_dashboard=lambda: None, on_quit=lambda: None, on_logout=lambda: None)
        assert _logout_item(tray).visible is True

    def test_hidden_while_recording_to_avoid_abandoning_a_call(self):
        tray = TrayIcon(_make_watcher("recording"), open_dashboard=lambda: None, on_quit=lambda: None, on_logout=lambda: None)
        assert _logout_item(tray).visible is False

    def test_hidden_while_paused(self):
        tray = TrayIcon(_make_watcher("paused"), open_dashboard=lambda: None, on_quit=lambda: None, on_logout=lambda: None)
        assert _logout_item(tray).visible is False

    def test_clicking_invokes_the_callback_and_stops_the_icon(self):
        on_logout = MagicMock()
        tray = TrayIcon(_make_watcher("idle"), open_dashboard=lambda: None, on_quit=lambda: None, on_logout=on_logout)
        tray._icon.stop = MagicMock()

        _logout_item(tray)(tray._icon)

        on_logout.assert_called_once()
        tray._icon.stop.assert_called_once()
