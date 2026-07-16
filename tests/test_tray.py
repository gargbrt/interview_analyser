"""Tests for the tray icon's pure status-text formatting (the actual
pystray.Icon/PIL image drawing isn't unit-tested here -- it's covered by
manual verification, same as the other real-hardware/real-display
boundaries noted in test_end_to_end.py).

Recording/paused and background processing are independent (a new call can
record while an earlier one is still being transcribed/analyzed), reflected
in status as a top-level `state` (idle/recording/paused) plus an always-
present `processing_jobs` dict, rather than "processing" being a `state`."""
from __future__ import annotations

from interview_analyzer.tray import _visual_state, status_text


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
