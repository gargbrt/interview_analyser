"""Tests for the tray icon's pure status-text formatting (the actual
pystray.Icon/PIL image drawing isn't unit-tested here -- it's covered by
manual verification, same as the other real-hardware/real-display
boundaries noted in test_end_to_end.py)."""
from __future__ import annotations

from interview_analyzer.tray import status_text


def test_idle_status_text():
    assert status_text({"state": "idle"}) == "Idle — watching for calls"


def test_recording_status_text_includes_app_name():
    assert status_text({"state": "recording", "app_name": "Zoom"}) == "Recording — Zoom"


def test_paused_status_text_includes_app_name():
    assert status_text({"state": "paused", "app_name": "Teams"}) == "Paused — Teams"


def test_recording_status_text_falls_back_when_app_name_missing():
    assert status_text({"state": "recording"}) == "Recording — call"
