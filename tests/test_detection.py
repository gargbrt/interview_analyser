"""Tests for detect_active_meeting() itself -- the browser-tab-title
matching logic that caused a real production bug: Google Meet's actual tab
title uses an EN DASH ("Meet – code", U+2013), not a plain hyphen, so
the original "Meet - " keyword never matched a real live call. Verified
against a real Meet window title captured during manual testing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from interview_analyzer.config_loader import Config
from interview_analyzer.watcher import detect_active_meeting

REAL_MEET_TITLE = "Meet – mtz-ofiz-ukh - Google Chrome"  # captured from an actual live call


def _cfg(**overrides) -> Config:
    raw = {
        "watched_processes": {
            "desktop_apps": ["Zoom.exe", "Teams.exe"],
            "browser_tab_keywords": ["Meet – ", "Zoom Meeting"],
            "browser_processes": ["chrome.exe", "msedge.exe"],
        },
    }
    raw.update(overrides)
    return Config(raw=raw)


def _mock_process(name):
    p = MagicMock()
    p.name.return_value = name
    return p


def test_detects_real_google_meet_title_with_en_dash():
    """The exact regression: this title was NOT detected before the fix."""
    with patch("interview_analyzer.watcher.psutil.process_iter", return_value=[_mock_process("chrome.exe")]), \
         patch("interview_analyzer.watcher._enumerate_window_titles", return_value=[REAL_MEET_TITLE]):
        result = detect_active_meeting(_cfg())

    assert result == ("Meet", False)


def test_plain_meet_homepage_tab_does_not_match():
    """The bare Meet homepage/lobby-less tab, titled "Google Meet", becomes
    "Google Meet - Google Chrome" once Chrome appends its own suffix --
    which would coincidentally contain a plain-hyphen "Meet - " keyword.
    This is exactly why that variant isn't in the default keyword list;
    only the real en-dash title should match."""
    with patch("interview_analyzer.watcher.psutil.process_iter", return_value=[_mock_process("chrome.exe")]), \
         patch("interview_analyzer.watcher._enumerate_window_titles",
               return_value=["Google Meet - Google Chrome"]):
        result = detect_active_meeting(_cfg())

    assert result is None


def test_desktop_app_match_returns_is_desktop_app_true():
    with patch("interview_analyzer.watcher.psutil.process_iter", return_value=[_mock_process("Zoom.exe")]):
        result = detect_active_meeting(_cfg())

    assert result == ("Zoom", True)


def test_desktop_app_takes_precedence_over_browser_tab():
    with patch("interview_analyzer.watcher.psutil.process_iter",
               return_value=[_mock_process("Zoom.exe"), _mock_process("chrome.exe")]), \
         patch("interview_analyzer.watcher._enumerate_window_titles", return_value=[REAL_MEET_TITLE]):
        result = detect_active_meeting(_cfg())

    assert result == ("Zoom", True)


def test_no_match_returns_none():
    with patch("interview_analyzer.watcher.psutil.process_iter", return_value=[_mock_process("notepad.exe")]):
        result = detect_active_meeting(_cfg())

    assert result is None


def test_browser_running_but_no_matching_tab_title_returns_none():
    with patch("interview_analyzer.watcher.psutil.process_iter", return_value=[_mock_process("chrome.exe")]), \
         patch("interview_analyzer.watcher._enumerate_window_titles",
               return_value=["Inbox - Gmail - Google Chrome", "Google Meet - Google Chrome"]):
        # note: bare "Google Meet" (the homepage/lobby-less title) does NOT
        # contain either "Meet – " or "Meet - " -- must not match
        result = detect_active_meeting(_cfg())

    assert result is None
