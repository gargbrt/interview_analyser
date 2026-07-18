"""Tests for detect_active_meeting() itself -- the browser-tab-title
matching logic that caused a real production bug: Google Meet's actual tab
title uses an EN DASH ("Meet – code", U+2013), not a plain hyphen, so
the original "Meet - " keyword never matched a real live call. Verified
against a real Meet window title captured during manual testing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import win32con

from interview_analyzer.config_loader import Config
from interview_analyzer.watcher import (
    _is_real_app_window,
    _windows_process_has_a_visible_window,
    detect_active_meeting,
)

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


def _mock_process(name, pid=1234):
    p = MagicMock()
    p.name.return_value = name
    p.info = {"name": name, "pid": pid}
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
    with patch("interview_analyzer.watcher.psutil.process_iter", return_value=[_mock_process("Zoom.exe")]), \
         patch("interview_analyzer.watcher._windows_process_has_a_visible_window", return_value=True):
        result = detect_active_meeting(_cfg())

    assert result == ("Zoom", True)


def test_desktop_app_takes_precedence_over_browser_tab():
    with patch("interview_analyzer.watcher.psutil.process_iter",
               return_value=[_mock_process("Zoom.exe"), _mock_process("chrome.exe")]), \
         patch("interview_analyzer.watcher._enumerate_window_titles", return_value=[REAL_MEET_TITLE]), \
         patch("interview_analyzer.watcher._windows_process_has_a_visible_window", return_value=True):
        result = detect_active_meeting(_cfg())

    assert result == ("Zoom", True)


def test_desktop_app_running_with_no_visible_window_is_not_treated_as_a_call():
    """Reproduces the real false-positive bug: Zoom.exe sitting in the
    background/system tray (no visible window anywhere, e.g. an empty
    MainWindowTitle observed directly on a real machine) must not be
    treated as an active call just because the process exists."""
    with patch("interview_analyzer.watcher.psutil.process_iter", return_value=[_mock_process("Zoom.exe")]), \
         patch("interview_analyzer.watcher._windows_process_has_a_visible_window", return_value=False):
        result = detect_active_meeting(_cfg())

    assert result is None


def test_desktop_app_falls_back_to_process_presence_when_win32_apis_unavailable():
    """On a platform (or environment) without pywin32, the visible-window
    check gracefully degrades to today's process-presence-only behavior
    instead of erroring out or always saying no."""
    with patch("interview_analyzer.watcher.psutil.process_iter", return_value=[_mock_process("Zoom.exe")]), \
         patch("interview_analyzer.watcher.win32gui", None), \
         patch("interview_analyzer.watcher.win32process", None):
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


class TestIsRealAppWindow:
    """Covers the wake-from-sleep false positive: a conferencing app can own
    a technically-"visible" window that isn't a real, on-screen app window
    at all -- most likely a small always-on-top "meeting starting" reminder
    popup (common for scheduled meetings whose start time passed while the
    machine was asleep), or a window DWM is cloaking because it lives on a
    different virtual desktop."""

    def test_not_visible_is_excluded(self):
        with patch("interview_analyzer.watcher.win32gui.IsWindowVisible", return_value=False):
            assert _is_real_app_window(123) is False

    def test_ordinary_visible_window_is_real(self):
        with patch("interview_analyzer.watcher.win32gui.IsWindowVisible", return_value=True), \
             patch("interview_analyzer.watcher.win32gui.GetWindowLong", return_value=0), \
             patch("interview_analyzer.watcher._is_window_cloaked", return_value=False):
            assert _is_real_app_window(123) is True

    def test_toolwindow_only_popup_is_excluded(self):
        """Reminder/notification popups are typically built with
        WS_EX_TOOLWINDOW specifically so they don't appear in the
        taskbar/alt-tab list -- the same signal used here."""
        with patch("interview_analyzer.watcher.win32gui.IsWindowVisible", return_value=True), \
             patch("interview_analyzer.watcher.win32gui.GetWindowLong", return_value=win32con.WS_EX_TOOLWINDOW), \
             patch("interview_analyzer.watcher._is_window_cloaked", return_value=False):
            assert _is_real_app_window(123) is False

    def test_toolwindow_that_is_also_an_appwindow_is_real(self):
        combined = win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_APPWINDOW
        with patch("interview_analyzer.watcher.win32gui.IsWindowVisible", return_value=True), \
             patch("interview_analyzer.watcher.win32gui.GetWindowLong", return_value=combined), \
             patch("interview_analyzer.watcher._is_window_cloaked", return_value=False):
            assert _is_real_app_window(123) is True

    def test_cloaked_window_is_excluded(self):
        """A window can be cloaked by DWM (e.g. it lives on a different
        virtual desktop) while IsWindowVisible() still reports True."""
        with patch("interview_analyzer.watcher.win32gui.IsWindowVisible", return_value=True), \
             patch("interview_analyzer.watcher.win32gui.GetWindowLong", return_value=0), \
             patch("interview_analyzer.watcher._is_window_cloaked", return_value=True):
            assert _is_real_app_window(123) is False


def test_process_owning_only_a_toolwindow_popup_is_not_a_visible_window():
    """End-to-end through _windows_process_has_a_visible_window: a process
    that owns exactly one window, and that window is a reminder-style
    popup (not a real app window), should not count as having a visible
    window at all."""
    def fake_enum(callback, _):
        callback(999, None)

    with patch("interview_analyzer.watcher.win32gui.EnumWindows", side_effect=fake_enum), \
         patch("interview_analyzer.watcher.win32process.GetWindowThreadProcessId", return_value=(0, 4321)), \
         patch("interview_analyzer.watcher._is_real_app_window", return_value=False):
        assert _windows_process_has_a_visible_window(4321) is False
