"""Tests for MeetingWatcher._maybe_refresh_model_catalog -- the gating
logic that decides when the background model-catalog check (see
model_catalog.py, covering Groq/Anthropic/OpenAI) actually fires from the
poll loop. The real network call is never exercised here; only whether/when
a background thread gets started."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from interview_analyzer.config_loader import Config
from interview_analyzer.watcher import MeetingWatcher


def _test_config(tmp_path) -> Config:
    return Config(raw={
        "retention_days": 3,
        "poll_interval_seconds": 0.01,
        "start_debounce_polls": 1,
        "stop_debounce_polls": 1,
        "watched_processes": {"desktop_apps": [], "browser_tab_keywords": [], "browser_processes": []},
        "audio": {"sample_rate": 16000, "channels": 1, "bitrate_kbps": 64,
                   "format": "opus", "raw_dir": str(tmp_path / "audio")},
        "transcription": {"engine": "faster-whisper", "whisper_model": "tiny",
                           "device": "cpu", "diarization": False},
        "analysis": {"engine": "groq_api", "llm_model": "llama-3.3-70b-versatile"},
        "storage": {"db_path": str(tmp_path / "interviews.db")},
        "output": {"output_dir": str(tmp_path / "output"), "reports_subdir": "reports",
                    "trends_filename": "trends.md"},
    })


class _FakeThread:
    """Stands in for threading.Thread -- records what it was started with
    instead of actually running it, and reports alive/dead on demand so
    the "already running" guard can be exercised deterministically."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self._alive = False

    def start(self):
        self._alive = True

    def is_alive(self):
        return self._alive


class TestMaybeRefreshModelCatalog:
    def test_starts_a_background_refresh_when_due(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        with patch("interview_analyzer.watcher.model_catalog.any_check_due", return_value=True), \
             patch("interview_analyzer.watcher.threading.Thread", _FakeThread):
            watcher._maybe_refresh_model_catalog()

        assert watcher._model_catalog_refresh_thread is not None
        assert watcher._model_catalog_refresh_thread.target.__name__ == "refresh_due_catalogs"
        assert watcher._model_catalog_refresh_thread.args == (watcher.cfg,)

    def test_does_nothing_when_not_due(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        with patch("interview_analyzer.watcher.model_catalog.any_check_due", return_value=False), \
             patch("interview_analyzer.watcher.threading.Thread", _FakeThread):
            watcher._maybe_refresh_model_catalog()

        assert watcher._model_catalog_refresh_thread is None

    def test_does_not_start_a_second_thread_while_one_is_still_running(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        in_flight = MagicMock()
        in_flight.is_alive.return_value = True
        watcher._model_catalog_refresh_thread = in_flight

        with patch("interview_analyzer.watcher.model_catalog.any_check_due", return_value=True), \
             patch("interview_analyzer.watcher.threading.Thread", _FakeThread):
            watcher._maybe_refresh_model_catalog()

        assert watcher._model_catalog_refresh_thread is in_flight  # unchanged

    def test_starts_a_new_thread_once_the_previous_one_finished(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        finished = MagicMock()
        finished.is_alive.return_value = False
        watcher._model_catalog_refresh_thread = finished

        with patch("interview_analyzer.watcher.model_catalog.any_check_due", return_value=True), \
             patch("interview_analyzer.watcher.threading.Thread", _FakeThread):
            watcher._maybe_refresh_model_catalog()

        assert watcher._model_catalog_refresh_thread is not finished

    def test_respects_the_retry_cooldown_between_attempts(self, tmp_path):
        """Without this, a persistently-missing key for every engine
        (any_check_due() stays True forever, since a skipped attempt never
        bumps last_checked -- see refresh_one's docstring) would spawn a
        new thread on every single poll tick."""
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        watcher._model_catalog_last_attempt_at = time.time()  # "just attempted"

        with patch("interview_analyzer.watcher.model_catalog.any_check_due", return_value=True), \
             patch("interview_analyzer.watcher.threading.Thread", _FakeThread):
            watcher._maybe_refresh_model_catalog()

        assert watcher._model_catalog_refresh_thread is None

    def test_tries_again_once_the_cooldown_has_elapsed(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        watcher._model_catalog_last_attempt_at = time.time() - 9999  # long ago

        with patch("interview_analyzer.watcher.model_catalog.any_check_due", return_value=True), \
             patch("interview_analyzer.watcher.threading.Thread", _FakeThread):
            watcher._maybe_refresh_model_catalog()

        assert watcher._model_catalog_refresh_thread is not None
