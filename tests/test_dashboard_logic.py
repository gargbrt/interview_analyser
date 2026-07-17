"""Tests for a few Dashboard behaviors that don't actually need a real Tk
window to exercise -- constructing a Dashboard and calling its methods
directly is safe as long as nothing touches self._root (only .open()/._run()
do that). The rest of the dashboard's Tk UI remains a manual-verification
boundary, same as elsewhere in this project."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from interview_analyzer.config_loader import Config
from interview_analyzer.dashboard import Dashboard, _friendly_error_markdown, _open_with_os_default
from interview_analyzer.db import InterviewDB


class _ImmediateThread:
    """Stands in for threading.Thread so background work started by the
    Ollama status/start/stop handlers below runs synchronously and
    deterministically within the test instead of racing the assertions."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


def _watcher(tmp_path, user_id=1):
    watcher = MagicMock()
    watcher.user_id = user_id
    watcher.cfg = Config(raw={
        "output": {"output_dir": str(tmp_path / "output"), "reports_subdir": "reports", "trends_filename": "trends.md"},
    })
    watcher.db = InterviewDB(tmp_path / "interviews.db")
    return watcher


class TestLogoutButton:
    def test_clicking_invokes_the_logout_callback(self, tmp_path):
        on_logout = MagicMock()
        dashboard = Dashboard(_watcher(tmp_path), on_logout=on_logout)

        dashboard._on_click_logout()

        on_logout.assert_called_once()

    def test_no_op_when_no_logout_callback_was_given(self, tmp_path):
        dashboard = Dashboard(_watcher(tmp_path))
        dashboard._on_click_logout()  # must not raise


class TestRefreshTrendsRegeneratesFromDb:
    """Regression coverage for a real bug: a profile with existing
    interview history saw "No trends yet" because its per-user trends file
    had never been generated (it previously only got written as a side
    effect of a *new* interview finishing analysis). Trends must now be
    regenerated straight from the DB on every refresh instead of trusting
    whatever file happens to already be on disk."""

    def test_regenerates_trends_for_a_user_with_only_pre_existing_history(self, tmp_path):
        watcher = _watcher(tmp_path, user_id=1)
        iid = watcher.db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        watcher.db.save_analysis(iid, {
            "qa_pairs": [],
            "session_summary": {"top_strengths": [], "top_issues": ["Rambling"], "one_thing_to_practice_next": ""},
        })
        # deliberately no write_trends_report() call here -- simulates a
        # profile whose trends file was never (re)generated after this
        # per-user split landed

        dashboard = Dashboard(watcher)
        dashboard._trends_text = MagicMock()

        dashboard._refresh_trends()

        inserted_text = "".join(
            call.args[1] for call in dashboard._trends_text.insert.call_args_list if len(call.args) > 1
        )
        assert "Rambling" in inserted_text

    def test_does_nothing_when_trends_text_widget_not_built_yet(self, tmp_path):
        dashboard = Dashboard(_watcher(tmp_path))
        dashboard._trends_text = None
        dashboard._refresh_trends()  # must not raise


def _dashboard_with_selected_interview(tmp_path):
    """Builds a Dashboard with a real interview selected in a faked-out
    history tree/feedback-panel state, without needing a real Tk window --
    enough to exercise the feedback button handlers' actual DB-facing logic."""
    watcher = _watcher(tmp_path)
    iid = watcher.db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)

    dashboard = Dashboard(watcher)
    dashboard._history_tree = MagicMock()
    dashboard._history_tree.selection.return_value = [str(iid)]
    dashboard._fb_transcript_var = MagicMock()
    dashboard._fb_analysis_var = MagicMock()
    dashboard._fb_comment_entry = MagicMock()
    dashboard._fb_status_label = MagicMock()
    return dashboard, iid


class TestFeedbackScoreConversion:
    def test_not_rated_becomes_none(self):
        var = MagicMock()
        var.get.return_value = "Not rated"
        assert Dashboard._feedback_score_from_var(var) is None

    def test_numeric_string_becomes_int(self):
        var = MagicMock()
        var.get.return_value = "7"
        assert Dashboard._feedback_score_from_var(var) == 7


class TestSubmitFeedback:
    def test_saves_scores_and_comment_for_the_selected_interview(self, tmp_path):
        dashboard, iid = _dashboard_with_selected_interview(tmp_path)
        dashboard._fb_transcript_var.get.return_value = "9"
        dashboard._fb_analysis_var.get.return_value = "4"
        dashboard._fb_comment_entry.get.return_value = "  missed a detail  "

        dashboard._on_submit_feedback()

        fb = dashboard.watcher.db.get_feedback(iid)
        assert fb.transcript_score == 9
        assert fb.analysis_score == 4
        assert fb.comment == "missed a detail"

    def test_no_op_when_nothing_selected(self, tmp_path):
        dashboard = Dashboard(_watcher(tmp_path))
        dashboard._history_tree = MagicMock()
        dashboard._history_tree.selection.return_value = []
        dashboard._on_submit_feedback()  # must not raise


class TestClearFeedbackRatings:
    def test_resets_the_ui_and_saves_an_unrated_row(self, tmp_path):
        dashboard, iid = _dashboard_with_selected_interview(tmp_path)
        dashboard.watcher.db.save_feedback(iid, user_id=1, transcript_score=8, analysis_score=6, comment="notes")

        dashboard._on_clear_feedback_ratings()

        dashboard._fb_transcript_var.set.assert_called_with("Not rated")
        dashboard._fb_analysis_var.set.assert_called_with("Not rated")
        dashboard._fb_comment_entry.delete.assert_called_with(0, "end")
        fb = dashboard.watcher.db.get_feedback(iid)
        assert fb is not None  # row still exists, just cleared
        assert fb.transcript_score is None
        assert fb.analysis_score is None


class TestDeleteFeedback:
    def test_removes_the_feedback_row_entirely(self, tmp_path):
        dashboard, iid = _dashboard_with_selected_interview(tmp_path)
        dashboard.watcher.db.save_feedback(iid, user_id=1, transcript_score=8, analysis_score=6, comment="notes")

        dashboard._on_delete_feedback()

        assert dashboard.watcher.db.get_feedback(iid) is None

    def test_no_op_when_nothing_selected(self, tmp_path):
        dashboard = Dashboard(_watcher(tmp_path))
        dashboard._history_tree = MagicMock()
        dashboard._history_tree.selection.return_value = []
        dashboard._on_delete_feedback()  # must not raise


class TestOpenWithOsDefault:
    """Regression coverage: os.startfile doesn't exist on macOS at all
    (an AttributeError, not a no-op) -- _open_with_os_default must branch
    before ever touching it there."""

    def test_uses_os_startfile_on_windows(self):
        with patch("interview_analyzer.dashboard.sys.platform", "win32"), \
             patch("interview_analyzer.dashboard.os.startfile", create=True) as mock_startfile:
            _open_with_os_default("C:\\some\\path")
        mock_startfile.assert_called_once_with("C:\\some\\path")

    def test_uses_open_command_on_macos(self):
        with patch("interview_analyzer.dashboard.sys.platform", "darwin"), \
             patch("interview_analyzer.dashboard.subprocess.run") as mock_run:
            _open_with_os_default("/some/path")
        mock_run.assert_called_once_with(["open", "/some/path"], check=True)

    def test_raises_on_an_unsupported_platform(self):
        with patch("interview_analyzer.dashboard.sys.platform", "linux"):
            try:
                _open_with_os_default("/some/path")
                assert False, "expected RuntimeError"
            except RuntimeError as e:
                assert "linux" in str(e)


class TestFriendlyErrorMarkdown:
    """The headline shown for a reprocessing/analysis failure should be
    something a non-technical user can act on, with the raw exception
    still available below for anyone who wants the exact error."""

    def test_ollama_not_running_gets_an_actionable_headline(self):
        exc = RuntimeError(
            "Ollama isn't running and couldn't be started automatically at "
            "http://localhost:11434. Install it from https://ollama.com, or start it "
            "manually, then try again."
        )
        md = _friendly_error_markdown("Reprocessing failed", exc)
        assert md.startswith("# Reprocessing failed")
        assert "Status" in md and "Start" in md
        assert "**Technical details:** RuntimeError: Ollama isn't running" in md

    def test_connection_error_gets_an_actionable_headline(self):
        md = _friendly_error_markdown("Reprocessing failed", ConnectionError("Connection refused"))
        assert "Couldn't reach the analysis model" in md
        assert "**Technical details:** ConnectionError: Connection refused" in md

    def test_missing_audio_gets_an_actionable_headline(self):
        exc = ValueError("No audio was recorded for this interview -- nothing to reprocess.")
        md = _friendly_error_markdown("Reprocessing failed", exc)
        assert "nothing to reprocess" in md

    def test_unrecognized_error_gets_a_generic_headline_with_technical_details(self):
        md = _friendly_error_markdown("Reprocessing failed", KeyError("session_summary"))
        assert "Something went wrong" in md
        assert "**Technical details:** KeyError:" in md


class TestOllamaStatusRow:
    """The Status tab's "Local analysis model" row -- see
    _apply_ollama_status/_on_start_ollama/_on_stop_ollama in dashboard.py.
    _root is faked with a synchronous .after so the background-thread
    handlers' UI updates land immediately within the test."""

    def _dashboard(self, tmp_path):
        dashboard = Dashboard(_watcher(tmp_path))
        dashboard._root = MagicMock()
        dashboard._root.after = lambda _ms, cb: cb()
        dashboard._ollama_status_label = MagicMock()
        dashboard._ollama_start_btn = MagicMock()
        dashboard._ollama_stop_btn = MagicMock()
        return dashboard

    def test_apply_status_running(self, tmp_path):
        dashboard = self._dashboard(tmp_path)
        dashboard._apply_ollama_status(True)
        dashboard._ollama_status_label.config.assert_called_with(text="● Running", foreground="#2f6f5e")
        dashboard._ollama_start_btn.config.assert_called_with(state="disabled")
        dashboard._ollama_stop_btn.config.assert_called_with(state="normal")

    def test_apply_status_not_running(self, tmp_path):
        dashboard = self._dashboard(tmp_path)
        dashboard._apply_ollama_status(False)
        dashboard._ollama_status_label.config.assert_called_with(text="● Not running", foreground="#c0392b")
        dashboard._ollama_start_btn.config.assert_called_with(state="normal")
        dashboard._ollama_stop_btn.config.assert_called_with(state="disabled")

    def test_apply_status_none_means_not_applicable_for_a_cloud_engine(self, tmp_path):
        dashboard = self._dashboard(tmp_path)
        dashboard.watcher.cfg = Config(raw={"analysis": {"engine": "anthropic_api"}})
        dashboard._apply_ollama_status(None)
        dashboard._ollama_status_label.config.assert_called_with(
            text="n/a (using anthropic_api)", foreground="#5b645f"
        )
        dashboard._ollama_start_btn.config.assert_called_with(state="disabled")
        dashboard._ollama_stop_btn.config.assert_called_with(state="disabled")

    def test_start_button_calls_ensure_ollama_running_and_updates_status(self, tmp_path):
        dashboard = self._dashboard(tmp_path)
        with patch("interview_analyzer.dashboard.threading.Thread", _ImmediateThread), \
             patch("interview_analyzer.dashboard.ensure_ollama_running", return_value=True) as mock_ensure:
            dashboard._on_start_ollama()

        mock_ensure.assert_called_once_with("http://localhost:11434")
        dashboard._ollama_status_label.config.assert_called_with(text="● Running", foreground="#2f6f5e")

    def test_stop_button_calls_stop_ollama_and_updates_status(self, tmp_path):
        dashboard = self._dashboard(tmp_path)
        with patch("interview_analyzer.dashboard.threading.Thread", _ImmediateThread), \
             patch("interview_analyzer.dashboard.stop_ollama") as mock_stop, \
             patch("interview_analyzer.dashboard.ollama_is_reachable", return_value=False):
            dashboard._on_stop_ollama()

        mock_stop.assert_called_once_with("http://localhost:11434")
        dashboard._ollama_status_label.config.assert_called_with(text="● Not running", foreground="#c0392b")

    def test_wake_ollama_async_is_a_no_op_for_a_cloud_engine(self, tmp_path):
        dashboard = self._dashboard(tmp_path)
        dashboard.watcher.cfg = Config(raw={"analysis": {"engine": "openai_api"}})
        with patch("interview_analyzer.dashboard.threading.Thread", _ImmediateThread), \
             patch("interview_analyzer.dashboard.ensure_ollama_running") as mock_ensure:
            dashboard._wake_ollama_async()

        mock_ensure.assert_not_called()

    def test_wake_ollama_async_starts_ollama_for_the_ollama_engine(self, tmp_path):
        dashboard = self._dashboard(tmp_path)
        with patch("interview_analyzer.dashboard.threading.Thread", _ImmediateThread), \
             patch("interview_analyzer.dashboard.ensure_ollama_running", return_value=True) as mock_ensure:
            dashboard._wake_ollama_async()

        mock_ensure.assert_called_once_with("http://localhost:11434")


class TestRefreshButtonsAlsoWakeOllama:
    """Refresh (History and Trends tabs) should give a stopped local model
    a head start warming up, not just refresh the displayed data."""

    def test_history_refresh_also_wakes_ollama(self, tmp_path):
        dashboard = Dashboard(_watcher(tmp_path))
        dashboard._refresh_history = MagicMock()
        dashboard._wake_ollama_async = MagicMock()

        dashboard._on_refresh_history()

        dashboard._refresh_history.assert_called_once()
        dashboard._wake_ollama_async.assert_called_once()

    def test_trends_refresh_also_wakes_ollama(self, tmp_path):
        dashboard = Dashboard(_watcher(tmp_path))
        dashboard._refresh_trends = MagicMock()
        dashboard._wake_ollama_async = MagicMock()

        dashboard._on_refresh_trends()

        dashboard._refresh_trends.assert_called_once()
        dashboard._wake_ollama_async.assert_called_once()
