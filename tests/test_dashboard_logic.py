"""Tests for a few Dashboard behaviors that don't actually need a real Tk
window to exercise -- constructing a Dashboard and calling its methods
directly is safe as long as nothing touches self._root (only .open()/._run()
do that). The rest of the dashboard's Tk UI remains a manual-verification
boundary, same as elsewhere in this project."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from interview_analyzer.config_loader import Config
from interview_analyzer.dashboard import Dashboard, _open_with_os_default
from interview_analyzer.db import InterviewDB


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
