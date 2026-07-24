"""Tests for a few Dashboard behaviors that don't actually need a real Tk
window to exercise -- constructing a Dashboard and calling its methods
directly is safe as long as nothing touches self._root (only .open()/._run()
do that). The rest of the dashboard's Tk UI remains a manual-verification
boundary, same as elsewhere in this project."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import json

from interview_analyzer.config_loader import Config
from interview_analyzer.dashboard import (
    Dashboard,
    _friendly_error_markdown,
    _group_transcript_by_speaker,
    _open_with_os_default,
    _parse_transcript_lines,
    _speaker_color,
    can_reprocess,
)
from interview_analyzer.db import InterviewDB
from interview_analyzer.profiles import CORE_COMPETENCIES, AssessmentProfile

VALID_ANALYSIS = {
    "qa_pairs": [],
    "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
}


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


class TestViewTrendsInfographicButton:
    def test_clicking_generates_and_opens_the_trends_infographic(self, tmp_path):
        watcher = _watcher(tmp_path)
        iid = watcher.db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        watcher.db.save_analysis(iid, VALID_ANALYSIS)
        dashboard = Dashboard(watcher)

        with patch("interview_analyzer.dashboard._open_with_os_default") as mock_open:
            dashboard._on_view_trends_infographic()

        mock_open.assert_called_once()
        opened_path = mock_open.call_args.args[0]
        assert opened_path.exists()
        assert "trends" in opened_path.name and "infographic" in opened_path.name

    def test_shows_an_error_in_the_trends_pane_if_opening_fails(self, tmp_path):
        dashboard = Dashboard(_watcher(tmp_path))
        dashboard._trends_text = MagicMock()

        with patch("interview_analyzer.dashboard._open_with_os_default", side_effect=OSError("no default browser")):
            dashboard._on_view_trends_infographic()

        inserted_text = "".join(
            call.args[1] for call in dashboard._trends_text.insert.call_args_list if len(call.args) > 1
        )
        assert "Couldn't open infographic" in inserted_text


class TestParseTranscriptLines:
    def test_parses_speaker_and_text(self):
        transcript = "[Interviewer] Hello there\n[You] Hi, nice to meet you"
        assert _parse_transcript_lines(transcript) == [
            ("Interviewer", "Hello there"),
            ("You", "Hi, nice to meet you"),
        ]

    def test_skips_blank_lines(self):
        transcript = "[You] Hi\n\n\n[Interviewer] Hello"
        assert _parse_transcript_lines(transcript) == [("You", "Hi"), ("Interviewer", "Hello")]

    def test_keeps_an_unrecognized_line_under_an_empty_speaker_rather_than_dropping_it(self):
        transcript = "not in the expected format\n[You] Hi"
        assert _parse_transcript_lines(transcript) == [
            ("", "not in the expected format"),
            ("You", "Hi"),
        ]


class TestGroupTranscriptBySpeaker:
    def test_merges_consecutive_same_speaker_lines_into_one_paragraph(self):
        transcript = (
            "[Interviewer] Welcome to the interview.\n"
            "[Interviewer] Today we'll cover a few topics.\n"
            "[You] Sounds good.\n"
        )
        assert _group_transcript_by_speaker(transcript) == [
            ("Interviewer", "Welcome to the interview. Today we'll cover a few topics."),
            ("You", "Sounds good."),
        ]

    def test_does_not_merge_across_a_different_speaker(self):
        transcript = "[You] A\n[Interviewer] B\n[You] C\n"
        assert _group_transcript_by_speaker(transcript) == [
            ("You", "A"), ("Interviewer", "B"), ("You", "C"),
        ]

    def test_nothing_to_merge_when_transcript_has_a_single_line(self):
        assert _group_transcript_by_speaker("[Speaker] hi") == [("Speaker", "hi")]

    def test_empty_transcript_produces_no_paragraphs(self):
        assert _group_transcript_by_speaker("") == []


class TestSpeakerColor:
    def test_you_and_interviewer_get_their_fixed_colors_case_insensitively(self):
        assigned = {}
        assert _speaker_color("You", assigned) == _speaker_color("you", assigned)
        assert _speaker_color("Interviewer", assigned) != _speaker_color("You", assigned)

    def test_an_unrecognized_speaker_gets_a_stable_color_within_one_render(self):
        assigned = {}
        first = _speaker_color("SPEAKER_00", assigned)
        second = _speaker_color("SPEAKER_00", assigned)
        assert first == second

    def test_different_unrecognized_speakers_get_different_colors(self):
        assigned = {}
        color_a = _speaker_color("SPEAKER_00", assigned)
        color_b = _speaker_color("SPEAKER_01", assigned)
        assert color_a != color_b

    def test_assigned_dict_is_not_shared_across_separate_render_calls(self):
        """A fresh `assigned` dict per render call means an unrecognized
        speaker's color depends only on the order speakers first appear
        *within that transcript*, not on some other transcript rendered
        earlier in the session."""
        assert _speaker_color("SPEAKER_00", {}) == _speaker_color("SPEAKER_00", {})


class TestRenderTranscriptWithSpeakerColors:
    def test_creates_one_tag_pair_per_distinct_speaker_and_inserts_grouped_text(self, tmp_path):
        dashboard = Dashboard(_watcher(tmp_path))
        text_widget = MagicMock()
        text_widget.tag_names.return_value = []

        transcript = "[Interviewer] Welcome.\n[Interviewer] Let's begin.\n[You] Thanks!\n"
        dashboard._render_transcript_with_speaker_colors(text_widget, transcript)

        insert_calls = [call.args for call in text_widget.insert.call_args_list]
        inserted_text = "".join(args[1] for args in insert_calls)
        assert "Welcome. Let's begin." in inserted_text
        assert "Thanks!" in inserted_text
        assert inserted_text.count("[Interviewer]") == 1  # merged, not repeated per line
        assert inserted_text.count("[You]") == 1

        configured_tags = {call.args[0] for call in text_widget.tag_configure.call_args_list}
        assert "speaker_label::Interviewer" in configured_tags
        assert "speaker_body::Interviewer" in configured_tags
        assert "speaker_label::You" in configured_tags

    def test_does_not_reconfigure_a_tag_that_already_exists(self, tmp_path):
        """Reusing an already-open History detail pane across selections
        must not spam tag_configure calls (or, more importantly, thrash
        colors) for speakers already seen."""
        dashboard = Dashboard(_watcher(tmp_path))
        text_widget = MagicMock()
        text_widget.tag_names.return_value = ["speaker_label::You", "speaker_body::You"]

        dashboard._render_transcript_with_speaker_colors(text_widget, "[You] Hi again")

        configured_tags = [call.args[0] for call in text_widget.tag_configure.call_args_list]
        assert "speaker_label::You" not in configured_tags


class TestViewInfographicButton:
    def _dashboard_with_selected(self, tmp_path, analysis=None):
        watcher = _watcher(tmp_path)
        iid = watcher.db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        if analysis is not None:
            watcher.db.save_analysis(iid, analysis)

        dashboard = Dashboard(watcher)
        dashboard._history_tree = MagicMock()
        dashboard._history_tree.selection.return_value = [str(iid)]
        for attr in (
            "_reprocess_btn", "_reprocess_with_profile_btn", "_open_audio_btn", "_view_transcript_btn",
            "_view_infographic_btn", "_delete_btn", "_cancel_btn",
        ):
            setattr(dashboard, attr, MagicMock())
        dashboard._history_text = MagicMock()
        return dashboard, iid

    def test_button_enabled_when_analysis_is_valid(self, tmp_path):
        dashboard, _ = self._dashboard_with_selected(tmp_path, analysis=VALID_ANALYSIS)

        dashboard._update_action_buttons()

        dashboard._view_infographic_btn.config.assert_called_with(state="normal")

    def test_button_disabled_when_analysis_is_malformed(self, tmp_path):
        dashboard, _ = self._dashboard_with_selected(
            tmp_path, analysis={"raw": "not the right shape", "parse_error": True}
        )

        dashboard._update_action_buttons()

        dashboard._view_infographic_btn.config.assert_called_with(state="disabled")

    def test_button_disabled_when_no_analysis_yet(self, tmp_path):
        dashboard, _ = self._dashboard_with_selected(tmp_path, analysis=None)

        dashboard._update_action_buttons()

        dashboard._view_infographic_btn.config.assert_called_with(state="disabled")

    def test_clicking_generates_and_opens_the_infographic(self, tmp_path):
        dashboard, iid = self._dashboard_with_selected(tmp_path, analysis=VALID_ANALYSIS)

        with patch("interview_analyzer.dashboard._open_with_os_default") as mock_open:
            dashboard._on_view_infographic()

        mock_open.assert_called_once()
        opened_path = mock_open.call_args.args[0]
        assert opened_path.exists()
        assert f"_{iid}_infographic.html" in opened_path.name

    def test_clicking_with_no_usable_analysis_does_not_try_to_open_anything(self, tmp_path):
        dashboard, _ = self._dashboard_with_selected(tmp_path, analysis=None)

        with patch("interview_analyzer.dashboard._open_with_os_default") as mock_open:
            dashboard._on_view_infographic()

        mock_open.assert_not_called()


class TestAssessmentProfileSettings:
    """The Settings tab's "Assessment profile" section (profiles.py's
    role/seniority/industry/company + competency selection, saved as named
    templates under this logged-in user). Widgets are mocked directly
    (same philosophy as this file's other tests -- see its docstring)
    rather than building the real Tk form."""

    def _dashboard_with_profile_widgets(
        self, tmp_path, role="(not specified)", seniority="(not specified)",
        industry="(not specified)", company="(not specified)", selected_competencies=None,
    ):
        watcher = _watcher(tmp_path)
        dashboard = Dashboard(watcher)
        dashboard._profile_role_var = MagicMock(get=MagicMock(return_value=role))
        dashboard._profile_seniority_var = MagicMock(get=MagicMock(return_value=seniority))
        dashboard._profile_industry_var = MagicMock(get=MagicMock(return_value=industry))
        dashboard._profile_company_var = MagicMock(get=MagicMock(return_value=company))
        selected = set(selected_competencies or [])
        dashboard._profile_competency_vars = {
            c: MagicMock(get=MagicMock(return_value=c in selected)) for c in CORE_COMPETENCIES
        }
        dashboard._profile_template_name_entry = MagicMock()
        dashboard._profile_template_picker = MagicMock()
        dashboard._profile_status_label = MagicMock()
        return dashboard

    def test_profile_from_settings_widgets_converts_not_specified_to_none(self, tmp_path):
        dashboard = self._dashboard_with_profile_widgets(tmp_path)
        profile = dashboard._profile_from_settings_widgets()
        assert profile.role is None
        assert profile.seniority is None
        assert profile.industry is None
        assert profile.company_type is None

    def test_profile_from_settings_widgets_reads_real_selections(self, tmp_path):
        dashboard = self._dashboard_with_profile_widgets(
            tmp_path, role="Product", seniority="Senior/Lead", industry="FinTech", company="FAANG / Big Tech",
            selected_competencies=["Leadership", "Execution"],
        )
        profile = dashboard._profile_from_settings_widgets()
        assert profile.role == "Product"
        assert profile.seniority == "Senior/Lead"
        assert profile.industry == "FinTech"
        assert profile.company_type == "FAANG / Big Tech"
        assert set(profile.competencies) == {"Leadership", "Execution"}

    def test_profile_from_settings_widgets_falls_back_to_all_competencies_when_none_checked(self, tmp_path):
        dashboard = self._dashboard_with_profile_widgets(tmp_path, selected_competencies=[])
        profile = dashboard._profile_from_settings_widgets()
        assert profile.competencies == CORE_COMPETENCIES

    def test_apply_profile_to_settings_widgets_sets_every_var(self, tmp_path):
        dashboard = self._dashboard_with_profile_widgets(tmp_path)
        profile = AssessmentProfile(competencies=["Leadership"], role="Sales", seniority="Entry Level")

        dashboard._apply_profile_to_settings_widgets(profile)

        dashboard._profile_role_var.set.assert_called_with("Sales")
        dashboard._profile_seniority_var.set.assert_called_with("Entry Level")
        dashboard._profile_industry_var.set.assert_called_with("(not specified)")
        dashboard._profile_competency_vars["Leadership"].set.assert_called_with(True)
        dashboard._profile_competency_vars["Execution"].set.assert_called_with(False)

    def test_save_profile_template_creates_it_from_current_widget_state(self, tmp_path):
        dashboard = self._dashboard_with_profile_widgets(tmp_path, role="Data", selected_competencies=["Execution"])
        dashboard._profile_template_name_entry.get.return_value = "My Template"

        dashboard._on_save_profile_template()

        templates = dashboard.watcher.db.list_profile_templates(user_id=1)
        assert len(templates) == 1
        assert templates[0].name == "My Template"
        assert templates[0].profile.role == "Data"

    def test_save_profile_template_with_a_blank_name_does_not_save(self, tmp_path):
        dashboard = self._dashboard_with_profile_widgets(tmp_path)
        dashboard._profile_template_name_entry.get.return_value = "   "

        dashboard._on_save_profile_template()

        assert dashboard.watcher.db.list_profile_templates(user_id=1) == []
        dashboard._profile_status_label.config.assert_called()

    def test_load_profile_template_applies_it_to_the_widgets(self, tmp_path):
        dashboard = self._dashboard_with_profile_widgets(tmp_path)
        dashboard.watcher.db.create_profile_template(
            user_id=1, name="My Template", profile=AssessmentProfile(role="Consultant"),
        )
        dashboard._profile_template_picker.get.return_value = "My Template"

        dashboard._on_load_profile_template()

        dashboard._profile_role_var.set.assert_called_with("Consultant")

    def test_set_active_profile_template_marks_it_active(self, tmp_path):
        dashboard = self._dashboard_with_profile_widgets(tmp_path)
        dashboard.watcher.db.create_profile_template(
            user_id=1, name="My Template", profile=AssessmentProfile(),
        )
        dashboard._profile_template_picker.get.return_value = "My Template"

        dashboard._on_set_active_profile_template()

        active = dashboard.watcher.db.get_active_profile_template(user_id=1)
        assert active is not None
        assert active.name == "My Template"

    def test_delete_profile_template_removes_it(self, tmp_path):
        dashboard = self._dashboard_with_profile_widgets(tmp_path)
        dashboard.watcher.db.create_profile_template(
            user_id=1, name="My Template", profile=AssessmentProfile(),
        )
        dashboard._profile_template_picker.get.return_value = "My Template"

        dashboard._on_delete_profile_template()

        assert dashboard.watcher.db.list_profile_templates(user_id=1) == []


class TestReprocessWithDifferentProfile:
    def _dashboard_with_selected_interview(self, tmp_path):
        watcher = _watcher(tmp_path)
        audio_path = tmp_path / "a.wav"
        audio_path.write_bytes(b"RIFF....WAVEreal audio bytes")  # can_reprocess needs real, non-empty audio
        iid = watcher.db.start_interview("Zoom", str(audio_path), retention_days=3, user_id=1)
        watcher.db.end_interview(iid)
        watcher.status = {"processing_jobs": {}}

        dashboard = Dashboard(watcher)
        dashboard._history_tree = MagicMock()
        dashboard._history_tree.selection.return_value = [str(iid)]
        for attr in (
            "_reprocess_btn", "_reprocess_with_profile_btn", "_open_audio_btn",
            "_view_transcript_btn", "_view_infographic_btn", "_delete_btn", "_cancel_btn",
        ):
            setattr(dashboard, attr, MagicMock())
        return dashboard, iid

    def test_shows_the_confirm_dialog_prefilled_from_the_current_profile_then_reprocesses(self, tmp_path):
        dashboard, iid = self._dashboard_with_selected_interview(tmp_path)
        existing = AssessmentProfile(role="Sales")
        dashboard.watcher.db.save_profile_snapshot(iid, existing)
        chosen = AssessmentProfile(competencies=["Execution"], role="Data")

        with patch("interview_analyzer.dashboard.threading.Thread", _ImmediateThread), \
             patch("interview_analyzer.dashboard.confirm_profile", return_value=chosen) as mock_confirm:
            dashboard._on_reprocess_with_profile()

        mock_confirm.assert_called_once()
        assert mock_confirm.call_args.args[0] == existing
        dashboard.watcher.reprocess_interview.assert_called_once_with(iid, profile=chosen)

    def test_intro_text_says_current_analysis_when_a_profile_is_saved(self, tmp_path):
        dashboard, iid = self._dashboard_with_selected_interview(tmp_path)
        dashboard.watcher.db.save_profile_snapshot(iid, AssessmentProfile(role="Sales"))

        with patch("interview_analyzer.dashboard.threading.Thread", _ImmediateThread), \
             patch("interview_analyzer.dashboard.confirm_profile", return_value=AssessmentProfile()) as mock_confirm:
            dashboard._on_reprocess_with_profile()

        assert "current analysis was run with" in mock_confirm.call_args.kwargs["intro_text"]

    def test_intro_text_says_no_saved_profile_when_none_exists(self, tmp_path):
        dashboard, iid = self._dashboard_with_selected_interview(tmp_path)
        # no save_profile_snapshot call -- interview predates the feature

        with patch("interview_analyzer.dashboard.threading.Thread", _ImmediateThread), \
             patch("interview_analyzer.dashboard.confirm_profile", return_value=AssessmentProfile()) as mock_confirm:
            dashboard._on_reprocess_with_profile()

        assert "no saved profile yet" in mock_confirm.call_args.kwargs["intro_text"]

    def test_no_op_when_nothing_selected(self, tmp_path):
        watcher = _watcher(tmp_path)
        watcher.status = {"processing_jobs": {}}
        dashboard = Dashboard(watcher)
        dashboard._history_tree = MagicMock()
        dashboard._history_tree.selection.return_value = []

        dashboard._on_reprocess_with_profile()  # must not raise

        dashboard.watcher.reprocess_interview.assert_not_called()

    def test_available_even_when_a_usable_report_already_exists(self, tmp_path):
        """Regression coverage for a real gap: plain Reprocess hides itself
        once a usable report exists (can_reprocess -- it only exists to
        recover a *missing* report), but redoing the analysis under
        different profile settings is a deliberate choice the user can make
        on an already-successful interview too, so it must not share that
        same gating."""
        dashboard, iid = self._dashboard_with_selected_interview(tmp_path)
        report_path = tmp_path / "report.md"
        report_path.write_text("# Report", encoding="utf-8")
        dashboard.watcher.db.save_report_path(iid, str(report_path))
        record = dashboard.watcher.db.get(iid)
        assert can_reprocess(record) is False  # sanity check: plain Reprocess would be hidden

        dashboard._update_action_buttons()

        dashboard._reprocess_btn.config.assert_called_with(state="disabled")
        dashboard._reprocess_with_profile_btn.config.assert_called_with(state="normal")

    def test_button_disabled_when_no_audio_at_all(self, tmp_path):
        dashboard, iid = self._dashboard_with_selected_interview(tmp_path)
        dashboard.watcher.db.update_audio_path(iid, str(tmp_path / "does_not_exist.wav"))

        dashboard._update_action_buttons()

        dashboard._reprocess_with_profile_btn.config.assert_called_with(state="disabled")


class TestPreviousAssessmentsSection:
    """The History tab's collapsible "Previous assessments" list, backed by
    db.list_analysis_history (see db.py's analysis_history table)."""

    def _dashboard_with_history_widgets(self, tmp_path):
        watcher = _watcher(tmp_path)
        iid = watcher.db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)

        dashboard = Dashboard(watcher)
        dashboard._assessment_history_tree = MagicMock()
        dashboard._assessment_history_tree.get_children.return_value = []
        dashboard._assessment_history_preview = MagicMock()
        dashboard._history_toggle_btn = MagicMock()
        return dashboard, iid

    def test_describe_profile_lists_the_non_empty_fields(self, tmp_path):
        dashboard, _ = self._dashboard_with_history_widgets(tmp_path)
        profile = AssessmentProfile(role="Product", seniority="Senior/Lead", industry=None, company_type=None)
        assert dashboard._describe_profile(profile) == "Product · Senior/Lead"

    def test_describe_profile_none_or_fully_generic_says_generic(self, tmp_path):
        dashboard, _ = self._dashboard_with_history_widgets(tmp_path)
        assert dashboard._describe_profile(None) == "Generic"
        assert dashboard._describe_profile(AssessmentProfile()) == "Generic"

    def test_refresh_populates_one_row_per_history_entry_newest_first(self, tmp_path):
        dashboard, iid = self._dashboard_with_history_widgets(tmp_path)
        dashboard.watcher.db.append_analysis_history(
            iid, {"session_summary": {"hire_recommendation": {"level": "Lean Hire"}}},
            profile=AssessmentProfile(role="Sales"),
        )
        dashboard.watcher.db.append_analysis_history(
            iid, {"session_summary": {"hire_recommendation": {"level": "Strong Hire"}}},
            profile=AssessmentProfile(role="Data"),
        )
        record = dashboard.watcher.db.get(iid)

        dashboard._refresh_assessment_history_section(record)

        assert dashboard._assessment_history_tree.insert.call_count == 2
        first_call_values = dashboard._assessment_history_tree.insert.call_args_list[0].kwargs["values"]
        assert first_call_values[1] == "Data"  # most recent (Strong Hire/Data) inserted first
        assert first_call_values[2] == "Strong Hire"
        dashboard._history_toggle_btn.config.assert_called_with(text="▶ Previous assessments (2)")

    def test_refresh_clears_the_list_when_nothing_is_selected(self, tmp_path):
        dashboard, _ = self._dashboard_with_history_widgets(tmp_path)

        dashboard._refresh_assessment_history_section(None)

        dashboard._assessment_history_tree.insert.assert_not_called()
        dashboard._history_toggle_btn.config.assert_called_with(text="▶ Previous assessments (0)")

    def test_toggle_expands_and_collapses(self, tmp_path):
        dashboard, iid = self._dashboard_with_history_widgets(tmp_path)
        dashboard._assessment_history_frame = MagicMock()
        dashboard._history_tree = MagicMock()
        dashboard._history_tree.selection.return_value = [str(iid)]

        assert dashboard._assessment_history_expanded is False
        dashboard._on_toggle_assessment_history()
        assert dashboard._assessment_history_expanded is True
        dashboard._assessment_history_frame.pack.assert_called_once()

        dashboard._on_toggle_assessment_history()
        assert dashboard._assessment_history_expanded is False
        dashboard._assessment_history_frame.pack_forget.assert_called_once()

    def test_selecting_an_entry_renders_its_historical_analysis(self, tmp_path):
        dashboard, iid = self._dashboard_with_history_widgets(tmp_path)
        dashboard._history_tree = MagicMock()
        dashboard._history_tree.selection.return_value = [str(iid)]
        dashboard.watcher.db.append_analysis_history(
            iid,
            {
                "qa_pairs": [],
                "session_summary": {
                    "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
                    "hire_recommendation": {"level": "Strong Hire", "rationale": "A past take."},
                },
            },
            profile=AssessmentProfile(role="Product"),
        )
        entry = dashboard.watcher.db.list_analysis_history(iid)[0]
        dashboard._assessment_history_tree.selection.return_value = [str(entry.id)]

        with patch("interview_analyzer.dashboard.render_into_text_widget") as mock_render:
            dashboard._on_select_history_entry()

        mock_render.assert_called_once()
        rendered_content = mock_render.call_args.args[1]
        assert "Strong Hire" in rendered_content
        assert "A past take." in rendered_content

    def test_selecting_nothing_is_a_no_op(self, tmp_path):
        dashboard, iid = self._dashboard_with_history_widgets(tmp_path)
        dashboard._history_tree = MagicMock()
        dashboard._history_tree.selection.return_value = [str(iid)]
        dashboard._assessment_history_tree.selection.return_value = []

        with patch("interview_analyzer.dashboard.render_into_text_widget") as mock_render:
            dashboard._on_select_history_entry()  # must not raise

        mock_render.assert_not_called()
