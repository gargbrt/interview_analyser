"""Tests for profile_confirm.py.

Like consent.py (which has no direct tests of its own -- see watcher.py's
tests, which mock ask_consent at the boundary instead), the live Tk dialog
itself isn't unit-tested here. What IS tested directly:
  - _profile_from_selections, the pure conversion from raw widget values to
    an AssessmentProfile (extracted specifically so this logic doesn't need
    a real Tk window to verify).
  - confirm_profile's fail-safe contract: recording has already started by
    the time this is shown, so a timeout/failure must return the active
    profile unchanged, never block or lose anything.
"""
from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

from interview_analyzer.profile_confirm import _DEFAULT_INTRO, _build_popup, _profile_from_selections, confirm_profile
from interview_analyzer.profiles import CORE_COMPETENCIES, GENERIC_PROFILE, AssessmentProfile


class TestProfileFromSelections:
    def test_not_specified_placeholder_becomes_none(self):
        profile = _profile_from_selections(
            "(not specified)", "(not specified)", "(not specified)", "(not specified)",
            selected_competencies=["Leadership"],
        )
        assert profile.role is None
        assert profile.seniority is None
        assert profile.industry is None
        assert profile.company_type is None

    def test_real_selections_pass_through(self):
        profile = _profile_from_selections(
            "Software Engineer", "Senior/Lead", "FinTech", "FAANG / Big Tech",
            selected_competencies=["Leadership", "Execution"],
        )
        assert profile.role == "Software Engineer"
        assert profile.seniority == "Senior/Lead"
        assert profile.industry == "FinTech"
        assert profile.company_type == "FAANG / Big Tech"
        assert set(profile.competencies) == {"Leadership", "Execution"}

    def test_no_competencies_checked_falls_back_to_all(self):
        """Matches the "no parameters selected -> generic" requirement --
        unchecking every box shouldn't silently produce an empty,
        unscoreable profile."""
        profile = _profile_from_selections(
            "(not specified)", "(not specified)", "(not specified)", "(not specified)",
            selected_competencies=[],
        )
        assert profile.competencies == CORE_COMPETENCIES

    def test_preserves_the_fixed_competency_order_regardless_of_input_order(self):
        profile = _profile_from_selections(
            "(not specified)", "(not specified)", "(not specified)", "(not specified)",
            selected_competencies=["Execution", "Leadership"],  # reversed vs. CORE_COMPETENCIES order
        )
        assert profile.competencies == [c for c in CORE_COMPETENCIES if c in ("Execution", "Leadership")]


class TestConfirmProfileFailSafe:
    """confirm_profile must never lose the recording it's attached to --
    recording has already started by the time this is shown (see
    watcher.py), so every failure/timeout path returns the active profile
    unchanged rather than raising or blocking indefinitely."""

    def test_returns_active_profile_when_shared_root_never_responds(self):
        active = AssessmentProfile(competencies=["Leadership"], role="Product")
        fake_root = MagicMock()  # .after(...) never actually invokes the scheduled popup

        result = confirm_profile(active, ui_root=fake_root, timeout_seconds=0)

        assert result == active

    def test_returns_active_profile_when_scheduling_on_the_shared_root_raises(self):
        active = AssessmentProfile(competencies=["Execution"], role="Sales")
        fake_root = MagicMock()
        fake_root.after.side_effect = RuntimeError("root already destroyed")

        result = confirm_profile(active, ui_root=fake_root, timeout_seconds=5)

        assert result == active

    def test_returns_active_profile_when_standalone_thread_never_responds(self):
        active = AssessmentProfile(competencies=["Collaboration"])

        with patch("interview_analyzer.profile_confirm.threading.Thread") as MockThread:
            MockThread.return_value.join.return_value = None  # thread "runs" but never posts a result
            result = confirm_profile(active, ui_root=None, timeout_seconds=0)

        assert result == active


class TestBuildPopupIntroText:
    """A real (offscreen) Tk window -- verifies the actual widget text,
    which is exactly what dashboard.py's "Reprocess with different
    profile" flow relies on to make clear the prefilled values reflect the
    interview's *current* analysis (see profile_confirm.confirm_profile's
    intro_text param)."""

    def _first_label_text(self, popup) -> str:
        return popup.winfo_children()[0].cget("text")

    def test_uses_the_default_intro_when_none_given(self):
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            popup = tk.Toplevel(root)
            _build_popup(popup, GENERIC_PROFILE, queue.Queue())
            assert self._first_label_text(popup) == _DEFAULT_INTRO
        finally:
            root.destroy()

    def test_uses_a_custom_intro_when_given(self):
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            popup = tk.Toplevel(root)
            _build_popup(popup, GENERIC_PROFILE, queue.Queue(), intro_text="Custom reprocess explanation")
            assert self._first_label_text(popup) == "Custom reprocess explanation"
        finally:
            root.destroy()
