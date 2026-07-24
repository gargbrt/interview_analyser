"""Confirms (or lets the user adjust) which AssessmentProfile a new
interview will be scored against, right after recording consent is granted.

Unlike consent.py's Yes/No gate, failing to respond here must never lose a
recording -- recording has already started by the time this shows (see
watcher.py). So there is no punishing auto-decline: on a timeout, on
Tkinter being unavailable, or on any other failure, this simply returns the
profile it was shown with (the currently active default), unedited.

Mirrors consent.py's shared-root-vs-own-thread pattern for the same reason
consent.py does: Tcl/Tk doesn't support multiple `Tk()` interpreters
concurrently on different OS threads in one process.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from .profiles import (
    COMPANY_TYPES,
    CORE_COMPETENCIES,
    INDUSTRIES,
    ROLES,
    SENIORITIES,
    AssessmentProfile,
)

logger = logging.getLogger(__name__)

_NOT_SPECIFIED = "(not specified)"


def _profile_from_selections(
    role: str, seniority: str, industry: str, company_type: str, selected_competencies: list[str],
) -> AssessmentProfile:
    """Pure conversion from the dialog's raw widget values into an
    AssessmentProfile -- separated from _build_popup so this logic (the
    "(not specified)" placeholder -> None, and "nothing checked -> fall
    back to every competency" rules) is directly testable without a live
    Tk window."""
    def _none_if_blank(value: str) -> Optional[str]:
        return None if value == _NOT_SPECIFIED else value

    return AssessmentProfile(
        competencies=[c for c in CORE_COMPETENCIES if c in selected_competencies] or list(CORE_COMPETENCIES),
        role=_none_if_blank(role),
        seniority=_none_if_blank(seniority),
        industry=_none_if_blank(industry),
        company_type=_none_if_blank(company_type),
    )


_DEFAULT_INTRO = (
    "Confirm the assessment profile for this interview.\n"
    "Adjust anything below, or just confirm the prefilled values."
)


def _build_popup(
    window, active_profile: AssessmentProfile, result_queue: "queue.Queue[AssessmentProfile]",
    intro_text: Optional[str] = None,
) -> None:
    import tkinter as tk
    from tkinter import ttk

    window.title("Interview Analyzer")
    window.attributes("-topmost", True)
    window.resizable(False, False)

    tk.Label(window, text=intro_text or _DEFAULT_INTRO, justify="left").pack(padx=16, pady=(12, 6))

    form = ttk.Frame(window, padding=(16, 4))
    form.pack(fill="x")

    def _dropdown(row: int, label: str, options: list[str], current: Optional[str]) -> tk.StringVar:
        ttk.Label(form, text=label, width=14).grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=current or _NOT_SPECIFIED)
        combo = ttk.Combobox(form, textvariable=var, values=[_NOT_SPECIFIED] + list(options), state="readonly", width=24)
        combo.grid(row=row, column=1, sticky="w", pady=2)
        return var

    role_var = _dropdown(0, "Role", ROLES, active_profile.role)
    seniority_var = _dropdown(1, "Seniority", SENIORITIES, active_profile.seniority)
    industry_var = _dropdown(2, "Industry", INDUSTRIES, active_profile.industry)
    company_var = _dropdown(3, "Company type", COMPANY_TYPES, active_profile.company_type)

    ttk.Label(window, text="Competencies to score:", justify="left").pack(anchor="w", padx=16, pady=(10, 2))
    checks_frame = ttk.Frame(window, padding=(16, 0))
    checks_frame.pack(fill="x")
    selected = set(active_profile.competencies)
    competency_vars: dict[str, tk.BooleanVar] = {}
    for i, competency in enumerate(CORE_COMPETENCIES):
        var = tk.BooleanVar(value=competency in selected)
        competency_vars[competency] = var
        tk.Checkbutton(checks_frame, text=competency, variable=var).grid(
            row=i // 2, column=i % 2, sticky="w", padx=(0, 12)
        )

    def _confirm() -> None:
        selected_competencies = [c for c in CORE_COMPETENCIES if competency_vars[c].get()]
        profile = _profile_from_selections(
            role_var.get(), seniority_var.get(), industry_var.get(), company_var.get(), selected_competencies,
        )
        result_queue.put(profile)
        window.destroy()

    btn_frame = tk.Frame(window)
    btn_frame.pack(pady=14)
    tk.Button(btn_frame, text="Confirm", width=14, command=_confirm).pack()

    # Closing the window (or a timeout, see confirm_profile) must still
    # produce a usable profile -- fall back to the active one exactly as
    # shown, never lose the recording over an unanswered dialog.
    window.protocol("WM_DELETE_WINDOW", lambda: (result_queue.put(active_profile), window.destroy()))


def confirm_profile(
    active_profile: AssessmentProfile, ui_root: Optional[object] = None, timeout_seconds: int = 60,
    intro_text: Optional[str] = None,
) -> AssessmentProfile:
    """Shows a confirm/edit popup prefilled from `active_profile`. Returns
    the (possibly edited) profile the user confirmed, or `active_profile`
    unchanged if the dialog times out, Tkinter isn't available, or anything
    else goes wrong -- recording has already started by the time this is
    called, so there is no fail-safe-by-declining here, only a fail-safe
    default.

    `intro_text`, if given, replaces the dialog's default explanatory
    label -- e.g. dashboard.py's "Reprocess with different profile" flow
    uses it to make clear the prefilled values are what the *existing*
    analysis was actually run with, not just some arbitrary default."""
    result_queue: "queue.Queue[AssessmentProfile]" = queue.Queue()

    if ui_root is not None:
        try:
            import tkinter as tk

            def _schedule():
                _build_popup(tk.Toplevel(ui_root), active_profile, result_queue, intro_text=intro_text)

            ui_root.after(0, _schedule)
        except Exception:  # noqa: BLE001
            logger.warning("Shared UI root unavailable for profile confirmation; using the active profile as-is.")
            return active_profile
        else:
            try:
                return result_queue.get(timeout=timeout_seconds + 2)
            except queue.Empty:
                logger.info("Profile confirmation dialog didn't respond in time; using the active profile as-is.")
                return active_profile

    def _show_popup():
        try:
            import tkinter as tk
        except ImportError:  # pragma: no cover
            logger.warning("Tkinter not available; using the active profile as-is.")
            result_queue.put(active_profile)
            return

        root = tk.Tk()
        _build_popup(root, active_profile, result_queue, intro_text=intro_text)
        root.mainloop()

    thread = threading.Thread(target=_show_popup, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds + 2)

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        logger.info("Profile confirmation dialog didn't respond in time; using the active profile as-is.")
        return active_profile
