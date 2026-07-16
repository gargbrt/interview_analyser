"""Asks the user for explicit permission before recording *each* detected
meeting. Not every Teams/Meet/Zoom/Webex/Chime call is an interview — this
gate is what stops the tool from silently recording personal or unrelated
work calls.

Uses a small always-on-top Tkinter popup with Yes/No and an auto-timeout.
Default-on-timeout is **No** (fail safe: never record without a positive
response).

If `ui_root` is given (the app's shared dashboard Tk root -- see
dashboard.py/app.py), the popup is built as a `Toplevel` on that root's
existing thread instead of spinning up a second `Tk()` interpreter on a new
thread. Tcl/Tk does not support multiple `Tk()` interpreters running
concurrently on different OS threads within one process -- doing so can
trigger a hard process abort (`Tcl_Panic`, not a catchable Python
exception) when one interpreter tears down while another is active
elsewhere. `python -m interview_analyzer.watcher`'s headless mode has no
shared root, so it still runs the popup on its own dedicated thread
(nothing else is using Tk concurrently there, so that's safe).
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def _build_popup(window, app_name: str, timeout_seconds: int, result_queue: "queue.Queue[bool]") -> None:
    import tkinter as tk

    window.title("Interview Analyzer")
    window.attributes("-topmost", True)
    window.resizable(False, False)

    def _decide(value: bool) -> None:
        result_queue.put(value)
        window.destroy()

    tk.Label(
        window,
        text=(
            f"Detected a call in {app_name}.\n\n"
            "Record this call for interview analysis?\n"
            f"(auto-declines in {timeout_seconds}s if unanswered)"
        ),
        padx=20, justify="left",
    ).pack(pady=15)

    btn_frame = tk.Frame(window)
    btn_frame.pack(pady=(0, 15))
    tk.Button(btn_frame, text="Yes, record", width=12, command=lambda: _decide(True)).pack(
        side="left", padx=10
    )
    tk.Button(btn_frame, text="No", width=12, command=lambda: _decide(False)).pack(side="left", padx=10)

    window.protocol("WM_DELETE_WINDOW", lambda: _decide(False))
    window.after(timeout_seconds * 1000, lambda: _decide(False))


def ask_consent(app_name: str, timeout_seconds: int = 20, ui_root: Optional[object] = None) -> bool:
    """Show a popup asking whether to record this detected meeting.

    Returns True only on an explicit "Yes" click within the timeout.
    Any other outcome (No, closed, timeout) returns False.
    """
    result_queue: "queue.Queue[bool]" = queue.Queue()

    if ui_root is not None:
        try:
            import tkinter as tk

            def _schedule():
                _build_popup(tk.Toplevel(ui_root), app_name, timeout_seconds, result_queue)

            ui_root.after(0, _schedule)
        except Exception:  # noqa: BLE001
            logger.warning("Shared UI root unavailable for consent prompt; falling back to a standalone window.")
        else:
            try:
                return result_queue.get(timeout=timeout_seconds + 2)
            except queue.Empty:
                logger.warning("Consent prompt did not respond in time; defaulting to NOT recording.")
                return False

    def _show_popup():
        try:
            import tkinter as tk
        except ImportError:  # pragma: no cover
            logger.warning(
                "Tkinter not available; defaulting to NOT recording. "
                "Install Python with tk support to enable the consent prompt."
            )
            result_queue.put(False)
            return

        root = tk.Tk()
        _build_popup(root, app_name, timeout_seconds, result_queue)
        root.mainloop()

    thread = threading.Thread(target=_show_popup, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds + 2)

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        logger.warning("Consent prompt did not respond in time; defaulting to NOT recording.")
        return False
