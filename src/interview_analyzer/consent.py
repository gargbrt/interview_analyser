"""Asks the user for explicit permission before recording *each* detected
meeting. Not every Teams/Meet/Zoom/Webex/Chime call is an interview — this
gate is what stops the tool from silently recording personal or unrelated
work calls.

Uses a small always-on-top Tkinter popup with Yes/No and an auto-timeout.
Default-on-timeout is **No** (fail safe: never record without a positive
response). Runs in its own thread/process-safe call so it doesn't block the
watcher's polling loop for long.
"""
from __future__ import annotations

import logging
import queue
import threading

logger = logging.getLogger(__name__)


def ask_consent(app_name: str, timeout_seconds: int = 20) -> bool:
    """Show a popup asking whether to record this detected meeting.

    Returns True only on an explicit "Yes" click within the timeout.
    Any other outcome (No, closed, timeout) returns False.
    """
    result_queue: "queue.Queue[bool]" = queue.Queue()

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
        root.title("Interview Analyzer")
        root.attributes("-topmost", True)
        root.resizable(False, False)

        answered = {"value": False}

        def _yes():
            answered["value"] = True
            root.destroy()

        def _no():
            answered["value"] = False
            root.destroy()

        tk.Label(
            root,
            text=(
                f"Detected a call in {app_name}.\n\n"
                "Record this call for interview analysis?\n"
                f"(auto-declines in {timeout_seconds}s if unanswered)"
            ),
            padx=20, pady=15, justify="left",
        ).pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=(0, 15))
        tk.Button(btn_frame, text="Yes, record", width=12, command=_yes).pack(
            side="left", padx=10
        )
        tk.Button(btn_frame, text="No", width=12, command=_no).pack(side="left", padx=10)

        root.after(timeout_seconds * 1000, root.destroy)
        root.mainloop()
        result_queue.put(answered["value"])

    thread = threading.Thread(target=_show_popup, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds + 2)

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        logger.warning("Consent prompt did not respond in time; defaulting to NOT recording.")
        return False
