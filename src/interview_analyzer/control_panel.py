"""Small always-on-top control panel shown while a call is being recorded,
letting you pause/resume audio capture or stop the recording (and kick off
transcription/analysis) early, without waiting for the meeting app to close.
Shows a live elapsed-recording timer and an activity indicator that's only
animated while actually capturing (it stops while paused).

If `ui_root` is given (the app's shared dashboard Tk root -- see
dashboard.py/app.py), the panel is built as a `Toplevel` on that root's
existing thread instead of spinning up a second `Tk()` interpreter on a new
thread -- see consent.py's module docstring for why running multiple `Tk()`
interpreters concurrently on different threads is unsafe (it can hard-crash
the whole process, not just raise a catchable exception).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class RecordingControlPanel:
    """Always-on-top Pause/Resume + Stop control for an in-progress recording.

    `on_pause` / `on_resume` / `on_stop` are invoked on the panel's Tk
    thread -- keep them fast and thread-safe (e.g. setting a threading.Event
    or calling SystemAudioRecorder.pause()/resume(), both safe from any
    thread).
    """

    def __init__(
        self,
        app_name: str,
        on_pause: Callable[[], None],
        on_resume: Callable[[], None],
        on_stop: Callable[[], None],
        ui_root: Optional[object] = None,
    ):
        self.app_name = app_name
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_stop = on_stop
        self._paused = False
        self._stopped = False
        self._root = None
        self._pause_btn = None
        self._status_label = None
        self._timer_label = None
        self._activity_bar = None
        self._recorded_seconds = 0  # excludes paused time -- mirrors what's actually on disk
        self._ready = threading.Event()

        if ui_root is not None and self._try_build_on_shared_root(ui_root):
            return
        self._thread = threading.Thread(target=self._run_standalone, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _try_build_on_shared_root(self, ui_root) -> bool:
        try:
            import tkinter as tk

            ui_root.after(0, lambda: self._build(tk.Toplevel(ui_root)))
        except Exception:  # noqa: BLE001
            logger.warning("Shared UI root unavailable for control panel; falling back to a standalone window.")
            return False
        if not self._ready.wait(timeout=5):
            logger.warning("Control panel window did not become ready in time.")
        return True

    def _run_standalone(self) -> None:
        try:
            import tkinter as tk
        except ImportError:  # pragma: no cover
            logger.warning(
                "Tkinter not available; recording pause/stop controls are disabled. "
                "The recording will still stop automatically when the call ends."
            )
            self._ready.set()
            return

        root = tk.Tk()
        self._build(root)
        root.mainloop()

    def _build(self, window) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._root = window
        window.title("Interview Analyzer")
        window.attributes("-topmost", True)
        window.resizable(False, False)

        tk.Label(
            window, text=f"Recording {self.app_name}", padx=15, justify="left",
        ).pack(pady=(10, 0))
        self._status_label = tk.Label(window, text="● Recording", fg="red", padx=15)
        self._status_label.pack(pady=(0, 2))

        self._timer_label = tk.Label(window, text="00:00", font=("Consolas", 13), padx=15)
        self._timer_label.pack(pady=(0, 6))

        self._activity_bar = ttk.Progressbar(window, mode="indeterminate", length=200)
        self._activity_bar.pack(padx=15, pady=(0, 10))
        self._activity_bar.start(80)

        btn_frame = tk.Frame(window)
        btn_frame.pack(pady=(0, 12))

        def _toggle_pause():
            if self._paused:
                self._paused = False
                self._pause_btn.config(text="Pause")
                self._status_label.config(text="● Recording", fg="red")
                self._activity_bar.start(80)
                self._on_resume()
            else:
                self._paused = True
                self._pause_btn.config(text="Resume")
                self._status_label.config(text="⏸ Paused", fg="gray")
                self._activity_bar.stop()
                self._on_pause()

        def _stop():
            if self._stopped:
                return
            self._stopped = True
            self._on_stop()
            self.close()

        self._pause_btn = tk.Button(btn_frame, text="Pause", width=10, command=_toggle_pause)
        self._pause_btn.pack(side="left", padx=8)
        tk.Button(btn_frame, text="Stop", width=10, command=_stop).pack(side="left", padx=8)

        # Closing the window via the titlebar X behaves like Stop rather
        # than silently leaving a live recording with no visible control.
        window.protocol("WM_DELETE_WINDOW", _stop)

        self._tick_timer(window)
        self._ready.set()

    def _tick_timer(self, window) -> None:
        if self._root is None:
            return
        if not self._paused and not self._stopped:
            self._recorded_seconds += 1
            mins, secs = divmod(self._recorded_seconds, 60)
            self._timer_label.config(text=f"{mins:02d}:{secs:02d}")
        window.after(1000, lambda: self._tick_timer(window))

    def close(self) -> None:
        """Close the panel. Safe to call from any thread, and safe to call
        more than once (e.g. both from its own Stop button and from the
        watcher once the meeting ends on its own)."""
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:  # noqa: BLE001
                pass
            self._root = None
