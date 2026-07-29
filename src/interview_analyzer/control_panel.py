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

Can also be collapsed to a small pill (see minimize()/_show_compact) so it
sits out of the way of whatever you're actually looking at (e.g. the
interview window itself) instead of the full-size panel -- hovering over
the pill temporarily reveals the full Pause/Resume/Stop controls again
without needing to restore it first, and it collapses back once the mouse
moves away.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Indeterminate-progress-bar step interval, in ms, for the "capturing"
# activity animation. Was 80ms; bumped up since this is purely decorative
# (it has zero bearing on the recording itself, which never touches this
# widget) and runs continuously for the entire length of a recording --
# 80ms means ~45,000 redraw ticks over a single 1-hour interview. 150ms
# still reads as "clearly animating" while cutting that by ~45%.
_ACTIVITY_STEP_MS = 150

# How long to wait, after the mouse leaves the panel, before collapsing a
# hover-revealed full view back down to the compact pill. Not just a UX nicety:
# Tk's crossing events can fire a spurious <Leave> on a container the instant
# the pointer moves onto one of its own child widgets (e.g. the Pause
# button), immediately followed by a corresponding <Enter> -- a delay here
# absorbs that blip instead of visibly flickering between compact and full
# on every mouse movement over the controls.
_COLLAPSE_DELAY_MS = 300


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
        self._full_frame = None
        self._compact_frame = None
        self._pause_btn = None
        self._status_label = None
        self._timer_label = None
        self._activity_bar = None
        self._compact_dot = None
        self._compact_timer_label = None
        self._recorded_seconds = 0  # excludes paused time -- mirrors what's actually on disk
        self._ready = threading.Event()
        # True once minimize() has been used -- the panel's "chosen" display
        # mode. Independent of whether the full controls are momentarily
        # visible right now (see _peeking) -- hovering over a compact panel
        # shows the full controls without leaving compact mode.
        self._compact = False
        self._collapse_after_id = None

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

        self._full_frame = tk.Frame(window)

        tk.Label(
            self._full_frame, text=f"Recording {self.app_name}", padx=15, justify="left",
        ).pack(pady=(10, 0))
        self._status_label = tk.Label(self._full_frame, text="● Recording", fg="red", padx=15)
        self._status_label.pack(pady=(0, 2))

        self._timer_label = tk.Label(self._full_frame, text="00:00", font=("Consolas", 13), padx=15)
        self._timer_label.pack(pady=(0, 6))

        self._activity_bar = ttk.Progressbar(self._full_frame, mode="indeterminate", length=200)
        self._activity_bar.pack(padx=15, pady=(0, 10))
        self._activity_bar.start(_ACTIVITY_STEP_MS)

        btn_frame = tk.Frame(self._full_frame)
        btn_frame.pack(pady=(0, 12))

        def _toggle_pause():
            if self._paused:
                self._paused = False
                self._pause_btn.config(text="Pause")
                self._status_label.config(text="● Recording", fg="red")
                self._activity_bar.start(_ACTIVITY_STEP_MS)
                self._on_resume()
            else:
                self._paused = True
                self._pause_btn.config(text="Resume")
                self._status_label.config(text="⏸ Paused", fg="gray")
                self._activity_bar.stop()
                self._on_pause()
            self._sync_compact_status()

        def _stop():
            if self._stopped:
                return
            self._stopped = True
            self._on_stop()
            self.close()

        self._pause_btn = tk.Button(btn_frame, text="Pause", width=10, command=_toggle_pause)
        self._pause_btn.pack(side="left", padx=8)
        tk.Button(btn_frame, text="Stop", width=10, command=_stop).pack(side="left", padx=8)
        tk.Button(btn_frame, text="–", width=2, command=self.minimize).pack(side="left", padx=(8, 0))

        # Compact "pill" view -- shown instead of _full_frame once minimize()
        # is used, so the panel takes up a lot less screen space (e.g. it no
        # longer needs to sit on top of whatever window you're actually
        # looking at) while still surfacing at a glance whether it's
        # recording/paused and for how long.
        self._compact_frame = tk.Frame(window)
        self._compact_dot = tk.Label(self._compact_frame, text="●", fg="red")
        self._compact_dot.pack(side="left", padx=(10, 4))
        self._compact_timer_label = tk.Label(self._compact_frame, text="00:00", font=("Consolas", 10))
        self._compact_timer_label.pack(side="left")
        tk.Label(self._compact_frame, text="  (hover for controls)", fg="gray", font=("Segoe UI", 8)).pack(
            side="left", padx=(0, 10),
        )

        # Hovering over the panel -- in either mode -- reveals the full
        # controls; moving the mouse away collapses back to the pill after
        # a short delay (see _COLLAPSE_DELAY_MS) if minimize() was used.
        window.bind("<Enter>", self._on_hover_enter)
        window.bind("<Leave>", self._on_hover_leave)

        self._show_full()

        # Closing the window via the titlebar X behaves like Stop rather
        # than silently leaving a live recording with no visible control.
        window.protocol("WM_DELETE_WINDOW", _stop)

        self._tick_timer(window)
        self._ready.set()

    def minimize(self) -> None:
        """Collapse the panel to a small pill showing just the recording
        status and elapsed time. Safe to call from any thread."""
        if self._root is None:
            return
        self._compact = True
        self._root.after(0, self._show_compact)

    def restore(self) -> None:
        """Return to the full panel permanently (not just a hover peek).
        Safe to call from any thread."""
        if self._root is None:
            return
        self._compact = False
        self._root.after(0, self._show_full)

    def _show_full(self) -> None:
        if self._root is None:
            return
        if self._collapse_after_id is not None:
            self._root.after_cancel(self._collapse_after_id)
            self._collapse_after_id = None
        self._compact_frame.pack_forget()
        self._full_frame.pack()

    def _show_compact(self) -> None:
        if self._root is None:
            return
        self._collapse_after_id = None
        self._full_frame.pack_forget()
        self._compact_frame.pack()

    def _on_hover_enter(self, _event=None) -> None:
        if self._collapse_after_id is not None:
            self._root.after_cancel(self._collapse_after_id)
            self._collapse_after_id = None
        if self._compact:
            # peek: show full controls without leaving compact mode, so
            # moving the mouse away (see _on_hover_leave) collapses it again
            self._compact_frame.pack_forget()
            self._full_frame.pack()

    def _on_hover_leave(self, _event=None) -> None:
        if not self._compact or self._root is None:
            return
        if self._collapse_after_id is not None:
            self._root.after_cancel(self._collapse_after_id)
        self._collapse_after_id = self._root.after(_COLLAPSE_DELAY_MS, self._show_compact)

    def _sync_compact_status(self) -> None:
        """Keeps the compact pill's status dot in sync with pause/resume,
        even while it isn't currently visible."""
        if self._compact_dot is None:
            return
        if self._paused:
            self._compact_dot.config(fg="gray")
        else:
            self._compact_dot.config(fg="red")

    def _tick_timer(self, window) -> None:
        if self._root is None:
            return
        if not self._paused and not self._stopped:
            self._recorded_seconds += 1
            mins, secs = divmod(self._recorded_seconds, 60)
            text = f"{mins:02d}:{secs:02d}"
            self._timer_label.config(text=text)
            self._compact_timer_label.config(text=text)
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
