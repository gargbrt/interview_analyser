"""Small always-on-top control panel shown while a call is being recorded,
letting you pause/resume audio capture or stop the recording (and kick off
transcription/analysis) early, without waiting for the meeting app to close.

Uses the same lightweight Tkinter-in-a-thread pattern as consent.py, but
stays open for the duration of the recording instead of showing once.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class RecordingControlPanel:
    """Always-on-top Pause/Resume + Stop control for an in-progress recording.

    `on_pause` / `on_resume` / `on_stop` are invoked on the panel's own Tk
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
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self) -> None:
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
        self._root = root
        root.title("Interview Analyzer")
        root.attributes("-topmost", True)
        root.resizable(False, False)

        tk.Label(
            root, text=f"Recording {self.app_name}", padx=15, justify="left",
        ).pack(pady=(10, 0))
        self._status_label = tk.Label(root, text="● Recording", fg="red", padx=15)
        self._status_label.pack(pady=(0, 10))

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=(0, 12))

        def _toggle_pause():
            if self._paused:
                self._paused = False
                self._pause_btn.config(text="Pause")
                self._status_label.config(text="● Recording", fg="red")
                self._on_resume()
            else:
                self._paused = True
                self._pause_btn.config(text="Resume")
                self._status_label.config(text="⏸ Paused", fg="gray")
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
        root.protocol("WM_DELETE_WINDOW", _stop)

        self._ready.set()
        root.mainloop()

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
