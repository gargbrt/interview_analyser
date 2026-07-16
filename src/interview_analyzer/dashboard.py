"""The main dashboard window: current status + pause/resume/stop, browsable
interview history with a readable report view, the cross-interview trends
report, and an editable settings form.

Opened on demand from the tray icon's "Open dashboard" item (see tray.py)
or the app's own request. Like consent.py/control_panel.py, it runs in its
own thread with its own Tk root -- Tkinter isn't thread-safe for arbitrary
cross-thread calls, so every window in this app owns its thread rather than
sharing one Tk mainloop.
"""
from __future__ import annotations

import logging
import pathlib
import threading
from typing import Optional

from .report_view import render_into_text_widget
from .settings_editor import load_editable_settings, save_editable_settings

logger = logging.getLogger(__name__)

_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
_ANALYSIS_ENGINES = ["ollama", "anthropic_api", "openai_api"]


def _configure_report_tags(text_widget) -> None:
    text_widget.tag_configure("h1", font=("Georgia", 15, "bold"), spacing3=6)
    text_widget.tag_configure("h2", font=("Georgia", 12, "bold"), spacing1=10, spacing3=4)
    text_widget.tag_configure("h3", font=("Georgia", 11, "bold"), spacing1=8, spacing3=2)
    text_widget.tag_configure("bullet", lmargin1=14, lmargin2=28, spacing3=2)
    text_widget.tag_configure("quote", font=("Segoe UI", 9, "italic"), foreground="#5b645f")
    text_widget.tag_configure("text", spacing3=2)


class Dashboard:
    def __init__(self, watcher):
        self.watcher = watcher
        self._lock = threading.Lock()
        self._root = None
        # set once the window is fully built and about to enter mainloop();
        # cleared on close. Guards other threads (tray, watcher notify,
        # tests) from touching widgets while they're still being built --
        # Tkinter raises "main thread is not in main loop" if you call into
        # a Tk instance from another thread before its loop is running.
        self._ready = threading.Event()
        # widgets refreshed by _refresh_status / notify callbacks; only
        # touched from the dashboard's own Tk thread
        self._status_label = None
        self._detail_label = None
        self._pause_btn = None
        self._stop_btn = None
        self._history_tree = None
        self._history_text = None
        self._trends_text = None
        self._settings_widgets: dict[str, object] = {}
        self._settings_status = None

    @property
    def is_open(self) -> bool:
        return self._root is not None

    def open(self) -> None:
        """Show the dashboard, creating it if not already open. If it's
        already open this just logs and returns -- bringing an
        already-open Tk window on another thread to the foreground isn't
        reliably safe to do cross-thread, so we don't attempt it."""
        with self._lock:
            if self._root is not None:
                logger.info("Dashboard already open.")
                return
            thread = threading.Thread(target=self._run, daemon=True)
            thread.start()

    def notify_state_change(self) -> None:
        """Called (from any thread) whenever the watcher's recording state
        changes, so an open dashboard can refresh itself."""
        if self._ready.is_set() and self._root is not None:
            try:
                self._root.after(0, self._refresh_status)
            except Exception:  # noqa: BLE001
                pass

    # -- window setup -----------------------------------------------------

    def _run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError:  # pragma: no cover
            logger.warning("Tkinter not available; cannot open the dashboard.")
            return

        root = tk.Tk()
        self._root = root
        root.title("Interview Analyzer")
        root.geometry("720x520")
        root.minsize(560, 400)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        notebook.add(self._build_status_tab(notebook, tk, ttk), text="Status")
        notebook.add(self._build_history_tab(notebook, tk, ttk), text="History")
        notebook.add(self._build_trends_tab(notebook, tk, ttk), text="Trends")
        notebook.add(self._build_settings_tab(notebook, tk, ttk), text="Settings")

        def _on_close():
            self._ready.clear()
            self._root = None
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _on_close)

        self._refresh_status()
        self._refresh_history()
        self._refresh_trends()
        self._ready.set()
        root.mainloop()

    # -- Status tab ---------------------------------------------------------

    def _build_status_tab(self, notebook, tk, ttk):
        frame = ttk.Frame(notebook, padding=16)

        self._status_label = ttk.Label(frame, text="", font=("Segoe UI", 14, "bold"))
        self._status_label.pack(anchor="w")
        self._detail_label = ttk.Label(frame, text="", foreground="#5b645f")
        self._detail_label.pack(anchor="w", pady=(2, 16))

        btn_row = ttk.Frame(frame)
        btn_row.pack(anchor="w")
        self._pause_btn = ttk.Button(btn_row, text="Pause", command=self._on_toggle_pause)
        self._pause_btn.pack(side="left", padx=(0, 8))
        self._stop_btn = ttk.Button(btn_row, text="Stop recording", command=self._on_stop)
        self._stop_btn.pack(side="left")

        ttk.Label(
            frame,
            text="The app watches for Teams/Meet/Webex/Zoom/Chime in the background.\n"
                 "You'll be asked for consent before each recording starts.",
            foreground="#5b645f", justify="left",
        ).pack(anchor="w", pady=(20, 0))

        return frame

    def _on_toggle_pause(self) -> None:
        if self.watcher.status.get("state") == "paused":
            self.watcher.resume_recording()
        else:
            self.watcher.pause_recording()
        self._refresh_status()

    def _on_stop(self) -> None:
        self.watcher.request_stop_recording()
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self._root is None or self._status_label is None:
            return
        status = self.watcher.status
        state = status.get("state", "idle")

        if state == "idle":
            self._status_label.config(text="● Idle", foreground="#5b645f")
            self._detail_label.config(text="Watching for a meeting to begin.")
            self._pause_btn.config(text="Pause", state="disabled")
            self._stop_btn.config(state="disabled")
        elif state == "recording":
            self._status_label.config(text="● Recording", foreground="#c0392b")
            self._detail_label.config(text=f"{status.get('app_name', 'call')}")
            self._pause_btn.config(text="Pause", state="normal")
            self._stop_btn.config(state="normal")
        else:  # paused
            self._status_label.config(text="⏸ Paused", foreground="#c8892c")
            self._detail_label.config(text=f"{status.get('app_name', 'call')} — capture paused")
            self._pause_btn.config(text="Resume", state="normal")
            self._stop_btn.config(state="normal")

        if self._root is not None:
            self._root.after(1000, self._refresh_status)

    # -- History tab ----------------------------------------------------

    def _build_history_tab(self, notebook, tk, ttk):
        frame = ttk.Frame(notebook, padding=10)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(1, weight=1)

        ttk.Button(frame, text="Refresh", command=self._refresh_history).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        columns = ("date", "app", "top_issue")
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        tree.heading("date", text="Date")
        tree.heading("app", text="App")
        tree.heading("top_issue", text="Top issue")
        tree.column("date", width=110, stretch=False)
        tree.column("app", width=90, stretch=False)
        tree.column("top_issue", width=220)
        tree.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        tree.bind("<<TreeviewSelect>>", lambda e: self._on_history_select())
        self._history_tree = tree

        text = tk.Text(frame, wrap="word", padx=10, pady=8, state="disabled", relief="flat")
        _configure_report_tags(text)
        text.grid(row=1, column=1, sticky="nsew")
        self._history_text = text

        return frame

    def _refresh_history(self) -> None:
        if self._history_tree is None:
            return
        self._history_tree.delete(*self._history_tree.get_children())
        records = list(reversed(self.watcher.db.list_all(user_id=self.watcher.user_id)))
        for record in records:
            top_issue = "—"
            analysis = record.analysis
            if analysis and not analysis.get("parse_error"):
                issues = analysis.get("session_summary", {}).get("top_issues") or []
                if issues:
                    top_issue = issues[0]
            date_str = record.started_at.split("T")[0]
            self._history_tree.insert(
                "", "end", iid=str(record.id),
                values=(date_str, record.source_app or "—", top_issue),
            )

    def _on_history_select(self) -> None:
        selection = self._history_tree.selection()
        if not selection:
            return
        record = self.watcher.db.get(int(selection[0]))
        self._history_text.config(state="normal")
        if record is None:
            self._history_text.delete("1.0", "end")
        elif record.report_path and pathlib.Path(record.report_path).exists():
            content = pathlib.Path(record.report_path).read_text(encoding="utf-8")
            render_into_text_widget(self._history_text, content)
        else:
            render_into_text_widget(
                self._history_text,
                "# Report not available\n\nThis interview hasn't finished processing yet, "
                "or its report file is missing.",
            )
        self._history_text.config(state="disabled")

    # -- Trends tab -------------------------------------------------------

    def _build_trends_tab(self, notebook, tk, ttk):
        frame = ttk.Frame(notebook, padding=10)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Button(frame, text="Refresh", command=self._refresh_trends).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        text = tk.Text(frame, wrap="word", padx=10, pady=8, state="disabled", relief="flat")
        _configure_report_tags(text)
        text.grid(row=1, column=0, sticky="nsew")
        self._trends_text = text

        return frame

    def _refresh_trends(self) -> None:
        if self._trends_text is None:
            return
        cfg = self.watcher.cfg
        trends_path = cfg.resolve(cfg.output.get("output_dir", "output")) / cfg.output.get(
            "trends_filename", "trends.md"
        )
        self._trends_text.config(state="normal")
        if trends_path.exists():
            render_into_text_widget(self._trends_text, trends_path.read_text(encoding="utf-8"))
        else:
            render_into_text_widget(self._trends_text, "# No trends yet\n\nComplete an interview to start tracking recurring issues.")
        self._trends_text.config(state="disabled")

    # -- Settings tab -----------------------------------------------------

    def _build_settings_tab(self, notebook, tk, ttk):
        frame = ttk.Frame(notebook, padding=16)
        cfg = self.watcher.cfg

        if cfg.path is None:
            ttk.Label(frame, text="No config file path available to edit.").pack(anchor="w")
            return frame

        current = load_editable_settings(cfg.path)
        self._settings_widgets = {}

        form = ttk.Frame(frame)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        def _row(r, label, dotted, widget):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", pady=4, padx=(0, 12))
            widget.grid(row=r, column=1, sticky="ew", pady=4)
            self._settings_widgets[dotted] = widget

        r = 0
        e = ttk.Spinbox(form, from_=0, to=365, width=8)
        e.set(current.get("retention_days", 3))
        _row(r, "Retention (days)", "retention_days", e); r += 1

        e = ttk.Entry(form, width=10)
        e.insert(0, str(current.get("poll_interval_seconds", 5)))
        _row(r, "Poll interval (seconds)", "poll_interval_seconds", e); r += 1

        e = ttk.Spinbox(form, from_=1, to=20, width=8)
        e.set(current.get("start_debounce_polls", 2))
        _row(r, "Start debounce (polls)", "start_debounce_polls", e); r += 1

        e = ttk.Spinbox(form, from_=16, to=320, width=8)
        e.set(current.get("audio.bitrate_kbps", 64))
        _row(r, "Audio bitrate (kbps)", "audio.bitrate_kbps", e); r += 1

        e = ttk.Combobox(form, values=_WHISPER_MODELS, state="readonly", width=12)
        e.set(current.get("transcription.whisper_model", "small"))
        _row(r, "Whisper model", "transcription.whisper_model", e); r += 1

        diarization_var = tk.BooleanVar(value=bool(current.get("transcription.diarization", True)))
        e = ttk.Checkbutton(form, variable=diarization_var, text="Separate speakers")
        e.var = diarization_var  # keep a reference so it isn't garbage-collected
        _row(r, "Diarization", "transcription.diarization", e); r += 1

        e = ttk.Combobox(form, values=_ANALYSIS_ENGINES, state="readonly", width=12)
        e.set(current.get("analysis.engine", "ollama"))
        _row(r, "Analysis engine", "analysis.engine", e); r += 1

        e = ttk.Entry(form)
        e.insert(0, current.get("analysis.llm_model", ""))
        _row(r, "Model name", "analysis.llm_model", e); r += 1

        e = ttk.Entry(form)
        e.insert(0, current.get("output.output_dir", "output"))
        _row(r, "Reports output dir", "output.output_dir", e); r += 1

        self._settings_status = ttk.Label(frame, text="", foreground="#2f6f5e")
        ttk.Button(frame, text="Save", command=self._on_save_settings).pack(anchor="w", pady=(16, 4))
        self._settings_status.pack(anchor="w")

        return frame

    def _on_save_settings(self) -> None:
        updates: dict[str, object] = {}
        for dotted, widget in self._settings_widgets.items():
            if hasattr(widget, "var"):
                updates[dotted] = bool(widget.var.get())
            else:
                updates[dotted] = widget.get()

        try:
            save_editable_settings(self.watcher.cfg.path, updates)
        except (ValueError, TypeError) as e:
            self._settings_status.config(text=f"Couldn't save: {e}", foreground="#a8722a")
            return

        self._settings_status.config(
            text="Saved. Restart the app for changes to take effect.", foreground="#2f6f5e"
        )
