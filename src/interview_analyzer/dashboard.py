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

import datetime as dt
import logging
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

from . import api_keys
from .confidence import format_confidence
from .infographic import write_interview_infographic, write_trends_infographic
from .language_packs import LANGUAGE_PACKS, PackActionDialog, is_pack_installed
from .model_setup import (
    MODEL_CATALOG,
    ModelInstallDialog,
    ensure_ollama_running,
    is_model_installed,
    ollama_is_reachable,
    size_label,
    stop_ollama,
)
from .report import write_trends_report
from .report_view import render_into_text_widget
from .settings_editor import load_editable_settings, save_editable_settings
from .tray import job_text

logger = logging.getLogger(__name__)

_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
_TRANSCRIPTION_ENGINES = ["faster-whisper", "groq"]
_ANALYSIS_ENGINES = ["ollama", "groq_api", "anthropic_api", "openai_api"]
_TRANSCRIPTION_LANGUAGES = ["auto", "en", "hi", "hinglish"]
# "Not rated" (index 0) doubles as the clear-a-rating value for the
# feedback panel's Comboboxes -- selecting it and saving is how you clear
# a previously-given score.
_FEEDBACK_SCORE_VALUES = ["Not rated"] + [str(i) for i in range(1, 11)]


def _open_with_os_default(path) -> None:
    """Opens `path` with whatever the OS would use for a double-click --
    Explorer/the default player on Windows, Finder/the default player on
    macOS (`open`, not `os.startfile`, which doesn't exist there)."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True)
    else:
        raise RuntimeError(f"Don't know how to open a path with the OS default on {sys.platform!r}.")


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    mins, secs = divmod(total, 60)
    return f"{mins:02d}:{secs:02d}"


def format_started(record) -> str:
    try:
        started = dt.datetime.fromisoformat(record.started_at)
    except (TypeError, ValueError):
        return record.started_at or "—"
    return started.strftime("%Y-%m-%d %H:%M")


def format_duration(record) -> str:
    """'—' when the recording never cleanly finished (ended_at unset --
    e.g. it was interrupted by a crash) rather than 0:00, since those are
    different situations worth distinguishing at a glance."""
    if not record.ended_at:
        return "—"
    try:
        started = dt.datetime.fromisoformat(record.started_at)
        ended = dt.datetime.fromisoformat(record.ended_at)
    except (TypeError, ValueError):
        return "—"
    seconds = max(0, int((ended - started).total_seconds()))
    mins, secs = divmod(seconds, 60)
    return f"{mins:02d}:{secs:02d}"


def _analysis_is_malformed(record) -> bool:
    """True if analysis ran and a report file exists, but the analysis
    itself was unusable (parse_error) -- e.g. the LLM returned syntactically
    valid JSON that didn't match the expected qa_pairs/session_summary
    shape at all (reproduced on a real interview: a long transcript
    against llama3.1:8b returned an unrelated generic {"title": ...,
    "topics": [...]} object). report.py still writes a report file in this
    case, but it's a "could not be parsed" placeholder, not real content --
    without this check, that placeholder file counted as "finished",
    permanently graying out Reprocess with no way to retry from the UI."""
    return bool(record.analysis and record.analysis.get("parse_error"))


def history_status_label(record) -> str:
    """One-line status for the History list's Status column and as the
    basis for the detail pane's placeholder text when there's no report."""
    if record.report_path and pathlib.Path(record.report_path).exists():
        analysis = record.analysis
        if analysis and analysis.get("no_speech_detected"):
            return "No speech detected"
        if _analysis_is_malformed(record):
            return "Analysis failed"
        if analysis:
            issues = analysis.get("session_summary", {}).get("top_issues") or []
            return issues[0] if issues else "No issues flagged"
        return "Report generated"
    if record.analysis:
        return "Report pending"
    if record.transcript:
        return "Analysis failed"
    if record.ended_at is None:
        return "Interrupted — no report"
    return "Not processed"


def has_audio(record) -> bool:
    """True if this interview's raw audio is still on disk (it's deleted
    automatically after the configured retention window)."""
    if not record.audio_path:
        return False
    path = pathlib.Path(record.audio_path)
    return path.exists() and path.stat().st_size > 0


def can_reprocess(record) -> bool:
    """True if this interview has recoverable audio and no *usable*
    finished report yet -- i.e. the History tab's Reprocess button should
    be enabled for it (subject also to the watcher not currently being
    busy -- see Dashboard._update_action_buttons). A report that exists
    but came from a malformed analysis (see _analysis_is_malformed) does
    NOT count as finished -- it's exactly the case Reprocess exists to
    recover from."""
    has_a_usable_report = (
        record.report_path
        and pathlib.Path(record.report_path).exists()
        and not _analysis_is_malformed(record)
    )
    if has_a_usable_report:
        return False
    return has_audio(record)


def _friendly_error_markdown(heading: str, exc: BaseException) -> str:
    """Renders a failure as a human-readable headline first, with the raw
    exception folded in below as "Technical details" -- so someone who
    just wants to know what to *do* isn't stuck parsing a Python exception,
    while the exact error is still there for anyone who wants to dig in or
    paste it into a bug report. The full traceback still goes to the log
    file (see the callers' logger.exception calls) -- this is deliberately
    just the type+message, not a stack trace, since that's the app's own
    log, not something the user reads in normal operation."""
    message = str(exc)
    lower = message.lower()
    if "ollama" in lower and ("couldn't be started" in lower or "isn't running" in lower):
        headline = (
            "The local analysis model (Ollama) isn't running, and couldn't be started "
            "automatically. Go to the **Status** tab and check \"Local analysis model\" -- "
            "click **Start** there, then try again."
        )
    elif "connection" in lower or "refused" in lower or isinstance(exc, ConnectionError):
        headline = (
            "Couldn't reach the analysis model. If you're using Ollama, check its status on "
            "the **Status** tab; if you're using a cloud engine, check your internet connection "
            "and API key in the Settings tab."
        )
    elif "no audio was recorded" in lower:
        headline = "This interview has no recorded audio to work with, so there's nothing to reprocess."
    else:
        headline = "Something went wrong while processing this interview."
    return f"# {heading}\n\n{headline}\n\n**Technical details:** {type(exc).__name__}: {message}"


def _configure_report_tags(text_widget) -> None:
    text_widget.tag_configure("h1", font=("Georgia", 15, "bold"), spacing3=6)
    text_widget.tag_configure("h2", font=("Georgia", 12, "bold"), spacing1=10, spacing3=4)
    text_widget.tag_configure("h3", font=("Georgia", 11, "bold"), spacing1=8, spacing3=2)
    text_widget.tag_configure("bullet", lmargin1=14, lmargin2=28, spacing3=2)
    text_widget.tag_configure("quote", font=("Segoe UI", 9, "italic"), foreground="#5b645f")
    text_widget.tag_configure("text", spacing3=2)


# Transcript speaker labels: [Speaker] text, one line per transcribed
# segment -- faster-whisper often splits one continuous utterance across
# several consecutive segments/lines, which is why _group_transcript_by_
# speaker below merges consecutive same-speaker lines into one paragraph.
_TRANSCRIPT_LINE_RE = re.compile(r"^\[(?P<speaker>[^\]]+)\]\s*(?P<text>.*)$")

# Fixed, muted, professional colors (not saturated/neon -- easy to read
# for long stretches) for the two speakers that appear in almost every
# transcript (dual-channel recording -- see recorder.py). "speaker" covers
# the mono/diarization-disabled fallback's generic single label.
_FIXED_SPEAKER_COLORS = {
    "you": "#2b6fa8",           # muted blue
    "interviewer": "#7a4fa0",   # muted purple -- clearly distinct from blue, not clashing
    "speaker": "#4a7a6a",       # muted teal -- the generic/no-diarization fallback label
}
# Rotated for any other speaker label this app didn't itself define (e.g.
# raw pyannote diarization output like "SPEAKER_00", "SPEAKER_01").
_SPEAKER_COLOR_PALETTE = ["#8a5a3a", "#5a5aa0", "#8a4a6a", "#3a7a3a", "#a05a5a"]


def _parse_transcript_lines(transcript: str) -> list[tuple[str, str]]:
    """Parses '[Speaker] text' lines into (speaker, text) pairs, skipping
    blank lines. A line that doesn't match the expected format (shouldn't
    happen for a transcript this app generated, but real-world text is
    real-world text) is kept under an empty speaker label rather than
    dropped, so nothing from the original transcript is ever lost."""
    parsed: list[tuple[str, str]] = []
    for line in transcript.splitlines():
        if not line.strip():
            continue
        m = _TRANSCRIPT_LINE_RE.match(line)
        if m:
            parsed.append((m.group("speaker").strip(), m.group("text").strip()))
        else:
            parsed.append(("", line.strip()))
    return parsed


def _group_transcript_by_speaker(transcript: str) -> list[tuple[str, str]]:
    """Merges consecutive same-speaker lines into one paragraph -- so a
    continuous turn shows as one flowing paragraph instead of the same
    `[Speaker]` label repeated on every line faster-whisper happened to
    split it into."""
    grouped: list[tuple[str, str]] = []
    for speaker, text in _parse_transcript_lines(transcript):
        if grouped and grouped[-1][0] == speaker:
            prev_speaker, prev_text = grouped[-1]
            grouped[-1] = (prev_speaker, f"{prev_text} {text}".strip())
        else:
            grouped.append((speaker, text))
    return grouped


def _speaker_color(speaker: str, assigned: dict[str, str]) -> str:
    """A stable color per speaker for one transcript render -- `assigned`
    is a fresh dict per render call (not shared/global), so an unrecognized
    speaker label always gets the same palette color within one transcript
    but colors aren't "reserved" globally across different interviews."""
    fixed = _FIXED_SPEAKER_COLORS.get(speaker.strip().lower())
    if fixed is not None:
        return fixed
    if speaker not in assigned:
        assigned[speaker] = _SPEAKER_COLOR_PALETTE[len(assigned) % len(_SPEAKER_COLOR_PALETTE)]
    return assigned[speaker]


class Dashboard:
    def __init__(self, watcher, on_logout: Optional[Callable[[], None]] = None):
        self.watcher = watcher
        self._on_logout = on_logout
        self._logout_btn = None
        self._lock = threading.Lock()
        self._root = None
        # set once the window is fully built and about to enter mainloop();
        # cleared on close. Guards other threads (tray, watcher notify,
        # tests) from touching widgets while they're still being built --
        # Tkinter raises "main thread is not in main loop" if you call into
        # a Tk instance from another thread before its loop is running.
        self._ready = threading.Event()
        # set once the dashboard's Tk thread has fully exited (mainloop
        # returned) -- used by app.py's logout flow to know it's safe to
        # show a new login dialog (only one Tk root may exist at a time,
        # see consent.py's docstring)
        self._closed = threading.Event()
        self._closed.set()
        self._thread: Optional[threading.Thread] = None
        # widgets refreshed by _refresh_status / notify callbacks; only
        # touched from the dashboard's own Tk thread
        self._status_label = None
        self._detail_label = None
        self._timer_label = None
        self._activity_bar = None
        self._activity_running = False
        self._background_jobs_label = None
        self._pause_btn = None
        self._stop_btn = None
        # Ollama status/start/stop row on the Status tab -- see
        # _start_ollama_status_polling. _ollama_poll_started guards against
        # starting a second polling thread if the dashboard is closed and
        # reopened (a fresh Dashboard instance isn't created for that --
        # see open()/close() -- so this flag must survive across runs).
        self._ollama_status_label = None
        self._ollama_start_btn = None
        self._ollama_stop_btn = None
        self._ollama_poll_started = False
        self._manual_start_entry = None
        self._manual_start_btn = None
        self._history_tree = None
        self._history_text = None
        self._reprocess_btn = None
        self._open_audio_btn = None
        self._view_transcript_btn = None
        self._view_infographic_btn = None
        self._delete_btn = None
        self._cancel_btn = None
        self._history_progress_label = None
        self._history_progress_bar = None
        self._history_progress_running = False
        self._trends_text = None
        self._settings_widgets: dict[str, object] = {}
        self._settings_status = None
        self._fb_transcript_var = None
        self._fb_analysis_var = None
        self._fb_comment_entry = None
        self._fb_frame = None
        self._fb_status_label = None
        self._fb_confidence_label = None
        self._model_name_entry = None
        self._model_status_label = None
        self._api_key_provider = None
        self._api_key_entry = None
        self._api_key_status_label = None
        self._lang_pack_rows: dict[str, dict[str, object]] = {}

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
            self._closed.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def wait_until_ready(self, timeout: float = 10) -> bool:
        """Block until the dashboard window is fully built (or `timeout`
        elapses). Used by app.py to make sure the watcher doesn't start
        polling -- and potentially spin up a *second* Tk() interpreter for
        a consent popup -- before the dashboard's shared root exists."""
        return self._ready.wait(timeout=timeout)

    def close(self) -> None:
        """Close the dashboard window from another thread (e.g. on logout,
        before showing a new login dialog). Safe to call even if already
        closed. Does not block -- call `wait_until_closed()` afterward if
        the caller needs to know the Tk root is fully torn down before
        doing anything else Tk-related."""
        root = self._root
        if root is not None:
            try:
                root.after(0, self._handle_close)
            except Exception:  # noqa: BLE001
                pass

    def wait_until_closed(self, timeout: float = 10) -> bool:
        """Block until the dashboard's Tk thread has fully exited (or
        `timeout` elapses). Only one Tk root may exist at a time in this
        process (see consent.py's docstring), so app.py's logout flow must
        wait for this before showing a new login dialog."""
        return self._closed.wait(timeout=timeout)

    def notify_state_change(self) -> None:
        """Called (from any thread) whenever the watcher's recording state
        changes, so an open dashboard can refresh itself -- including
        History/Trends, since a transition back to idle is exactly when a
        new report (or a reprocessed one) may have just appeared."""
        if self._ready.is_set() and self._root is not None:
            try:
                self._root.after(0, self._refresh_status)
                self._root.after(0, self._refresh_history)
                self._root.after(0, self._refresh_trends)
            except Exception:  # noqa: BLE001
                pass

    # -- window setup -----------------------------------------------------

    def _run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError:  # pragma: no cover
            logger.warning("Tkinter not available; cannot open the dashboard.")
            self._closed.set()
            return

        root = tk.Tk()
        self._root = root
        root.title("Interview Analyzer")
        root.geometry("820x540")
        root.minsize(620, 400)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        notebook.add(self._build_status_tab(notebook, tk, ttk), text="Status")
        notebook.add(self._build_history_tab(notebook, tk, ttk), text="History")
        notebook.add(self._build_trends_tab(notebook, tk, ttk), text="Trends")
        notebook.add(self._build_settings_tab(notebook, tk, ttk), text="Settings")

        root.protocol("WM_DELETE_WINDOW", self._handle_close)

        self._refresh_status()
        self._refresh_history()
        self._refresh_trends()
        self._start_ollama_status_polling()
        self.watcher.set_ui_root(root)
        self._ready.set()
        root.mainloop()
        self._closed.set()

    def _handle_close(self) -> None:
        """Runs on the dashboard's own Tk thread -- either via the window's
        titlebar close button, or scheduled by `close()` from another
        thread. Tears down the window; `mainloop()` returning is what lets
        `_run()` reach `self._closed.set()` above."""
        self._ready.clear()
        self.watcher.set_ui_root(None)
        if self._root is not None:
            self._root.destroy()
        self._root = None

    # -- shared helpers -----------------------------------------------------

    def _make_scrollable_tab(self, notebook, tk, ttk):
        """Wraps a tab's content in a vertically-scrollable canvas, so
        content/buttons further down (e.g. Settings' Save button) stay
        reachable via a scrollbar or mouse wheel instead of getting clipped
        off in a window smaller than the content -- both tabs' content only
        grows as more settings/sections get added over time, so a fixed
        window size can't be relied on to always fit everything.

        Returns (outer_frame_to_add_to_notebook, inner_frame_to_build_content_in) --
        callers build their tab exactly as before, just into the inner frame."""
        outer = ttk.Frame(notebook)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        inner = ttk.Frame(canvas, padding=16)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            # the inner frame always matches the canvas's width -- only
            # vertical scrolling is needed, horizontal would just look broken
            canvas.itemconfig(inner_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # only captures the mouse wheel while actually hovering this tab's
        # canvas, so scrolling a different tab/widget elsewhere isn't affected
        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        return outer, inner

    # -- Status tab ---------------------------------------------------------

    def _build_status_tab(self, notebook, tk, ttk):
        outer, frame = self._make_scrollable_tab(notebook, tk, ttk)

        self._status_label = ttk.Label(frame, text="", font=("Segoe UI", 14, "bold"))
        self._status_label.pack(anchor="w")
        self._detail_label = ttk.Label(frame, text="", foreground="#5b645f")
        self._detail_label.pack(anchor="w", pady=(2, 4))
        self._timer_label = ttk.Label(frame, text="", font=("Consolas", 12))
        self._timer_label.pack(anchor="w", pady=(0, 8))

        self._activity_bar = ttk.Progressbar(frame, mode="indeterminate", length=260)
        self._activity_bar.pack(anchor="w", pady=(0, 16))

        btn_row = ttk.Frame(frame)
        btn_row.pack(anchor="w")
        self._pause_btn = ttk.Button(btn_row, text="Pause", command=self._on_toggle_pause)
        self._pause_btn.pack(side="left", padx=(0, 8))
        self._stop_btn = ttk.Button(btn_row, text="Stop recording", command=self._on_stop)
        self._stop_btn.pack(side="left")

        # manual fallback for when automatic detection doesn't pick up a
        # real call -- clicking this IS the consent, so it skips straight
        # to recording rather than showing the usual consent popup
        manual_row = ttk.Frame(frame)
        manual_row.pack(anchor="w", pady=(14, 0))
        ttk.Label(manual_row, text="Not detected automatically?").pack(side="left", padx=(0, 8))
        self._manual_start_entry = ttk.Entry(manual_row, width=14)
        self._manual_start_entry.insert(0, "Meet")
        self._manual_start_entry.pack(side="left", padx=(0, 8))
        self._manual_start_btn = ttk.Button(manual_row, text="Start recording", command=self._on_manual_start)
        self._manual_start_btn.pack(side="left")

        # Local analysis model (Ollama) status -- shown here since a
        # not-running model is a common, previously-confusing cause of
        # "reprocessing failed"/"analysis failed" (see _friendly_error_markdown
        # and history_status_label's hint text). Only meaningful for the
        # "ollama" analysis engine; see _apply_ollama_status for the
        # cloud-engine case.
        model_row = ttk.Frame(frame)
        model_row.pack(anchor="w", pady=(14, 0))
        ttk.Label(model_row, text="Local analysis model:").pack(side="left", padx=(0, 6))
        self._ollama_status_label = ttk.Label(model_row, text="checking…", foreground="#5b645f")
        self._ollama_status_label.pack(side="left", padx=(0, 10))
        self._ollama_start_btn = ttk.Button(model_row, text="Start", command=self._on_start_ollama)
        self._ollama_start_btn.pack(side="left", padx=(0, 4))
        self._ollama_stop_btn = ttk.Button(model_row, text="Stop", command=self._on_stop_ollama)
        self._ollama_stop_btn.pack(side="left")

        ttk.Button(
            frame, text="Open recordings folder", command=self._on_open_recordings_folder
        ).pack(anchor="w", pady=(14, 0))

        if self._on_logout is not None:
            self._logout_btn = ttk.Button(frame, text="Log out", command=self._on_click_logout)
            self._logout_btn.pack(anchor="w", pady=(8, 0))

        # background transcribe/analyze/report jobs are independent of the
        # live recording state above -- a new call can be recording while an
        # earlier one is still processing -- so they get their own line
        self._background_jobs_label = ttk.Label(frame, text="", foreground="#2f6fa8")
        self._background_jobs_label.pack(anchor="w", pady=(16, 0))

        ttk.Label(
            frame,
            text="The app watches for Teams/Meet/Webex/Zoom/Chime in the background.\n"
                 "You'll be asked for consent before each recording starts.",
            foreground="#5b645f", justify="left",
        ).pack(anchor="w", pady=(20, 0))

        return outer

    def _on_toggle_pause(self) -> None:
        if self.watcher.status.get("state") == "paused":
            self.watcher.resume_recording()
        else:
            self.watcher.pause_recording()
        self._refresh_status()

    def _on_stop(self) -> None:
        self.watcher.request_stop_recording()
        self._refresh_status()

    def _on_manual_start(self) -> None:
        app_name = self._manual_start_entry.get().strip() or "Manual"
        try:
            self.watcher.request_start_recording(app_name)
        except RuntimeError:
            pass  # already recording -- the button should be disabled then anyway
        self._refresh_status()

    def _on_open_recordings_folder(self) -> None:
        cfg = self.watcher.cfg
        folder = cfg.resolve(cfg.audio.get("raw_dir", "data/audio"))
        try:
            folder.mkdir(parents=True, exist_ok=True)
            _open_with_os_default(folder)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to open recordings folder %s", folder)

    def _ollama_host(self) -> str:
        return self.watcher.cfg.analysis.get("ollama_host", "http://localhost:11434")

    def _using_ollama_engine(self) -> bool:
        return self.watcher.cfg.analysis.get("engine", "ollama") == "ollama"

    def _start_ollama_status_polling(self) -> None:
        """Background thread that checks Ollama's reachability every few
        seconds and pushes the result to the Status tab -- run off the Tk
        thread since ollama_is_reachable() makes a real HTTP request
        (blocking the UI thread for up to its 3s timeout on every refresh
        would freeze the whole window while Ollama is down, exactly when
        the status indicator matters most)."""
        if self._ollama_poll_started:
            return
        self._ollama_poll_started = True

        def _poll() -> None:
            while self._root is not None:
                if self._using_ollama_engine():
                    reachable = ollama_is_reachable(self._ollama_host())
                else:
                    reachable = None  # not applicable -- see _apply_ollama_status
                root = self._root
                if root is None:
                    break
                try:
                    root.after(0, lambda r=reachable: self._apply_ollama_status(r))
                except Exception:  # noqa: BLE001
                    break
                time.sleep(8)

        threading.Thread(target=_poll, daemon=True).start()

    def _apply_ollama_status(self, reachable: Optional[bool]) -> None:
        """`reachable` is None when the configured analysis engine isn't
        Ollama at all (a cloud engine) -- there's no local model to show a
        running/stopped state for or to start/stop."""
        if self._ollama_status_label is None:
            return
        if reachable is None:
            engine = self.watcher.cfg.analysis.get("engine", "ollama")
            self._ollama_status_label.config(text=f"n/a (using {engine})", foreground="#5b645f")
            self._ollama_start_btn.config(state="disabled")
            self._ollama_stop_btn.config(state="disabled")
        elif reachable:
            self._ollama_status_label.config(text="● Running", foreground="#2f6f5e")
            self._ollama_start_btn.config(state="disabled")
            self._ollama_stop_btn.config(state="normal")
        else:
            self._ollama_status_label.config(text="● Not running", foreground="#c0392b")
            self._ollama_start_btn.config(state="normal")
            self._ollama_stop_btn.config(state="disabled")

    def _on_start_ollama(self) -> None:
        self._ollama_start_btn.config(state="disabled")
        self._ollama_status_label.config(text="Starting…", foreground="#5b645f")
        host = self._ollama_host()

        def _run() -> None:
            started = ensure_ollama_running(host)
            if self._root is not None:
                self._root.after(0, lambda: self._apply_ollama_status(started))

        threading.Thread(target=_run, daemon=True).start()

    def _on_stop_ollama(self) -> None:
        self._ollama_stop_btn.config(state="disabled")
        self._ollama_status_label.config(text="Stopping…", foreground="#5b645f")
        host = self._ollama_host()

        def _run() -> None:
            stop_ollama(host)
            reachable = ollama_is_reachable(host)
            if self._root is not None:
                self._root.after(0, lambda: self._apply_ollama_status(reachable))

        threading.Thread(target=_run, daemon=True).start()

    def _wake_ollama_async(self) -> None:
        """Fired from the History/Trends Refresh buttons so that clicking
        Refresh also proactively wakes up a stopped local model, instead of
        only starting it lazily the moment an actual analysis request is
        made (see OllamaEngine.run() in analyzer.py) -- by the time you've
        clicked Reprocess it's had a head start warming up."""
        if not self._using_ollama_engine():
            return
        host = self._ollama_host()

        def _run() -> None:
            reachable = ensure_ollama_running(host)
            if self._root is not None:
                self._root.after(0, lambda: self._apply_ollama_status(reachable))

        threading.Thread(target=_run, daemon=True).start()

    def _on_click_logout(self) -> None:
        # only enabled while idle (see _refresh_status) -- logging out
        # mid-call would silently abandon an active recording, same rule as
        # the tray menu's "Log out" item
        if self._on_logout is not None:
            self._on_logout()

    def _refresh_status(self) -> None:
        if self._root is None or self._status_label is None:
            return
        status = self.watcher.status
        state = status.get("state", "idle")
        jobs = status.get("processing_jobs") or {}

        self._manual_start_btn.config(state="normal" if state == "idle" else "disabled")
        self._manual_start_entry.config(state="normal" if state == "idle" else "disabled")
        if self._logout_btn is not None:
            self._logout_btn.config(state="normal" if state == "idle" else "disabled")

        if state == "idle":
            self._status_label.config(text="● Idle", foreground="#5b645f")
            self._detail_label.config(text="Watching for a meeting to begin.")
            self._timer_label.config(text="")
            self._pause_btn.config(text="Pause", state="disabled")
            self._stop_btn.config(state="disabled")
        elif state == "recording":
            self._status_label.config(text="● Recording", foreground="#c0392b")
            self._detail_label.config(text=f"{status.get('app_name', 'call')}")
            self._timer_label.config(text=_format_elapsed(status.get("elapsed_seconds", 0)))
            self._pause_btn.config(text="Pause", state="normal")
            self._stop_btn.config(state="normal")
        else:  # paused
            self._status_label.config(text="⏸ Paused", foreground="#c8892c")
            self._detail_label.config(text=f"{status.get('app_name', 'call')} — capture paused")
            self._timer_label.config(text=_format_elapsed(status.get("elapsed_seconds", 0)))
            self._pause_btn.config(text="Resume", state="normal")
            self._stop_btn.config(state="normal")

        self._apply_activity_bar(self._activity_bar, "_activity_running", state == "recording", None)

        if jobs:
            if len(jobs) == 1:
                job = next(iter(jobs.values()))
                app = job.get("source_app") or "an interview"
                self._background_jobs_label.config(text=f"⋯ Processing {app} — {job_text(job)}")
            else:
                self._background_jobs_label.config(text=f"⋯ {len(jobs)} interviews processing in the background")
        else:
            self._background_jobs_label.config(text="")

        self._update_history_progress_indicator(jobs)

        if self._root is not None:
            self._root.after(1000, self._refresh_status)

    def _apply_activity_bar(self, bar, running_attr: str, should_run: bool, progress: Optional[float]) -> None:
        """Shared indeterminate/determinate switching, used by both the
        Status tab's activity bar and the History tab's inline one.
        `progress` (0.0-1.0) switches the bar to a real percentage --
        e.g. transcription, which faster-whisper reports incrementally --
        rather than an oscillating "something is happening" animation that
        can't say how much is left."""
        is_running = getattr(self, running_attr)
        if progress is not None:
            if is_running:
                bar.stop()
                setattr(self, running_attr, False)
            bar.config(mode="determinate", maximum=100)
            bar["value"] = progress * 100
        else:
            bar.config(mode="indeterminate")
            if should_run and not is_running:
                bar.start(80)
                setattr(self, running_attr, True)
            elif not should_run and is_running:
                bar.stop()
                setattr(self, running_attr, False)

    def _update_history_progress_indicator(self, jobs: dict) -> None:
        """A general "something's processing" bar for the History tab
        (per-row status text in the tree itself -- see _refresh_history --
        shows which specific interview and its stage/percentage). Also
        keeps the action buttons' enabled state in sync with busy-ness
        (re-selecting a row must not re-enable Reprocess while it's
        genuinely still running)."""
        if self._history_progress_label is None:
            return
        processing = bool(jobs)
        progress = None
        label = ""
        if processing:
            if len(jobs) == 1:
                job = next(iter(jobs.values()))
                progress = job.get("progress")
                label = job_text(job)
            else:
                label = f"{len(jobs)} interviews processing"

        if processing and not self._history_progress_bar.winfo_ismapped():
            self._history_progress_bar.pack(side="left", padx=(8, 0))
        elif not processing and self._history_progress_bar.winfo_ismapped():
            self._history_progress_bar.pack_forget()

        self._history_progress_label.config(text=label)
        self._apply_activity_bar(self._history_progress_bar, "_history_progress_running", processing, progress)
        self._update_action_buttons()

    # -- History tab ----------------------------------------------------

    def _build_history_tab(self, notebook, tk, ttk):
        frame = ttk.Frame(notebook, padding=10)
        # the tree's columns need ~460px to show fully without scrolling --
        # give it more relative share than a plain 1:2 split would, so the
        # default window size rarely needs horizontal scrolling at all
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(frame)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Button(toolbar, text="Refresh", command=self._on_refresh_history).pack(side="left")
        self._reprocess_btn = ttk.Button(
            toolbar, text="Reprocess (generate report)", command=self._on_reprocess, state="disabled"
        )
        self._reprocess_btn.pack(side="left", padx=(8, 0))
        self._open_audio_btn = ttk.Button(
            toolbar, text="Play audio", command=self._on_open_audio, state="disabled"
        )
        self._open_audio_btn.pack(side="left", padx=(8, 0))
        self._view_transcript_btn = ttk.Button(
            toolbar, text="View transcript", command=self._on_view_transcript, state="disabled"
        )
        self._view_transcript_btn.pack(side="left", padx=(8, 0))
        self._view_infographic_btn = ttk.Button(
            toolbar, text="View infographic", command=self._on_view_infographic, state="disabled"
        )
        self._view_infographic_btn.pack(side="left", padx=(8, 0))
        self._delete_btn = ttk.Button(
            toolbar, text="Delete", command=self._on_delete, state="disabled"
        )
        self._delete_btn.pack(side="left", padx=(8, 0))
        self._cancel_btn = ttk.Button(
            toolbar, text="Cancel processing", command=self._on_cancel, state="disabled"
        )
        self._cancel_btn.pack(side="left", padx=(8, 0))

        progress_row = ttk.Frame(frame)
        progress_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._history_progress_label = ttk.Label(progress_row, text="", foreground="#2f6fa8")
        self._history_progress_label.pack(side="left")
        self._history_progress_bar = ttk.Progressbar(progress_row, mode="indeterminate", length=180)
        # packed on demand by _update_history_progress_indicator, not here

        tree_frame = ttk.Frame(frame)
        tree_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        columns = ("started", "duration", "app", "status")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
        tree.heading("started", text="Started")
        tree.heading("duration", text="Duration")
        tree.heading("app", text="App")
        tree.heading("status", text="Status")
        tree.column("started", width=125, minwidth=90, stretch=False)
        tree.column("duration", width=65, minwidth=55, stretch=False)
        tree.column("app", width=75, minwidth=55, stretch=False)
        tree.column("status", width=190, minwidth=100, stretch=True)

        # explicit scrollbars, properly bound to the tree's view -- without
        # these, dragging to scroll horizontally goes through Tk's default
        # (unbound) fallback, which is what made it feel slow/unresponsive
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tree.bind("<<TreeviewSelect>>", lambda e: self._on_history_select())
        self._history_tree = tree

        detail_container = ttk.Frame(frame)
        detail_container.grid(row=2, column=1, sticky="nsew")
        detail_container.columnconfigure(0, weight=1)
        detail_container.rowconfigure(0, weight=1)

        text = tk.Text(detail_container, wrap="word", padx=10, pady=8, state="disabled", relief="flat")
        _configure_report_tags(text)
        # explicit scrollbar (not just relying on the mouse wheel, which
        # works but gives no visual indicator there's more content below --
        # e.g. a long transcript) so it's obvious there's more to scroll to
        text_vsb = ttk.Scrollbar(detail_container, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=text_vsb.set)
        text.grid(row=0, column=0, sticky="nsew")
        text_vsb.grid(row=0, column=1, sticky="ns")
        self._history_text = text

        self._fb_frame = self._build_feedback_panel(detail_container, tk, ttk)
        self._fb_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._fb_frame.grid_remove()  # hidden until a report is actually selected

        return frame

    def _build_feedback_panel(self, parent, tk, ttk):
        """A compact "how good was this?" panel shown under a selected
        interview's report. Feedback here is what calibrates the
        confidence score shown on *future* reports and feeds corrective
        notes back into future analysis prompts -- see confidence.py.
        Scores are 1-10 (10 highest); "Not rated" is both the default and
        how to clear a rating (see _on_clear_feedback_ratings)."""
        panel = ttk.LabelFrame(parent, text="Feedback", padding=8)

        row1 = ttk.Frame(panel)
        row1.pack(fill="x", pady=(0, 4))
        ttk.Label(row1, text="Transcription quality (1-10)", width=24).pack(side="left")
        self._fb_transcript_var = tk.StringVar(value=_FEEDBACK_SCORE_VALUES[0])
        ttk.Combobox(
            row1, textvariable=self._fb_transcript_var, values=_FEEDBACK_SCORE_VALUES,
            state="readonly", width=10,
        ).pack(side="left")

        row2 = ttk.Frame(panel)
        row2.pack(fill="x", pady=(0, 4))
        ttk.Label(row2, text="Analysis quality (1-10)", width=24).pack(side="left")
        self._fb_analysis_var = tk.StringVar(value=_FEEDBACK_SCORE_VALUES[0])
        ttk.Combobox(
            row2, textvariable=self._fb_analysis_var, values=_FEEDBACK_SCORE_VALUES,
            state="readonly", width=10,
        ).pack(side="left")

        row3 = ttk.Frame(panel)
        row3.pack(fill="x", pady=(0, 4))
        ttk.Label(row3, text="Comment (optional)", width=24).pack(side="left")
        self._fb_comment_entry = ttk.Entry(row3)
        self._fb_comment_entry.pack(side="left", fill="x", expand=True)

        row4 = ttk.Frame(panel)
        row4.pack(fill="x")
        ttk.Button(row4, text="Submit feedback", command=self._on_submit_feedback).pack(side="left")
        ttk.Button(row4, text="Clear ratings", command=self._on_clear_feedback_ratings).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(row4, text="Delete feedback", command=self._on_delete_feedback).pack(
            side="left", padx=(6, 0)
        )
        self._fb_status_label = ttk.Label(row4, text="", foreground="#2f6f5e")
        self._fb_status_label.pack(side="left", padx=(10, 0))

        self._fb_confidence_label = ttk.Label(panel, text="", foreground="#5b645f")
        self._fb_confidence_label.pack(anchor="w", pady=(4, 0))

        return panel

    @staticmethod
    def _feedback_score_from_var(var) -> Optional[int]:
        value = var.get()
        return None if value == _FEEDBACK_SCORE_VALUES[0] else int(value)

    def _on_submit_feedback(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        comment = self._fb_comment_entry.get().strip()
        self.watcher.db.save_feedback(
            record.id,
            user_id=self.watcher.user_id,
            transcript_score=self._feedback_score_from_var(self._fb_transcript_var),
            analysis_score=self._feedback_score_from_var(self._fb_analysis_var),
            comment=comment,
        )
        self._fb_status_label.config(text="Saved -- thanks, this improves future confidence scoring.")

    def _on_clear_feedback_ratings(self) -> None:
        """Resets both scores to "Not rated" and the comment to empty, and
        saves that immediately -- the requested "clear the rating and save"
        action. Distinct from Delete feedback below, which removes the row
        entirely rather than leaving an explicitly-unrated one."""
        record = self._selected_record()
        if record is None:
            return
        self._fb_transcript_var.set(_FEEDBACK_SCORE_VALUES[0])
        self._fb_analysis_var.set(_FEEDBACK_SCORE_VALUES[0])
        self._fb_comment_entry.delete(0, "end")
        self.watcher.db.save_feedback(
            record.id, user_id=self.watcher.user_id,
            transcript_score=None, analysis_score=None, comment="",
        )
        self._fb_status_label.config(text="Ratings cleared.")

    def _on_delete_feedback(self) -> None:
        """Removes this interview's feedback row entirely -- for a rating
        given by mistake, e.g. clicking the wrong interview's row."""
        record = self._selected_record()
        if record is None:
            return
        self.watcher.db.delete_feedback(record.id)
        self._fb_transcript_var.set(_FEEDBACK_SCORE_VALUES[0])
        self._fb_analysis_var.set(_FEEDBACK_SCORE_VALUES[0])
        self._fb_comment_entry.delete(0, "end")
        self._fb_status_label.config(text="Feedback deleted.")

    def _refresh_feedback_panel(self, record) -> None:
        if self._fb_frame is None:
            return
        if record is None or not record.analysis:
            self._fb_frame.grid_remove()
            return

        self._fb_frame.grid()
        self._fb_status_label.config(text="")

        existing = self.watcher.db.get_feedback(record.id)

        def _to_display(score: Optional[int]) -> str:
            return _FEEDBACK_SCORE_VALUES[0] if score is None else str(score)

        self._fb_transcript_var.set(_to_display(existing.transcript_score if existing else None))
        self._fb_analysis_var.set(_to_display(existing.analysis_score if existing else None))
        self._fb_comment_entry.delete(0, "end")
        if existing and existing.comment:
            self._fb_comment_entry.insert(0, existing.comment)

        confidence_info = (record.analysis or {}).get("confidence_info")
        self._fb_confidence_label.config(text=f"Confidence in this assessment: {format_confidence(confidence_info)}")

    def _refresh_history(self) -> None:
        if self._history_tree is None:
            return
        selected = self._history_tree.selection()
        jobs = self.watcher.status.get("processing_jobs", {})
        self._history_tree.delete(*self._history_tree.get_children())
        records = list(reversed(self.watcher.db.list_all(user_id=self.watcher.user_id)))
        for record in records:
            job = jobs.get(record.id)
            status_cell = job_text(job) if job else history_status_label(record)
            self._history_tree.insert(
                "", "end", iid=str(record.id),
                values=(
                    format_started(record),
                    format_duration(record),
                    record.source_app or "—",
                    status_cell,
                ),
            )
        if selected and self._history_tree.exists(selected[0]):
            self._history_tree.selection_set(selected[0])
        self._update_action_buttons()

    def _on_refresh_history(self) -> None:
        self._refresh_history()
        self._wake_ollama_async()

    def _selected_record(self):
        selection = self._history_tree.selection()
        if not selection:
            return None
        return self.watcher.db.get(int(selection[0]))

    def _update_action_buttons(self) -> None:
        record = self._selected_record()
        status = self.watcher.status
        is_recording_this = record is not None and status.get("interview_id") == record.id
        is_processing_this = record is not None and record.id in (status.get("processing_jobs") or {})
        busy = is_recording_this or is_processing_this

        can_do_reprocess = record is not None and can_reprocess(record) and not busy
        self._reprocess_btn.config(state="normal" if can_do_reprocess else "disabled")
        can_play = record is not None and has_audio(record)
        self._open_audio_btn.config(state="normal" if can_play else "disabled")
        can_view_transcript = record is not None and bool(record.transcript)
        self._view_transcript_btn.config(state="normal" if can_view_transcript else "disabled")
        can_view_infographic = (
            record is not None and bool(record.analysis)
            and not _analysis_is_malformed(record)
            and not record.analysis.get("no_speech_detected")
        )
        self._view_infographic_btn.config(state="normal" if can_view_infographic else "disabled")
        can_delete = record is not None and not busy
        self._delete_btn.config(state="normal" if can_delete else "disabled")
        self._cancel_btn.config(state="normal" if is_processing_this else "disabled")

    def _on_cancel(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        self.watcher.cancel_processing(record.id)

    def _on_history_select(self) -> None:
        record = self._selected_record()
        self._update_action_buttons()
        self._history_text.config(state="normal")
        jobs = self.watcher.status.get("processing_jobs", {})
        show_feedback = False
        if record is None:
            self._history_text.delete("1.0", "end")
        elif record.id in jobs:
            render_into_text_widget(
                self._history_text,
                f"# Processing…\n\n{job_text(jobs[record.id])}\n\n"
                "Click **Cancel processing** above to stop it.",
            )
        elif record.report_path and pathlib.Path(record.report_path).exists():
            content = pathlib.Path(record.report_path).read_text(encoding="utf-8")
            render_into_text_widget(self._history_text, content)
            show_feedback = bool(record.analysis) and not (
                record.analysis.get("parse_error") or record.analysis.get("no_speech_detected")
            )
        else:
            reason = {
                "Interrupted — no report": "The recording was interrupted before it finished (e.g. a crash or "
                                            "a force-quit), so it never reached the report stage.",
                "Analysis failed": "The transcript was produced, but analysis didn't complete -- most often "
                                    "because the local analysis model (Ollama) wasn't running at the time. "
                                    "Check its status on the **Status** tab (it should start automatically "
                                    "the next time you try, but you can also start it there yourself).",
                "Not processed": "This interview hasn't been processed yet.",
                "Report pending": "Analysis finished, but the report file wasn't written yet.",
            }.get(history_status_label(record), "The report file is missing.")
            hint = ("\n\nIts audio is still available -- click **Reprocess** above to try generating "
                    "a report from it now." if can_reprocess(record) else
                    "\n\nIts audio is no longer available, so this can't be recovered.")
            if record.transcript:
                hint += "\n\nThe raw transcript was saved though -- click **View transcript** above to read it."
            render_into_text_widget(self._history_text, f"# Report not available\n\n{reason}{hint}")
        self._history_text.config(state="disabled")
        self._refresh_feedback_panel(record if show_feedback else None)

    def _on_reprocess(self) -> None:
        record = self._selected_record()
        if record is None or not can_reprocess(record):
            return
        if record.id in (self.watcher.status.get("processing_jobs") or {}):
            return  # already busy (button should be disabled already; this is a defensive belt-and-braces check)
        interview_id = record.id
        self._reprocess_btn.config(state="disabled")

        def _run():
            try:
                self.watcher.reprocess_interview(interview_id)
            except Exception as e:  # noqa: BLE001
                logger.exception("Reprocessing interview #%s failed", interview_id)
                # `e` is deleted when this except block exits (standard
                # Python behavior for `except ... as e`), so the rendered
                # markdown must be built now -- a lambda referring to `e`
                # directly would raise NameError once .after() runs it
                # later, since by then the name no longer exists.
                error_markdown = _friendly_error_markdown("Reprocessing failed", e)
                if self._root is not None:
                    # only re-render the row list (to re-enable Reprocess,
                    # since the audio is still there) -- NOT the detail
                    # pane, which already shows the error message below and
                    # would otherwise get immediately overwritten by
                    # _on_history_select()'s normal "not processed" text
                    self._root.after(0, self._refresh_history)
                    self._root.after(0, lambda msg=error_markdown: self._show_reprocess_error(msg))
            else:
                # success -- re-render the row list and the now-available report
                if self._root is not None:
                    self._root.after(0, self._refresh_history)
                    self._root.after(0, self._on_history_select)

        threading.Thread(target=_run, daemon=True).start()

    def _show_reprocess_error(self, message: str) -> None:
        """`message` is already fully-formed markdown from
        _friendly_error_markdown -- built in _on_reprocess while the
        original exception object was still alive."""
        self._history_text.config(state="normal")
        render_into_text_widget(self._history_text, message)
        self._history_text.config(state="disabled")

    def _on_open_audio(self) -> None:
        record = self._selected_record()
        if record is None or not has_audio(record):
            return
        try:
            _open_with_os_default(record.audio_path)
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to open audio for interview #%s", record.id)
            self._history_text.config(state="normal")
            render_into_text_widget(self._history_text, f"# Couldn't open audio\n\n{e}")
            self._history_text.config(state="disabled")

    def _on_view_transcript(self) -> None:
        record = self._selected_record()
        if record is None or not record.transcript:
            return
        self._history_text.config(state="normal")
        self._history_text.delete("1.0", "end")
        self._history_text.insert("end", "Transcript\n", ("h1",))
        self._history_text.insert("end", "\n")
        self._render_transcript_with_speaker_colors(self._history_text, record.transcript)
        self._history_text.config(state="disabled")

    def _on_view_infographic(self) -> None:
        """Generates (or regenerates, so it always reflects the current
        analysis) the HTML infographic for the selected interview and
        opens it with the OS's default browser -- see infographic.py."""
        record = self._selected_record()
        if record is None:
            return
        try:
            path = write_interview_infographic(record, self.watcher.cfg)
            if path is None:
                return  # no usable analysis to visualize -- button should already be disabled then
            _open_with_os_default(path)
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to open infographic for interview #%s", record.id)
            self._history_text.config(state="normal")
            render_into_text_widget(self._history_text, f"# Couldn't open infographic\n\n{e}")
            self._history_text.config(state="disabled")

    def _render_transcript_with_speaker_colors(self, text_widget, transcript: str) -> None:
        """Renders '[Speaker] text' lines as color-coded, per-speaker
        paragraphs (see _group_transcript_by_speaker/_speaker_color) --
        consecutive same-speaker lines become one flowing paragraph
        instead of repeating the label on every line. Colors are assigned
        fresh each call (see _speaker_color), and Tk tags are created
        on demand per speaker name actually present in this transcript."""
        assigned_colors: dict[str, str] = {}
        existing_tags = set(text_widget.tag_names())
        for speaker, paragraph_text in _group_transcript_by_speaker(transcript):
            if not speaker:
                # a line that didn't match "[Speaker] text" at all --
                # inserted as-is rather than dropped (see
                # _parse_transcript_lines)
                text_widget.insert("end", paragraph_text + "\n\n", ("text",))
                continue
            color = _speaker_color(speaker, assigned_colors)
            label_tag = f"speaker_label::{speaker}"
            body_tag = f"speaker_body::{speaker}"
            if label_tag not in existing_tags:
                text_widget.tag_configure(label_tag, foreground=color, font=("Segoe UI", 10, "bold"))
                text_widget.tag_configure(body_tag, foreground=color)
                existing_tags.add(label_tag)
                existing_tags.add(body_tag)
            text_widget.insert("end", f"[{speaker}] ", (label_tag,))
            text_widget.insert("end", paragraph_text + "\n\n", (body_tag,))

    def _on_delete(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        if record.id in (self.watcher.status.get("processing_jobs") or {}):
            return  # still being processed -- shouldn't happen since the button is disabled then, but be safe
        import tkinter.messagebox as messagebox

        confirmed = messagebox.askyesno(
            "Delete interview",
            f"Delete the {format_started(record)} interview ({record.source_app or 'unknown app'})?\n\n"
            "This also removes its audio and report files from disk, and can't be undone.",
            parent=self._root,
        )
        if not confirmed:
            return
        for path_str in (record.audio_path, record.report_path):
            if path_str:
                try:
                    pathlib.Path(path_str).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Couldn't delete file %s for interview #%s", path_str, record.id)
        self.watcher.db.delete_interview(record.id)
        self._refresh_history()
        self._history_text.config(state="normal")
        self._history_text.delete("1.0", "end")
        self._history_text.config(state="disabled")
        self._refresh_feedback_panel(None)

    # -- Trends tab -------------------------------------------------------

    def _build_trends_tab(self, notebook, tk, ttk):
        frame = ttk.Frame(notebook, padding=10)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(frame)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Button(toolbar, text="Refresh", command=self._on_refresh_trends).pack(side="left")
        ttk.Button(toolbar, text="View infographic", command=self._on_view_trends_infographic).pack(
            side="left", padx=(8, 0)
        )

        text = tk.Text(frame, wrap="word", padx=10, pady=8, state="disabled", relief="flat")
        _configure_report_tags(text)
        text_vsb = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=text_vsb.set)
        text.grid(row=1, column=0, sticky="nsew")
        text_vsb.grid(row=1, column=1, sticky="ns")
        self._trends_text = text

        return frame

    def _refresh_trends(self) -> None:
        if self._trends_text is None:
            return
        # Regenerated from the DB every time rather than just reading
        # whatever file happens to be on disk -- it's a cheap in-memory
        # aggregation (no LLM calls), and relying on a stale/possibly-never
        # -written file is exactly what caused a real bug: a profile with
        # real interview history saw "No trends yet" because its per-user
        # trends file had never been generated (the file previously only
        # got (re)written as a side effect of finishing a *new* interview's
        # analysis, so profiles whose most recent interview predated that
        # per-user split never got one written for them retroactively).
        records = self.watcher.db.list_all(user_id=self.watcher.user_id)
        trends_path = write_trends_report(records, self.watcher.cfg, user_id=self.watcher.user_id)
        self._trends_text.config(state="normal")
        render_into_text_widget(self._trends_text, trends_path.read_text(encoding="utf-8"))
        self._trends_text.config(state="disabled")

    def _on_refresh_trends(self) -> None:
        self._refresh_trends()
        self._wake_ollama_async()

    def _on_view_trends_infographic(self) -> None:
        """Regenerates (so it always reflects the current DB state, same
        "always fresh" approach as _refresh_trends) and opens the HTML
        trends infographic -- see infographic.py's write_trends_infographic."""
        records = self.watcher.db.list_all(user_id=self.watcher.user_id)
        try:
            path = write_trends_infographic(records, self.watcher.cfg, user_id=self.watcher.user_id)
            _open_with_os_default(path)
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to open the trends infographic")
            self._trends_text.config(state="normal")
            render_into_text_widget(self._trends_text, f"# Couldn't open infographic\n\n{e}")
            self._trends_text.config(state="disabled")

    # -- Settings tab -----------------------------------------------------

    def _build_settings_tab(self, notebook, tk, ttk):
        outer, frame = self._make_scrollable_tab(notebook, tk, ttk)
        cfg = self.watcher.cfg

        if cfg.path is None:
            ttk.Label(frame, text="No config file path available to edit.").pack(anchor="w")
            return outer

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
        _row(r, "Start debounce, apps (polls)", "start_debounce_polls", e); r += 1

        e = ttk.Spinbox(form, from_=1, to=60, width=8)
        e.set(current.get("browser_start_debounce_polls", 6))
        _row(r, "Start debounce, browser tabs (polls)", "browser_start_debounce_polls", e); r += 1

        e = ttk.Spinbox(form, from_=1, to=60, width=8)
        e.set(current.get("stop_debounce_polls", 12))
        _row(r, "Stop debounce (polls)", "stop_debounce_polls", e); r += 1

        e = ttk.Entry(form, width=10)
        e.insert(0, str(current.get("declined_cooldown_seconds", 300)))
        _row(r, "Re-prompt after declining (seconds)", "declined_cooldown_seconds", e); r += 1

        e = ttk.Spinbox(form, from_=16, to=320, width=8)
        e.set(current.get("audio.bitrate_kbps", 64))
        _row(r, "Audio bitrate (kbps)", "audio.bitrate_kbps", e); r += 1

        mic_var = tk.BooleanVar(value=bool(current.get("audio.include_microphone", True)))
        e = ttk.Checkbutton(form, variable=mic_var, text="Include your microphone (recommended)")
        e.var = mic_var
        _row(r, "Record your voice", "audio.include_microphone", e); r += 1

        e = ttk.Combobox(form, values=_TRANSCRIPTION_ENGINES, state="readonly", width=14)
        e.set(current.get("transcription.engine", "faster-whisper"))
        _row(r, "Transcription engine", "transcription.engine", e); r += 1
        ttk.Label(
            form,
            text="faster-whisper = local, free, fully private (default). groq = Groq's hosted\n"
                 "Whisper API -- free (no credit card), much faster on a slow machine, but your\n"
                 "audio leaves this machine and goes to Groq's servers. Needs a Groq API key\n"
                 "below (get one free at console.groq.com/keys).",
            foreground="#6b6b6b", justify="left",
        ).grid(row=r, column=1, sticky="w"); r += 1

        e = ttk.Combobox(form, values=_WHISPER_MODELS, state="readonly", width=12)
        e.set(current.get("transcription.whisper_model", "small"))
        _row(r, "Whisper model", "transcription.whisper_model", e); r += 1

        diarization_var = tk.BooleanVar(value=bool(current.get("transcription.diarization", True)))
        e = ttk.Checkbutton(form, variable=diarization_var, text="Separate speakers")
        e.var = diarization_var  # keep a reference so it isn't garbage-collected
        _row(r, "Diarization", "transcription.diarization", e); r += 1

        live_transcribe_var = tk.BooleanVar(value=bool(current.get("transcription.live_during_recording", False)))
        e = ttk.Checkbutton(form, variable=live_transcribe_var, text="Transcribe while recording")
        e.var = live_transcribe_var
        _row(r, "Live transcription", "transcription.live_during_recording", e); r += 1
        ttk.Label(
            form,
            text="Off by default. When on, most of the transcript is already done by the time\n"
                 "a call ends instead of only starting afterward -- but it's a newer path through\n"
                 "the code, so it's opt-in until you've tried it. Any failure falls back "
                 "automatically to\ntranscribing normally after the call ends; recording itself is "
                 "never affected either way.",
            foreground="#6b6b6b", justify="left",
        ).grid(row=r, column=1, sticky="w"); r += 1

        e = ttk.Combobox(form, values=_TRANSCRIPTION_LANGUAGES, state="readonly", width=12)
        e.set(current.get("transcription.language", "auto"))
        _row(r, "Language", "transcription.language", e); r += 1
        ttk.Label(
            form,
            text="auto = detect automatically. hinglish = Hindi speech, romanized where possible.",
            foreground="#6b6b6b",
        ).grid(row=r, column=1, sticky="w"); r += 1

        e = ttk.Combobox(form, values=_ANALYSIS_ENGINES, state="readonly", width=12)
        e.set(current.get("analysis.engine", "ollama"))
        _row(r, "Analysis engine", "analysis.engine", e); r += 1

        model_row = ttk.Frame(form)
        # editable (not readonly) so a model outside the curated catalog can
        # still be typed in directly -- the catalog is a convenience list,
        # not the only thing this app can run
        e = ttk.Combobox(model_row, values=list(MODEL_CATALOG.keys()), width=18)
        e.set(current.get("analysis.llm_model", ""))
        e.pack(side="left")
        self._model_name_entry = e
        ttk.Button(model_row, text="Install model...", command=self._on_install_model).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(form, text="Model name").grid(row=r, column=0, sticky="w", pady=4, padx=(0, 12))
        model_row.grid(row=r, column=1, sticky="ew", pady=4)
        self._settings_widgets["analysis.llm_model"] = e
        r += 1

        self._model_status_label = ttk.Label(form, text="", foreground="#5b645f")
        self._model_status_label.grid(row=r, column=1, sticky="w"); r += 1
        e.bind("<<ComboboxSelected>>", lambda _e: self._refresh_model_status_label())
        e.bind("<KeyRelease>", lambda _e: self._refresh_model_status_label())
        self._refresh_model_status_label()

        r = self._build_api_key_row(form, tk, ttk, r)

        e = ttk.Entry(form)
        e.insert(0, current.get("output.output_dir", "output"))
        _row(r, "Reports output dir", "output.output_dir", e); r += 1

        ttk.Separator(form, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=10); r += 1

        ttk.Label(form, text="Language packs", font=("Segoe UI", 10, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w"
        ); r += 1
        r = self._build_language_pack_rows(form, ttk, r)

        self._settings_status = ttk.Label(frame, text="", foreground="#2f6f5e")
        ttk.Button(frame, text="Save", command=self._on_save_settings).pack(anchor="w", pady=(16, 4))
        self._settings_status.pack(anchor="w")

        return outer

    def _build_language_pack_rows(self, form, ttk, r: int) -> int:
        """One row per optional language pack (see language_packs.py) --
        install/uninstall any time, not just during first-run setup."""
        for pack_id, entry in LANGUAGE_PACKS.items():
            row = ttk.Frame(form)
            row.grid(row=r, column=0, columnspan=2, sticky="ew", pady=2)
            ttk.Label(row, text=entry["label"], width=32).pack(side="left")
            status_label = ttk.Label(row, text="", foreground="#5b645f", width=14)
            status_label.pack(side="left")
            action_btn = ttk.Button(row, text="", command=lambda pid=pack_id: self._on_toggle_language_pack(pid))
            action_btn.pack(side="left")
            self._lang_pack_rows[pack_id] = {"status_label": status_label, "action_btn": action_btn}
            r += 1
        self._refresh_language_pack_rows()
        return r

    def _refresh_language_pack_rows(self) -> None:
        for pack_id, widgets in self._lang_pack_rows.items():
            installed = is_pack_installed(pack_id)
            widgets["status_label"].config(text="Installed" if installed else "Not installed")
            widgets["action_btn"].config(text="Uninstall" if installed else "Install")

    def _on_toggle_language_pack(self, pack_id: str) -> None:
        from tkinter import messagebox

        entry = LANGUAGE_PACKS[pack_id]
        installed = is_pack_installed(pack_id)
        action = "uninstall" if installed else "install"
        verb = "Uninstall" if installed else "Install"
        confirmed = messagebox.askyesno(
            "Interview Analyzer",
            f"{verb} the '{entry['label']}' language pack ({entry['pip_package']})?\n\n{entry['description']}",
            parent=self._root,
        )
        if not confirmed:
            return
        PackActionDialog(pack_id, action, ui_root=self._root, on_done=lambda _ok: self._refresh_language_pack_rows())

    def _build_api_key_row(self, form, tk, ttk, r: int) -> int:
        """Cloud API key entry for groq/anthropic_api/openai_api -- saved
        locally, encrypted with Windows DPAPI (see api_keys.py), never
        written into config.yaml. A claude.ai/ChatGPT *subscription* does
        not grant API access -- this only ever stores a real API key you
        paste in from console.groq.com / console.anthropic.com /
        platform.openai.com. One saved "groq" key covers both the
        transcription engine and the "groq_api" analysis engine above --
        Groq issues one key per account for everything."""
        row = ttk.Frame(form)
        row.grid(row=r, column=0, columnspan=2, sticky="ew", pady=2)

        self._api_key_provider = ttk.Combobox(
            row, values=["groq", "anthropic_api", "openai_api"], state="readonly", width=13
        )
        self._api_key_provider.set("groq")
        self._api_key_provider.pack(side="left")
        self._api_key_entry = ttk.Entry(row, show="*", width=24)
        self._api_key_entry.pack(side="left", padx=(6, 6))
        ttk.Button(row, text="Save key", command=self._on_save_api_key).pack(side="left")
        ttk.Button(row, text="Clear key", command=self._on_clear_api_key).pack(side="left", padx=(6, 0))
        self._api_key_provider.bind("<<ComboboxSelected>>", lambda _e: self._refresh_api_key_status())

        ttk.Label(form, text="Cloud API key").grid(row=r, column=0, sticky="w", pady=4, padx=(0, 12))
        r += 1
        self._api_key_status_label = ttk.Label(form, text="", foreground="#5b645f")
        self._api_key_status_label.grid(row=r, column=1, sticky="w"); r += 1
        ttk.Label(
            form,
            text="groq: free, no credit card -- get one at console.groq.com/keys. A claude.ai /\n"
                 "ChatGPT subscription does NOT work for anthropic_api/openai_api -- those need a\n"
                 "real API key from console.anthropic.com / platform.openai.com (separately billed).",
            foreground="#6b6b6b", justify="left",
        ).grid(row=r, column=1, sticky="w"); r += 1

        self._refresh_api_key_status()
        return r

    def _refresh_api_key_status(self) -> None:
        if self._api_key_status_label is None:
            return
        provider = self._api_key_provider.get()
        if api_keys.has_key(provider):
            key = api_keys.load_key(provider)
            shown = api_keys.masked(key) if key else "saved"
            self._api_key_status_label.config(text=f"{provider}: key saved ({shown})", foreground="#2f6f5e")
        else:
            self._api_key_status_label.config(text=f"{provider}: not set", foreground="#5b645f")

    def _on_save_api_key(self) -> None:
        from tkinter import messagebox

        provider = self._api_key_provider.get()
        key = self._api_key_entry.get().strip()
        if not key:
            return
        if api_keys.save_key(provider, key):
            self._api_key_entry.delete(0, "end")
            self._refresh_api_key_status()
        else:
            messagebox.showwarning(
                "Interview Analyzer",
                "Couldn't save the key locally (Windows DPAPI unavailable). "
                "Use the INTERVIEW_ANALYZER_API_KEY environment variable instead -- "
                "see docs/using_cloud_apis.md.",
                parent=self._root,
            )

    def _on_clear_api_key(self) -> None:
        provider = self._api_key_provider.get()
        api_keys.clear_key(provider)
        self._refresh_api_key_status()

    def _refresh_model_status_label(self) -> None:
        if self._model_status_label is None:
            return
        model_name = self._model_name_entry.get().strip()
        if not model_name:
            self._model_status_label.config(text="")
            return
        self._model_status_label.config(text=size_label(model_name))

    def _on_install_model(self) -> None:
        from tkinter import messagebox

        model_name = self._model_name_entry.get().strip()
        if not model_name:
            return
        host = self.watcher.cfg.analysis.get("ollama_host", "http://localhost:11434")
        catalog_note = MODEL_CATALOG.get(model_name, {}).get("description", "")
        if ollama_is_reachable(host) and is_model_installed(model_name, host):
            messagebox.showinfo("Interview Analyzer", f"{model_name} is already installed.", parent=self._root)
            return
        confirmed = messagebox.askyesno(
            "Interview Analyzer",
            f"Download {model_name} ({size_label(model_name)}) via Ollama?\n\n{catalog_note}\n\n"
            "This downloads to your machine and runs fully locally afterwards.",
            parent=self._root,
        )
        if not confirmed:
            return
        ModelInstallDialog(model_name, host, ui_root=self._root)

    def _on_save_settings(self) -> None:
        updates: dict[str, object] = {}
        for dotted, widget in self._settings_widgets.items():
            if hasattr(widget, "var"):
                updates[dotted] = bool(widget.var.get())
            else:
                updates[dotted] = widget.get()

        # Switching to an Ollama model that isn't downloaded yet is a
        # multi-GB action -- confirm it explicitly rather than silently
        # kicking off a background pull the user didn't ask for.
        new_engine = updates.get("analysis.engine", self.watcher.cfg.analysis.get("engine", "ollama"))
        new_model = updates.get("analysis.llm_model", self.watcher.cfg.analysis.get("llm_model"))
        if new_engine == "ollama" and new_model:
            host = self.watcher.cfg.analysis.get("ollama_host", "http://localhost:11434")
            if ollama_is_reachable(host) and not is_model_installed(new_model, host):
                from tkinter import messagebox

                confirmed = messagebox.askyesno(
                    "Interview Analyzer",
                    f"{new_model} isn't installed yet ({size_label(new_model)} download). "
                    "Install it now via the Settings tab's \"Install model...\" button, "
                    "then Save again?\n\nSaving without installing it first will make analysis "
                    "fail until you do.",
                    parent=self._root,
                )
                if not confirmed:
                    return

        try:
            save_editable_settings(self.watcher.cfg.path, updates)
        except (ValueError, TypeError) as e:
            self._settings_status.config(text=f"Couldn't save: {e}", foreground="#a8722a")
            return

        self._settings_status.config(
            text="Saved. Restart the app for changes to take effect.", foreground="#2f6f5e"
        )
