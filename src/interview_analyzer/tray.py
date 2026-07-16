"""System tray icon: the app's always-available control surface. Shows the
current state at a glance (idle / recording / paused) and lets you
pause/resume/stop a recording or open the dashboard without hunting for a
window on the taskbar.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    pystray = None
    Image = None
    ImageDraw = None

# state -> fill color
_STATE_COLOR = {
    "idle": "#8a8f8a",
    "recording": "#c0392b",
    "paused": "#c8892c",
    "processing": "#2f6fa8",
}

STAGE_LABEL = {
    "compressing": "Compressing",
    "transcribing": "Transcribing",
    "analyzing": "Analyzing",
    "generating_report": "Generating report",
}


def job_text(job: dict) -> str:
    """Human-readable one-liner for a single processing_jobs entry, e.g.
    'Transcribing… 42%'. Shared by the tray and the dashboard's History tab
    (which shows this per-row for whichever interview is being processed)."""
    stage = STAGE_LABEL.get(job.get("stage"), "Processing")
    progress = job.get("progress")
    if progress is not None:
        return f"{stage}… {round(progress * 100)}%"
    return f"{stage}…"


def status_text(status: dict) -> str:
    """Human-readable one-liner for a watcher.status snapshot, shared by
    the tray tooltip/menu label and the dashboard. Recording/paused and
    background processing are independent now (a new call can be recording
    while an earlier one is still being transcribed/analyzed), so both are
    reflected together when both are happening."""
    state = status.get("state", "idle")
    jobs = status.get("processing_jobs") or {}

    if state == "idle":
        base = "Idle — watching for calls"
    else:
        app_name = status.get("app_name") or "call"
        base = f"{'Paused' if state == 'paused' else 'Recording'} — {app_name}"

    if not jobs:
        return base

    if len(jobs) == 1:
        job_summary = job_text(next(iter(jobs.values())))
    else:
        job_summary = f"{len(jobs)} interviews processing"

    return job_summary if state == "idle" else f"{base} · {job_summary}"


def _visual_state(status: dict) -> str:
    """Which of idle/recording/paused/processing the icon should actually
    look like -- recording/paused takes visual priority (it's the state
    that needs attention), background processing shows only when nothing
    is actively being recorded."""
    state = status.get("state", "idle")
    if state in ("recording", "paused"):
        return state
    return "processing" if status.get("processing_jobs") else "idle"


def _make_icon_image(state: str):
    color = _STATE_COLOR.get(state, _STATE_COLOR["idle"])
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad = 6
    draw.ellipse((pad, pad, size - pad, size - pad), fill=color)
    if state == "paused":
        # two vertical bars, like a pause glyph, so paused is
        # distinguishable from recording at a glance even without color
        bar_w, bar_h = 7, 26
        cy = size // 2
        draw.rectangle((22, cy - bar_h // 2, 22 + bar_w, cy + bar_h // 2), fill="white")
        draw.rectangle((size - 22 - bar_w, cy - bar_h // 2, size - 22, cy + bar_h // 2), fill="white")
    elif state == "processing":
        # three dots, like a "working on it" glyph
        cy = size // 2
        for dx in (-14, 0, 14):
            cx = size // 2 + dx
            draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill="white")
    return img


class TrayIcon:
    """Wraps a pystray.Icon reflecting a MeetingWatcher's live status.

    `watcher` drives all state; this class only reads `watcher.status` and
    calls `watcher.pause_recording()` / `.resume_recording()` /
    `.request_stop_recording()`. `open_dashboard` and `on_quit` are
    provided by the app entry point.
    """

    def __init__(
        self,
        watcher,
        open_dashboard: Callable[[], None],
        on_quit: Callable[[], None],
        on_logout: Optional[Callable[[], None]] = None,
    ):
        if pystray is None:
            raise RuntimeError(
                "pystray and Pillow are required for the tray icon. "
                "Install with `pip install pystray Pillow`."
            )
        self.watcher = watcher
        self._open_dashboard = open_dashboard
        self._on_quit = on_quit
        self._on_logout = on_logout
        self._icon = pystray.Icon(
            "interview_analyzer",
            icon=_make_icon_image("idle"),
            title=status_text(watcher.status),
            menu=self._build_menu(),
        )

    def _build_menu(self):
        return pystray.Menu(
            pystray.MenuItem(lambda item: status_text(self.watcher.status), None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Start recording now (not detected?)",
                lambda icon, item: self._start_manually(),
                visible=lambda item: self.watcher.status.get("state") == "idle",
            ),
            pystray.MenuItem(
                lambda item: "Resume recording" if self.watcher.status.get("state") == "paused" else "Pause recording",
                self._toggle_pause,
                visible=lambda item: self.watcher.status.get("state") in ("recording", "paused"),
            ),
            pystray.MenuItem(
                "Stop recording",
                lambda icon, item: self.watcher.request_stop_recording(),
                visible=lambda item: self.watcher.status.get("state") in ("recording", "paused"),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open dashboard", lambda icon, item: self._open_dashboard(), default=True),
            pystray.MenuItem(
                "Log out",
                lambda icon, item: self._logout(),
                visible=lambda item: self._on_logout is not None and self.watcher.status.get("state") == "idle",
            ),
            pystray.MenuItem("Quit", lambda icon, item: self._quit()),
        )

    def _start_manually(self) -> None:
        # a fixed generic label -- the dashboard's Status tab has a text
        # field if you want to name the app/platform yourself
        try:
            self.watcher.request_start_recording("Manual")
        except RuntimeError:
            pass  # already recording -- the menu item should be hidden then anyway

    def _toggle_pause(self, icon, item) -> None:
        if self.watcher.status.get("state") == "paused":
            self.watcher.resume_recording()
        else:
            self.watcher.pause_recording()

    def _quit(self) -> None:
        self._on_quit()
        self._icon.stop()

    def _logout(self) -> None:
        # only visible while idle (see _build_menu) -- logging out mid-call
        # would silently abandon an active recording, so that's blocked at
        # the menu level rather than needing a confirmation dialog here
        if self._on_logout is not None:
            self._on_logout()
        self._icon.stop()

    def refresh(self) -> None:
        """Re-read watcher.status and update the icon image/tooltip. Safe
        to call from any thread -- pass this as MeetingWatcher's
        `on_state_change` callback."""
        status = self.watcher.status
        try:
            self._icon.icon = _make_icon_image(_visual_state(status))
            self._icon.title = status_text(status)
            self._icon.update_menu()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to refresh tray icon")

    def run(self) -> None:
        """Blocking call -- run on the main thread."""
        self._icon.run()
