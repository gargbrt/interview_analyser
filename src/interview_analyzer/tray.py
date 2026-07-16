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

# state -> (fill color, tooltip label)
_STATE_STYLE = {
    "idle": ("#8a8f8a", "Idle — watching for calls"),
    "recording": ("#c0392b", "Recording"),
    "paused": ("#c8892c", "Paused"),
}


def status_text(status: dict) -> str:
    """Human-readable one-liner for a watcher.status snapshot, shared by
    the tray tooltip/menu label and (indirectly) the dashboard."""
    state = status.get("state", "idle")
    if state == "idle":
        return "Idle — watching for calls"
    app_name = status.get("app_name") or "call"
    return f"{'Paused' if state == 'paused' else 'Recording'} — {app_name}"


def _make_icon_image(state: str):
    color, _ = _STATE_STYLE.get(state, _STATE_STYLE["idle"])
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
    ):
        if pystray is None:
            raise RuntimeError(
                "pystray and Pillow are required for the tray icon. "
                "Install with `pip install pystray Pillow`."
            )
        self.watcher = watcher
        self._open_dashboard = open_dashboard
        self._on_quit = on_quit
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
                lambda item: "Resume recording" if self.watcher.status.get("state") == "paused" else "Pause recording",
                self._toggle_pause,
                visible=lambda item: self.watcher.status.get("state") != "idle",
            ),
            pystray.MenuItem(
                "Stop recording",
                lambda icon, item: self.watcher.request_stop_recording(),
                visible=lambda item: self.watcher.status.get("state") != "idle",
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open dashboard", lambda icon, item: self._open_dashboard(), default=True),
            pystray.MenuItem("Quit", lambda icon, item: self._quit()),
        )

    def _toggle_pause(self, icon, item) -> None:
        if self.watcher.status.get("state") == "paused":
            self.watcher.resume_recording()
        else:
            self.watcher.pause_recording()

    def _quit(self) -> None:
        self._on_quit()
        self._icon.stop()

    def refresh(self) -> None:
        """Re-read watcher.status and update the icon image/tooltip. Safe
        to call from any thread -- pass this as MeetingWatcher's
        `on_state_change` callback."""
        status = self.watcher.status
        state = status.get("state", "idle")
        try:
            self._icon.icon = _make_icon_image(state)
            self._icon.title = status_text(status)
            self._icon.update_menu()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to refresh tray icon")

    def run(self) -> None:
        """Blocking call -- run on the main thread."""
        self._icon.run()
