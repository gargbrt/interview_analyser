"""Primary entry point: system tray icon + dashboard window.

    python -m interview_analyzer.app [--username NAME]

Shows a GUI login dialog (unless --username is given), then runs the
meeting watcher in the background with a tray icon for at-a-glance status
and pause/resume/stop, and a dashboard window (opened from the tray, and
once automatically on first run) for browsing report history, trends, and
settings.

"Log out" (tray menu, visible while idle) ends the current profile's
session and returns to the login dialog without restarting the process --
see `_run_session()`. "Quit" exits the process entirely.

For a headless/console-only run with no GUI at all (e.g. an unattended
server-style startup task), use `python -m interview_analyzer.watcher`
instead -- see docs/run_at_startup.md.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import Optional

from .auth import User, create_user, get_user_by_username
from .config_loader import Config, load_config
from .dashboard import Dashboard
from .db import InterviewDB
from .login_dialog import gui_login_or_create
from .model_setup import maybe_run_first_time_setup
from .single_instance import acquire_single_instance_lock
from .tray import TrayIcon
from .watcher import MeetingWatcher

logger = logging.getLogger(__name__)


def _login(cfg: Config, args: argparse.Namespace) -> Optional[User]:
    """Opens its own short-lived DB connection just to resolve/create the
    user profile -- the session itself (once logged in) uses MeetingWatcher's
    own connection instead (see `_run_session`)."""
    db = InterviewDB(cfg.resolve(cfg.storage.get("db_path", "data/interviews.db")))
    try:
        if args.username:
            return get_user_by_username(db._conn, args.username) or create_user(db._conn, args.username)
        return gui_login_or_create(db._conn, cfg=cfg)
    finally:
        db.close()


def _run_session(cfg: Config, user: User) -> str:
    """Runs one logged-in profile's watcher + dashboard + tray until the
    user clicks Quit or Log out. Returns "quit" or "logout" so the caller
    knows whether to loop back to the login dialog."""
    watcher = MeetingWatcher(cfg, user_id=user.id)
    result = {"action": "quit"}

    def _do_logout() -> None:
        # shared by both the tray menu's "Log out" item and the dashboard's
        # own Log out button (Status tab) -- either one stops the tray icon,
        # which is what unblocks tray.run() below regardless of which
        # triggered it.
        result["action"] = "logout"
        watcher.shutdown()
        tray.stop()

    dashboard = Dashboard(watcher, on_logout=_do_logout)
    tray = TrayIcon(
        watcher,
        open_dashboard=dashboard.open,
        on_quit=watcher.shutdown,
        on_logout=_do_logout,
    )
    watcher.set_on_state_change(lambda: (tray.refresh(), dashboard.notify_state_change()))

    # Open the dashboard (its Tk root becomes the one shared UI thread that
    # consent/control-panel popups build Toplevels on -- see consent.py's
    # docstring) and wait for it before the watcher starts polling, so
    # there's no window where a popup might spin up its own competing Tk()
    # interpreter concurrently with the dashboard's.
    dashboard.open()
    if not dashboard.wait_until_ready(timeout=10):
        logger.warning(
            "Dashboard did not open in time; consent/control-panel popups will run as standalone windows."
        )

    # One-time (per install) prompt to install the local analysis model or
    # skip it for a cloud API instead -- before the watcher starts polling,
    # same reasoning as the dashboard-readiness wait above: no popup should
    # spin up a competing Tk() interpreter concurrently with the dashboard's.
    maybe_run_first_time_setup(cfg, ui_root=watcher.ui_root)

    watcher_thread = threading.Thread(target=watcher.run_forever, daemon=True)
    watcher_thread.start()

    tray.run()  # blocking; returns once Quit or Log out stops the icon

    # Tear down fully before returning -- if this was a logout, the caller
    # is about to show a new login dialog, which needs its own standalone
    # Tk() root and must not run concurrently with a still-live dashboard
    # root (see consent.py's docstring on why that's unsafe).
    watcher.shutdown()
    watcher_thread.join(timeout=cfg.poll_interval_seconds + 5)
    dashboard.close()
    dashboard.wait_until_closed(timeout=5)

    return result["action"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Interview Analyzer (tray + dashboard)")
    parser.add_argument("--username", help="Skip the login dialog with this profile name")
    args = parser.parse_args()

    cfg = load_config()

    lock_path = cfg.resolve(cfg.storage.get("db_path", "data/interviews.db")).with_name(".app.lock")
    if not acquire_single_instance_lock(lock_path):
        logger.error(
            "Interview Analyzer is already running (lock held at %s). "
            "Use the existing tray icon instead of starting a second copy.",
            lock_path,
        )
        sys.exit(1)

    while True:
        user = _login(cfg, args)
        if user is None:
            logger.info("Login cancelled; exiting.")
            return

        logger.info("Logged in as '%s' (profile #%s).", user.username, user.id)
        action = _run_session(cfg, user)

        if action != "logout":
            return
        if args.username:
            # --username mode skipped the login dialog entirely, so there's
            # nothing to log back out *to* -- just exit like Quit would.
            logger.info("Logged out.")
            return
        logger.info("Logged out; returning to login.")


if __name__ == "__main__":
    main()
