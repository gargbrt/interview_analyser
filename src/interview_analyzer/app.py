"""Primary entry point: system tray icon + dashboard window.

    python -m interview_analyzer.app [--username NAME]

Shows a GUI login dialog (unless --username is given), then runs the
meeting watcher in the background with a tray icon for at-a-glance status
and pause/resume/stop, and a dashboard window (opened from the tray, and
once automatically on first run) for browsing report history, trends, and
settings.

For a headless/console-only run with no GUI at all (e.g. an unattended
server-style startup task), use `python -m interview_analyzer.watcher`
instead -- see docs/run_at_startup.md.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading

from .auth import create_user, get_user_by_username
from .config_loader import load_config
from .dashboard import Dashboard
from .db import InterviewDB
from .login_dialog import gui_login_or_create
from .model_setup import maybe_run_first_time_setup
from .single_instance import acquire_single_instance_lock
from .tray import TrayIcon
from .watcher import MeetingWatcher

logger = logging.getLogger(__name__)


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

    db = InterviewDB(cfg.resolve(cfg.storage.get("db_path", "data/interviews.db")))

    if args.username:
        user = get_user_by_username(db._conn, args.username) or create_user(db._conn, args.username)
    else:
        user = gui_login_or_create(db._conn)
        if user is None:
            logger.info("Login cancelled; exiting.")
            db.close()
            return

    logger.info("Logged in as '%s' (profile #%s).", user.username, user.id)
    db.close()

    watcher = MeetingWatcher(cfg, user_id=user.id)
    dashboard = Dashboard(watcher)
    tray = TrayIcon(watcher, open_dashboard=dashboard.open, on_quit=watcher.shutdown)
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

    tray.run()  # blocking; runs on the main thread


if __name__ == "__main__":
    main()
