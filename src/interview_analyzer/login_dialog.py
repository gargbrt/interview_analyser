"""GUI counterpart to `auth.login_or_create`'s console prompts, for the
tray+dashboard app (`app.py`). Same login rules, just asked via a small
Tkinter dialog instead of `input()`/`getpass` so the app never needs a
console window.
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from typing import Optional

from .auth import User, create_user, ensure_users_table, get_user_by_username, verify_password

logger = logging.getLogger(__name__)


def gui_login_or_create(conn: sqlite3.Connection) -> Optional[User]:
    """Show a login dialog; returns the logged-in/created User, or None if
    the user closed the dialog without completing login."""
    ensure_users_table(conn)
    result_queue: "queue.Queue[Optional[User]]" = queue.Queue()

    def _show():
        try:
            import tkinter as tk
        except ImportError:  # pragma: no cover
            logger.warning("Tkinter not available; cannot show the login dialog.")
            result_queue.put(None)
            return

        root = tk.Tk()
        root.title("Interview Analyzer — Log in")
        root.attributes("-topmost", True)
        root.resizable(False, False)

        tk.Label(
            root,
            text="Local profile name\n(scopes your interview history and trend tracking)",
            padx=20, justify="left",
        ).pack(pady=(18, 6))

        username_var = tk.StringVar()
        username_entry = tk.Entry(root, textvariable=username_var, width=28)
        username_entry.pack(padx=20)
        username_entry.focus_set()

        password_label = tk.Label(root, text="Password (only if this profile has one)", padx=20)
        password_var = tk.StringVar()
        password_entry = tk.Entry(root, textvariable=password_var, show="*", width=28)
        password_label.pack(pady=(10, 4))
        password_entry.pack(padx=20)

        error_var = tk.StringVar()
        tk.Label(root, textvariable=error_var, fg="red", padx=20, wraplength=260, justify="left").pack(pady=(6, 0))

        def _submit(_event=None):
            username = username_var.get().strip()
            if not username:
                error_var.set("Enter a profile name.")
                return

            existing = get_user_by_username(conn, username)
            if existing is None:
                user = create_user(conn, username, password_var.get() or None)
                result_queue.put(user)
                root.destroy()
                return

            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row["password_hash"]:
                if not verify_password(conn, username, password_var.get()):
                    error_var.set("Incorrect password for that profile.")
                    return
            result_queue.put(existing)
            root.destroy()

        def _cancel():
            result_queue.put(None)
            root.destroy()

        root.bind("<Return>", _submit)
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="Continue", width=12, command=_submit).pack(side="left", padx=8)
        tk.Button(btn_frame, text="Cancel", width=12, command=_cancel).pack(side="left", padx=8)
        root.protocol("WM_DELETE_WINDOW", _cancel)

        root.mainloop()

    thread = threading.Thread(target=_show, daemon=True)
    thread.start()
    thread.join()

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return None
