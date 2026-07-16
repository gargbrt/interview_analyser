"""Optional per-language transcription enhancements, installable/uninstallable
from the Settings tab at any time -- not just during first-run setup.

Currently there's one pack (Hindi/Hinglish romanization, see
transcriber.py's `_to_latin_if_available` and docs/language_support.md),
but this is a small registry so more can be added the same way later
without touching the dashboard code that lists/installs them.
"""
from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

LANGUAGE_PACKS: dict[str, dict[str, str]] = {
    "hindi_hinglish": {
        "label": "Hindi / Hinglish romanization",
        "pip_package": "indic-transliteration",
        "import_name": "indic_transliteration",
        "description": (
            "Romanizes Hindi (Devanagari) transcripts to Latin script for the "
            "\"hinglish\" transcription language setting. See docs/language_support.md."
        ),
    },
}


def is_pack_installed(pack_id: str) -> bool:
    entry = LANGUAGE_PACKS.get(pack_id)
    if entry is None:
        return False
    return importlib.util.find_spec(entry["import_name"]) is not None


def _run_pip(args: list[str], on_output: Optional[Callable[[str], None]] = None) -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "pip", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    for line in process.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        lines.append(line)
        if on_output is not None:
            on_output(line)
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError("\n".join(lines[-15:]) or f"pip exited with code {returncode}")


def install_pack(pack_id: str, on_output: Optional[Callable[[str], None]] = None) -> None:
    entry = LANGUAGE_PACKS.get(pack_id)
    if entry is None:
        raise ValueError(f"Unknown language pack '{pack_id}'.")
    _run_pip(["install", entry["pip_package"]], on_output=on_output)
    importlib.invalidate_caches()


def uninstall_pack(pack_id: str, on_output: Optional[Callable[[str], None]] = None) -> None:
    entry = LANGUAGE_PACKS.get(pack_id)
    if entry is None:
        raise ValueError(f"Unknown language pack '{pack_id}'.")
    _run_pip(["uninstall", "-y", entry["pip_package"]], on_output=on_output)
    importlib.invalidate_caches()


class PackActionDialog:
    """Small progress dialog for installing/uninstalling a language pack via
    pip -- pip doesn't report byte-level progress the way Ollama's pull API
    does, so this shows a scrolling log of pip's own output instead of a
    percentage bar. Same Toplevel-on-shared-root pattern as
    model_setup.py's dialogs (see consent.py's docstring for why)."""

    def __init__(self, pack_id: str, action: str, ui_root: Optional[object] = None, on_done: Optional[Callable[[bool], None]] = None):
        assert action in ("install", "uninstall")
        self.pack_id = pack_id
        self.action = action
        self._on_done = on_done
        self._root = None

        if ui_root is not None:
            try:
                import tkinter as tk

                ui_root.after(0, lambda: self._build(tk.Toplevel(ui_root)))
                return
            except Exception:  # noqa: BLE001
                logger.warning("Shared UI root unavailable for language pack dialog; falling back to a standalone window.")

        threading.Thread(target=self._run_standalone, daemon=True).start()

    def _run_standalone(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        self._build(root)
        root.mainloop()

    def _build(self, window) -> None:
        from tkinter import ttk

        entry = LANGUAGE_PACKS[self.pack_id]
        self._root = window
        verb = "Installing" if self.action == "install" else "Uninstalling"
        window.title(f"Interview Analyzer -- {verb} Language Pack")
        window.attributes("-topmost", True)
        window.resizable(False, False)

        body = ttk.Frame(window, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=f"{verb} {entry['label']} ({entry['pip_package']})...").pack(anchor="w", pady=(0, 8))
        self._bar = ttk.Progressbar(body, mode="indeterminate", length=380)
        self._bar.pack(fill="x", pady=(0, 8))
        self._bar.start(80)
        self._log = tk.Text(body, height=8, width=60, state="disabled", font=("Consolas", 8))
        self._log.pack(fill="both", expand=True)
        self._close_btn = ttk.Button(body, text="Cancel", state="disabled", command=self._root.destroy)
        self._close_btn.pack(anchor="w", pady=(10, 0))

        threading.Thread(target=self._do_action, daemon=True).start()

    def _append_log(self, line: str) -> None:
        if self._root is None:
            return
        self._root.after(0, lambda: self._apply_log(line))

    def _apply_log(self, line: str) -> None:
        if self._log is None:
            return
        self._log.config(state="normal")
        self._log.insert("end", line + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _do_action(self) -> None:
        try:
            if self.action == "install":
                install_pack(self.pack_id, on_output=self._append_log)
            else:
                uninstall_pack(self.pack_id, on_output=self._append_log)
        except Exception as e:  # noqa: BLE001
            logger.exception("Language pack %s failed for %s", self.action, self.pack_id)
            self._finish(f"Failed: {e}", False)
            return
        self._finish(f"{self.action.capitalize()} complete.", True)

    def _finish(self, message: str, success: bool) -> None:
        if self._root is None:
            return
        self._root.after(0, lambda: self._show_result(message, success))

    def _show_result(self, message: str, success: bool) -> None:
        self._bar.stop()
        self._append_log(message)
        self._close_btn.config(text="Close", state="normal")
        if self._on_done is not None:
            self._on_done(success)
