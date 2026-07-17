"""Installing/selecting the local Ollama analysis model.

The app's analysis engine defaults to a free, fully-local model via Ollama
(see engines.py). Pulling that model is a multi-gigabyte download, so this
module makes sure the user always explicitly agrees to it and always sees
the approximate size first -- both the very first time the app runs, and
any later time a Settings change would trigger downloading a *different*
model. Nothing here ever downloads a model without a prior "yes" click.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import psutil
import requests

logger = logging.getLogger(__name__)

# Approx download size in GB for the models this app curates/recommends
# (see config.yaml's "Free-tier model guidance" comment). Sizes are
# ballparked from Ollama's public model library so the user can make an
# informed call before a large download starts, not an exact byte count.
MODEL_CATALOG: dict[str, dict[str, object]] = {
    "llama3.1:8b": {
        "size_gb": 4.7,
        "description": "Fast, runs on most laptops. Default -- recommended for most users.",
    },
    "qwen2.5:14b": {
        "size_gb": 9.0,
        "description": "Noticeably better reasoning for rubric scoring. Needs ~16GB+ RAM.",
    },
    "llama3.2:3b": {
        "size_gb": 2.0,
        "description": "Smallest/fastest option; lower analysis quality. Good for slow machines.",
    },
    "phi3:mini": {
        "size_gb": 2.3,
        "description": "Small and fast with decent quality -- a middle ground.",
    },
}

DEFAULT_MODEL = "llama3.1:8b"


def approx_size_gb(model_name: str) -> Optional[float]:
    entry = MODEL_CATALOG.get(model_name)
    return float(entry["size_gb"]) if entry else None


def size_label(model_name: str) -> str:
    size = approx_size_gb(model_name)
    return f"~{size:.1f} GB" if size is not None else "size unknown until download starts"


def ollama_is_reachable(host: str) -> bool:
    try:
        requests.get(f"{host}/api/tags", timeout=3)
        return True
    except requests.RequestException:
        return False


def _ollama_executable_candidates() -> list[pathlib.Path]:
    """Likely locations of the Ollama server executable, per platform --
    used only to auto-start it if it's not already running (see
    ensure_ollama_running). Ollama itself doesn't register as an OS
    auto-start service on install, on either platform, which is *why*
    "not running" is a real, common situation this needs to handle."""
    if sys.platform == "win32":
        base = pathlib.Path(os.environ.get("LOCALAPPDATA", ""))
        return [base / "Programs" / "Ollama" / "ollama.exe"]
    if sys.platform == "darwin":
        return [
            pathlib.Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            pathlib.Path("/opt/homebrew/bin/ollama"),
            pathlib.Path("/usr/local/bin/ollama"),
        ]
    return []


def ensure_ollama_running(host: str, timeout: float = 20) -> bool:
    """Makes sure Ollama is reachable at `host`, starting it in the
    background first if it isn't. Returns True once reachable (either it
    already was, or this successfully started it), False if it couldn't be
    reached even after trying (e.g. Ollama isn't installed at all).

    Called automatically before every Ollama analysis request (see
    analyzer.py's OllamaEngine) -- this is what lets background processing
    and "Reprocess" just work even if Ollama wasn't already running,
    instead of failing outright with a connection error the user then has
    to notice and fix by hand.
    """
    if ollama_is_reachable(host):
        return True

    exe = shutil.which("ollama") or next(
        (str(p) for p in _ollama_executable_candidates() if p.exists()), None
    )
    if exe is None:
        logger.warning(
            "Ollama isn't reachable at %s, and no Ollama installation was found "
            "to start automatically. Install it from https://ollama.com.", host,
        )
        return False

    try:
        subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        logger.info("Ollama wasn't running; starting it automatically (%s).", exe)
    except Exception:  # noqa: BLE001
        logger.warning("Found Ollama at %s but couldn't start it.", exe, exc_info=True)
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ollama_is_reachable(host):
            return True
        time.sleep(0.5)
    logger.warning("Started Ollama but it didn't become reachable within %ss.", timeout)
    return False


def stop_ollama(host: str, timeout: float = 10) -> bool:
    """Best-effort stop of the local Ollama server process, for the
    dashboard's Status tab "Stop" button. Returns True once `host` is
    unreachable (either it wasn't running to begin with, or this
    successfully stopped it).

    There's no cross-platform "shut down the server" API call -- Ollama's
    own CLI only offers `ollama stop <model>` (unloads a model from memory,
    leaves the server itself running), so this finds and terminates the
    server process directly by name. Note: on Windows, if Ollama's tray app
    is installed and running, it may relaunch the server shortly after this
    kills it -- that's outside this app's control.
    """
    found_any = False
    for proc in psutil.process_iter(["name"]):
        name = (proc.info.get("name") or "").lower()
        if name in ("ollama.exe", "ollama", "ollama_llama_server.exe", "ollama_llama_server"):
            found_any = True
            try:
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    if not found_any:
        return not ollama_is_reachable(host)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not ollama_is_reachable(host):
            return True
        time.sleep(0.5)
    return not ollama_is_reachable(host)


def list_installed_models(host: str) -> list[str]:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", [])]
    except requests.RequestException:
        return []


def is_model_installed(model_name: str, host: str) -> bool:
    installed = list_installed_models(host)
    # Ollama's library sometimes reports a pulled tag with an explicit
    # ":latest" suffix even when it was requested bare -- compare loosely.
    return model_name in installed or f"{model_name}:latest" in installed


def pull_model(
    model_name: str,
    host: str,
    on_progress: Optional[Callable[[Optional[float], str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Downloads `model_name` via Ollama's streaming pull API.

    `on_progress(fraction, status_text)` is called as Ollama reports real
    completed/total byte counts for the layer currently downloading --
    `fraction` is None for status lines that don't carry byte counts (e.g.
    "verifying sha256 digest"), so callers should fall back to an
    indeterminate indicator in that case rather than treating it as 0%.

    Raises on failure (network error, Ollama not running, unknown model
    name) -- callers should catch and show the user a clear message.
    Raises InterruptedError if `cancel_event` is set mid-download.
    """
    resp = requests.post(
        f"{host}/api/pull",
        json={"name": model_name, "stream": True},
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    try:
        for line in resp.iter_lines():
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Model download cancelled.")
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("error"):
                raise RuntimeError(data["error"])
            status = data.get("status", "")
            total = data.get("total")
            completed = data.get("completed")
            if on_progress is not None:
                if total and completed is not None and total > 0:
                    on_progress(min(completed / total, 1.0), status)
                else:
                    on_progress(None, status)
    finally:
        resp.close()


def _setup_marker_path(cfg) -> pathlib.Path:
    db_path = cfg.resolve(cfg.storage.get("db_path", "data/interviews.db"))
    return db_path.with_name(".model_setup_complete")


def setup_already_done(cfg) -> bool:
    return _setup_marker_path(cfg).exists()


def mark_setup_done(cfg) -> None:
    path = _setup_marker_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class FirstRunSetupWizard:
    """One-time (per install) dialog offered before the watcher starts:
    install the default local analysis model now, pick a different one from
    the curated catalog, or skip and use a cloud API instead. Blocks the
    caller (via `.wait()`) until the user decides, same pattern as
    control_panel.py's readiness handshake.

    Built as a Toplevel on the shared dashboard Tk root when available (see
    consent.py's module docstring for why -- concurrent Tk() interpreters on
    different threads can hard-crash the process), else as a standalone
    root, same fallback every other popup in this app uses.
    """

    def __init__(self, cfg, ui_root: Optional[object] = None):
        self.cfg = cfg
        self._acfg = cfg.analysis
        self._host = self._acfg.get("ollama_host", "http://localhost:11434")
        self._done = threading.Event()
        self._root = None
        self._body = None
        self._model_var = None
        self._cancel_event = threading.Event()

        if ui_root is not None and self._try_build_on_shared_root(ui_root):
            return
        self._thread = threading.Thread(target=self._run_standalone, daemon=True)
        self._thread.start()

    def _try_build_on_shared_root(self, ui_root) -> bool:
        try:
            import tkinter as tk

            ui_root.after(0, lambda: self._build(tk.Toplevel(ui_root)))
        except Exception:  # noqa: BLE001
            logger.warning("Shared UI root unavailable for setup wizard; falling back to a standalone window.")
            return False
        return True

    def _run_standalone(self) -> None:
        try:
            import tkinter as tk
        except ImportError:  # pragma: no cover
            logger.warning("Tkinter not available; skipping the analysis model setup wizard.")
            self._done.set()
            return

        root = tk.Tk()
        self._build(root)
        root.mainloop()

    def wait(self, timeout: Optional[float] = None) -> None:
        self._done.wait(timeout=timeout)

    # -- choice screen ----------------------------------------------------

    def _build(self, window) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._root = window
        window.title("Interview Analyzer -- Analysis Model Setup")
        window.attributes("-topmost", True)
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", self._on_remind_later)

        self._body = ttk.Frame(window, padding=18)
        self._body.pack(fill="both", expand=True)
        self._show_choice_screen()

    def _clear_body(self) -> None:
        for child in self._body.winfo_children():
            child.destroy()

    def _show_choice_screen(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._clear_body()
        body = self._body

        ttk.Label(
            body,
            text=(
                "Interview Analyzer needs an AI model to analyze your interviews.\n\n"
                "By default it uses a free, local model via Ollama -- your transcripts\n"
                "never leave your computer. This is a one-time download."
            ),
            justify="left",
            wraplength=420,
        ).pack(anchor="w", pady=(0, 12))

        display_to_model = {}
        values = []
        for name, entry in MODEL_CATALOG.items():
            display = f"{name}  ({size_label(name)}) -- {entry['description']}"
            display_to_model[display] = name
            values.append(display)
        self._display_to_model = display_to_model

        ttk.Label(body, text="Model to install:").pack(anchor="w")
        combo = ttk.Combobox(body, values=values, state="readonly", width=60)
        default_model = self._acfg.get("llm_model", DEFAULT_MODEL)
        default_display = next((d for d, m in display_to_model.items() if m == default_model), values[0])
        combo.set(default_display)
        combo.pack(anchor="w", pady=(2, 14), fill="x")
        self._combo = combo

        btn_frame = ttk.Frame(body)
        btn_frame.pack(fill="x")
        ttk.Button(btn_frame, text="Install locally", command=self._on_install_clicked).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(
            btn_frame, text="Skip -- I'll use a cloud API", command=self._on_skip_clicked
        ).pack(side="left", padx=(0, 8))
        ttk.Button(btn_frame, text="Remind me later", command=self._on_remind_later).pack(side="left")

        ttk.Label(
            body,
            text=(
                "Skipping leaves analysis unset until you configure a cloud API key\n"
                "(anthropic_api or openai_api) in the Settings tab -- see docs/using_cloud_apis.md."
            ),
            justify="left",
            wraplength=420,
            foreground="#6b6b6b",
        ).pack(anchor="w", pady=(14, 0))

    def _on_skip_clicked(self) -> None:
        mark_setup_done(self.cfg)
        self._close()

    def _on_remind_later(self) -> None:
        # deliberately does NOT mark setup done -- ask again next launch
        # rather than silently leaving analysis unconfigured forever
        self._close()

    def _on_install_clicked(self) -> None:
        display = self._combo.get()
        model_name = self._display_to_model.get(display, DEFAULT_MODEL)
        self._show_progress_screen(model_name)
        thread = threading.Thread(target=self._do_install, args=(model_name,), daemon=True)
        thread.start()

    # -- progress screen ----------------------------------------------------

    def _show_progress_screen(self, model_name: str) -> None:
        from tkinter import ttk

        self._clear_body()
        body = self._body

        ttk.Label(body, text=f"Downloading {model_name} ({size_label(model_name)})...").pack(
            anchor="w", pady=(0, 10)
        )
        self._progress_bar = ttk.Progressbar(body, mode="indeterminate", length=380, maximum=100)
        self._progress_bar.pack(fill="x", pady=(0, 8))
        self._progress_bar.start(80)
        self._progress_running = True
        self._status_label = ttk.Label(body, text="Starting download...", foreground="#2f6fa8")
        self._status_label.pack(anchor="w")
        self._close_btn = ttk.Button(body, text="Cancel", command=self._on_cancel_install)
        self._close_btn.pack(anchor="w", pady=(14, 0))

    def _on_cancel_install(self) -> None:
        self._cancel_event.set()

    def _do_install(self, model_name: str) -> None:
        try:
            if not ollama_is_reachable(self._host):
                self._on_install_error(
                    "Ollama isn't running or isn't installed. Install it from "
                    "https://ollama.com, start it, then reopen this from the Settings tab."
                )
                return
            pull_model(model_name, self._host, on_progress=self._on_pull_progress, cancel_event=self._cancel_event)
        except InterruptedError:
            self._on_install_error("Download cancelled.")
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("Model download failed")
            self._on_install_error(f"Download failed: {e}")
            return

        if model_name != self._acfg.get("llm_model"):
            try:
                from .settings_editor import save_editable_settings

                save_editable_settings(self.cfg.path, {"analysis.llm_model": model_name, "analysis.engine": "ollama"})
            except Exception:  # noqa: BLE001
                logger.exception("Failed to persist chosen model to config.yaml")

        mark_setup_done(self.cfg)
        self._on_install_success(model_name)

    def _on_pull_progress(self, fraction: Optional[float], status: str) -> None:
        if self._root is None:
            return
        self._root.after(0, lambda: self._apply_progress(fraction, status))

    def _apply_progress(self, fraction: Optional[float], status: str) -> None:
        if self._status_label is None:
            return
        self._status_label.config(text=status or "Downloading...")
        if fraction is not None:
            if self._progress_running:
                self._progress_bar.stop()
                self._progress_running = False
            self._progress_bar.config(mode="determinate")
            self._progress_bar["value"] = fraction * 100
        elif not self._progress_running:
            self._progress_bar.config(mode="indeterminate")
            self._progress_bar.start(80)
            self._progress_running = True

    def _on_install_success(self, model_name: str) -> None:
        if self._root is None:
            return
        self._root.after(0, lambda: self._show_result(f"{model_name} installed and ready.", success=True))

    def _on_install_error(self, message: str) -> None:
        if self._root is None:
            return
        self._root.after(0, lambda: self._show_result(message, success=False))

    def _show_result(self, message: str, success: bool) -> None:
        from tkinter import ttk

        if self._progress_running:
            self._progress_bar.stop()
        self._progress_bar.config(mode="determinate")
        self._progress_bar["value"] = 100 if success else self._progress_bar["value"]
        self._status_label.config(text=message, foreground="#2f6f5e" if success else "#a8722a")
        self._close_btn.config(text="Close", command=self._close)

    def _close(self) -> None:
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._root = None
        self._done.set()


class ModelInstallDialog:
    """Standalone download-progress dialog for installing/updating a model
    on demand -- used by the Settings tab's "Install model" button, once the
    user has already confirmed the download via a Yes/No prompt showing its
    approximate size (see Dashboard._on_install_model). Unlike
    FirstRunSetupWizard, this never writes to config.yaml itself -- the
    Settings tab's own Save button is what persists engine/model choices.
    """

    def __init__(self, model_name: str, host: str, ui_root: Optional[object] = None):
        self.model_name = model_name
        self.host = host
        self._root = None
        self._progress_running = False
        self._cancel_event = threading.Event()

        if ui_root is not None:
            try:
                import tkinter as tk

                ui_root.after(0, lambda: self._build(tk.Toplevel(ui_root)))
                return
            except Exception:  # noqa: BLE001
                logger.warning("Shared UI root unavailable for model install dialog; falling back to a standalone window.")

        threading.Thread(target=self._run_standalone, daemon=True).start()

    def _run_standalone(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        self._build(root)
        root.mainloop()

    def _build(self, window) -> None:
        from tkinter import ttk

        self._root = window
        window.title("Interview Analyzer -- Installing Model")
        window.attributes("-topmost", True)
        window.resizable(False, False)

        body = ttk.Frame(window, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=f"Downloading {self.model_name} ({size_label(self.model_name)})...").pack(
            anchor="w", pady=(0, 10)
        )
        self._progress_bar = ttk.Progressbar(body, mode="indeterminate", length=380, maximum=100)
        self._progress_bar.pack(fill="x", pady=(0, 8))
        self._progress_bar.start(80)
        self._progress_running = True
        self._status_label = ttk.Label(body, text="Starting download...", foreground="#2f6fa8")
        self._status_label.pack(anchor="w")
        self._close_btn = ttk.Button(body, text="Cancel", command=self._cancel_event.set)
        self._close_btn.pack(anchor="w", pady=(14, 0))
        window.protocol("WM_DELETE_WINDOW", self._cancel_event.set)

        threading.Thread(target=self._do_install, daemon=True).start()

    def _do_install(self) -> None:
        try:
            if not ollama_is_reachable(self.host):
                self._finish("Ollama isn't running or isn't installed. Install it from https://ollama.com first.", False)
                return
            pull_model(self.model_name, self.host, on_progress=self._on_progress, cancel_event=self._cancel_event)
        except InterruptedError:
            self._finish("Download cancelled.", False)
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("Model download failed")
            self._finish(f"Download failed: {e}", False)
            return
        self._finish(f"{self.model_name} installed and ready.", True)

    def _on_progress(self, fraction: Optional[float], status: str) -> None:
        if self._root is None:
            return
        self._root.after(0, lambda: self._apply_progress(fraction, status))

    def _apply_progress(self, fraction: Optional[float], status: str) -> None:
        if self._status_label is None:
            return
        self._status_label.config(text=status or "Downloading...")
        if fraction is not None:
            if self._progress_running:
                self._progress_bar.stop()
                self._progress_running = False
            self._progress_bar.config(mode="determinate")
            self._progress_bar["value"] = fraction * 100
        elif not self._progress_running:
            self._progress_bar.config(mode="indeterminate")
            self._progress_bar.start(80)
            self._progress_running = True

    def _finish(self, message: str, success: bool) -> None:
        if self._root is None:
            return
        self._root.after(0, lambda: self._show_result(message, success))

    def _show_result(self, message: str, success: bool) -> None:
        if self._progress_running:
            self._progress_bar.stop()
        self._progress_bar.config(mode="determinate")
        if success:
            self._progress_bar["value"] = 100
        self._status_label.config(text=message, foreground="#2f6f5e" if success else "#a8722a")
        self._close_btn.config(text="Close", command=self._root.destroy)


def maybe_run_first_time_setup(cfg, ui_root: Optional[object] = None, timeout: float = 300) -> None:
    """Entry point called once from app.py before the watcher starts. A
    no-op if setup was already completed (or explicitly skipped) before, or
    if the configured engine isn't "ollama" (cloud engines need an API key
    the user supplies themselves, not a download this app can offer)."""
    if cfg.analysis.get("engine", "ollama") != "ollama":
        mark_setup_done(cfg)
        return
    if setup_already_done(cfg):
        return

    model_name = cfg.analysis.get("llm_model", DEFAULT_MODEL)
    host = cfg.analysis.get("ollama_host", "http://localhost:11434")
    if ollama_is_reachable(host) and is_model_installed(model_name, host):
        # already set up outside this app (e.g. `ollama pull` run manually)
        mark_setup_done(cfg)
        return

    wizard = FirstRunSetupWizard(cfg, ui_root=ui_root)
    wizard.wait(timeout=timeout)
