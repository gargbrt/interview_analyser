"""Watches running processes / browser window titles for known conferencing
apps (Teams, Meet, Webex, Zoom, Chime — desktop app or browser tab) and
drives the record -> transcribe -> analyze -> report pipeline automatically,
with no manual step per interview.
"""
from __future__ import annotations

import datetime as dt
import logging
import pathlib
import threading
import time
from typing import Any, Callable, Optional

import psutil

from .analyzer import analyze_transcript
from .cleanup import run_cleanup
from .compress import compress_audio
from .config_loader import Config, load_config
from .consent import ask_consent
from .control_panel import RecordingControlPanel
from .db import InterviewDB
from .recorder import SystemAudioRecorder
from .report import write_interview_report, write_trends_report
from .transcriber import transcribe

logger = logging.getLogger(__name__)

try:
    import win32gui  # type: ignore
except ImportError:  # pragma: no cover
    win32gui = None


def _enumerate_window_titles() -> list[str]:
    if win32gui is None:
        return []
    titles: list[str] = []

    def _cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)

    win32gui.EnumWindows(_cb, None)
    return titles


def detect_active_meeting(cfg: Config) -> Optional[str]:
    """Return a human-readable app name if a meeting appears active, else None."""
    watched = cfg.watched_processes
    running = {p.name() for p in psutil.process_iter(["name"])}

    for proc_name in watched.get("desktop_apps", []):
        if proc_name in running:
            return proc_name.replace(".exe", "")

    browser_running = any(b in running for b in watched.get("browser_processes", []))
    if browser_running:
        titles = _enumerate_window_titles()
        for title in titles:
            for keyword in watched.get("browser_tab_keywords", []):
                if keyword.lower() in title.lower():
                    return keyword.strip(" -")
    return None


class MeetingWatcher:
    def __init__(
        self,
        cfg: Config,
        user_id: Optional[int] = None,
        on_state_change: Optional[Callable[[], None]] = None,
    ):
        self.cfg = cfg
        self.user_id = user_id
        self.db = InterviewDB(cfg.resolve(cfg.storage.get("db_path", "data/interviews.db")))
        self._recorder: Optional[SystemAudioRecorder] = None
        self._control_panel: Optional[RecordingControlPanel] = None
        self._current_interview_id: Optional[int] = None
        self._current_app_name: Optional[str] = None
        self._present_polls = 0
        self._absent_polls = 0
        self._audio_dir = cfg.resolve(cfg.audio.get("raw_dir", "data/audio"))
        # tracks meetings the user already said "No" to, so we don't re-prompt
        # every poll interval for the same still-running call
        self._declined_this_session: set[str] = set()
        # set by the control panel's Stop button (runs on its own Tk thread);
        # consumed by _tick() on the watcher's own thread on the next poll
        self._manual_stop_requested = threading.Event()
        self._shutdown_requested = threading.Event()
        # called (from whichever thread triggered the change) after every
        # idle/recording/paused transition -- the tray icon and dashboard
        # use this to refresh instead of polling internal state
        self._on_state_change = on_state_change

    def set_on_state_change(self, callback: Optional[Callable[[], None]]) -> None:
        """Register (or clear, with None) the callback invoked after every
        idle/recording/paused transition. Used by app.py to wire the tray
        icon and dashboard up once they exist (they can't be constructed
        before the watcher, so this can't just be a constructor arg)."""
        self._on_state_change = callback

    def _notify(self) -> None:
        if self._on_state_change is not None:
            try:
                self._on_state_change()
            except Exception:  # noqa: BLE001
                logger.exception("on_state_change callback failed")

    @property
    def status(self) -> dict[str, Any]:
        """A snapshot of current watcher state, safe to read from any thread
        (e.g. the tray icon or dashboard, which never mutate it)."""
        if self._current_interview_id is None:
            return {"state": "idle"}
        paused = self._recorder is not None and self._recorder.is_paused
        return {
            "state": "paused" if paused else "recording",
            "app_name": self._current_app_name,
            "interview_id": self._current_interview_id,
        }

    def pause_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.pause()
            self._notify()

    def resume_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.resume()
            self._notify()

    def request_stop_recording(self) -> None:
        """Ask the watcher to end the in-progress recording (and run the
        transcribe/analyze/report pipeline) on its next poll tick. Safe to
        call from any thread -- the control panel, the tray icon, and the
        dashboard's Stop button all funnel through here."""
        self._manual_stop_requested.set()

    def shutdown(self) -> None:
        """Stop `run_forever()`'s loop after the current tick. Used by the
        tray icon's Quit action."""
        self._shutdown_requested.set()

    def run_forever(self) -> None:
        logger.info("Watcher started. Waiting for a meeting to begin...")
        while not self._shutdown_requested.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                logger.exception("Watcher tick failed; continuing")
            run_cleanup(self.db)
            time.sleep(self.cfg.poll_interval_seconds)

    def _tick(self) -> None:
        if self._manual_stop_requested.is_set():
            self._manual_stop_requested.clear()
            if self._current_interview_id is not None:
                logger.info("Recording stopped manually via control panel.")
                self._stop_and_process()
            return

        app_name = detect_active_meeting(self.cfg)

        if app_name and self._current_interview_id is None:
            if app_name in self._declined_this_session:
                return  # already asked and declined for this ongoing call
            self._present_polls += 1
            if self._present_polls >= self.cfg.start_debounce_polls:
                self._present_polls = 0
                if ask_consent(app_name, timeout_seconds=20):
                    self._start_recording(app_name)
                else:
                    logger.info("Recording declined for %s; will not prompt again until "
                                "this call ends.", app_name)
                    self._declined_this_session.add(app_name)
        elif not app_name:
            self._declined_this_session.clear()
            if self._current_interview_id is not None:
                self._absent_polls += 1
                if self._absent_polls >= self.cfg.start_debounce_polls:
                    self._stop_and_process()
        else:
            self._present_polls = 0
            self._absent_polls = 0

    def _start_recording(self, app_name: str) -> None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = self._audio_dir / f"{timestamp}_{app_name}.wav"

        self._recorder = SystemAudioRecorder(
            sample_rate=self.cfg.audio.get("sample_rate", 16000),
            channels=self.cfg.audio.get("channels", 1),
        )
        self._recorder.start(wav_path)
        self._current_app_name = app_name
        self._current_interview_id = self.db.start_interview(
            source_app=app_name,
            audio_path=str(wav_path),
            retention_days=self.cfg.retention_days,
            user_id=self.user_id,
        )
        self._control_panel = RecordingControlPanel(
            app_name,
            on_pause=self.pause_recording,
            on_resume=self.resume_recording,
            on_stop=self.request_stop_recording,
        )
        logger.info("Consent given for %s. Recording started (interview #%s).",
                     app_name, self._current_interview_id)
        self._notify()

    def _stop_and_process(self) -> None:
        self._absent_polls = 0
        interview_id = self._current_interview_id
        self._current_interview_id = None
        self._current_app_name = None
        self._notify()

        if self._control_panel is not None:
            self._control_panel.close()
            self._control_panel = None

        wav_path = self._recorder.stop()
        self.db.end_interview(interview_id)
        logger.info("Meeting ended. Processing interview #%s...", interview_id)

        audio_path = compress_audio(
            wav_path,
            bitrate_kbps=self.cfg.audio.get("bitrate_kbps", 64),
            fmt=self.cfg.audio.get("format", "opus"),
        )
        # keep DB pointing at the (now-compressed) file so cleanup can find it
        self.db.update_audio_path(interview_id, str(audio_path))

        transcript = transcribe(audio_path, self.cfg)
        self.db.save_transcript(interview_id, transcript)

        analysis = analyze_transcript(transcript, self.cfg)
        self.db.save_analysis(interview_id, analysis)

        record = self.db.get(interview_id)
        report_path = write_interview_report(record, self.cfg)
        self.db.save_report_path(interview_id, str(report_path))

        write_trends_report(self.db.list_all(user_id=self.user_id), self.cfg)

        logger.info("Interview #%s processed. Report: %s", interview_id, report_path)
        self._notify()  # lets an open dashboard refresh its history/trends tabs


def main() -> None:
    import argparse

    from .auth import login_or_create

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Interview Analyzer watcher")
    parser.add_argument("--username", help="Skip the login prompt with this profile name")
    args = parser.parse_args()

    cfg = load_config()
    db = InterviewDB(cfg.resolve(cfg.storage.get("db_path", "data/interviews.db")))
    user = login_or_create(db._conn, non_interactive_username=args.username)
    logger.info("Logged in as '%s' (profile #%s).", user.username, user.id)
    db.close()

    watcher = MeetingWatcher(cfg, user_id=user.id)
    watcher.run_forever()


if __name__ == "__main__":
    main()
