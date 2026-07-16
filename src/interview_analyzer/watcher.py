"""Watches running processes / browser window titles for known conferencing
apps (Teams, Meet, Webex, Zoom, Chime — desktop app or browser tab) and
drives the record -> transcribe -> analyze -> report pipeline automatically,
with no manual step per interview.

Recording is single-threaded (one active call at a time, since there's one
system-audio stream to capture) but transcribe/analyze/report runs in its
own background thread per interview, so the watcher keeps polling -- and
can start recording a *new* call -- while a previous one is still being
processed. Multiple interviews can be mid-processing concurrently; each is
tracked independently in `_processing_jobs`.
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
from .confidence import calibrated_confidence, calibration_notes as build_calibration_notes
from .consent import ask_consent
from .control_panel import RecordingControlPanel
from .db import InterviewDB
from .recorder import SystemAudioRecorder
from .report import write_interview_report, write_trends_report
from .transcriber import TranscriptionCancelled, transcribe

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


def detect_active_meeting(cfg: Config) -> Optional[tuple[str, bool]]:
    """Returns (app_name, is_desktop_app) if a meeting appears active, else
    None.

    `is_desktop_app` distinguishes a native app process match (Zoom.exe
    etc.) from a browser tab title match. The former is a strong signal --
    that process generally only runs while the app is actually open. The
    latter is much weaker: e.g. a Google Meet tab sitting on the pre-join
    lobby/camera-check screen matches the exact same title pattern as an
    actual call, so a browser tab alone isn't reliable evidence you're
    actually *in* a call. Callers use this to require a longer sustained
    presence before treating a browser tab as a real meeting.
    """
    watched = cfg.watched_processes
    running = {p.name() for p in psutil.process_iter(["name"])}

    for proc_name in watched.get("desktop_apps", []):
        if proc_name in running:
            return proc_name.replace(".exe", ""), True

    browser_running = any(b in running for b in watched.get("browser_processes", []))
    if browser_running:
        titles = _enumerate_window_titles()
        for title in titles:
            for keyword in watched.get("browser_tab_keywords", []):
                if keyword.lower() in title.lower():
                    # strips both a plain hyphen and an en dash (U+2013,
                    # what Google Meet's real tab title actually uses) so
                    # "Meet – " and "Meet - " both clean up to "Meet"
                    return keyword.strip(" -–"), False
    return None


class MeetingWatcher:
    def __init__(
        self,
        cfg: Config,
        user_id: Optional[int] = None,
        on_state_change: Optional[Callable[[], None]] = None,
        ui_root: Optional[object] = None,
    ):
        self.cfg = cfg
        self.user_id = user_id
        self.db = InterviewDB(cfg.resolve(cfg.storage.get("db_path", "data/interviews.db")))
        # the app's shared dashboard Tk root, if there is one -- passed through
        # to ask_consent()/RecordingControlPanel() so their popups build as
        # Toplevels on that one interpreter instead of spinning up their own
        # Tk() on a new thread (see consent.py's docstring for why running
        # multiple Tk() interpreters on different threads is unsafe). None in
        # headless mode (watcher.py's own main()), where popups run standalone.
        self._ui_root = ui_root
        self._recorder: Optional[SystemAudioRecorder] = None
        self._control_panel: Optional[RecordingControlPanel] = None
        self._current_interview_id: Optional[int] = None
        self._current_app_name: Optional[str] = None
        self._present_polls = 0
        self._absent_polls = 0
        self._audio_dir = cfg.resolve(cfg.audio.get("raw_dir", "data/audio"))
        # app_name -> timestamp of the last "No" answer, so we don't re-prompt
        # every poll interval for the same still-running call. Expires after
        # declined_cooldown_seconds (see _is_declined_and_not_expired)
        # rather than staying declined forever, since browser-tab detection
        # can keep matching well past when the user actually left the call
        # (see detect_active_meeting's is_desktop_app docstring) -- without
        # an expiry, a stuck-positive detection would mean never being able
        # to record again without restarting the app.
        self._declined_this_session: dict[str, float] = {}
        # tracks the app+timestamp of the interview that just ended, so a
        # stale/lingering window-title match (e.g. a Meet tab still open on
        # the post-call screen) doesn't immediately re-trigger a consent
        # prompt for a call that's actually already over
        self._last_ended_app: Optional[str] = None
        self._last_ended_at: float = 0.0
        # interview_id -> {"stage": str, "progress": Optional[float],
        # "source_app": Optional[str]} for every transcribe/analyze/report
        # job currently running in the background. Multiple entries can
        # coexist (a new recording can start while an earlier interview is
        # still processing); guarded by _processing_lock since jobs run on
        # their own threads.
        self._processing_jobs: dict[int, dict[str, Any]] = {}
        self._cancel_events: dict[int, threading.Event] = {}
        self._processing_lock = threading.Lock()
        # set by the control panel's Stop button (runs on its own Tk thread);
        # consumed by _tick() on the watcher's own thread on the next poll
        self._manual_stop_requested = threading.Event()
        # app_name if a manual "Start recording" was requested (e.g. from
        # the dashboard or tray, when automatic detection missed a real
        # call); consumed by _tick() on the watcher's own thread, same
        # reasoning as _manual_stop_requested above
        self._manual_start_requested: Optional[str] = None
        self._shutdown_requested = threading.Event()
        # called (from whichever thread triggered the change) after every
        # recording/processing-job transition -- the tray icon and
        # dashboard use this to refresh instead of polling internal state
        self._on_state_change = on_state_change

    def set_ui_root(self, ui_root: Optional[object]) -> None:
        """Register (or clear, with None) the shared dashboard Tk root that
        consent/control-panel popups should build on. Called by Dashboard
        itself whenever it opens/closes, so this always reflects the
        *current* dashboard window rather than a possibly-destroyed one
        from an earlier open/close cycle."""
        self._ui_root = ui_root

    @property
    def ui_root(self) -> Optional[object]:
        """The shared dashboard Tk root, if a dashboard is currently open --
        used by app.py to build the first-run model setup wizard on the same
        root as every other popup (see consent.py's docstring for why)."""
        return self._ui_root

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
        (e.g. the tray icon or dashboard, which never mutate it).

        `processing_jobs` is always present (a dict, possibly empty) and is
        independent of `state`/`app_name`/etc: recording and background
        processing now happen concurrently, so e.g. state="recording" with
        a non-empty processing_jobs means "recording a new call while an
        earlier one is still being transcribed/analyzed" -- both true at once.
        """
        with self._processing_lock:
            processing_jobs = {iid: dict(job) for iid, job in self._processing_jobs.items()}

        if self._current_interview_id is None:
            base: dict[str, Any] = {"state": "idle"}
        else:
            paused = self._recorder is not None and self._recorder.is_paused
            elapsed = self._recorder.elapsed_seconds if self._recorder is not None else 0.0
            base = {
                "state": "paused" if paused else "recording",
                "app_name": self._current_app_name,
                "interview_id": self._current_interview_id,
                "elapsed_seconds": elapsed,
            }
        base["processing_jobs"] = processing_jobs
        return base

    def request_start_recording(self, app_name: str) -> None:
        """Ask the watcher to start recording `app_name` immediately, on
        its next poll tick, without waiting for automatic detection and
        without asking for consent again (calling this IS the consent) --
        the manual fallback for when automatic detection doesn't pick up
        a real call. Safe to call from any thread, including the
        dashboard's own Tk thread: actually starting the recording (which
        builds the pause/resume/stop control panel) must happen on the
        watcher's own thread, or building that panel as a Toplevel on the
        dashboard's root while blocked *inside* a dashboard button-click
        handler would deadlock (see consent.py's docstring for why Tk
        cross-thread calls need care)."""
        if self._current_interview_id is not None:
            raise RuntimeError("Already recording.")
        self._manual_start_requested = app_name.strip() or "Manual"

    def pause_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.pause()
            self._notify()

    def resume_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.resume()
            self._notify()

    def request_stop_recording(self) -> None:
        """Ask the watcher to end the in-progress recording (and hand it
        off for background processing) on its next poll tick. Safe to call
        from any thread -- the control panel, the tray icon, and the
        dashboard's Stop button all funnel through here."""
        self._manual_stop_requested.set()

    def cancel_processing(self, interview_id: int) -> bool:
        """Cancel an in-progress (or queued) transcribe/analyze/report job
        for `interview_id`. Safe to call from any thread. Returns True if a
        running job was found and signaled, False if there was nothing to
        cancel (already finished, or never started). Whatever was
        transcribed so far is discarded -- faster-whisper's batch API has
        no meaningful "resume from here"."""
        with self._processing_lock:
            event = self._cancel_events.get(interview_id)
        if event is None:
            return False
        event.set()
        return True

    def shutdown(self) -> None:
        """Stop `run_forever()`'s loop after the current tick. Used by the
        tray icon's Quit action. Background processing jobs already running
        are not interrupted by this -- only new polling stops."""
        self._shutdown_requested.set()

    def run_forever(self) -> None:
        logger.info("Watcher started. Waiting for a meeting to begin...")
        while not self._shutdown_requested.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                logger.exception("Watcher tick failed; continuing")
            try:
                run_cleanup(self.db)
            except Exception:  # noqa: BLE001
                logger.exception("Retention cleanup failed; continuing")
            time.sleep(self.cfg.poll_interval_seconds)

    def _is_declined_and_not_expired(self, app_name: str) -> bool:
        declined_at = self._declined_this_session.get(app_name)
        if declined_at is None:
            return False
        if time.time() - declined_at >= self.cfg.declined_cooldown_seconds:
            del self._declined_this_session[app_name]
            return False
        return True

    def _tick(self) -> None:
        if self._manual_start_requested is not None:
            app_name = self._manual_start_requested
            self._manual_start_requested = None
            if self._current_interview_id is None:
                self._declined_this_session.pop(app_name, None)
                logger.info("Recording started manually for %s.", app_name)
                self._start_recording(app_name)
            return

        if self._manual_stop_requested.is_set():
            self._manual_stop_requested.clear()
            if self._current_interview_id is not None:
                logger.info("Recording stopped manually via control panel.")
                self._stop_and_process()
            return

        detection = detect_active_meeting(self.cfg)
        app_name, is_desktop_app = detection if detection else (None, False)

        if app_name and self._current_interview_id is None:
            if self._is_declined_and_not_expired(app_name):
                return  # already asked and declined for this ongoing call
            if (
                app_name == self._last_ended_app
                and time.time() - self._last_ended_at < self.cfg.post_call_cooldown_seconds
            ):
                # a just-ended call's window/tab can linger for a bit (e.g. a
                # Meet tab still open on the post-call screen) and keep
                # matching the same detection keywords -- don't immediately
                # re-prompt for what's actually already over
                return
            self._present_polls += 1
            # a browser tab match is a much weaker signal than a real app
            # process (see detect_active_meeting) -- e.g. sitting on Meet's
            # pre-join lobby screen matches identically to an actual call --
            # so require it to hold for longer before treating it as real
            required_polls = self.cfg.start_debounce_polls if is_desktop_app else self.cfg.browser_start_debounce_polls
            if self._present_polls >= required_polls:
                self._present_polls = 0
                if ask_consent(app_name, timeout_seconds=20, ui_root=self._ui_root):
                    self._start_recording(app_name)
                else:
                    logger.info("Recording declined for %s; won't prompt again for %ss.",
                                app_name, self.cfg.declined_cooldown_seconds)
                    self._declined_this_session[app_name] = time.time()
        elif not app_name:
            self._declined_this_session.clear()
            if self._current_interview_id is not None:
                self._absent_polls += 1
                if self._absent_polls >= self.cfg.stop_debounce_polls:
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
            include_microphone=self.cfg.audio.get("include_microphone", True),
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
            ui_root=self._ui_root,
        )
        logger.info("Consent given for %s. Recording started (interview #%s).",
                     app_name, self._current_interview_id)
        self._notify()

    def _stop_and_process(self) -> None:
        """Ends the current recording and hands it off to a background
        thread for transcribe/analyze/report -- this method itself returns
        quickly, so `run_forever()`'s polling loop (and a brand new
        recording) isn't blocked for however long processing takes."""
        self._absent_polls = 0
        interview_id = self._current_interview_id
        app_name = self._current_app_name
        self._current_interview_id = None
        # start the post-call cooldown (see _tick) from the moment recording
        # actually stops, not from whenever processing eventually finishes
        self._last_ended_app = app_name
        self._last_ended_at = time.time()
        self._current_app_name = None
        self._notify()

        if self._control_panel is not None:
            self._control_panel.close()
            self._control_panel = None

        wav_path = self._recorder.stop()
        self.db.end_interview(interview_id)
        logger.info("Meeting ended. Processing interview #%s in the background...", interview_id)

        thread = threading.Thread(
            target=self._process_in_background,
            args=(interview_id, wav_path, app_name),
            daemon=True,
        )
        thread.start()

    def _process_in_background(
        self, interview_id: int, wav_path: pathlib.Path, source_app: Optional[str]
    ) -> None:
        """Runs compress -> transcribe -> analyze -> report for one
        interview on its own thread. Exceptions are logged, not raised
        further (there's no caller waiting on this thread)."""
        cancel_event = threading.Event()
        with self._processing_lock:
            self._cancel_events[interview_id] = cancel_event
        self._set_job(interview_id, "compressing", source_app=source_app)
        try:
            audio_path = compress_audio(
                wav_path,
                bitrate_kbps=self.cfg.audio.get("bitrate_kbps", 64),
                fmt=self.cfg.audio.get("format", "opus"),
            )
            # keep DB pointing at the (now-compressed) file so cleanup can find it
            self.db.update_audio_path(interview_id, str(audio_path))

            self._run_analysis_pipeline(interview_id, audio_path, cancel_event, source_app=source_app)
        except TranscriptionCancelled:
            logger.info("Processing for interview #%s was cancelled.", interview_id)
        except Exception:  # noqa: BLE001
            logger.exception("Background processing for interview #%s failed", interview_id)
        finally:
            self._clear_job(interview_id)

    def _set_job(
        self,
        interview_id: int,
        stage: str,
        progress: Optional[float] = None,
        source_app: Optional[str] = None,
    ) -> None:
        with self._processing_lock:
            job = self._processing_jobs.setdefault(interview_id, {})
            job["stage"] = stage
            job["progress"] = progress
            if source_app is not None:
                job["source_app"] = source_app
        self._notify()

    def _clear_job(self, interview_id: int) -> None:
        with self._processing_lock:
            self._processing_jobs.pop(interview_id, None)
            self._cancel_events.pop(interview_id, None)
        self._notify()  # lets an open dashboard refresh its history/trends tabs

    def _run_analysis_pipeline(
        self,
        interview_id: int,
        audio_path: pathlib.Path,
        cancel_event: threading.Event,
        source_app: Optional[str] = None,
    ) -> None:
        """transcribe -> analyze -> write report for audio that's already
        been recorded (and compressed, if applicable). Shared by
        `_process_in_background` and `reprocess_interview`; callers own
        registering/clearing this job in `_processing_jobs` (via
        `_set_job`/`_clear_job`) around the call, since a failure needs to
        clear it too."""
        self._set_job(interview_id, "transcribing", progress=None, source_app=source_app)

        last_notified_at = 0.0

        def _on_transcribe_progress(fraction: float) -> None:
            nonlocal last_notified_at
            now = time.monotonic()
            # throttle -- a long interview can yield hundreds of segments,
            # and flooding the UI thread with .after() calls per segment
            # isn't necessary for a progress bar humans are just glancing at
            if fraction < 1.0 and now - last_notified_at < 0.25:
                return
            last_notified_at = now
            self._set_job(interview_id, "transcribing", progress=fraction)

        transcript = transcribe(
            audio_path, self.cfg, on_progress=_on_transcribe_progress, cancel_event=cancel_event
        )
        self.db.save_transcript(interview_id, transcript)

        if not transcript.strip():
            # faster-whisper found no speech at all (e.g. a silent/noise-only
            # recording) -- there's nothing for the analysis engine to work
            # with, so skip straight to a report that says so plainly
            # instead of sending it an empty transcript
            self._set_job(interview_id, "generating_report")
            self.db.save_analysis(interview_id, {"no_speech_detected": True})
            record = self.db.get(interview_id)
            report_path = write_interview_report(record, self.cfg)
            self.db.save_report_path(interview_id, str(report_path))
            write_trends_report(self.db.list_all(user_id=self.user_id), self.cfg, user_id=self.user_id)
            logger.info("Interview #%s: no speech detected; skipped analysis.", interview_id)
            return

        self._set_job(interview_id, "analyzing", progress=None)

        last_analyze_notified_at = 0.0

        def _on_analyze_progress(fraction: float) -> None:
            nonlocal last_analyze_notified_at
            now = time.monotonic()
            if fraction < 1.0 and now - last_analyze_notified_at < 0.25:
                return
            last_analyze_notified_at = now
            self._set_job(interview_id, "analyzing", progress=fraction)

        notes = build_calibration_notes(self.db, self.user_id)
        analysis = analyze_transcript(
            transcript, self.cfg, on_progress=_on_analyze_progress, calibration_notes=notes
        )
        model_reported_confidence = None
        if isinstance(analysis.get("session_summary"), dict):
            model_reported_confidence = analysis["session_summary"].get("confidence")
        analysis["confidence_info"] = calibrated_confidence(self.db, self.user_id, model_reported_confidence)
        self.db.save_analysis(interview_id, analysis)

        self._set_job(interview_id, "generating_report")
        record = self.db.get(interview_id)
        report_path = write_interview_report(record, self.cfg)
        self.db.save_report_path(interview_id, str(report_path))

        write_trends_report(self.db.list_all(user_id=self.user_id), self.cfg, user_id=self.user_id)
        logger.info("Interview #%s processed. Report: %s", interview_id, report_path)

    def reprocess_interview(self, interview_id: int) -> None:
        """Re-run transcribe/analyze/report for an interview that has audio
        on disk but never finished processing -- e.g. a crash mid-pipeline,
        a missing dependency, or Ollama being unreachable at the time.
        Blocks like the main pipeline does (transcription/analysis can take
        a while); call it from a background thread, not the UI thread. Runs
        fine alongside a live recording or other interviews' background
        processing -- only reprocessing the *same* interview twice at once
        is rejected."""
        if self._current_interview_id == interview_id:
            raise RuntimeError("This interview is still being recorded.")
        with self._processing_lock:
            already_running = interview_id in self._processing_jobs
        if already_running:
            raise RuntimeError("This interview is already being processed.")

        record = self.db.get(interview_id)
        if record is None or not record.audio_path:
            raise ValueError("No audio was recorded for this interview -- nothing to reprocess.")
        audio_path = pathlib.Path(record.audio_path)
        if not audio_path.exists():
            raise ValueError(f"Audio file is missing: {audio_path}")
        if audio_path.stat().st_size == 0:
            raise ValueError(
                "Audio file is empty -- the recording was likely interrupted before "
                "any audio was saved, so there's nothing to transcribe."
            )

        cancel_event = threading.Event()
        with self._processing_lock:
            self._cancel_events[interview_id] = cancel_event
        try:
            self._run_analysis_pipeline(interview_id, audio_path, cancel_event, source_app=record.source_app)
        except TranscriptionCancelled:
            logger.info("Reprocessing interview #%s was cancelled.", interview_id)
            raise
        finally:
            self._clear_job(interview_id)


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
