"""Tests for MeetingWatcher.reprocess_interview -- recovering a report for
an interview whose audio was recorded but never finished processing (a
crash mid-pipeline, a missing dependency, Ollama being unreachable, etc.
at the time). Also covers running multiple interviews' processing
concurrently and cancelling an in-progress job -- reprocessing (and the
main record->process pipeline) now runs in a background thread per
interview rather than blocking the watcher, so more than one can be
in flight at once."""
from __future__ import annotations

import datetime as dt
import pathlib
import threading
import wave
from unittest.mock import patch

import pytest

from interview_analyzer.config_loader import Config
from interview_analyzer.transcriber import TranscriptionCancelled
from interview_analyzer.watcher import MeetingWatcher

FAKE_ANALYSIS = {
    "qa_pairs": [],
    "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
}


def _test_config(tmp_path) -> Config:
    return Config(raw={
        "retention_days": 3,
        "poll_interval_seconds": 0.01,
        "start_debounce_polls": 1,
        "stop_debounce_polls": 1,
        "watched_processes": {"desktop_apps": [], "browser_tab_keywords": [], "browser_processes": []},
        "audio": {"sample_rate": 16000, "channels": 1, "bitrate_kbps": 64,
                   "format": "opus", "raw_dir": str(tmp_path / "audio")},
        "transcription": {"engine": "faster-whisper", "whisper_model": "tiny",
                           "device": "cpu", "diarization": False},
        "analysis": {"engine": "ollama", "llm_model": "llama3.1:8b",
                     "ollama_host": "http://localhost:11434"},
        "storage": {"db_path": str(tmp_path / "interviews.db")},
        "output": {"output_dir": str(tmp_path / "output"), "reports_subdir": "reports",
                    "trends_filename": "trends.md"},
    })


def _seed_interview(watcher, tmp_path, name="orphaned.wav", app="Zoom") -> int:
    audio_path = tmp_path / "audio" / name
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"RIFF....WAVEreal audio bytes")
    iid = watcher.db.start_interview(app, str(audio_path), retention_days=3, user_id=1)
    watcher.db.end_interview(iid)
    return iid


def _seed_interview_with_real_audio_and_no_ended_at(watcher, tmp_path, seconds=5.0, name="crashed.wav") -> int:
    """Simulates a recording that crashed before _stop_and_process's
    end_interview() call ever ran -- real, decodable audio (needed for
    get_audio_duration_seconds to probe it) but ended_at left null."""
    audio_path = tmp_path / "audio" / name
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * int(seconds * 16000))
    return watcher.db.start_interview("Zoom", str(audio_path), retention_days=3, user_id=1)


def test_reprocess_interview_skips_analysis_when_no_speech_detected(tmp_path):
    """Some recordings (silence, background noise only) produce zero
    faster-whisper segments and an empty transcript -- there's nothing for
    the analysis engine to work with, so it must not even be called."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview(watcher, tmp_path, "silent.wav")

    with patch("interview_analyzer.watcher.transcribe", return_value=""), \
         patch("interview_analyzer.watcher.analyze_transcript") as mock_analyze:
        watcher.reprocess_interview(iid)

    mock_analyze.assert_not_called()
    record = watcher.db.get(iid)
    assert record.transcript == ""
    assert record.analysis == {"no_speech_detected": True}
    assert record.report_path is not None
    content = pathlib.Path(record.report_path).read_text(encoding="utf-8")
    assert "No speech was detected" in content
    assert watcher.status == {"state": "idle", "processing_jobs": {}}


def test_reprocess_interview_treats_whitespace_only_transcript_as_no_speech(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview(watcher, tmp_path, "silent.wav")

    with patch("interview_analyzer.watcher.transcribe", return_value="   \n  "), \
         patch("interview_analyzer.watcher.analyze_transcript") as mock_analyze:
        watcher.reprocess_interview(iid)

    mock_analyze.assert_not_called()
    assert watcher.db.get(iid).analysis == {"no_speech_detected": True}


def test_reprocess_interview_reports_live_transcription_progress(tmp_path):
    """Each on_progress() call updates the job's entry in
    processing_jobs in real time (not just at the end) -- verified with a
    controlled fake clock so the built-in throttle doesn't skip any of
    these updates, since segments spaced >=0.25s apart should always get
    through."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview(watcher, tmp_path)

    observed_jobs = []
    fake_clock = iter([10.0, 11.0, 12.0])  # >=0.25s apart -- nothing should be throttled

    def _fake_transcribe(path, cfg, on_progress=None, cancel_event=None):
        for fraction in (0.3, 0.7, 1.0):
            on_progress(fraction)
            observed_jobs.append(dict(watcher.status["processing_jobs"][iid]))
        return "[Interviewer] Hi\n[You] Hello"

    with patch("interview_analyzer.watcher.transcribe", side_effect=_fake_transcribe), \
         patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS), \
         patch("interview_analyzer.watcher.time.monotonic", side_effect=fake_clock):
        watcher.reprocess_interview(iid)

    assert [j["progress"] for j in observed_jobs] == [0.3, 0.7, 1.0]
    assert all(j["stage"] == "transcribing" for j in observed_jobs)
    # the job is removed entirely once processing finishes
    assert watcher.status == {"state": "idle", "processing_jobs": {}}


def test_reprocess_interview_throttles_rapid_progress_updates(tmp_path):
    """A long interview can yield hundreds of segments; updates that land
    within the same 0.25s window are coalesced (except the final 100%,
    which always gets through) so the UI thread isn't flooded."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview(watcher, tmp_path)

    observed_progress = []
    fake_clock = iter([10.0, 10.05, 10.10, 10.5])

    def _fake_transcribe(path, cfg, on_progress=None, cancel_event=None):
        for fraction in (0.2, 0.4, 0.6, 1.0):
            on_progress(fraction)
            observed_progress.append(watcher.status["processing_jobs"][iid]["progress"])
        return "[Interviewer] Hi\n[You] Hello"

    with patch("interview_analyzer.watcher.transcribe", side_effect=_fake_transcribe), \
         patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS), \
         patch("interview_analyzer.watcher.time.monotonic", side_effect=fake_clock):
        watcher.reprocess_interview(iid)

    assert observed_progress == [0.2, 0.2, 0.2, 1.0]


def test_reprocess_interview_reports_live_analysis_progress(tmp_path):
    """The analysis stage now reports an estimated % too (via Ollama's
    streaming eval_count, see analyzer.py), the same way transcription
    already did -- verified here at the watcher level with a controlled
    fake clock so the throttle doesn't skip any of these updates."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview(watcher, tmp_path)

    observed_jobs = []
    fake_clock = iter([10.0, 11.0, 12.0])

    def _fake_analyze(transcript, cfg, on_progress=None, calibration_notes=""):
        for fraction in (0.4, 0.8, 1.0):
            on_progress(fraction)
            observed_jobs.append(dict(watcher.status["processing_jobs"][iid]))
        return FAKE_ANALYSIS

    with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
         patch("interview_analyzer.watcher.analyze_transcript", side_effect=_fake_analyze), \
         patch("interview_analyzer.watcher.time.monotonic", side_effect=fake_clock):
        watcher.reprocess_interview(iid)

    assert [j["progress"] for j in observed_jobs] == [0.4, 0.8, 1.0]
    assert all(j["stage"] == "analyzing" for j in observed_jobs)
    assert watcher.status == {"state": "idle", "processing_jobs": {}}


def test_reprocess_interview_recovers_a_report_from_existing_audio(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview(watcher, tmp_path)

    with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
         patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):
        watcher.reprocess_interview(iid)

    record = watcher.db.get(iid)
    assert record.transcript == "[Interviewer] Hi\n[You] Hello"
    assert record.analysis == FAKE_ANALYSIS
    assert record.report_path is not None
    assert pathlib.Path(record.report_path).exists()
    assert watcher.status == {"state": "idle", "processing_jobs": {}}


def test_reprocess_interview_rejects_missing_audio(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)

    iid = watcher.db.start_interview("Zoom", str(tmp_path / "audio" / "gone.wav"), retention_days=3, user_id=1)
    watcher.db.end_interview(iid)

    try:
        watcher.reprocess_interview(iid)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "missing" in str(e).lower()


def test_reprocess_interview_rejects_empty_audio(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)

    audio_path = tmp_path / "audio" / "empty.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"")

    iid = watcher.db.start_interview("Zoom", str(audio_path), retention_days=3, user_id=1)
    watcher.db.end_interview(iid)

    try:
        watcher.reprocess_interview(iid)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_reprocess_interview_rejects_unknown_interview_id(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)

    try:
        watcher.reprocess_interview(999)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_reprocess_interview_refuses_when_that_same_interview_is_recording(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    watcher._current_interview_id = 42  # simulate interview #42 actively recording

    try:
        watcher.reprocess_interview(42)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "recorded" in str(e).lower()


def test_reprocess_interview_allowed_while_a_different_interview_is_recording(tmp_path):
    """Recording and background processing are independent now -- an
    active recording for one interview must not block reprocessing a
    completely different one."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview(watcher, tmp_path)
    watcher._current_interview_id = 999  # a different interview is "recording"

    with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
         patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):
        watcher.reprocess_interview(iid)  # must not raise

    assert watcher.db.get(iid).report_path is not None


def test_reprocess_interview_refuses_when_already_being_processed(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview(watcher, tmp_path)
    watcher._processing_jobs[iid] = {"stage": "transcribing", "progress": 0.5}

    try:
        watcher.reprocess_interview(iid)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "already" in str(e).lower()


def test_two_different_interviews_can_be_reprocessed_concurrently(tmp_path):
    """The whole point of the background-processing redesign: interview A
    finishing shouldn't be required before interview B can start."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid_a = _seed_interview(watcher, tmp_path, "a.wav")
    iid_b = _seed_interview(watcher, tmp_path, "b.wav")

    a_started = threading.Event()
    release_a = threading.Event()

    def _slow_transcribe_a(path, cfg, on_progress=None, cancel_event=None):
        a_started.set()
        release_a.wait(timeout=10)
        return "[Interviewer] Hi\n[You] A"

    with patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):
        with patch("interview_analyzer.watcher.transcribe", side_effect=_slow_transcribe_a):
            thread_a = threading.Thread(target=watcher.reprocess_interview, args=(iid_a,), daemon=True)
            thread_a.start()
            assert a_started.wait(timeout=10), "interview A's transcribe never started"

            # while A is still blocked mid-transcribe, B must be reprocessable too
            with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] B"):
                watcher.reprocess_interview(iid_b)

            assert watcher.db.get(iid_b).transcript == "[Interviewer] Hi\n[You] B"
            assert iid_a in watcher.status["processing_jobs"]  # A is still going

            release_a.set()
            thread_a.join(timeout=10)

    assert watcher.db.get(iid_a).transcript == "[Interviewer] Hi\n[You] A"
    assert watcher.status == {"state": "idle", "processing_jobs": {}}


def test_cancel_processing_stops_an_in_progress_reprocess(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview(watcher, tmp_path)

    started = threading.Event()

    def _fake_transcribe(path, cfg, on_progress=None, cancel_event=None):
        started.set()
        # simulate faster-whisper's own per-segment cancellation check
        while not cancel_event.is_set():
            cancel_event.wait(timeout=0.05)
        raise TranscriptionCancelled()

    result = {}

    def _run():
        try:
            watcher.reprocess_interview(iid)
        except TranscriptionCancelled:
            result["cancelled"] = True

    with patch("interview_analyzer.watcher.transcribe", side_effect=_fake_transcribe), \
         patch("interview_analyzer.watcher.analyze_transcript") as mock_analyze:
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        assert started.wait(timeout=10), "transcribe never started"

        cancelled = watcher.cancel_processing(iid)
        thread.join(timeout=10)

    assert cancelled is True
    assert result.get("cancelled") is True
    mock_analyze.assert_not_called()
    # no transcript/analysis/report was ever saved for a cancelled job
    record = watcher.db.get(iid)
    assert record.transcript is None
    assert record.report_path is None
    assert watcher.status == {"state": "idle", "processing_jobs": {}}


def test_cancel_processing_returns_false_when_nothing_to_cancel(tmp_path):
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    assert watcher.cancel_processing(12345) is False


def test_reprocess_interview_backfills_ended_at_from_real_audio_duration(tmp_path):
    """Regression coverage for a real bug: an interview whose recording
    crashed before _stop_and_process's end_interview() call ran left
    ended_at permanently null, so the History tab's Duration column showed
    blank forever (see dashboard.py's format_duration) -- even after a
    successful reprocess, since reprocessing alone never sets ended_at.
    It's now back-filled from the audio file's own real duration."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview_with_real_audio_and_no_ended_at(watcher, tmp_path, seconds=5.0)
    assert watcher.db.get(iid).ended_at is None

    with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
         patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):
        watcher.reprocess_interview(iid)

    record = watcher.db.get(iid)
    assert record.ended_at is not None
    started = dt.datetime.fromisoformat(record.started_at)
    ended = dt.datetime.fromisoformat(record.ended_at)
    assert (ended - started).total_seconds() == pytest.approx(5.0, abs=0.5)


def test_reprocess_interview_does_not_overwrite_a_real_ended_at(tmp_path):
    """A normally-ended interview being reprocessed for an unrelated reason
    (e.g. a bad model config at the time) must not have its real ended_at
    clobbered by a duration-based guess."""
    cfg = _test_config(tmp_path)
    watcher = MeetingWatcher(cfg, user_id=1)
    iid = _seed_interview(watcher, tmp_path, "normal.wav")
    original_ended_at = watcher.db.get(iid).ended_at

    with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
         patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS):
        watcher.reprocess_interview(iid)

    assert watcher.db.get(iid).ended_at == original_ended_at
