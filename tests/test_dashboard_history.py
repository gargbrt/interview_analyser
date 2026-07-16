"""Tests for the History tab's pure formatting/status-logic helpers (no Tk
needed -- these just take an InterviewRecord and return strings/bools)."""
from __future__ import annotations

import json

from interview_analyzer.dashboard import can_reprocess, format_duration, format_started, has_audio, history_status_label
from interview_analyzer.db import InterviewRecord

FAKE_ANALYSIS = {
    "qa_pairs": [],
    "session_summary": {"top_strengths": [], "top_issues": ["Lacks structure"], "one_thing_to_practice_next": ""},
}


def _record(**overrides) -> InterviewRecord:
    defaults = dict(
        id=1, user_id=1, started_at="2026-07-16T11:59:46.072781", ended_at=None,
        source_app="Zoom", audio_path=None, audio_expires_at=None, audio_deleted=False,
        transcript=None, analysis_json=None, report_path=None,
    )
    defaults.update(overrides)
    return InterviewRecord(**defaults)


def test_format_started_renders_readable_local_timestamp():
    assert format_started(_record()) == "2026-07-16 11:59"


def test_format_started_falls_back_to_raw_string_on_bad_input():
    assert format_started(_record(started_at="not-a-date")) == "not-a-date"


def test_format_duration_is_dash_when_never_ended():
    assert format_duration(_record(ended_at=None)) == "—"


def test_format_duration_computes_minutes_and_seconds():
    r = _record(started_at="2026-07-16T11:59:46", ended_at="2026-07-16T12:01:16")  # 90s
    assert format_duration(r) == "01:30"


def test_format_duration_handles_sub_minute_calls():
    r = _record(started_at="2026-07-16T11:59:46", ended_at="2026-07-16T11:59:56")  # 10s
    assert format_duration(r) == "00:10"


def test_history_status_label_report_generated_with_top_issue(tmp_path):
    report_path = tmp_path / "report.md"
    report_path.write_text("# report", encoding="utf-8")
    r = _record(ended_at="2026-07-16T12:00:00", analysis_json=json.dumps(FAKE_ANALYSIS),
                transcript="t", report_path=str(report_path))
    assert history_status_label(r) == "Lacks structure"


def test_history_status_label_interrupted_when_never_ended():
    r = _record(ended_at=None)
    assert history_status_label(r) == "Interrupted — no report"


def test_history_status_label_no_speech_detected(tmp_path):
    report_path = tmp_path / "report.md"
    report_path.write_text("# report", encoding="utf-8")
    r = _record(ended_at="2026-07-16T12:00:00", analysis_json=json.dumps({"no_speech_detected": True}),
                transcript="", report_path=str(report_path))
    assert history_status_label(r) == "No speech detected"


def test_history_status_label_analysis_failed_when_transcript_but_no_analysis():
    r = _record(ended_at="2026-07-16T12:00:00", transcript="some transcript")
    assert history_status_label(r) == "Analysis failed"


def test_history_status_label_not_processed_when_ended_but_nothing_else():
    r = _record(ended_at="2026-07-16T12:00:00")
    assert history_status_label(r) == "Not processed"


def test_can_reprocess_true_when_audio_exists_and_no_report(tmp_path):
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"some audio bytes")
    r = _record(ended_at="2026-07-16T12:00:00", audio_path=str(audio_path))
    assert can_reprocess(r) is True


def test_can_reprocess_false_when_report_already_exists(tmp_path):
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"some audio bytes")
    report_path = tmp_path / "report.md"
    report_path.write_text("# report", encoding="utf-8")
    r = _record(ended_at="2026-07-16T12:00:00", audio_path=str(audio_path), report_path=str(report_path))
    assert can_reprocess(r) is False


def test_can_reprocess_false_when_audio_missing():
    r = _record(ended_at="2026-07-16T12:00:00", audio_path=None)
    assert can_reprocess(r) is False


def test_can_reprocess_false_when_audio_file_empty(tmp_path):
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"")
    r = _record(ended_at="2026-07-16T12:00:00", audio_path=str(audio_path))
    assert can_reprocess(r) is False


def test_can_reprocess_false_when_audio_file_does_not_exist(tmp_path):
    r = _record(ended_at="2026-07-16T12:00:00", audio_path=str(tmp_path / "gone.wav"))
    assert can_reprocess(r) is False


def test_has_audio_true_when_file_exists_and_nonempty(tmp_path):
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"some audio bytes")
    assert has_audio(_record(audio_path=str(audio_path))) is True


def test_has_audio_false_when_missing_empty_or_no_path(tmp_path):
    assert has_audio(_record(audio_path=None)) is False
    assert has_audio(_record(audio_path=str(tmp_path / "gone.wav"))) is False
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert has_audio(_record(audio_path=str(empty))) is False


def test_has_audio_true_even_when_a_report_already_exists(tmp_path):
    """Playing the audio should stay available after a report exists too --
    only Reprocess is gated on "no report yet"."""
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"some audio bytes")
    report_path = tmp_path / "report.md"
    report_path.write_text("# report", encoding="utf-8")
    r = _record(audio_path=str(audio_path), report_path=str(report_path))
    assert has_audio(r) is True
    assert can_reprocess(r) is False
