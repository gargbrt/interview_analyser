"""Tests for infographic.py: the HTML "report card" generated alongside
the markdown report (see report.py) and opened via the History tab's
"View infographic" button."""
from __future__ import annotations

import json

from interview_analyzer.config_loader import Config
from interview_analyzer.db import InterviewRecord
from interview_analyzer.infographic import infographic_path, write_interview_infographic

VALID_ANALYSIS = {
    "qa_pairs": [
        {
            "question": "Tell me about a conflict you resolved.",
            "answer_summary": "Rambled without a clear structure.",
            "issues": [{"category": "structure", "detail": "No STAR structure used.", "excerpt": ""}],
            "suggested_improvement": "Lead with the situation, then action, then result.",
        }
    ],
    "session_summary": {
        "top_strengths": ["Clear technical vocabulary"],
        "top_issues": ["Lack of structure"],
        "one_thing_to_practice_next": "Practice STAR format.",
        "confidence": 72,
    },
    "confidence_info": {"score": 72, "source": "model", "sample_size": 0},
}


def _record(tmp_path, **overrides) -> InterviewRecord:
    defaults = dict(
        id=1, user_id=1, started_at="2026-07-16T11:59:46.072781", ended_at="2026-07-16T12:20:00",
        source_app="Zoom", audio_path=None, audio_expires_at=None, audio_deleted=False,
        transcript="[Interviewer] Hi\n[You] Hello", analysis_json=json.dumps(VALID_ANALYSIS), report_path=None,
    )
    defaults.update(overrides)
    return InterviewRecord(**defaults)


def _cfg(tmp_path) -> Config:
    return Config(raw={"output": {"output_dir": str(tmp_path / "output"), "reports_subdir": "reports"}})


def test_returns_none_and_writes_nothing_when_analysis_has_a_parse_error(tmp_path):
    record = _record(tmp_path, analysis_json=json.dumps({"raw": "not json shaped", "parse_error": True}))
    result = write_interview_infographic(record, _cfg(tmp_path))
    assert result is None
    assert not infographic_path(record, _cfg(tmp_path)).exists()


def test_returns_none_when_no_speech_was_detected(tmp_path):
    record = _record(tmp_path, analysis_json=json.dumps({"no_speech_detected": True}))
    result = write_interview_infographic(record, _cfg(tmp_path))
    assert result is None


def test_returns_none_when_there_is_no_analysis_at_all(tmp_path):
    record = _record(tmp_path, analysis_json=None)
    result = write_interview_infographic(record, _cfg(tmp_path))
    assert result is None


def test_writes_a_real_html_file_for_valid_analysis(tmp_path):
    record = _record(tmp_path)
    cfg = _cfg(tmp_path)

    result = write_interview_infographic(record, cfg)

    assert result is not None
    assert result.exists()
    assert result == infographic_path(record, cfg)
    content = result.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert "</html>" in content


def test_includes_real_content_from_the_analysis(tmp_path):
    record = _record(tmp_path)
    content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")

    assert "Tell me about a conflict you resolved." in content
    assert "Clear technical vocabulary" in content
    assert "Lack of structure" in content
    assert "Practice STAR format." in content
    assert "72" in content  # confidence score


def test_escapes_html_special_characters_in_model_generated_text(tmp_path):
    """This file is opened directly in a real browser -- unescaped
    model-generated text would be a script-injection risk, not just a
    cosmetic bug."""
    analysis = {
        "qa_pairs": [{
            "question": "<script>alert('xss')</script>",
            "answer_summary": "Answer with <b>tags</b> & \"quotes\"",
            "issues": [],
            "suggested_improvement": "",
        }],
        "session_summary": {
            "top_strengths": ["<img src=x onerror=alert(1)>"],
            "top_issues": [],
            "one_thing_to_practice_next": "",
        },
    }
    record = _record(tmp_path, analysis_json=json.dumps(analysis))

    content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")

    assert "<script>alert" not in content
    assert "&lt;script&gt;" in content
    assert "<img src=x" not in content
    assert "&lt;img" in content


def test_no_qa_pairs_renders_a_note_instead_of_crashing(tmp_path):
    analysis = {
        "qa_pairs": [],
        "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
    }
    record = _record(tmp_path, analysis_json=json.dumps(analysis))

    content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")

    assert "No individual questions were extracted" in content


def test_missing_confidence_score_renders_not_available_dial(tmp_path):
    analysis = {
        "qa_pairs": [],
        "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
        "confidence_info": {"score": None, "source": "unavailable", "sample_size": 0},
    }
    record = _record(tmp_path, analysis_json=json.dumps(analysis))

    content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")

    assert "N/A" in content


def test_dict_shaped_issues_and_strengths_are_stringified_not_crashed_on(tmp_path):
    """LLM output doesn't always exactly match the requested shape --
    report.py already handles this defensively (_stringify); the
    infographic reuses the same handling."""
    analysis = {
        "qa_pairs": [],
        "session_summary": {
            "top_strengths": [{"strength": "Good pacing"}],
            "top_issues": [{"issue": "Rambling"}],
            "one_thing_to_practice_next": "",
        },
    }
    record = _record(tmp_path, analysis_json=json.dumps(analysis))

    content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")

    assert "Good pacing" in content
    assert "Rambling" in content


def test_infographic_path_is_alongside_the_markdown_report(tmp_path):
    record = _record(tmp_path)
    cfg = _cfg(tmp_path)

    path = infographic_path(record, cfg)

    assert path.name == "2026-07-16_Zoom_1_infographic.html"
    assert path.parent == cfg.resolve(cfg.output.get("output_dir")) / "reports"
