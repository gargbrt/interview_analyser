import json

from interview_analyzer.config_loader import Config
from interview_analyzer.db import InterviewRecord
from interview_analyzer.report import write_interview_report, write_trends_report


def _config(tmp_path) -> Config:
    return Config(raw={
        "output": {
            "output_dir": str(tmp_path / "output"),
            "reports_subdir": "reports",
            "trends_filename": "trends.md",
        }
    })


def _record(id_, source_app, analysis, report_path=None) -> InterviewRecord:
    return InterviewRecord(
        id=id_,
        user_id=1,
        started_at="2026-07-10T10:00:00",
        ended_at="2026-07-10T10:30:00",
        source_app=source_app,
        audio_path=None,
        audio_expires_at=None,
        audio_deleted=True,
        transcript="[Interviewer] Hi\n[You] Hello",
        analysis_json=json.dumps(analysis) if analysis else None,
        report_path=report_path,
    )


def test_write_interview_report_creates_file_with_expected_sections(tmp_path):
    cfg = _config(tmp_path)
    analysis = {
        "qa_pairs": [
            {
                "question": "Why do you want this role?",
                "answer_summary": "Generic answer, no specifics about the company.",
                "issues": [{"category": "specificity", "detail": "No company-specific detail."}],
                "suggested_improvement": "Mention a specific product or value of the company.",
            }
        ],
        "session_summary": {
            "top_strengths": ["Confident tone"],
            "top_issues": ["Generic answers lacking specificity"],
            "one_thing_to_practice_next": "Research 3 company-specific talking points before next interview.",
        },
    }
    record = _record(1, "Zoom", analysis)

    report_path = write_interview_report(record, cfg)

    assert report_path.exists()
    content = report_path.read_text()
    assert "Why do you want this role?" in content
    assert "Generic answers lacking specificity" in content
    assert "Research 3 company-specific talking points" in content


def test_write_interview_report_handles_parse_error(tmp_path):
    cfg = _config(tmp_path)
    record = _record(2, "Teams", {"raw": "garbled output", "parse_error": True})

    report_path = write_interview_report(record, cfg)
    content = report_path.read_text(encoding="utf-8")

    assert "could not be parsed" in content
    assert "garbled output" in content


def test_write_trends_report_aggregates_across_interviews(tmp_path):
    cfg = _config(tmp_path)

    analysis_a = {
        "qa_pairs": [{"question": "Q1", "answer_summary": "", "issues": [
            {"category": "structure", "detail": "no STAR"}
        ], "suggested_improvement": ""}],
        "session_summary": {"top_strengths": ["clarity"], "top_issues": ["structure"],
                             "one_thing_to_practice_next": "practice STAR"},
    }
    analysis_b = {
        "qa_pairs": [{"question": "Q2", "answer_summary": "", "issues": [
            {"category": "structure", "detail": "rambled"}
        ], "suggested_improvement": ""}],
        "session_summary": {"top_strengths": ["clarity"], "top_issues": ["structure"],
                             "one_thing_to_practice_next": "practice STAR"},
    }
    records = [_record(1, "Zoom", analysis_a), _record(2, "Teams", analysis_b)]

    trends_path = write_trends_report(records, cfg)
    content = trends_path.read_text()

    assert "based on 2 analyzed interview(s)" in content
    # "structure" should show up as the most frequent issue, counted from both
    # the qa_pair-level issue tag and the session_summary top_issues entry
    assert "structure" in content
    assert "clarity" in content


def test_write_trends_report_handles_zero_analyzed(tmp_path):
    cfg = _config(tmp_path)
    trends_path = write_trends_report([], cfg)
    content = trends_path.read_text()
    assert "No analyzed interviews yet." in content
