"""Tests for infographic.py: the HTML "report card" generated alongside
the markdown report (see report.py) and opened via the History tab's
"View infographic" button, and its trends counterpart opened via the
Trends tab."""
from __future__ import annotations

import json

from interview_analyzer.config_loader import Config
from interview_analyzer.db import InterviewRecord
from interview_analyzer.infographic import (
    infographic_path,
    trends_infographic_path,
    write_interview_infographic,
    write_trends_infographic,
)

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


class TestInterviewInfographicProfileFields:
    """Assessment-profile additions: per-competency score bars, the hire-
    scale recommendation badge, and the selection-probability dial (see
    rubric.py/confidence.py)."""

    def _analysis_with_profile_fields(self, **overrides):
        analysis = {
            "qa_pairs": [],
            "session_summary": {
                "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
                "competency_scores": [
                    {"name": "Leadership", "score": 82, "remark": "Took clear ownership of the project."},
                ],
                "hire_recommendation": {"level": "Strong Hire", "rationale": "Consistently strong signal."},
            },
        }
        analysis.update(overrides)
        return analysis

    def test_renders_competency_score_bars_with_remarks(self, tmp_path):
        record = _record(tmp_path, analysis_json=json.dumps(self._analysis_with_profile_fields()))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert "Leadership" in content
        assert "82/100" in content
        assert "Took clear ownership" in content

    def test_renders_hire_recommendation_badge(self, tmp_path):
        record = _record(tmp_path, analysis_json=json.dumps(self._analysis_with_profile_fields()))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert "Strong Hire" in content
        assert "Consistently strong signal." in content

    def test_renders_selection_probability_dial(self, tmp_path):
        analysis = self._analysis_with_profile_fields()
        analysis["selection_probability"] = {"percent": 81, "label": "Strong Hire", "basis": "Anchored at 90%."}
        record = _record(tmp_path, analysis_json=json.dumps(analysis))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert "Selection probability" in content
        assert ">81<" in content
        assert "Anchored at 90%." in content

    def test_missing_selection_probability_renders_not_available_dial(self, tmp_path):
        record = _record(tmp_path, analysis_json=json.dumps(self._analysis_with_profile_fields()))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert "Selection probability" in content
        assert "not available" in content

    def test_renders_binary_recommendation_pill_alongside_percentage(self, tmp_path):
        analysis = self._analysis_with_profile_fields()
        analysis["selection_probability"] = {
            "percent": 81, "label": "Strong Hire", "basis": "Anchored at 90%.", "binary_recommendation": "Recommended",
        }
        record = _record(tmp_path, analysis_json=json.dumps(analysis))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert ">81<" in content  # percentage is still shown, not replaced
        assert "recommendation-good" in content
        assert "Recommended" in content

    def test_renders_not_recommended_pill_with_bad_styling(self, tmp_path):
        analysis = self._analysis_with_profile_fields()
        analysis["selection_probability"] = {
            "percent": 20, "label": "No Hire", "basis": "Anchored at 15%.", "binary_recommendation": "Not Recommended",
        }
        record = _record(tmp_path, analysis_json=json.dumps(analysis))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert "recommendation-bad" in content
        assert "Not Recommended" in content

    def test_no_binary_recommendation_omits_the_pill(self, tmp_path):
        analysis = self._analysis_with_profile_fields()
        analysis["selection_probability"] = {"percent": 60, "label": "Hire", "basis": "..."}
        record = _record(tmp_path, analysis_json=json.dumps(analysis))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert '<p class="recommendation-pill' not in content

    def test_no_competency_scores_omits_the_section_without_crashing(self, tmp_path):
        analysis = {
            "qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
        }
        record = _record(tmp_path, analysis_json=json.dumps(analysis))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert "Competency scores" not in content

    def test_escapes_html_in_competency_remarks_and_hire_rationale(self, tmp_path):
        analysis = self._analysis_with_profile_fields(
            session_summary={
                "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
                "competency_scores": [{"name": "Leadership", "score": 50, "remark": "<script>alert(1)</script>"}],
                "hire_recommendation": {"level": "Hire", "rationale": "<b>unsafe</b>"},
            },
        )
        record = _record(tmp_path, analysis_json=json.dumps(analysis))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert "<script>alert" not in content
        assert "&lt;script&gt;" in content
        assert "<b>unsafe</b>" not in content

    def test_renders_competency_weight_label_for_the_records_profile(self, tmp_path):
        # Entry Level -> Leadership is "minor" per profiles.SENIORITY_EMPHASIS
        profile = {
            "competencies": ["Leadership"], "role": None, "seniority": "Entry Level",
            "industry": None, "company_type": None, "name": None,
        }
        record = _record(
            tmp_path,
            analysis_json=json.dumps(self._analysis_with_profile_fields()),
            profile_snapshot_json=json.dumps(profile),
        )
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert "minor weight" in content

    def test_renders_overall_competency_score(self, tmp_path):
        analysis = self._analysis_with_profile_fields(
            session_summary={
                "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
                "competency_scores": [
                    {"name": "Leadership", "score": 80, "remark": ""},
                    {"name": "Execution", "score": 40, "remark": ""},
                ],
                "hire_recommendation": {"level": "Hire", "rationale": ""},
            },
        )
        record = _record(tmp_path, analysis_json=json.dumps(analysis))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert "Overall competency score" in content

    def test_omits_overall_competency_score_when_no_scores(self, tmp_path):
        analysis = {
            "qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
        }
        record = _record(tmp_path, analysis_json=json.dumps(analysis))
        content = write_interview_infographic(record, _cfg(tmp_path)).read_text(encoding="utf-8")
        assert "Overall competency score" not in content


def test_infographic_path_is_alongside_the_markdown_report(tmp_path):
    record = _record(tmp_path)
    cfg = _cfg(tmp_path)

    path = infographic_path(record, cfg)

    assert path.name == "2026-07-16_Zoom_1_infographic.html"
    assert path.parent == cfg.resolve(cfg.output.get("output_dir")) / "reports"


class TestTrendsInfographic:
    def test_path_is_alongside_the_markdown_trends_report(self, tmp_path):
        cfg = _cfg(tmp_path)
        path = trends_infographic_path(cfg, user_id=1)
        assert path.name == "trends_user1_infographic.html"

    def test_zero_analyzed_interviews_still_writes_a_real_page(self, tmp_path):
        cfg = _cfg(tmp_path)
        path = write_trends_infographic([], cfg, user_id=1)

        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert content.startswith("<!doctype html>")
        assert "No analyzed interviews yet" in content

    def test_includes_bar_rows_for_recurring_issues_and_strengths(self, tmp_path):
        cfg = _cfg(tmp_path)
        records = [
            _record(tmp_path, id=1, analysis_json=json.dumps({
                "qa_pairs": [],
                "session_summary": {
                    "top_strengths": ["Clear communication"],
                    "top_issues": ["Rambling"],
                    "one_thing_to_practice_next": "",
                },
            })),
            _record(tmp_path, id=2, analysis_json=json.dumps({
                "qa_pairs": [],
                "session_summary": {
                    "top_strengths": ["Clear communication"],
                    "top_issues": ["Rambling"],
                    "one_thing_to_practice_next": "",
                },
            })),
        ]

        content = write_trends_infographic(records, cfg, user_id=1).read_text(encoding="utf-8")

        assert "Rambling" in content
        assert "Clear communication" in content
        assert "based on 2 analyzed interview(s)" in content
        # both interviews flagged the same issue/strength -- count should show 2
        assert ">2<" in content

    def test_skips_records_with_no_usable_analysis(self, tmp_path):
        cfg = _cfg(tmp_path)
        records = [
            _record(tmp_path, id=1, analysis_json=json.dumps({"raw": "bad", "parse_error": True})),
            _record(tmp_path, id=2, analysis_json=json.dumps({"no_speech_detected": True})),
            _record(tmp_path, id=3, analysis_json=None),
        ]

        content = write_trends_infographic(records, cfg, user_id=1).read_text(encoding="utf-8")

        assert "based on 0 analyzed interview(s)" in content
        assert "No analyzed interviews yet" in content

    def test_escapes_html_special_characters(self, tmp_path):
        cfg = _cfg(tmp_path)
        records = [_record(tmp_path, id=1, analysis_json=json.dumps({
            "qa_pairs": [],
            "session_summary": {
                "top_strengths": ["<script>alert(1)</script>"],
                "top_issues": [],
                "one_thing_to_practice_next": "",
            },
        }))]

        content = write_trends_infographic(records, cfg, user_id=1).read_text(encoding="utf-8")

        assert "<script>alert" not in content
        assert "&lt;script&gt;" in content

    def test_lists_all_interviews_regardless_of_analysis_status(self, tmp_path):
        """The "all interviews" list itself (like the markdown trends
        report's) includes every interview regardless of whether it has
        usable analysis -- only the bar-chart counts are analysis-only."""
        cfg = _cfg(tmp_path)
        records = [
            _record(tmp_path, id=1, source_app="Zoom", started_at="2026-07-16T11:00:00"),  # has VALID_ANALYSIS
            _record(tmp_path, id=2, source_app="Meet", started_at="2026-07-17T09:00:00", analysis_json=None),
        ]

        content = write_trends_infographic(records, cfg, user_id=1).read_text(encoding="utf-8")

        assert "2026-07-16" in content
        assert "Zoom" in content
        assert "2026-07-17" in content
        assert "Meet" in content
        assert "not yet generated" in content
        assert "not yet generated" in content

    def test_includes_competency_averages_weakest_first(self, tmp_path):
        cfg = _cfg(tmp_path)
        records = [
            _record(tmp_path, id=1, analysis_json=json.dumps({
                "qa_pairs": [],
                "session_summary": {
                    "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
                    "competency_scores": [
                        {"name": "Leadership", "score": 40, "remark": ""},
                        {"name": "Execution", "score": 90, "remark": ""},
                    ],
                },
            })),
        ]

        content = write_trends_infographic(records, cfg, user_id=1).read_text(encoding="utf-8")

        assert "Competency averages" in content
        assert content.index("Leadership") < content.index("Execution")

    def test_no_competency_scores_omits_the_section(self, tmp_path):
        cfg = _cfg(tmp_path)
        records = [_record(tmp_path, id=1)]  # VALID_ANALYSIS has no competency_scores

        content = write_trends_infographic(records, cfg, user_id=1).read_text(encoding="utf-8")

        assert "Competency averages" not in content
