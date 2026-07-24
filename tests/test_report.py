import json

from interview_analyzer.config_loader import Config
from interview_analyzer.db import InterviewRecord
from interview_analyzer.profiles import AssessmentProfile
from interview_analyzer.report import (
    render_interview_report_markdown,
    trends_report_path,
    write_interview_report,
    write_trends_report,
)


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


def test_write_interview_report_includes_excerpt_quote_when_present(tmp_path):
    cfg = _config(tmp_path)
    analysis = {
        "qa_pairs": [{
            "question": "Tell me about a conflict.",
            "answer_summary": "Rambled without a clear resolution.",
            "issues": [{
                "category": "clarity",
                "detail": "Rambling, no clear resolution stated.",
                "excerpt": "so, um, there was this one time, it was kind of a whole thing",
            }],
            "suggested_improvement": "State the resolution directly.",
        }],
        "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
    }
    record = _record(3, "Zoom", analysis)

    report_path = write_interview_report(record, cfg)
    content = report_path.read_text(encoding="utf-8")

    assert '> "so, um, there was this one time, it was kind of a whole thing"' in content


def test_write_interview_report_omits_excerpt_line_when_absent(tmp_path):
    cfg = _config(tmp_path)
    analysis = {
        "qa_pairs": [{
            "question": "Tell me about a conflict.",
            "answer_summary": "No answer given.",
            "issues": [{"category": "specificity", "detail": "No concrete outcome given."}],
            "suggested_improvement": "Add a measurable result.",
        }],
        "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
    }
    record = _record(4, "Zoom", analysis)

    report_path = write_interview_report(record, cfg)
    content = report_path.read_text(encoding="utf-8")

    assert ">" not in content


def test_write_interview_report_handles_no_speech_detected(tmp_path):
    cfg = _config(tmp_path)
    record = _record(5, "Zoom", {"no_speech_detected": True})

    report_path = write_interview_report(record, cfg)
    content = report_path.read_text(encoding="utf-8")

    assert "No speech was detected" in content


def test_write_interview_report_shows_feedback_calibrated_confidence(tmp_path):
    cfg = _config(tmp_path)
    analysis = {
        "qa_pairs": [],
        "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
        "confidence_info": {"score": 82, "source": "feedback", "sample_size": 12},
    }
    record = _record(6, "Zoom", analysis)

    content = write_interview_report(record, cfg).read_text(encoding="utf-8")

    assert "Confidence in this assessment:" in content
    assert "82%" in content
    assert "12" in content


def test_write_interview_report_shows_model_reported_confidence(tmp_path):
    cfg = _config(tmp_path)
    analysis = {
        "qa_pairs": [],
        "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
        "confidence_info": {"score": 65, "source": "model", "sample_size": 0},
    }
    record = _record(7, "Zoom", analysis)

    content = write_interview_report(record, cfg).read_text(encoding="utf-8")

    assert "65%" in content
    assert "self-assessment" in content


def test_write_interview_report_handles_missing_confidence_info(tmp_path):
    """Older analyses recorded before this feature existed have no
    confidence_info at all -- must render gracefully, not crash."""
    cfg = _config(tmp_path)
    analysis = {
        "qa_pairs": [],
        "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
    }
    record = _record(8, "Zoom", analysis)

    content = write_interview_report(record, cfg).read_text(encoding="utf-8")

    assert "not available" in content


def test_write_trends_report_excludes_no_speech_detected_interviews(tmp_path):
    cfg = _config(tmp_path)
    records = [
        _record(1, "Zoom", {"no_speech_detected": True}),
        _record(2, "Teams", {
            "qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": ["Rambling"],
                                                  "one_thing_to_practice_next": ""},
        }),
    ]

    trends_path = write_trends_report(records, cfg)
    content = trends_path.read_text(encoding="utf-8")

    assert "based on 1 analyzed interview(s)" in content
    assert "Rambling" in content


def test_write_trends_report_handles_top_issues_returned_as_dicts_not_strings(tmp_path):
    """Reproduces a real crash: a local LLM (llama3.1:8b) returned
    top_issues as a list of dicts instead of the requested plain strings,
    and using one directly as a Counter key raised
    `TypeError: unhashable type: 'dict'`, taking down trend aggregation
    for an otherwise-successfully-processed interview."""
    cfg = _config(tmp_path)
    records = [_record(1, "Zoom", {
        "qa_pairs": [],
        "session_summary": {
            "top_strengths": [{"strength": "Clear communication"}],
            "top_issues": [{"issue": "Rambling answers", "detail": "..."}],
            "one_thing_to_practice_next": "",
        },
    })]

    trends_path = write_trends_report(records, cfg)  # must not raise
    content = trends_path.read_text(encoding="utf-8")

    assert "Rambling answers" in content
    assert "Clear communication" in content


def test_write_interview_report_handles_top_issues_returned_as_dicts_not_strings(tmp_path):
    cfg = _config(tmp_path)
    record = _record(3, "Zoom", {
        "qa_pairs": [],
        "session_summary": {
            "top_strengths": [{"strength": "Clear communication"}],
            "top_issues": [{"issue": "Rambling answers"}],
            "one_thing_to_practice_next": "",
        },
    })

    report_path = write_interview_report(record, cfg)  # must not raise
    content = report_path.read_text(encoding="utf-8")

    assert "Rambling answers" in content
    assert "Clear communication" in content


def test_write_interview_report_handles_a_qa_issue_returned_as_a_plain_string(tmp_path):
    """Same class of schema drift, but at the per-question issue level
    (a plain string instead of the requested {"category", "detail"} dict)."""
    cfg = _config(tmp_path)
    record = _record(4, "Zoom", {
        "qa_pairs": [{
            "question": "Tell me about yourself.",
            "answer_summary": "Rambled a bit.",
            "issues": ["Too long-winded"],
            "suggested_improvement": "",
        }],
        "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": ""},
    })

    report_path = write_interview_report(record, cfg)  # must not raise
    content = report_path.read_text(encoding="utf-8")

    assert "Too long-winded" in content


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


def test_write_trends_report_includes_competency_averages_weakest_first(tmp_path):
    cfg = _config(tmp_path)
    analysis_a = {
        "qa_pairs": [],
        "session_summary": {
            "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
            "competency_scores": [
                {"name": "Leadership", "score": 40, "remark": ""},
                {"name": "Execution", "score": 90, "remark": ""},
            ],
        },
    }
    analysis_b = {
        "qa_pairs": [],
        "session_summary": {
            "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
            "competency_scores": [{"name": "Leadership", "score": 60, "remark": ""}],
        },
    }
    records = [_record(1, "Zoom", analysis_a), _record(2, "Teams", analysis_b)]

    content = write_trends_report(records, cfg).read_text()

    assert "Competency averages" in content
    leadership_pos = content.index("Leadership")
    execution_pos = content.index("Execution")
    assert leadership_pos < execution_pos  # weaker average (50) listed before stronger (90)
    assert "across 2 interview(s)" in content  # Leadership scored in both


class TestInterviewReportProfileFields:
    """Session-summary additions from the assessment-profile feature:
    per-competency scores, the hire-scale recommendation, and the
    selection-probability estimate (see rubric.py/confidence.py)."""

    def _analysis(self, **overrides):
        base = {
            "qa_pairs": [],
            "session_summary": {
                "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
                "competency_scores": [{"name": "Leadership", "score": 72, "remark": "Showed clear ownership."}],
                "hire_recommendation": {"level": "Hire", "rationale": "Strong overall signal."},
            },
        }
        base.update(overrides)
        return base

    def test_renders_competency_scores_with_remarks(self, tmp_path):
        record = _record(1, "Zoom", self._analysis())
        content = write_interview_report(record, _config(tmp_path)).read_text()
        assert "Leadership" in content
        assert "72/100" in content
        assert "Showed clear ownership." in content

    def test_renders_hire_recommendation(self, tmp_path):
        record = _record(1, "Zoom", self._analysis())
        content = write_interview_report(record, _config(tmp_path)).read_text()
        assert "Hire recommendation" in content
        assert "Strong overall signal." in content

    def test_renders_selection_probability_when_present(self, tmp_path):
        analysis = self._analysis()
        analysis["selection_probability"] = {
            "percent": 68, "label": "Hire", "basis": "Hire-scale call: \"Hire\" (anchors 75%).",
        }
        record = _record(1, "Zoom", analysis)
        content = write_interview_report(record, _config(tmp_path)).read_text()
        assert "68%" in content
        assert "Hire-scale call" in content

    def test_omits_selection_probability_section_when_absent(self, tmp_path):
        record = _record(1, "Zoom", self._analysis())  # no selection_probability key at all
        content = write_interview_report(record, _config(tmp_path)).read_text()
        assert "selection probability" not in content.lower()

    def test_renders_binary_recommendation_alongside_the_percentage(self):
        analysis = self._analysis()
        analysis["selection_probability"] = {
            "percent": 68, "label": "Hire", "basis": "...", "binary_recommendation": "Recommended",
        }
        record = _record(1, "Zoom", analysis)
        content = render_interview_report_markdown(record, record.analysis)
        assert "68%" in content
        assert "**Recommendation:** Recommended" in content

    def test_renders_competency_weight_label_for_the_given_profile(self, tmp_path):
        # Entry Level -> Leadership is "minor" per profiles.SENIORITY_EMPHASIS
        profile = AssessmentProfile(competencies=["Leadership"], seniority="Entry Level")
        record = _record(1, "Zoom", self._analysis())
        content = render_interview_report_markdown(record, record.analysis, profile)
        assert "(minor weight)" in content

    def test_renders_overall_competency_score(self, tmp_path):
        analysis = self._analysis(session_summary={
            "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
            "competency_scores": [
                {"name": "Leadership", "score": 80, "remark": ""},
                {"name": "Execution", "score": 40, "remark": ""},
            ],
            "hire_recommendation": {"level": "Hire", "rationale": ""},
        })
        record = _record(1, "Zoom", analysis)
        content = write_interview_report(record, _config(tmp_path)).read_text()
        assert "Overall competency score" in content

    def test_omits_overall_competency_score_when_no_scores(self, tmp_path):
        analysis = self._analysis(session_summary={
            "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
            "hire_recommendation": {"level": "Hire", "rationale": ""},
        })
        record = _record(1, "Zoom", analysis)
        content = write_interview_report(record, _config(tmp_path)).read_text()
        assert "Overall competency score" not in content


class TestTrendsAreScopedPerUser:
    """Regression coverage for a real bug: write_trends_report always
    computed its CONTENT from a user-scoped record list, but wrote it to a
    single shared trends.md -- so whichever profile's interview finished
    processing last silently overwrote every other profile's trends file,
    and any profile's dashboard would read back whoever wrote it last
    regardless of who was actually logged in. Each user must get their own
    file now."""

    def test_different_users_get_different_trends_files(self, tmp_path):
        cfg = _config(tmp_path)
        path_a = write_trends_report([], cfg, user_id=1)
        path_b = write_trends_report([], cfg, user_id=2)

        assert path_a != path_b
        assert path_a.exists()
        assert path_b.exists()

    def test_trends_report_path_is_deterministic_per_user(self, tmp_path):
        cfg = _config(tmp_path)
        assert trends_report_path(cfg, user_id=1) == trends_report_path(cfg, user_id=1)
        assert trends_report_path(cfg, user_id=1) != trends_report_path(cfg, user_id=2)

    def test_writing_one_users_trends_does_not_touch_another_users_file(self, tmp_path):
        cfg = _config(tmp_path)
        record_a = _record(1, "Zoom", {
            "qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": ["Rambling"],
                                                  "one_thing_to_practice_next": ""},
        })
        write_trends_report([record_a], cfg, user_id=1)
        path_b = write_trends_report([], cfg, user_id=2)

        content_a = trends_report_path(cfg, user_id=1).read_text(encoding="utf-8")
        content_b = path_b.read_text(encoding="utf-8")
        assert "Rambling" in content_a
        assert "Rambling" not in content_b

    def test_no_user_id_falls_back_to_the_bare_shared_filename(self, tmp_path):
        """Only relevant for contexts with no login concept at all -- not
        used by the normal tray+dashboard app, which always has a user_id."""
        cfg = _config(tmp_path)
        path = trends_report_path(cfg, user_id=None)
        assert path.name == "trends.md"


class TestRenderInterviewReportMarkdown:
    """The in-memory (no file written) rendering used to preview a past
    assessment from db.AnalysisHistoryRecord -- must produce identical
    content to write_interview_report for the same (record, analysis),
    since it's the same underlying builder."""

    def test_matches_the_written_report_content(self, tmp_path):
        analysis = {
            "qa_pairs": [{"question": "Q1", "answer_summary": "A1", "issues": [], "suggested_improvement": ""}],
            "session_summary": {"top_strengths": ["Clear"], "top_issues": [], "one_thing_to_practice_next": ""},
        }
        record = _record(1, "Zoom", analysis)

        written_content = write_interview_report(record, _config(tmp_path)).read_text(encoding="utf-8")
        rendered = render_interview_report_markdown(record, analysis)

        assert rendered == written_content

    def test_can_render_a_different_analysis_than_the_records_own(self, tmp_path):
        """The whole point: render a *historical* analysis, not necessarily
        record.analysis itself."""
        record = _record(1, "Zoom", analysis={"qa_pairs": [], "session_summary": {}})
        different_analysis = {
            "qa_pairs": [],
            "session_summary": {
                "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "",
                "hire_recommendation": {"level": "Strong Hire", "rationale": "From a past attempt."},
            },
        }

        rendered = render_interview_report_markdown(record, different_analysis)

        assert "Strong Hire" in rendered
        assert "From a past attempt." in rendered

    def test_writes_nothing_to_disk(self, tmp_path):
        record = _record(1, "Zoom", analysis={"qa_pairs": [], "session_summary": {}})
        output_dir = tmp_path / "output"

        render_interview_report_markdown(record, {"qa_pairs": [], "session_summary": {}})

        assert not output_dir.exists()
