import json

import pytest

from interview_analyzer.analyzer import analyze_transcript
from interview_analyzer.config_loader import Config
from interview_analyzer.engines import AnalysisEngine, get_engine, register_engine


class FakeGoodEngine(AnalysisEngine):
    """Simulates an analysis engine that returns well-formed JSON."""

    def run(self, prompt: str) -> str:
        assert "[Interviewer]" in prompt or "Transcript" in prompt
        return json.dumps({
            "qa_pairs": [
                {
                    "question": "Tell me about a conflict you resolved.",
                    "answer_summary": "Rambled without a clear structure or outcome.",
                    "issues": [{"category": "structure", "detail": "No STAR structure used."}],
                    "suggested_improvement": "Lead with the situation, then action, then result.",
                }
            ],
            "session_summary": {
                "top_strengths": ["Clear technical vocabulary"],
                "top_issues": ["Lack of structure in behavioral answers"],
                "one_thing_to_practice_next": "Practice STAR format on 3 behavioral questions.",
            },
        })


class FakeBrokenEngine(AnalysisEngine):
    """Simulates an engine that fails to produce valid JSON."""

    def run(self, prompt: str) -> str:
        return "this is not json"


def _make_config(engine_name: str) -> Config:
    return Config(raw={"analysis": {"engine": engine_name}})


def test_custom_engine_registration_and_selection():
    register_engine("fake_good", lambda acfg: FakeGoodEngine())
    cfg = _make_config("fake_good")

    result = analyze_transcript("[Interviewer] Hi\n[You] Hello", cfg)

    assert "qa_pairs" in result
    assert result["session_summary"]["top_issues"] == ["Lack of structure in behavioral answers"]


def test_analyzer_handles_non_json_response_gracefully():
    register_engine("fake_broken", lambda acfg: FakeBrokenEngine())
    cfg = _make_config("fake_broken")

    result = analyze_transcript("[Interviewer] Hi\n[You] Hello", cfg)

    assert result["parse_error"] is True
    assert "raw" in result


def test_unknown_engine_raises_helpful_error():
    with pytest.raises(ValueError, match="Unknown analysis engine"):
        get_engine("totally_made_up_engine", {})
