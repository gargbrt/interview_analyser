import json
from unittest.mock import MagicMock, patch

import pytest

from interview_analyzer import api_keys
from interview_analyzer.analyzer import AnthropicEngine, OllamaEngine, OpenAIEngine, analyze_transcript
from interview_analyzer.config_loader import Config
from interview_analyzer.engines import AnalysisEngine, get_engine, register_engine


class FakeGoodEngine(AnalysisEngine):
    """Simulates an analysis engine that returns well-formed JSON, written
    against the pre-on_progress interface -- exercises analyze_transcript's
    TypeError fallback for third-party engines that predate that parameter."""

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


def _ndjson_lines(*objs) -> list[bytes]:
    return [json.dumps(o).encode("utf-8") for o in objs]


class TestOllamaEngineStreamingProgress:
    """Ollama's streaming /api/generate endpoint reports eval_count (tokens
    generated so far) per NDJSON line -- used to estimate an analyzing %
    for the dashboard, the same way transcription reports real progress."""

    def test_calls_on_progress_with_increasing_fractions_and_reaches_1_0(self):
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})
        lines = _ndjson_lines(
            {"response": '{"qa', "eval_count": 10, "done": False},
            {"response": '_pairs": []}', "eval_count": 400, "done": False},
            {"response": "", "eval_count": 420, "done": True},
        )
        fake_resp = MagicMock()
        fake_resp.iter_lines.return_value = lines
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            progress_calls = []
            raw = engine.run("How many questions? ", on_progress=progress_calls.append)

        assert raw == '{"qa_pairs": []}'
        assert progress_calls[-1] == 1.0
        assert progress_calls == sorted(progress_calls)  # monotonically increasing
        assert mock_post.call_args.kwargs["json"]["stream"] is True

    def test_without_on_progress_uses_the_plain_non_streaming_call(self):
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"response": '{"qa_pairs": []}'}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            raw = engine.run("prompt")

        assert raw == '{"qa_pairs": []}'
        assert mock_post.call_args.kwargs["json"]["stream"] is False

    def test_analyze_transcript_passes_progress_callback_through_to_ollama(self):
        cfg = Config(raw={"analysis": {"engine": "ollama"}})
        lines = _ndjson_lines({"response": '{"qa_pairs": []}', "eval_count": 100, "done": True})
        fake_resp = MagicMock()
        fake_resp.iter_lines.return_value = lines
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp):
            progress_calls = []
            result = analyze_transcript(
                "[Interviewer] Hi\n[You] Hello", cfg, on_progress=progress_calls.append
            )

        assert result == {"qa_pairs": []}
        assert progress_calls[-1] == 1.0


class TestCloudEngineApiKeyResolution:
    """A claude.ai/ChatGPT *subscription* does not grant API access -- both
    cloud engines need a real API key, resolved from either the environment
    variable (the original mechanism) or a key saved via the Settings tab's
    "Cloud API key" section (api_keys.py), with the env var taking
    precedence if both are set."""

    def test_anthropic_engine_uses_env_var_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "sk-ant-from-env")
        with patch.object(api_keys, "_STORE_PATH", tmp_path / ".api_keys.json"):
            engine = AnthropicEngine({})
        assert engine.api_key == "sk-ant-from-env"

    def test_anthropic_engine_falls_back_to_saved_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        with patch.object(api_keys, "_STORE_PATH", tmp_path / ".api_keys.json"):
            api_keys.save_key("anthropic_api", "sk-ant-saved")
            engine = AnthropicEngine({})
        assert engine.api_key == "sk-ant-saved"

    def test_anthropic_engine_env_var_takes_precedence_over_saved_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "sk-ant-from-env")
        with patch.object(api_keys, "_STORE_PATH", tmp_path / ".api_keys.json"):
            api_keys.save_key("anthropic_api", "sk-ant-saved")
            engine = AnthropicEngine({})
        assert engine.api_key == "sk-ant-from-env"

    def test_anthropic_engine_raises_a_clear_error_with_neither(self, monkeypatch, tmp_path):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        with patch.object(api_keys, "_STORE_PATH", tmp_path / ".api_keys.json"):
            with pytest.raises(RuntimeError, match="claude.ai subscription"):
                AnthropicEngine({})

    def test_openai_engine_falls_back_to_saved_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        with patch.object(api_keys, "_STORE_PATH", tmp_path / ".api_keys.json"):
            api_keys.save_key("openai_api", "sk-openai-saved")
            engine = OpenAIEngine({})
        assert engine.api_key == "sk-openai-saved"

    def test_openai_engine_raises_a_clear_error_with_neither(self, monkeypatch, tmp_path):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        with patch.object(api_keys, "_STORE_PATH", tmp_path / ".api_keys.json"):
            with pytest.raises(RuntimeError, match="ChatGPT subscription"):
                OpenAIEngine({})
