import json
from unittest.mock import MagicMock, patch

import pytest

from interview_analyzer import api_keys
from interview_analyzer.analyzer import AnthropicEngine, GroqEngine, OllamaEngine, OpenAIEngine, analyze_transcript
from interview_analyzer.config_loader import Config
from interview_analyzer.engines import AnalysisEngine, get_engine, register_engine
from interview_analyzer.rubric import RESULT_JSON_SCHEMA


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


class FakeWrongShapeEngine(AnalysisEngine):
    """Simulates a real failure mode reproduced on an actual interview:
    llama3.1:8b returned syntactically valid JSON that completely ignored
    the requested qa_pairs/session_summary schema -- a generic-looking
    {"title": ..., "topics": [...]} object instead. This used to sail
    through as a "successful" analysis and produce a blank report with no
    way to reprocess it."""

    def run(self, prompt: str) -> str:
        return json.dumps({
            "title": "Product Management Interview",
            "speaker": "Interviewee",
            "topics": ["product roadmap", "market analysis"],
            "time": 15,
            "date": "2023-03-16",
        })


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


def test_analyzer_treats_wrong_shaped_json_the_same_as_a_parse_error():
    register_engine("fake_wrong_shape", lambda acfg: FakeWrongShapeEngine())
    cfg = _make_config("fake_wrong_shape")

    result = analyze_transcript("[Interviewer] Hi\n[You] Hello", cfg)

    assert result["parse_error"] is True
    assert "raw" in result
    assert json.loads(result["raw"])["title"] == "Product Management Interview"


def test_analyzer_accepts_an_empty_qa_pairs_list_as_valid_shape():
    """qa_pairs=[] is legitimate (e.g. a very short call) -- must not be
    mistaken for the wrong-shape case just because it's empty."""
    class _EmptyButValidEngine(AnalysisEngine):
        def run(self, prompt: str) -> str:
            return json.dumps({"qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": []}})

    register_engine("fake_empty_valid", lambda acfg: _EmptyButValidEngine())
    cfg = _make_config("fake_empty_valid")

    result = analyze_transcript("[Interviewer] Hi\n[You] Hello", cfg)

    assert result.get("parse_error") is not True
    assert result["qa_pairs"] == []


def _ndjson_lines(*objs) -> list[bytes]:
    return [json.dumps(o).encode("utf-8") for o in objs]


class TestOllamaEngineStreamingProgress:
    """Ollama's streaming /api/generate endpoint reports eval_count (tokens
    generated so far) per NDJSON line -- used to estimate an analyzing %
    for the dashboard, the same way transcription reports real progress.

    Every test here patches ensure_ollama_running to True since it's called
    at the top of OllamaEngine.run() -- see TestOllamaEngineAutoStart for
    coverage of ensure_ollama_running itself."""

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

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=True), \
             patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            progress_calls = []
            raw = engine.run("How many questions? ", on_progress=progress_calls.append)

        assert raw == '{"qa_pairs": []}'
        assert progress_calls[-1] == 1.0
        assert progress_calls == sorted(progress_calls)  # monotonically increasing
        assert mock_post.call_args.kwargs["json"]["format"] == RESULT_JSON_SCHEMA
        assert mock_post.call_args.kwargs["json"]["stream"] is True

    def test_without_on_progress_uses_the_plain_non_streaming_call(self):
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"response": '{"qa_pairs": []}'}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=True), \
             patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            raw = engine.run("prompt")

        assert raw == '{"qa_pairs": []}'
        assert mock_post.call_args.kwargs["json"]["stream"] is False

    def test_sends_the_rubric_json_schema_not_just_a_bare_json_format_flag(self):
        """Regression coverage for a real bug: asking Ollama for merely
        "format": "json" only guarantees *some* valid JSON, not the
        expected shape -- reproduced on a real interview, where the model
        returned well-formed but completely unrelated JSON. Passing the
        full schema as `format` constrains decoding to actually match it
        (verified empirically against a real Ollama server)."""
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"response": '{"qa_pairs": []}'}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=True), \
             patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            engine.run("prompt")

        assert mock_post.call_args.kwargs["json"]["format"] == RESULT_JSON_SCHEMA

    def test_analyze_transcript_passes_progress_callback_through_to_ollama(self):
        cfg = Config(raw={"analysis": {"engine": "ollama"}})
        response_json = '{"qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": []}}'
        lines = _ndjson_lines({"response": response_json, "eval_count": 100, "done": True})
        fake_resp = MagicMock()
        fake_resp.iter_lines.return_value = lines
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=True), \
             patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp):
            progress_calls = []
            result = analyze_transcript(
                "[Interviewer] Hi\n[You] Hello", cfg, on_progress=progress_calls.append
            )

        assert result == {"qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": []}}
        assert progress_calls[-1] == 1.0


class TestOllamaEngineAutoStart:
    """OllamaEngine.run() checks/starts Ollama before making any request --
    see model_setup.ensure_ollama_running. Ollama doesn't auto-start with
    the OS by default, so without this a machine reboot (or Ollama simply
    not having been launched yet) surfaces as a raw ConnectionError instead
    of the app just handling it."""

    def test_run_starts_ollama_automatically_before_the_request(self):
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"response": '{"qa_pairs": []}'}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=True) as mock_ensure, \
             patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp):
            raw = engine.run("prompt")

        assert raw == '{"qa_pairs": []}'
        mock_ensure.assert_called_once_with("http://localhost:11434")

    def test_run_raises_a_clear_error_when_ollama_cannot_be_started(self):
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=False), \
             patch("interview_analyzer.analyzer.requests.post") as mock_post:
            with pytest.raises(RuntimeError, match="couldn't be started automatically"):
                engine.run("prompt")

        mock_post.assert_not_called()


class TestCloudEngineApiKeyResolution:
    """A claude.ai/ChatGPT *subscription* does not grant API access -- both
    cloud engines need a real API key, resolved from either the environment
    variable (the original mechanism) or a key saved via the Settings tab's
    "Cloud API key" section (api_keys.py), with the env var taking
    precedence if both are set.

    api_keys.load_key() itself is mocked directly (rather than exercising
    its real Windows-DPAPI/macOS-Keychain storage backend, which
    api_keys_test.py already covers) so these tests only verify engine
    resolution logic and stay identical regardless of which OS runs them."""

    def test_anthropic_engine_uses_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "sk-ant-from-env")
        monkeypatch.setattr(api_keys, "load_key", lambda provider: None)
        engine = AnthropicEngine({})
        assert engine.api_key == "sk-ant-from-env"

    def test_anthropic_engine_falls_back_to_saved_key(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: "sk-ant-saved" if provider == "anthropic_api" else None)
        engine = AnthropicEngine({})
        assert engine.api_key == "sk-ant-saved"

    def test_anthropic_engine_env_var_takes_precedence_over_saved_key(self, monkeypatch):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "sk-ant-from-env")
        monkeypatch.setattr(api_keys, "load_key", lambda provider: "sk-ant-saved")
        engine = AnthropicEngine({})
        assert engine.api_key == "sk-ant-from-env"

    def test_anthropic_engine_raises_a_clear_error_with_neither(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: None)
        with pytest.raises(RuntimeError, match="claude.ai subscription"):
            AnthropicEngine({})

    def test_openai_engine_falls_back_to_saved_key(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: "sk-openai-saved" if provider == "openai_api" else None)
        engine = OpenAIEngine({})
        assert engine.api_key == "sk-openai-saved"

    def test_openai_engine_raises_a_clear_error_with_neither(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: None)
        with pytest.raises(RuntimeError, match="ChatGPT subscription"):
            OpenAIEngine({})

    def test_groq_engine_falls_back_to_saved_key(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: "gsk-saved" if provider == "groq" else None)
        engine = GroqEngine({})
        assert engine.api_key == "gsk-saved"

    def test_groq_engine_raises_a_clear_error_with_neither(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: None)
        with pytest.raises(RuntimeError, match="console.groq.com"):
            GroqEngine({})

    def test_groq_engine_defaults_to_llama_4_scout_for_free_tier_rate_limit_headroom(self, monkeypatch):
        """Regression coverage for a real bug: the previous default,
        openai/gpt-oss-20b, is capped at 8K tokens/minute on Groq's free
        tier -- reproduced directly, a single real long-transcript request
        already needed more than that on its own (reasoning tokens alone
        ate a big chunk of the budget), failing with a 413 rate-limit
        error before even accounting for other usage that minute.
        llama-4-scout gets 30K tokens/minute and isn't a reasoning model,
        so it comfortably completes the same request."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({})
        assert engine.model == "meta-llama/llama-4-scout-17b-16e-instruct"


class TestGroqEngineRequest:
    def test_sends_the_rubric_json_schema_in_best_effort_mode(self, monkeypatch):
        """strict=False, not True: strict structured-output mode is only
        supported on Groq's GPT-OSS models (see the class docstring for
        why this engine doesn't default to one of those) -- sending
        strict=True for a model that doesn't support it risks an outright
        API error rather than a graceful downgrade. The rubric shape is
        still requested via the schema; analyze_transcript()'s own
        validation is the actual safety net for a non-compliant response."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": '{"qa_pairs": []}'}}]}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            raw = engine.run("prompt")

        assert raw == '{"qa_pairs": []}'
        sent = mock_post.call_args.kwargs["json"]
        assert sent["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
        assert sent["response_format"]["type"] == "json_schema"
        assert sent["response_format"]["json_schema"]["strict"] is False
        assert sent["response_format"]["json_schema"]["schema"] == RESULT_JSON_SCHEMA
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer gsk-from-env"

    def test_a_404_gets_a_clear_hint_about_the_model_name_not_a_bare_httperror(self, monkeypatch):
        """Regression coverage for a real, confusing failure: switching
        analysis.engine to "groq_api" doesn't reset analysis.llm_model
        (shared across all four engines) -- a name left over from Ollama
        (e.g. "llama3.1:8b") isn't a valid Groq model id, and reproduced
        directly, Groq's plain 404 for that gave no hint why."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({"llm_model": "llama3.1:8b"})
        fake_resp = MagicMock()
        fake_resp.status_code = 404

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp):
            with pytest.raises(RuntimeError, match="llama3.1:8b"):
                engine.run("prompt")

    def test_sets_a_generous_max_tokens_for_the_reasoning_model(self, monkeypatch):
        """Regression coverage for a real bug: GPT-OSS is a reasoning
        model that spends real tokens "thinking" before answering
        (reproduced directly: 2790 reasoning tokens on one real
        transcript) -- without an explicit, generous max_tokens, the
        reasoning phase alone could exhaust Groq's default budget, leaving
        nothing for the actual JSON answer and causing analysis to fail
        outright on a long transcript."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            engine.run("prompt")

        assert mock_post.call_args.kwargs["json"]["max_tokens"] >= 8000

    def test_uses_a_custom_model_when_configured(self, monkeypatch):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({"llm_model": "llama-3.1-8b-instant"})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            engine.run("prompt")

        assert mock_post.call_args.kwargs["json"]["model"] == "llama-3.1-8b-instant"
