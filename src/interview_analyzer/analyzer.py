"""Runs the rubric-based analysis over a transcript using a pluggable
AnalysisEngine (see engines.py). Ships with four built-in engines:

  - "ollama"        (default) - free, fully local. Best free-tier quality
                       tradeoff is Llama 3.1 8B for speed, or Qwen2.5 14B
                       for noticeably better reasoning if your machine can
                       run it (see config.yaml comment).
  - "groq_api"       - bring your own (free, no-credit-card) Groq API key.
                       Runs open models on Groq's own fast hardware --
                       much faster than local Ollama on a CPU-only
                       machine, but your transcript leaves your machine.
  - "anthropic_api"  - bring your own Anthropic API key (NOT the same as a
                       claude.ai subscription - this is billed separately
                       per token via console.anthropic.com).
  - "openai_api"     - bring your own OpenAI API key.

To add a different engine entirely, see engines.py / docs/using_cloud_apis.md.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Callable, Optional

import requests

from . import api_keys
from .config_loader import Config
from .engines import AnalysisEngine, get_engine, register_engine
from .model_setup import ensure_ollama_running
from .rubric import RESULT_JSON_SCHEMA, build_prompt

logger = logging.getLogger(__name__)


class OllamaEngine(AnalysisEngine):
    def __init__(self, acfg: dict):
        self.host = acfg.get("ollama_host", "http://localhost:11434")
        self.model = acfg.get("llm_model", "llama3.1:8b")

    def run(self, prompt: str, on_progress: Optional[Callable[[float], None]] = None) -> str:
        # Ollama is a separate local background service that doesn't
        # auto-start with the OS by default -- without this, a machine
        # reboot (or Ollama just not being launched yet) turns into a
        # confusing raw ConnectionError here instead of just working.
        if not ensure_ollama_running(self.host):
            raise RuntimeError(
                f"Ollama isn't running and couldn't be started automatically at {self.host}. "
                "Install it from https://ollama.com, or start it manually, then try again."
            )

        # A full JSON Schema (not just the string "json") -- constrains
        # Ollama's decoding to actually match the rubric's shape, not just
        # produce *some* valid JSON. See rubric.py's RESULT_JSON_SCHEMA
        # docstring for why: a plain "json" format let the model return a
        # syntactically valid but completely unrelated object on a real
        # long transcript, producing a blank report with no clear error.
        # Constrained decoding is measurably slower than free-form
        # generation (grammar-checked per token), hence the longer timeout
        # than a plain generation call would need.
        if on_progress is None:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "format": RESULT_JSON_SCHEMA},
                timeout=1200,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

        # Streaming mode: Ollama's rubric response is fixed-shape JSON, so
        # the number of tokens generated so far (eval_count) against a rough
        # expected-size estimate makes a reasonable progress signal -- an
        # estimate, not exact, same spirit as transcription's real
        # segment-timestamp progress but for a single LLM call there's no
        # exact total to report against ahead of time.
        expected_tokens = max(300, prompt.count("?") * 180 + 150)
        chunks = []
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True, "format": RESULT_JSON_SCHEMA},
            timeout=1200,
            stream=True,
        )
        resp.raise_for_status()
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunks.append(data.get("response", ""))
                eval_count = data.get("eval_count")
                if eval_count is not None:
                    on_progress(min(eval_count / expected_tokens, 0.97))
                if data.get("done"):
                    on_progress(1.0)
        finally:
            resp.close()
        return "".join(chunks)


class AnthropicEngine(AnalysisEngine):
    """Bring-your-own-API-key Anthropic engine. Requires a real API key from
    console.anthropic.com -- a claude.ai Pro/Max subscription does not grant
    API access, they're billed and provisioned separately.
    """

    def __init__(self, acfg: dict):
        env_var = acfg.get("cloud_api_key_env_var", "INTERVIEW_ANALYZER_API_KEY")
        # the environment variable (the original mechanism) always wins if
        # set; otherwise fall back to a key saved via the Settings tab's
        # "Cloud API key" section (see api_keys.py -- encrypted at rest,
        # never plaintext)
        self.api_key = os.environ.get(env_var) or api_keys.load_key("anthropic_api")
        if not self.api_key:
            raise RuntimeError(
                f"No Anthropic API key found. Set it in the Settings tab's \"Cloud API key\" "
                f"section, or set the {env_var} environment variable, to a real API key from "
                "console.anthropic.com (not your claude.ai subscription -- that doesn't grant API access)."
            )
        self.model = acfg.get("llm_model", "claude-sonnet-5")

    def run(self, prompt: str, on_progress: Optional[Callable[[float], None]] = None) -> str:
        # No incremental progress signal for this engine -- a single
        # non-streaming request; on_progress is accepted for interface
        # consistency but left unused, same as any engine that has nothing
        # partial to report.
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=600,
        )
        resp.raise_for_status()
        content = resp.json()["content"]
        return "".join(b["text"] for b in content if b.get("type") == "text")


class OpenAIEngine(AnalysisEngine):
    def __init__(self, acfg: dict):
        env_var = acfg.get("cloud_api_key_env_var", "INTERVIEW_ANALYZER_API_KEY")
        self.api_key = os.environ.get(env_var) or api_keys.load_key("openai_api")
        if not self.api_key:
            raise RuntimeError(
                f"No OpenAI API key found. Set it in the Settings tab's \"Cloud API key\" "
                f"section, or set the {env_var} environment variable, to a real API key from "
                "platform.openai.com (not your ChatGPT subscription -- that doesn't grant API access)."
            )
        self.model = acfg.get("llm_model", "gpt-4o-mini")

    def run(self, prompt: str, on_progress: Optional[Callable[[float], None]] = None) -> str:
        # No incremental progress signal for this engine -- see AnthropicEngine.run.
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class GroqEngine(AnalysisEngine):
    """Bring-your-own-API-key Groq engine. Unlike the other two cloud
    engines, Groq's free tier (console.groq.com/keys, no credit card) is
    genuinely usable, not just a trial -- it runs open models (Llama,
    GPT-OSS, Qwen, DeepSeek) on Groq's own fast hardware. The real
    tradeoff versus Ollama isn't cost, it's privacy: your transcript
    leaves your machine and goes to Groq's servers.

    Defaults to openai/gpt-oss-20b rather than a Llama model because Groq's
    *strict* structured-output mode (which guarantees the response matches
    RESULT_JSON_SCHEMA exactly, instead of just being valid-JSON-shaped --
    see rubric.py) is currently only supported on the GPT-OSS models. Even
    if a user points this at a model that doesn't support strict mode,
    analyze_transcript()'s own shape validation (see
    _has_the_expected_shape) still catches a malformed response and marks
    it reprocessable rather than silently shipping a blank report -- same
    safety net as the Ollama engine.
    """

    def __init__(self, acfg: dict):
        env_var = acfg.get("cloud_api_key_env_var", "INTERVIEW_ANALYZER_API_KEY")
        self.api_key = os.environ.get(env_var) or api_keys.load_key("groq")
        if not self.api_key:
            raise RuntimeError(
                f"No Groq API key found. Get a free one (no credit card) at "
                f"https://console.groq.com/keys, then set it in the Settings tab's "
                f"\"Cloud API key\" section, or set the {env_var} environment variable."
            )
        self.model = acfg.get("llm_model", "openai/gpt-oss-20b")

    def run(self, prompt: str, on_progress: Optional[Callable[[float], None]] = None) -> str:
        # No incremental progress signal for this engine -- see AnthropicEngine.run.
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "interview_analysis",
                        "strict": True,
                        "schema": RESULT_JSON_SCHEMA,
                    },
                },
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


register_engine("ollama", lambda acfg: OllamaEngine(acfg))
register_engine("anthropic_api", lambda acfg: AnthropicEngine(acfg))
register_engine("openai_api", lambda acfg: OpenAIEngine(acfg))
register_engine("groq_api", lambda acfg: GroqEngine(acfg))


def _has_the_expected_shape(parsed: dict) -> bool:
    """True if `parsed` actually looks like the rubric's qa_pairs +
    session_summary schema, not just any valid JSON. A real failure mode
    (reproduced on a real interview: a long transcript against
    llama3.1:8b) is the model returning well-formed JSON that ignores the
    requested schema entirely -- e.g. a generic {"title": ..., "topics":
    [...]} object -- which used to sail through as a "successful" analysis
    and silently produce a blank report with no way to reprocess it (see
    dashboard.py's can_reprocess/history_status_label, which key off this
    same parse_error flag)."""
    return isinstance(parsed.get("qa_pairs"), list) and isinstance(parsed.get("session_summary"), dict)


def analyze_transcript(
    transcript: str,
    cfg: Config,
    on_progress: Optional[Callable[[float], None]] = None,
    calibration_notes: str = "",
) -> dict:
    acfg = cfg.analysis
    engine_name = acfg.get("engine", "ollama")
    engine = get_engine(engine_name, acfg)

    prompt = build_prompt(transcript, calibration_notes=calibration_notes)
    try:
        raw = engine.run(prompt, on_progress=on_progress)
    except TypeError:
        # a third-party engine registered before on_progress existed, or
        # that simply doesn't implement it -- fall back to the old call
        # shape rather than breaking custom engines (see engines.py)
        raw = engine.run(prompt)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Analyzer returned non-JSON output; storing raw text under 'raw'.")
        return {"raw": raw, "parse_error": True}

    if not isinstance(parsed, dict) or not _has_the_expected_shape(parsed):
        logger.warning(
            "Analyzer returned valid JSON but not the expected qa_pairs/session_summary "
            "shape; treating it the same as a parse failure so it can be reprocessed."
        )
        return {"raw": raw, "parse_error": True}

    return parsed
