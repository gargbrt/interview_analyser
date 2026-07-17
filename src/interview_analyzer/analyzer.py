"""Runs the rubric-based analysis over a transcript using a pluggable
AnalysisEngine (see engines.py). Ships with three built-in engines:

  - "ollama"        (default) - free, fully local. Best free-tier quality
                       tradeoff is Llama 3.1 8B for speed, or Qwen2.5 14B
                       for noticeably better reasoning if your machine can
                       run it (see config.yaml comment).
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
from .rubric import build_prompt

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

        if on_progress is None:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=600,
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
            json={"model": self.model, "prompt": prompt, "stream": True, "format": "json"},
            timeout=600,
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


register_engine("ollama", lambda acfg: OllamaEngine(acfg))
register_engine("anthropic_api", lambda acfg: AnthropicEngine(acfg))
register_engine("openai_api", lambda acfg: OpenAIEngine(acfg))


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
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Analyzer returned non-JSON output; storing raw text under 'raw'.")
        return {"raw": raw, "parse_error": True}
