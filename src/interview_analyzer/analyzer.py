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

import requests

from .config_loader import Config
from .engines import AnalysisEngine, get_engine, register_engine
from .rubric import build_prompt

logger = logging.getLogger(__name__)


class OllamaEngine(AnalysisEngine):
    def __init__(self, acfg: dict):
        self.host = acfg.get("ollama_host", "http://localhost:11434")
        self.model = acfg.get("llm_model", "llama3.1:8b")

    def run(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


class AnthropicEngine(AnalysisEngine):
    """Bring-your-own-API-key Anthropic engine. Requires a real API key from
    console.anthropic.com -- a claude.ai Pro/Max subscription does not grant
    API access, they're billed and provisioned separately.
    """

    def __init__(self, acfg: dict):
        env_var = acfg.get("cloud_api_key_env_var", "INTERVIEW_ANALYZER_API_KEY")
        self.api_key = os.environ.get(env_var)
        if not self.api_key:
            raise RuntimeError(
                f"Set the {env_var} environment variable to an Anthropic API key "
                "(from console.anthropic.com, not your claude.ai subscription) to use anthropic_api."
            )
        self.model = acfg.get("llm_model", "claude-sonnet-5")

    def run(self, prompt: str) -> str:
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
        self.api_key = os.environ.get(env_var)
        if not self.api_key:
            raise RuntimeError(f"Set the {env_var} environment variable to use openai_api.")
        self.model = acfg.get("llm_model", "gpt-4o-mini")

    def run(self, prompt: str) -> str:
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


def analyze_transcript(transcript: str, cfg: Config) -> dict:
    acfg = cfg.analysis
    engine_name = acfg.get("engine", "ollama")
    engine = get_engine(engine_name, acfg)

    prompt = build_prompt(transcript)
    raw = engine.run(prompt)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Analyzer returned non-JSON output; storing raw text under 'raw'.")
        return {"raw": raw, "parse_error": True}
