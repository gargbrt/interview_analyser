"""Plugin interface for analysis engines.

Built-in engines: `ollama` (default, free, local), `anthropic_api`,
`openai_api`. To plug in your own (a different provider, a fine-tuned
model, a different prompt strategy), subclass `AnalysisEngine`, implement
`run`, and register it — see docs/using_cloud_apis.md for a full example.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional


class AnalysisEngine(ABC):
    """Base class for a pluggable analysis engine.

    Implementations receive the fully-built rubric prompt (transcript
    already embedded) and must return the raw text response from the
    model — JSON parsing/validation is handled centrally in analyzer.py.

    `on_progress`, if given, may be called with an estimated 0.0-1.0
    fraction as the response streams in (e.g. Ollama's streaming API
    reports tokens generated so far) — purely cosmetic for a progress bar,
    never required. It's optional so third-party engines that only
    implement `run(self, prompt)` keep working unchanged; analyzer.py falls
    back to calling them without it.
    """

    @abstractmethod
    def run(self, prompt: str, on_progress: Optional[Callable[[float], None]] = None) -> str:
        ...


_REGISTRY: dict[str, Callable[[dict], AnalysisEngine]] = {}


def register_engine(name: str, factory: Callable[[dict], AnalysisEngine]) -> None:
    """Register a custom engine factory under `name`, so it can be selected
    via `analysis.engine: "<name>"` in config.yaml.

    Example (in your own module, imported before the watcher starts):

        from interview_analyzer.engines import register_engine, AnalysisEngine

        class MyEngine(AnalysisEngine):
            def run(self, prompt: str) -> str:
                ...  # call your own API / local model

        register_engine("my_engine", lambda acfg: MyEngine())
    """
    _REGISTRY[name] = factory


def get_engine(name: str, acfg: dict) -> AnalysisEngine:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown analysis engine '{name}'. Registered engines: {list(_REGISTRY)}. "
            "Register a custom one with interview_analyzer.engines.register_engine()."
        )
    return _REGISTRY[name](acfg)
