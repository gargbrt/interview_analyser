"""Loads and validates config.yaml, resolving paths relative to project root."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    # the file this was loaded from, if any -- used by the dashboard's
    # Settings tab to know what to write edits back into
    path: Optional[pathlib.Path] = None

    @property
    def retention_days(self) -> int:
        return int(self.raw.get("retention_days", 3))

    @property
    def poll_interval_seconds(self) -> float:
        return float(self.raw.get("poll_interval_seconds", 5))

    @property
    def start_debounce_polls(self) -> int:
        return int(self.raw.get("start_debounce_polls", 2))

    @property
    def watched_processes(self) -> dict[str, list[str]]:
        return self.raw.get("watched_processes", {})

    @property
    def audio(self) -> dict[str, Any]:
        return self.raw.get("audio", {})

    @property
    def transcription(self) -> dict[str, Any]:
        return self.raw.get("transcription", {})

    @property
    def analysis(self) -> dict[str, Any]:
        return self.raw.get("analysis", {})

    @property
    def storage(self) -> dict[str, Any]:
        return self.raw.get("storage", {})

    @property
    def output(self) -> dict[str, Any]:
        return self.raw.get("output", {})

    def resolve(self, relative: str) -> pathlib.Path:
        """Resolve a config-relative path against the project root."""
        p = pathlib.Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_config(path: pathlib.Path | str = DEFAULT_CONFIG_PATH) -> Config:
    path = pathlib.Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw or {}, path=path)
