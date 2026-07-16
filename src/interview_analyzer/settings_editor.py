"""Reads/writes a whitelisted subset of config.yaml's scalar settings for
the dashboard's Settings tab, preserving the file's comments and layout.

Only leaf values under known keys are ever touched -- the document
structure (including every explanatory comment) is otherwise left exactly
as the user wrote it. This intentionally does not support adding/removing
watched apps or other list-shaped settings; those still need a text editor.
"""
from __future__ import annotations

import pathlib
from typing import Any

from ruamel.yaml import YAML

_yaml = YAML()
_yaml.preserve_quotes = True

# (path-of-keys, value type) for every setting the Settings tab can edit.
EDITABLE_FIELDS: list[tuple[tuple[str, ...], type]] = [
    (("retention_days",), int),
    (("poll_interval_seconds",), float),
    (("start_debounce_polls",), int),
    (("browser_start_debounce_polls",), int),
    (("stop_debounce_polls",), int),
    (("declined_cooldown_seconds",), float),
    (("audio", "bitrate_kbps"), int),
    (("audio", "include_microphone"), bool),
    (("transcription", "whisper_model"), str),
    (("transcription", "diarization"), bool),
    (("transcription", "language"), str),
    (("analysis", "engine"), str),
    (("analysis", "llm_model"), str),
    (("output", "output_dir"), str),
]


def _get_nested(doc: Any, path: tuple[str, ...]) -> Any:
    node = doc
    for key in path:
        node = node[key]
    return node


def _set_nested(doc: Any, path: tuple[str, ...], value: Any) -> None:
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def load_editable_settings(config_path: pathlib.Path) -> dict[str, Any]:
    """Return {"dotted.path": value} for every field in EDITABLE_FIELDS
    that's present in the file."""
    with open(config_path, "r", encoding="utf-8") as f:
        doc = _yaml.load(f)

    settings: dict[str, Any] = {}
    for path, _type in EDITABLE_FIELDS:
        try:
            settings[".".join(path)] = _get_nested(doc, path)
        except KeyError:
            continue
    return settings


def save_editable_settings(config_path: pathlib.Path, updates: dict[str, Any]) -> None:
    """Apply {"dotted.path": new_value} on top of the existing config.yaml,
    preserving every comment and the rest of the document untouched."""
    with open(config_path, "r", encoding="utf-8") as f:
        doc = _yaml.load(f)

    by_path = {path: _type for path, _type in EDITABLE_FIELDS}

    for dotted, value in updates.items():
        path = tuple(dotted.split("."))
        if path not in by_path:
            raise ValueError(f"'{dotted}' is not an editable setting.")
        expected_type = by_path[path]
        if expected_type is bool and isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes", "on")
        elif expected_type in (int, float) and isinstance(value, str):
            value = expected_type(value.strip())
        _set_nested(doc, path, value)

    with open(config_path, "w", encoding="utf-8") as f:
        _yaml.dump(doc, f)
