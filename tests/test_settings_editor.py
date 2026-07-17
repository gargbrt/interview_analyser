"""Settings editor must round-trip config.yaml without clobbering comments
-- the dashboard's Settings tab writes back into the user's real config
file, and config.yaml is full of explanatory comments that would be lost
by a naive `yaml.safe_load` / `yaml.safe_dump` round trip.
"""
from __future__ import annotations

import pathlib

from interview_analyzer.settings_editor import load_editable_settings, save_editable_settings

SAMPLE_CONFIG = """\
# Interview Analyzer configuration
# All settings here are safe defaults for a free, fully-local setup.

# --- Retention ---
# Number of days raw audio recordings are kept before automatic deletion.
retention_days: 3

poll_interval_seconds: 5
start_debounce_polls: 2

audio:
  bitrate_kbps: 64
  format: "opus"

transcription:
  whisper_model: "small"    # tiny|base|small|medium|large-v3
  diarization: true

analysis:
  engine: "ollama"
  llm_model: "llama3.1:8b"

output:
  output_dir: "output"
"""


def _write_sample(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "config.yaml"
    path.write_text(SAMPLE_CONFIG, encoding="utf-8")
    return path


def test_load_editable_settings_reads_known_fields(tmp_path):
    path = _write_sample(tmp_path)
    settings = load_editable_settings(path)

    assert settings["retention_days"] == 3
    assert settings["transcription.whisper_model"] == "small"
    assert settings["analysis.engine"] == "ollama"
    assert settings["output.output_dir"] == "output"


def test_save_editable_settings_updates_values_and_preserves_comments(tmp_path):
    path = _write_sample(tmp_path)

    save_editable_settings(path, {
        "retention_days": 7,
        "analysis.engine": "anthropic_api",
        "transcription.diarization": False,
    })

    content = path.read_text(encoding="utf-8")

    # comments and unrelated formatting survive untouched
    assert "# Interview Analyzer configuration" in content
    assert "# Number of days raw audio recordings are kept before automatic deletion." in content
    assert "# tiny|base|small|medium|large-v3" in content

    # values actually changed
    settings = load_editable_settings(path)
    assert settings["retention_days"] == 7
    assert settings["analysis.engine"] == "anthropic_api"
    assert settings["transcription.diarization"] is False
    # untouched fields stay exactly as they were
    assert settings["transcription.whisper_model"] == "small"
    assert settings["output.output_dir"] == "output"


def test_save_editable_settings_coerces_string_input_from_gui_widgets(tmp_path):
    path = _write_sample(tmp_path)

    # Tkinter Entry/Spinbox widgets hand back strings regardless of the
    # underlying type -- the editor must coerce them itself.
    save_editable_settings(path, {
        "retention_days": "10",
        "poll_interval_seconds": "2.5",
        "transcription.diarization": "false",
    })

    settings = load_editable_settings(path)
    assert settings["retention_days"] == 10
    assert settings["poll_interval_seconds"] == 2.5
    assert settings["transcription.diarization"] is False


def test_save_editable_settings_can_add_a_new_language_key(tmp_path):
    """transcription.language didn't exist in older config.yaml files (the
    Hindi/English/Hinglish language pack added it later) -- saving it must
    add the key rather than requiring it to already be present."""
    path = _write_sample(tmp_path)

    save_editable_settings(path, {"transcription.language": "hinglish"})

    settings = load_editable_settings(path)
    assert settings["transcription.language"] == "hinglish"
    assert settings["transcription.whisper_model"] == "small"  # untouched


def test_save_editable_settings_can_add_a_new_live_during_recording_key(tmp_path):
    """transcription.live_during_recording didn't exist in older
    config.yaml files (added with the live transcription feature) --
    saving it must add the key rather than requiring it to already be
    present, same as transcription.language above."""
    path = _write_sample(tmp_path)

    save_editable_settings(path, {"transcription.live_during_recording": True})

    settings = load_editable_settings(path)
    assert settings["transcription.live_during_recording"] is True
    assert settings["transcription.whisper_model"] == "small"  # untouched


def test_save_editable_settings_rejects_unknown_field(tmp_path):
    path = _write_sample(tmp_path)
    try:
        save_editable_settings(path, {"not_a_real_setting": 1})
        assert False, "expected ValueError"
    except ValueError:
        pass
