# Interview Analyzer

Automatically records your **online interview calls** (Teams, Google Meet, Webex,
Zoom, Amazon Chime — app or browser, any of them) on Windows, transcribes them,
and produces a structured feedback report highlighting communication and answer
-quality issues — free by default, fully local, no meeting-bot APIs required.

It does **not** join your meeting as a bot/participant, and it does **not**
record silently. It watches for a known conferencing app/tab to start,
**asks your explicit permission before recording** (not every call in these
apps is an interview), and only then records your system audio in the
background. When the app closes, it automatically transcribes, analyzes,
deletes the raw audio after N days (configurable), and keeps a lightweight
permanent record of the analysis so it can track recurring issues across
many interviews over time — scoped to a local login profile.

> ⚠️ **Consent**: Recording a call may be legally regulated depending on your
> location (e.g., two-party consent laws, GDPR). The in-app prompt covers
> *your* decision to record on *your* machine; whether you also need to tell
> the interviewer is a separate legal question — see `docs/consent.md`.

---

## How it works

```
watcher.py     →  detects meeting app/tab running
                       │
                       ▼
consent.py     →  pop-up: "Record this call for interview analysis?"
                   (times out to "No" if unanswered — never records silently)
                       │  Yes
                       ▼
recorder.py    →  starts system-audio (loopback) + mic recording
control_panel.py→ small always-on-top Pause/Resume/Stop control shown for
                   the duration of the recording
                       │  (compressed mono opus, ~1MB/min)
                       ▼
                  on app close (or clicking Stop) → recording stops
                  automatically -- no manual step needed to end capture
                       │
                       ▼
transcriber.py →  faster-whisper (local, free) + speaker diarization
                       │
                       ▼
analyzer.py    →  pluggable engine (local Ollama by default, free; or your
                   own Anthropic/OpenAI API key; or a custom engine you
                   register) scores each Q&A pair against a rubric:
                   structure, clarity, specificity, confidence, technical
                   accuracy -- shows live "Analyzing… N%" progress, and
                   asks the model for its own confidence + feeds in any
                   calibration notes from your past feedback (see below)
                       │
                       ▼
confidence.py  →  a confidence score for the assessment: your own
                   feedback-based accuracy track record once you've rated
                   enough reports, else the model's own self-reported figure
                       │
                       ▼
db.py          →  SQLite, scoped to your local login profile: stores
                   transcript + analysis JSON permanently, stores audio
                   file path + expiry date, stores your feedback ratings
                       │
                       ▼
cleanup.py     →  deletes audio files older than `retention_days` (default 3)
                       │
                       ▼
report.py      →  writes a markdown report per interview + an updated
                   cross-interview "recurring issues" trend report
```

Only the **transcript text and analysis JSON** are kept long-term (a few KB
per interview). Raw audio is deleted automatically after the configurable
retention window — it's only needed transiently for transcription.

### Recording controls

Once you say "Yes" to the consent prompt, a small always-on-top control
panel appears for the duration of the call:

- **Pause** — stops writing audio to disk (e.g. if the conversation turns
  personal); the underlying capture keeps running so resuming is instant.
  Paused segments are simply omitted from the recording, not stored anywhere.
- **Resume** — starts writing again.
- **Stop** — ends the recording immediately and kicks off transcription,
  analysis, and report generation right away, without waiting for the
  meeting app itself to close. Closing the panel's titlebar (✕) does the
  same thing, so there's no way to be left with an invisible in-progress
  recording.

The same Pause/Resume/Stop actions are also always available from the
**tray icon's menu** and the **dashboard's Status tab** (see "Run" below)
— all three surfaces control the same recording, so use whichever's handy.

Recording also **stops automatically** the moment the watcher detects the
meeting app/tab has closed — you never have to remember to turn it off.

---

## Requirements (all free)

- Windows 10/11
- Python 3.10+
- [Ollama](https://ollama.com) installed locally -- the app offers to pull
  a model for you on first run (see "First run" below), so a manual
  `ollama pull` beforehand is optional, not required
- A WASAPI loopback-capable audio backend (handled via `pyaudiowpatch`,
  no extra driver install needed on Windows 10/11)

No paid API keys are required for the default configuration. You can
optionally swap the analyzer to use a hosted LLM API by editing
`config/config.yaml` (or the Settings tab) — see `docs/using_cloud_apis.md`.

---

## Setup

```bash
git clone https://github.com/gargbrt/interview_analyser.git
cd interview_analyser
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

That's it — no model pull needed beforehand; the app walks you through that
on first launch (next section). If you'd rather do it upfront anyway:
`ollama pull llama3.1:8b` (~4.7GB).

Edit `config/config.yaml` (or the dashboard's Settings tab, once running) to adjust:
- `retention_days` — how long raw audio is kept before auto-deletion (default: 3)
- `watched_processes` — which apps/browser tabs trigger recording
- `whisper_model` — transcription model size (`tiny`/`base`/`small`/`medium`)
- `transcription.language` — `auto`/`en`/`hi`/`hinglish` (see `docs/language_support.md`)
- `llm_model` — which local Ollama model to use for analysis
- `output_dir` — where markdown reports are written

## Run

```bash
python -m interview_analyzer.app
```

This is the normal way to run it: a **system tray icon** plus a **dashboard
window**, both described below.

1. **Login dialog** — create/select your local profile (just a name,
   password optional). Use `--username yourname` to skip it.
2. **Dashboard** opens automatically.
3. **First run only**: an **analysis model setup** dialog, if the
   configured local model (`llama3.1:8b` by default) isn't already
   installed — pick a model (each shows its approximate download size),
   install it locally with a live progress bar, or skip and configure a
   cloud API key instead (`docs/using_cloud_apis.md`). This is a one-time
   prompt — see "Analysis model setup" below for what happens after.

Leave it running in the background — it does nothing until it detects a
meeting app, at which point it **asks you** whether to record (see
`docs/consent.md`). Only on "Yes" does it record, analyze, and report, fully
automatically from there. No manual step is required per interview beyond
that one Yes/No. If automatic detection ever misses a real call, the Status
tab (and the tray menu) has a manual **"Start recording"** fallback that
skips straight to recording — clicking it is treated as your consent.

Reports land in `output_dir/reports/<date>_<app>.md`. A continuously updated
`output_dir/trends.md` tracks recurring issues across all interviews for
your profile — both are also browsable right in the dashboard. The Status
tab also has an **"Open recordings folder"** button that opens the raw
audio folder directly in Explorer.

### Analysis model setup

The local Ollama model is a multi-gigabyte download, so the app always asks
before pulling one — both the first time (above) and any later time you
change it:

- **Settings tab → Model name**: pick from a dropdown of curated models
  (each with its approximate size and a one-line tradeoff description --
  e.g. `llama3.1:8b` ~4.7GB fast/default, `qwen2.5:14b` ~9GB better
  reasoning, `llama3.2:3b` ~2GB fastest/lower quality) or type any other
  Ollama model name. **Install model...** downloads whatever's currently
  typed/selected, always confirming the size first. Saving Settings with a
  model that isn't installed yet also prompts before it would let analysis
  fail silently.
- Nothing is ever downloaded without an explicit "yes" click.

### Feedback & confidence scoring

Every report has a **Feedback** panel (History tab → select an interview)
to rate whether the transcription and analysis were accurate, with an
optional comment. This calibrates a **confidence score** shown on every
report going forward (your own accuracy track record once you've rated a
few, the model's own self-reported confidence before that) and feeds
corrective notes from negative feedback back into future analysis prompts.
See `docs/feedback_and_confidence.md` for details.

### Language packs

`transcription.language` (Settings tab, or `config.yaml`) supports
`auto`/`en`/`hi`/`hinglish`. The Hindi/Hinglish romanization pack is
optional and installable/uninstallable any time from Settings → Language
packs (or `pip install/uninstall indic-transliteration` by hand). See
`docs/language_support.md` for Whisper's language coverage and constraints.

### The tray icon

Always visible once the app is running. The dot color/shape shows current
state (grey = idle, red = recording, amber with pause bars = paused).
Right/left-click for the menu:

- current status (e.g. "Recording — Zoom")
- **Pause recording** / **Resume recording** (only shown while recording)
- **Stop recording** (only shown while recording — ends the call immediately
  and runs the report pipeline, same as the in-call control panel's Stop)
- **Open dashboard**
- **Quit**

### The dashboard

Opens automatically on first launch, and any time from the tray icon. Four
tabs:

- **Status** — current state and the same Pause/Resume/Stop controls as the
  tray menu and the in-call control panel, a manual "Start recording"
  fallback, and an "Open recordings folder" button.
- **History** — every past interview (date, app, top issue); select one to
  read its full report right there, nicely formatted, no need to dig
  through the `output/reports/` folder. Below the report: a **Feedback**
  panel (see "Feedback & confidence scoring" above) and the report's
  confidence score.
- **Trends** — the recurring-issues report across all your interviews.
- **Settings** — edit the most common `config.yaml` options (retention
  days, poll interval, Whisper model, language, diarization, analysis
  engine/model, reports folder) from a form instead of a text editor, plus
  **Install model...** (with a curated model-size catalog) and
  **Language packs** install/uninstall. Saving preserves every comment in
  `config.yaml`; a restart picks up the new values. Less common settings
  (like the watched-app list) still need a text editor.

### Headless mode (no GUI)

For an unattended server-style setup with no tray icon, no dashboard, and a
console login prompt instead — e.g. a bare scheduled task — run
`python -m interview_analyzer.watcher` instead. The consent prompt and the
in-call pause/resume/stop control panel still appear either way (they're
independent of which entry point started the watcher); see
`docs/run_at_startup.md`.

---

## Project layout

```
config/config.yaml          - all user settings
src/interview_analyzer/
  app.py                      - GUI entry point: login dialog + tray + dashboard
  watcher.py                   - detects meeting app start/stop, drives the pipeline
  tray.py                       - system tray icon (status, pause/resume/stop, quit)
  dashboard.py                   - dashboard window (status/history/trends/settings tabs)
  login_dialog.py                 - GUI login dialog (app.py's counterpart to auth.py's console prompt)
  single_instance.py                - prevents two copies of the app running at once
  settings_editor.py                  - comment-preserving config.yaml edits for the Settings tab
  model_setup.py                        - first-run + on-demand local model install (size-confirmed downloads)
  language_packs.py                       - optional per-language transcription packs, install/uninstall any time
  report_view.py                            - renders report/trend markdown into the dashboard's Text widgets
  consent.py                                  - pop-up permission prompt before recording
  auth.py                                       - local login/profile system
  recorder.py                                     - system-audio loopback recording (with pause/resume)
  control_panel.py                                  - Pause/Resume/Stop control shown during recording
  compress.py                                         - shrinks WAV to small opus/mp3
  transcriber.py                                        - faster-whisper transcription + diarization + language packs
  engines.py                                              - pluggable AnalysisEngine base + registry
  analyzer.py                                               - built-in engines (ollama/anthropic/openai) + rubric runner
  rubric.py                                                   - the evaluation rubric/prompts (editable)
  confidence.py                                                 - confidence scoring + feedback-based calibration notes
  db.py                                                           - SQLite storage layer (per-user scoped, thread-safe)
  cleanup.py                                                        - retention/auto-delete of audio
  report.py                                                           - per-interview + trend markdown reports
  config_loader.py                                                      - loads config.yaml
tests/                        - automated test suite (180+ tests; see "Testing" below)
```

## Testing

The suite covers auth/login, the DB layer (including retention expiry,
per-user scoping, and feedback storage), cleanup, the pluggable
analyzer/engine registry, confidence scoring/calibration, report generation,
recorder pause/resume/stop behavior, watcher status/notify wiring, meeting
detection, the comment-preserving settings editor, model/language-pack
install logic, and a full mocked end-to-end pipeline run (consent → record
→ pause/resume/manual-stop → transcribe → analyze → report → trend update →
cleanup).

The tray icon and dashboard window are also manually verified end-to-end on
a real Windows session (not just unit-tested): constructing and running the
actual `pystray` icon, opening the actual dashboard window and driving its
widgets, editing and saving a real settings form, and confirming rendered
report/trend content — since none of that renders meaningfully under a
mocked Tk/pystray backend.

**What's verified automatically:** all orchestration logic, file lifecycle,
consent gating (including "don't re-prompt for the same ongoing call"),
retention deletion, and report/trend content.

**What's *not* verified automatically** (needs a real Windows machine with a
live call, which isn't available in CI/dev sandboxes): actual WASAPI
loopback audio capture, actual faster-whisper transcription quality, and
actual Ollama/API responses. Those three boundaries are mocked in tests —
please treat first real-world runs as a manual verification step, and open
an issue if `recorder.py`'s device selection doesn't work on your audio
setup (this is the most hardware-dependent part).

Run tests with pytest (recommended, once installed):
```bash
pip install pytest
pytest tests/
```
Or, dependency-free, with the included minimal runner:
```bash
python tests/run_tests.py
```

## Customizing the rubric

`src/interview_analyzer/rubric.py` contains the prompt and scoring categories
sent to the LLM per answer (structure, clarity, specificity, confidence,
technical accuracy). Edit this file to add categories relevant to your field
(e.g., system design depth, SQL correctness) — it's plain text, no code
changes needed elsewhere.

## Contributing

Issues and PRs welcome. Particularly useful contributions: Linux/macOS
watcher support, additional conferencing-app process signatures, better
diarization, a small web UI for browsing reports.

## License

MIT
