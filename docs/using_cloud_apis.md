# Using Cloud APIs or a Custom Analysis Engine

By default, transcription and analysis both run fully free and local (via
faster-whisper and [Ollama](https://ollama.com)). You can swap in a hosted
option instead for either or both:

- **Groq** — genuinely free (no credit card), and much faster than running
  locally on a CPU-only machine, for both transcription and analysis. The
  real tradeoff isn't cost, it's privacy: your audio/transcript leaves this
  machine. See "Free, faster alternative: Groq" below.
- **Anthropic / OpenAI** — bring your own paid API key, for higher-quality
  analysis if you're OK with a small per-analysis cost.
- A completely custom engine you write yourself (analysis only, for now).

## Free, faster alternative: Groq

[Groq](https://groq.com) hosts open models (Whisper, Llama, GPT-OSS, and
others) on its own fast hardware, with a free tier that's actually usable
day to day, not just a trial (no credit card required). It's a meaningfully
different tradeoff than Ollama: you get real speed instead of a slow
CPU-only local model, but your audio and/or transcript is sent to Groq's
servers over the internet, rather than never leaving your machine.

**Setup (one-time):**
1. Create a free account and API key at
   [console.groq.com/keys](https://console.groq.com/keys).
2. In the dashboard's Settings tab → **Cloud API key** section: pick
   `groq`, paste the key, **Save key**. (One key covers both uses below —
   Groq issues one key per account for everything.)
3. Pick which part(s) you want to speed up:
   ```yaml
   transcription:
     engine: "groq"   # instead of "faster-whisper"
   analysis:
     engine: "groq_api"   # instead of "ollama"
   ```
   Or from the Settings tab: **Transcription engine** and **Analysis
   engine** dropdowns. You can switch either one independently — e.g. keep
   transcription local but use Groq for analysis, or vice versa.

**Notes:**
- Transcription defaults to `whisper-large-v3-turbo` (fast; comparable
  accuracy to the local "medium"/"large-v3" models). Dual-channel speaker
  separation (You/Interviewer) works the same way as the local engine —
  each channel is uploaded separately and merged by timestamp.
- Analysis defaults to `openai/gpt-oss-20b` rather than a Llama model,
  because Groq's *strict* structured-output mode (which guarantees the
  response matches the expected rubric shape exactly) is currently only
  supported on the GPT-OSS models — this avoids the malformed-response
  failure mode described below for Ollama. Other Groq models work too, just
  without that guarantee (a malformed response still gets caught and made
  reprocessable rather than silently producing a blank report).
- Live transcription (`transcription.live_during_recording`) only applies
  to the local engine — Groq's hosted API is already fast enough at the end
  of a call that there's nothing useful for it to save.
- Free-tier rate limits apply (requests/tokens per day) — see
  [Groq's docs](https://console.groq.com/docs) for current numbers. If you
  hit them, analysis/transcription will fail with a clear error from Groq's
  API rather than silently hanging.

## Important: a claude.ai subscription is not an API key

If you have a **Claude Pro/Max subscription** (claude.ai / the Claude app),
that does **not** give you API access. API usage is billed separately,
per token, through [console.anthropic.com](https://console.anthropic.com).
You'll need to create an API key there (a few dollars of credit covers a
large number of interview analyses — a single interview's analysis is a few
thousand tokens).

The same applies to OpenAI: a ChatGPT Plus subscription ≠ an OpenAI API key.
You need a key from platform.openai.com.

There is no supported way to "log in" with a claude.ai/ChatGPT account
instead of using a real API key -- no such integration exists for either
provider (and wouldn't be reliable/supported even unofficially), so this
app never asks for your consumer account credentials anywhere.

## Setting your API key (two ways)

**From the dashboard (recommended)** — Settings tab → **Cloud API key**
section: pick the provider (`groq`/`anthropic_api`/`openai_api`), paste the
key, **Save key**. It's encrypted at rest with Windows DPAPI (tied to your
Windows account, never written in plaintext -- same mechanism as the login
dialog's "Remember me", see `api_keys.py`), and never touches
`config.yaml`. **Clear key** removes it.

**Environment variable** (still supported, e.g. for CI/scripted setups; if
both are set, the environment variable wins):
```powershell
setx INTERVIEW_ANALYZER_API_KEY "sk-ant-..."
```
(open a new terminal after `setx` for it to take effect)

Either way, also set the engine in `config/config.yaml` (or the Settings
tab's Analysis engine dropdown):

```yaml
analysis:
  engine: "anthropic_api"       # or "openai_api", or "groq_api" (free)
  llm_model: "claude-sonnet-5"  # or "gpt-4o-mini" for openai_api, "openai/gpt-oss-20b" for groq_api
```

## Getting the best free (local) results

The default `llama3.1:8b` is fast and works on most laptops. If your
machine has headroom (roughly 16GB+ RAM), `qwen2.5:14b` tends to follow
the structured rubric-scoring instructions more reliably:

```bash
ollama pull qwen2.5:14b
```
```yaml
analysis:
  llm_model: "qwen2.5:14b"
```

Or from the dashboard: Settings tab → pick `qwen2.5:14b` from the Model
name dropdown (it lists a small curated catalog with approximate download
sizes) → **Install model...**, which downloads it with a progress bar after
confirming the size with you.

## Writing a completely custom engine

Any provider, any local model, any prompt strategy — implement
`AnalysisEngine` and register it before the watcher starts:

```python
# my_engine.py
from interview_analyzer.engines import AnalysisEngine, register_engine
import requests

class MyEngine(AnalysisEngine):
    def run(self, prompt: str) -> str:
        # call whatever you want; must return the raw text response
        # (the caller will attempt to json.loads() it)
        response = requests.post("https://my-provider/api", json={"prompt": prompt})
        return response.json()["text"]

register_engine("my_engine", lambda acfg: MyEngine())
```

(Optionally implement `run(self, prompt, on_progress=None)` and call
`on_progress(fraction)` as your response streams in, if your provider
supports streaming, for a live "Analyzing… N%" indicator in the dashboard --
see `analyzer.py`'s `OllamaEngine` for a worked example. It's entirely
optional; engines that only implement `run(self, prompt)` still work fine.)

Then import `my_engine` before `interview_analyzer.watcher.main()` runs
(e.g. add `import my_engine` at the top of a small wrapper script you point
Task Scheduler at instead of `-m interview_analyzer.watcher` directly), and
set in config.yaml:
```yaml
analysis:
  engine: "my_engine"
```

The rubric prompt itself (what categories are scored, what JSON shape is
requested) lives in `src/interview_analyzer/rubric.py` and is provider-agnostic
— edit it once and it applies no matter which engine you choose.
