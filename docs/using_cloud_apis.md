# Using Cloud APIs or a Custom Analysis Engine

By default, analysis runs fully free and local via [Ollama](https://ollama.com).
You can swap in a hosted model instead — useful if you want higher-quality
feedback and are OK with a small per-analysis cost — or plug in a completely
custom engine.

## Important: a claude.ai subscription is not an API key

If you have a **Claude Pro/Max subscription** (claude.ai / the Claude app),
that does **not** give you API access. API usage is billed separately,
per token, through [console.anthropic.com](https://console.anthropic.com).
You'll need to create an API key there (a few dollars of credit covers a
large number of interview analyses — a single interview's analysis is a few
thousand tokens).

The same applies to OpenAI: a ChatGPT Plus subscription ≠ an OpenAI API key.
You need a key from platform.openai.com.

## Using Anthropic's API

1. Get an API key from console.anthropic.com.
2. Set it as an environment variable (PowerShell):
   ```powershell
   setx INTERVIEW_ANALYZER_API_KEY "sk-ant-..."
   ```
   (open a new terminal after `setx` for it to take effect)
3. Edit `config/config.yaml`:
   ```yaml
   analysis:
     engine: "anthropic_api"
     llm_model: "claude-sonnet-5"
   ```

## Using OpenAI's API

Same idea:
```powershell
setx INTERVIEW_ANALYZER_API_KEY "sk-..."
```
```yaml
analysis:
  engine: "openai_api"
  llm_model: "gpt-4o-mini"
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
