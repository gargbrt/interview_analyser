# Language Support

## What Whisper (the transcription engine) supports

faster-whisper (the local, free transcription engine this app uses) is
built on OpenAI's Whisper model, which supports **~99 languages**, including
English and Hindi. Accuracy isn't uniform across all of them though:

- **High accuracy**: English, and other languages with lots of training
  data (Spanish, French, German, Hindi, Mandarin, Japanese, etc.).
- **Lower accuracy**: less-represented languages, heavy accents, or noisy
  audio -- expect more transcription errors.
- **Code-switching (mixing two languages in the same sentence, e.g.
  "Hinglish")**: Whisper does not have a dedicated language code for this.
  It was trained on real-world audio that includes some code-switching, so
  it often handles embedded English words reasonably well even when pinned
  to Hindi, but it's not guaranteed and there's no official mode for it.

Set `transcription.language` in `config/config.yaml` (or the Settings tab)
to control this:

| Value      | Behavior                                                        |
|------------|-------------------------------------------------------------------|
| `auto`     | Detect the spoken language automatically (default).               |
| `en`       | Force English.                                                    |
| `hi`       | Force Hindi (output in Devanagari script).                        |
| `hinglish` | Force Hindi decoding (embedded English words transcribed inline), then romanize the result to Latin script -- see below. |

## The optional Hindi / English / Hinglish pack

Selecting `hinglish` does two things:

1. Passes `language="hi"` to Whisper, which in practice lets it transcribe
   embedded English words in Latin script within an otherwise-Hindi
   transcript, rather than forcing everything into Hindi.
2. Runs the result through the optional
   [`indic-transliteration`](https://pypi.org/project/indic-transliteration/)
   package to romanize any Devanagari text to Latin script, so the whole
   transcript reads as Romanized Hinglish (e.g. "aap kaise ho") instead of
   mixed Devanagari/Latin script.

`indic-transliteration` is a lightweight, pure-Python package listed in
`requirements.txt` (so a normal `pip install -r requirements.txt` already
includes it). If it isn't installed, `hinglish` mode still works --
transcription proceeds normally, it just isn't romanized (a warning is
logged), same graceful-fallback pattern as
[diarization](diarization_setup.md) when its optional dependency is
missing.

### Installing/uninstalling it from the app

The dashboard's **Settings** tab has a **Language packs** section listing
every optional pack (currently just this one), each with an **Install** /
**Uninstall** button and a live "Installed" / "Not installed" status --
usable any time, not just during first-run setup. Installing/uninstalling
runs `pip install`/`pip uninstall` for you in the background with a
progress/log dialog, and always asks for confirmation first.

Or by hand, any time:

```bash
pip install indic-transliteration      # install
pip uninstall indic-transliteration    # uninstall
```

## Analysis (the LLM rubric scoring)

The analysis engine (Ollama or a cloud API, see `docs/using_cloud_apis.md`)
receives whatever language the transcript ends up in and evaluates it in
that language -- no separate configuration needed there. Larger/more
capable models generally handle non-English and code-switched transcripts
more reliably than smaller ones; if analysis quality on a Hindi/Hinglish
transcript seems off, trying `qwen2.5:14b` instead of the default
`llama3.1:8b` (see the Settings tab's "Install model..." button) is worth
testing.
