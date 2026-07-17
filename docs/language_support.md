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
| `auto`     | Detect the spoken language automatically.                         |
| `en`       | Force English (default).                                          |
| `hi`       | Force Hindi (output in Devanagari script).                        |
| `hinglish` | Force Hindi decoding (embedded English words transcribed inline), then romanize the result to Latin script -- see below. |

## Accented English (e.g. Indian English)

Whisper's accuracy is noticeably lower on strongly accented English than on
US/UK-accented English, out of the box. Three settings here specifically
help with that, all on by default:

1. **`transcription.language: "en"`** -- Whisper's "auto" mode detects the
   spoken language from only the first ~30 seconds of audio, and a strong
   accent is a common way for that step to misdetect the language entirely,
   garbling everything downstream. Pinning to `"en"` skips that guesswork.
2. **`transcription.whisper_model: "medium"`** (or `"large-v3"` if your
   machine can run it fast enough) -- Whisper's accuracy on accented speech
   improves with model size more than its accuracy on standard-accent
   speech does; `"small"` and below visibly mis-hear more on accented
   audio. This is slower to transcribe on CPU than `"small"` was --
   roughly 2-3x for `"medium"` -- so drop back down if it's too slow for
   your machine.
3. **`transcription.initial_prompt`** -- a short text hint passed to
   Whisper before each decode, biasing it towards transcribing informal,
   disfluent spoken audio (including filler words) accurately rather than
   "cleaning it up" into a different, more formal-sounding — and often
   wrong — sentence. Set it to `""` to disable.

### Why not a dedicated Indian-accent/Indic ASR model?

[Sarvam AI](https://www.sarvam.ai/blogs/sarvam-translate) and others offer
ASR specifically tuned for Indian accents and languages, but Sarvam's
strongest speech-to-text models (Saarika/Saaras) are a paid cloud API only
(~₹1.5/min after a small free-credit trial) — not a free, local model, so
they don't fit this app's fully-local/free design and would add an
internet dependency and ongoing cost. Free, locally-runnable alternatives
like [IndicWhisper](https://huggingface.co/vasista22/whisper-hindi-large-v2)
exist but are fine-tuned for *Hindi speech*, not Indian-accented *English*
speech, and aren't published in the CTranslate2 format faster-whisper (this
app's transcription engine) needs -- using one would mean converting it
yourself first. If you want to try one anyway,
`transcription.whisper_model` accepts any Hugging Face repo already in
CTranslate2 format (not just the built-in size names), including one you
convert yourself -- see [SYSTRAN/faster-whisper's conversion
instructions](https://github.com/SYSTRAN/faster-whisper#model-conversion).

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
