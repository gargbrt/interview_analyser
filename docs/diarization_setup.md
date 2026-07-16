# Enabling Speaker Diarization

Diarization separates the transcript into `[Interviewer]` vs `[You]` lines.
Without it, transcription still works, just unlabeled (`[Speaker]` for
everyone) — the analyzer can usually still infer question/answer pairs from
turn-taking, but labeled speakers are more reliable.

Diarization uses [pyannote.audio](https://github.com/pyannote/pyannote-audio),
which is free but requires a free Hugging Face account + token to download
the pretrained model the first time.

## Setup (one-time)

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Accept the user agreement on these two model pages (required by pyannote):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Create an access token: huggingface.co → Settings → Access Tokens →
   New token (read access is enough).
4. Set it as an environment variable:
   ```powershell
   setx HUGGING_FACE_HUB_TOKEN "hf_..."
   ```
5. Run once manually so the model downloads (a few hundred MB, one-time):
   ```bash
   python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')"
   ```

After this, `diarization: true` in `config/config.yaml` (the default) will
work automatically for every future interview.

## If you'd rather skip it

Set `diarization: false` in `config/config.yaml`. Transcripts will still be
produced and analyzed, just without speaker labels. The analyzer prompt
still works reasonably well off turn structure alone, but labeled speakers
are recommended if you can spare the five minutes of setup.
