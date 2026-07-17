# Speaker Labels (`[Interviewer]` vs `[You]`)

The transcript labels who said what. There are two ways this happens,
tried in this order automatically -- **you don't need to configure which
one is used**, only whether speaker labels happen at all
(`transcription.diarization` in `config/config.yaml`, on by default).

## 1. Channel separation (default, no setup needed)

When your microphone is captured (`audio.include_microphone: true`, the
default), it's recorded on its **own channel**, completely separate from
system audio (what you hear from the call) -- see `recorder.py`. Your
microphone channel is transcribed and labeled `[You]`; the system-audio
channel is transcribed and labeled `[Interviewer]`; the two are merged by
timestamp into one chronological transcript.

This is the default and needs **no setup at all** -- no account, no
token, no model download -- and is more reliable than acoustic diarization
below, since which channel a word came from *is* who said it, not a guess.

**One caveat**: a laptop microphone often picks up *some* of the
interviewer's audio through your speakers too, unless you're using
headphones/earbuds -- there's no acoustic echo cancellation here.
`transcriber.py` filters out the disruptive case automatically (a whole
sentence picked up nearly verbatim and mislabeled under "You" -- reproduced
on a real interview without headphones), but a stray word or two blending
into a real answer can still slip through undetected, since that's
indistinguishable from something you actually said. Using headphones
during the call avoids the problem at the source rather than relying on
this after-the-fact filter.

This only applies when the microphone was actually captured -- if it
wasn't (e.g. `include_microphone: false`, or the OS blocked microphone
access), there's no separate channel to label from, and diarization falls
back to option 2 below.

## 2. Acoustic diarization via pyannote.audio (fallback, needs setup)

Used only for recordings with no separate microphone channel. Uses
[pyannote.audio](https://github.com/pyannote/pyannote-audio), which is
free but requires a free Hugging Face account + token to download the
pretrained model the first time.

### Setup (one-time)

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

Without this setup, a mono recording (no mic channel) just falls back
further to a single generic `[Speaker]` label for everyone -- transcription
still works, just unlabeled.

## If you'd rather skip speaker labels entirely

Set `diarization: false` in `config/config.yaml`. Transcripts will still be
produced and analyzed, just without speaker labels (`[Speaker]` for
everyone, even for a mic-captured recording). The analyzer prompt still
works reasonably well off turn structure alone, but labeled speakers are
recommended.
