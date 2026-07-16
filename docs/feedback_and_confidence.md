# Feedback & Confidence Scoring

Every report has a small **Feedback** panel (dashboard → History tab →
select an interview) asking two things, each on a **1-10 scale** (10 =
highest quality):

- **Transcription quality** — did it capture what was actually said?
- **Analysis quality** — was the assessment accurate/helpful?

Both default to **"Not rated"**, plus an optional free-text comment,
submitted together with **Submit feedback**. You can resubmit any time —
it replaces your previous rating for that interview rather than stacking
duplicates.

## Clearing or deleting feedback

- **Clear ratings** — resets both scores to "Not rated" and clears the
  comment, then saves that immediately. Use this if you want to take back
  a rating but don't want to lose the feedback row's existence (e.g. you're
  unsure and want to reconsider later).
- **Delete feedback** — removes the feedback row for that interview
  entirely. Use this to clean up a rating given by mistake (e.g. you rated
  the wrong interview).

Both are on the same panel as Submit feedback.

## What your feedback is used for

1. **Confidence scoring.** Each report shows a *"Confidence in this
   assessment"* line. Once you've rated at least 3 analyses, that score is
   derived from your own average analysis-quality score (normalized to a
   percentage — e.g. an average of 7/10 shows as 70%) rather than the
   model's own guess — a track record beats a self-assessment. Before that
   (or if the feedback history can't be read for any reason), it falls
   back to the model's own self-reported confidence, which every built-in
   analysis run now asks for as part of its JSON output (see `rubric.py`'s
   `confidence` field).

   | Shown as                                          | Meaning |
   |----------------------------------------------------|---------|
   | `82% (calibrated from your last 12 feedback ratings)` | Your own track record |
   | `74% (model's own self-assessment — rate this report to start calibrating from your feedback instead)` | Not enough feedback yet |
   | `not available`                                     | Neither is available (e.g. an older analysis engine that doesn't report confidence, and no feedback yet) |

2. **Calibration notes.** If you leave a comment on feedback scored 4 or
   below (transcription or analysis), that comment is summarized and
   injected into the prompt for *future* analyses, so the model has a
   chance to avoid repeating the same mistake — e.g. "missed that I gave a
   specific metric in my answer" gets carried forward as a note the model
   sees on your next interview's analysis. This only affects future
   analyses; it doesn't retroactively change any past report.

Both mechanisms are scoped to your local login profile, same as everything
else in this app (no data leaves your machine unless you're using a cloud
analysis engine — see `docs/using_cloud_apis.md`).

## Why a rating without a comment still matters

Even a bare score with no comment feeds the confidence calculation — it's
what builds up your accuracy track record. Comments are only needed to
also feed the calibration-notes mechanism above, and only when the score
is 4 or below.
