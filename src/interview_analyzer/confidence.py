"""Confidence scoring for each interview's analysis, and turning past
feedback into corrective notes for future analysis prompts.

Two independent things live here:

  - `calibrated_confidence`: a 0-100 score, shown on each report, saying how
    much to trust *this* assessment. Once there's enough of the user's own
    "was this accurate?" feedback (see db.py's feedback table), it's derived
    from their actual track record with this analysis engine/model instead
    of the model's own self-assessment. Before that (or if the feedback
    table can't be read for any reason), it falls back to the model's own
    self-reported confidence from its JSON output (see rubric.py).

  - `calibration_notes`: a short block of text built from recent negative
    feedback comments, injected into the analysis prompt so the model has a
    chance to actually act on past corrections -- the closest thing to
    "learning" available without fine-tuning the model itself.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Below this many rated interviews, the user's own track record is too thin
# to be a more meaningful signal than the model's own self-assessment.
MIN_FEEDBACK_SAMPLES = 3

# Feedback is a 1-10 quality score (10 = highest, see db.py); a score at or
# below this is "negative" enough to be worth feeding back into future
# analysis prompts as a corrective note (see calibration_notes below).
NEGATIVE_SCORE_THRESHOLD = 4


def calibrated_confidence(db, user_id: Optional[int], model_reported: Optional[float]) -> dict:
    """Returns {"score": int|None, "source": "feedback"|"model"|"unavailable",
    "sample_size": int}. `score` is None only when there's neither usable
    feedback history nor a model-reported figure to fall back to."""
    try:
        feedback = db.list_feedback(user_id=user_id)
        rated = [f for f in feedback if f.analysis_score is not None]
    except Exception:  # noqa: BLE001
        # feedback table unreadable for any reason -- fall back to the
        # model's own figure rather than let confidence scoring break
        # analysis entirely
        logger.warning("Couldn't read feedback history for confidence calibration; falling back to model-reported confidence.", exc_info=True)
        rated = None

    if rated is not None and len(rated) >= MIN_FEEDBACK_SAMPLES:
        # average 1-10 score, normalized to a 0-100 scale
        avg_score = sum(f.analysis_score for f in rated) / len(rated)
        return {"score": round(avg_score / 10 * 100), "source": "feedback", "sample_size": len(rated)}

    if model_reported is not None:
        try:
            score = max(0, min(100, round(float(model_reported))))
        except (TypeError, ValueError):
            return {"score": None, "source": "unavailable", "sample_size": 0}
        return {"score": score, "source": "model", "sample_size": len(rated) if rated else 0}

    return {"score": None, "source": "unavailable", "sample_size": len(rated) if rated else 0}


def format_confidence(confidence_info: Optional[dict]) -> str:
    """Human-readable one-liner for a report -- see report.py."""
    if not confidence_info or confidence_info.get("score") is None:
        return "not available"
    score = confidence_info["score"]
    source = confidence_info.get("source")
    n = confidence_info.get("sample_size", 0)
    if source == "feedback":
        return f"{score}% (calibrated from your last {n} feedback ratings)"
    if source == "model":
        return f"{score}% (model's own self-assessment -- rate this report to start calibrating from your feedback instead)"
    return f"{score}%"


def calibration_notes(db, user_id: Optional[int], limit: int = 5) -> str:
    """Builds a short, prompt-injectable summary of recent feedback the user
    marked as inaccurate, with their comment -- empty string if there's
    nothing usable (no feedback yet, no comments, or the table can't be
    read), in which case the prompt is simply unchanged from before this
    feature existed."""
    try:
        feedback = db.list_feedback(user_id=user_id)
    except Exception:  # noqa: BLE001
        logger.warning("Couldn't read feedback history for calibration notes.", exc_info=True)
        return ""

    negative = [
        f for f in feedback
        if f.comment and f.comment.strip() and (
            (f.analysis_score is not None and f.analysis_score <= NEGATIVE_SCORE_THRESHOLD)
            or (f.transcript_score is not None and f.transcript_score <= NEGATIVE_SCORE_THRESHOLD)
        )
    ]
    if not negative:
        return ""

    lines = [f"- {f.comment.strip()}" for f in negative[-limit:]]
    return (
        "The user has previously flagged issues with analyses of their interviews. "
        "Take these into account and avoid repeating the same mistakes:\n" + "\n".join(lines)
    )
