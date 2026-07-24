"""Confidence scoring for each interview's analysis, and turning past
feedback into corrective notes for future analysis prompts.

Three independent things live here:

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

  - `estimate_selection_probability`: a distinct 0-100 estimate of how
    likely the candidate would be selected/hired -- NOT the same thing as
    `calibrated_confidence` above (that measures trust in the assessment's
    *accuracy*; this measures the assessed *outcome*). Deliberately blends
    three inputs (the model's own hire-scale call, how the profile weights
    the scored competencies, and the assessment's own calibrated
    confidence) so it can never look more precise than the assessment
    backing it actually is.
"""
from __future__ import annotations

import logging
from typing import Optional

from .profiles import AssessmentProfile, GENERIC_PROFILE, competency_emphasis_map
from .rubric import HIRE_RECOMMENDATION_LEVELS

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


# Baseline percentage anchor per hire-scale level (rubric.py's
# HIRE_RECOMMENDATION_LEVELS) -- fixed, not computed, so the same level
# always anchors to the same starting point.
_HIRE_LEVEL_ANCHOR: dict[str, int] = {
    "Strong No Hire": 5,
    "No Hire": 15,
    "Lean No Hire": 30,
    "Lean Hire": 55,
    "Hire": 75,
    "Strong Hire": 90,
    "Exceptional": 97,
}
assert set(_HIRE_LEVEL_ANCHOR) == set(HIRE_RECOMMENDATION_LEVELS)

# How much an emphasis tier (profiles.py) counts toward the competency
# weighted average below -- a "critical" competency's score should move the
# needle far more than a "minor" one.
_EMPHASIS_WEIGHT: dict[str, float] = {
    "critical": 3.0, "high": 2.0, "moderate": 1.5, "low": 1.0, "minor": 0.5,
}

_NEUTRAL_PERCENT = 50
_MAX_COMPETENCY_NUDGE = 15  # caps how far the competency weighting alone can move the anchor

# The percent at/above which estimate_selection_probability's binary
# recommendation reads "Recommended" -- deliberately the same pivot as
# _NEUTRAL_PERCENT, so "better than neutral" and "recommended" always agree.
_RECOMMENDED_THRESHOLD = 50


def competency_weight(name: str, profile: AssessmentProfile = GENERIC_PROFILE) -> float:
    """The numeric weight (see _EMPHASIS_WEIGHT) this profile's context
    gives `name` -- a "critical" competency counts far more toward the
    weighted total than a "minor" one. Falls back to "moderate" for a
    competency the profile has no emphasis data for."""
    emphasis = competency_emphasis_map(profile).get(name, "moderate")
    return _EMPHASIS_WEIGHT.get(emphasis, 1.5)


def weighted_competency_total(
    competency_scores: Optional[list[dict]], profile: AssessmentProfile = GENERIC_PROFILE,
) -> Optional[float]:
    """The overall 0-100 competency score: each competency's score weighted
    by how much this profile's context (role/seniority/industry/company --
    see profiles.py) emphasizes it, so a "critical" competency counts far
    more toward the total than a "minor" one. None if there are no usable
    scores to average. Shared by estimate_selection_probability (which
    nudges a hire-scale anchor by this) and report.py/infographic.py
    (which show it directly as an "Overall competency score")."""
    if not competency_scores:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    for entry in competency_scores:
        if not isinstance(entry, dict):
            continue
        score = entry.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        weight = competency_weight(entry.get("name", ""), profile)
        weighted_sum += score * weight
        weight_total += weight
    return weighted_sum / weight_total if weight_total > 0 else None


def estimate_selection_probability(
    hire_recommendation: Optional[dict],
    competency_scores: Optional[list[dict]],
    profile: AssessmentProfile = GENERIC_PROFILE,
    confidence_info: Optional[dict] = None,
) -> dict:
    """A distinct-from-`calibrated_confidence` estimate of how likely this
    candidate would be selected, expressed as {"percent": int (1-99),
    "label": str, "basis": str, "binary_recommendation": "Recommended"|"Not
    Recommended"}. Never returns 0 or 100 -- this is always an estimate,
    never a certainty. `binary_recommendation` is a simple >=50% reading of
    the same estimate -- shown ALONGSIDE the percentage/label (per explicit
    user request), never in place of it, since collapsing to a single yes/no
    is exactly what the percentage was added to avoid.

    Combines three inputs:
      1. hire_recommendation["level"] (the model's own 7-point hire-scale
         call, see rubric.py) anchors a baseline percentage.
      2. competency_scores, weighted by how much this profile's context
         (role/seniority/industry/company -- see profiles.py) emphasizes
         each one (see weighted_competency_total), nudges that baseline up
         or down.
      3. confidence_info (from calibrated_confidence) pulls the whole
         estimate toward a neutral 50% the less trustworthy the underlying
         assessment is -- a low-confidence assessment must not produce a
         falsely precise-looking selection probability.
    """
    level = (hire_recommendation or {}).get("level") or ""
    anchor = _HIRE_LEVEL_ANCHOR.get(level, _NEUTRAL_PERCENT)

    nudge = 0.0
    weighted_avg = weighted_competency_total(competency_scores, profile)
    if weighted_avg is not None:
        nudge = max(-_MAX_COMPETENCY_NUDGE, min(_MAX_COMPETENCY_NUDGE, (weighted_avg - _NEUTRAL_PERCENT) * 0.3))

    baseline = max(0, min(100, anchor + nudge))

    confidence_score = (confidence_info or {}).get("score")
    # No usable confidence figure at all -> treat as low-trust (0.5), same
    # spirit as calibrated_confidence's own "unavailable" fallback -- don't
    # let a missing confidence signal accidentally read as full trust.
    confidence_weight = 0.5 if confidence_score is None else max(0.0, min(1.0, confidence_score / 100))
    pulled = _NEUTRAL_PERCENT + (baseline - _NEUTRAL_PERCENT) * confidence_weight
    percent = max(1, min(99, round(pulled)))

    basis = (
        f"Hire-scale call: \"{level or 'not given'}\" (anchors {anchor}%); "
        f"competency weighting nudged it by {nudge:+.0f} points; "
        f"pulled toward a neutral 50% at {round(confidence_weight * 100)}% strength based on "
        f"assessment confidence."
    )
    binary_recommendation = "Recommended" if percent >= _RECOMMENDED_THRESHOLD else "Not Recommended"
    return {"percent": percent, "label": level or None, "basis": basis, "binary_recommendation": binary_recommendation}
