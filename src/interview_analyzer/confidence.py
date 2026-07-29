"""Confidence scoring for each interview's analysis, and turning past
feedback into corrective notes for future analysis prompts.

Independent things live here:

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

  - `transcript_specificity_nudge`: a small, self-calibrating signal read
    directly from the raw transcript text (never from anything the model
    produced) -- see its own docstring for why this exists: the model's
    hire_recommendation and competency_scores are NOT independent evidence
    of each other, since rubric.py's prompt explicitly tells the model to
    ground the former in the latter, so nudging a hire-scale anchor with a
    re-aggregation of the same competency scores was closer to self-
    consistency correction than genuinely combining two separate readings.

  - `hire_outcome_calibration`: ground truth, not a proxy -- this user's
    own submitted real hired/not-hired outcomes (db.py's FeedbackRecord.
    hire_outcome) for past interviews of a similar type (matched by role/
    seniority, backing off to a coarser match when there isn't enough
    labeled history yet). Produces the two anchors (this user's own
    average estimate for interviews that turned out Hired vs. Not Hired)
    that estimate_selection_probability rescales its final number against.

  - `estimate_selection_probability`: a distinct 0-100 estimate of how
    likely the candidate would be selected/hired -- NOT the same thing as
    `calibrated_confidence` above (that measures trust in the assessment's
    *accuracy*; this measures the assessed *outcome*). Blends the model's
    own hire-scale call (nudged by transcript_specificity_nudge, the
    primary independent signal) with a small, capped competency-
    disagreement sanity check -- weighted_competency_total is allowed a
    limited say again, but only when it diverges sharply from the hire-
    scale anchor, not as a primary driver (see estimate_selection_
    probability's own docstring for why the earlier all-or-nothing designs
    -- competency scores as the main nudge, then no competency influence at
    all -- each went too far) -- then pulls the whole result toward a
    neutral 50% based on the assessment's own calibrated confidence, so it
    can never look more precise than the assessment backing it actually is.
    Finally, if hire_outcome_calibration found enough labeled history,
    rescales the result against this user's own real-world track record --
    the one input here that's ground truth rather than a proxy for it.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from .profiles import TIER_ORDER, AssessmentProfile, GENERIC_PROFILE, competency_emphasis_value
from .rubric import HIRE_RECOMMENDATION_LEVELS

logger = logging.getLogger(__name__)

# Below this many rated interviews, the user's own track record is too thin
# for feedback to have ANY say at all -- below this, it's purely the
# model's own self-assessment.
MIN_FEEDBACK_SAMPLES = 3

# Above MIN_FEEDBACK_SAMPLES, feedback's influence on the final score grows
# SMOOTHLY with how much of it exists, rather than switching from "100%
# trust the model's self-report" to "100% trust the feedback average" the
# instant the count crosses MIN_FEEDBACK_SAMPLES. Reproduced as a real,
# confusing incident: a model-reported 70% became a feedback-based 45% the
# moment a 3rd rating came in -- an abrupt cliff the user has no way to
# anticipate, and one bad early rating (from when the app was rougher)
# got as much permanent say as every later one, forever. Saturates at
# FEEDBACK_TRUST_SATURATION_SAMPLES (a real track record IS more
# meaningful than the model's own self-assessment once there's enough of
# it), but MAX_FEEDBACK_TRUST_WEIGHT stops short of 100% even then, so the
# model's own per-interview signal is never fully silenced.
FEEDBACK_TRUST_SATURATION_SAMPLES = 15
MAX_FEEDBACK_TRUST_WEIGHT = 0.9

# How much each step further back in time discounts a rating in the
# feedback average -- 1.0 would be a plain unweighted average (a rating
# from months ago counts exactly as much as one from today, forever).
# Gentle by design (0.85, not something aggressive) so ordinary week-to-
# week noise doesn't swing the score, while genuine improvement (or
# decline) in more recent ratings still shows up instead of being
# permanently capped by one old outlier.
FEEDBACK_RECENCY_DECAY = 0.85

# Feedback is a 1-10 quality score (10 = highest, see db.py); a score at or
# below this is "negative" enough to be worth feeding back into future
# analysis prompts as a corrective note (see calibration_notes below).
NEGATIVE_SCORE_THRESHOLD = 4


def calibrated_confidence(db, user_id: Optional[int], model_reported: Optional[float]) -> dict:
    """Returns {"score": int|None, "source": "feedback"|"model"|"unavailable",
    "sample_size": int}. `score` is None only when there's neither usable
    feedback history nor a model-reported figure to fall back to.

    `source: "feedback"` means feedback had SOME say (sample_size >=
    MIN_FEEDBACK_SAMPLES), not that it's the only input -- see
    FEEDBACK_TRUST_SATURATION_SAMPLES above for why it's blended with
    model_reported rather than replacing it outright."""
    try:
        # already created_at ASC (oldest first) -- see db.py's list_feedback
        feedback = db.list_feedback(user_id=user_id)
        rated = [f for f in feedback if f.analysis_score is not None]
    except Exception:  # noqa: BLE001
        # feedback table unreadable for any reason -- fall back to the
        # model's own figure rather than let confidence scoring break
        # analysis entirely
        logger.warning("Couldn't read feedback history for confidence calibration; falling back to model-reported confidence.", exc_info=True)
        rated = None

    if rated is not None and len(rated) >= MIN_FEEDBACK_SAMPLES:
        n = len(rated)
        # newest entry gets weight 1.0; each one further back is
        # discounted by another factor of FEEDBACK_RECENCY_DECAY
        weights = [FEEDBACK_RECENCY_DECAY ** (n - 1 - i) for i in range(n)]
        weighted_avg = sum(f.analysis_score * w for f, w in zip(rated, weights)) / sum(weights)
        feedback_score = weighted_avg / 10 * 100  # 0-100 scale

        model_score: Optional[float] = None
        if model_reported is not None:
            try:
                model_score = max(0.0, min(100.0, float(model_reported)))
            except (TypeError, ValueError):
                model_score = None

        if model_score is None:
            # nothing to blend with -- feedback is all there is
            blended = feedback_score
        else:
            feedback_trust = min(1.0, n / FEEDBACK_TRUST_SATURATION_SAMPLES) * MAX_FEEDBACK_TRUST_WEIGHT
            blended = feedback_score * feedback_trust + model_score * (1 - feedback_trust)
        return {"score": round(blended), "source": "feedback", "sample_size": n}

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
        return f"{score}% (blends your last {n} feedback ratings with the model's own self-assessment)"
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
_MAX_NUDGE = 15  # caps how far transcript_specificity_nudge alone can move the anchor

# The percent at/above which estimate_selection_probability's binary
# recommendation reads "Recommended" -- deliberately the same pivot as
# _NEUTRAL_PERCENT, so "better than neutral" and "recommended" always agree.
_RECOMMENDED_THRESHOLD = 50

# How far the weighted competency total (0-100) must diverge from the
# hire-scale anchor (also 0-100) before it's treated as a genuine
# disagreement worth a small correction -- roughly one hire-scale "step"
# (the anchors are ~15-25 points apart), so this only fires when the
# competency math and the model's own holistic verdict meaningfully
# disagree, not on ordinary noise.
_COMPETENCY_DISAGREEMENT_THRESHOLD = 20
# The disagreement, past the threshold, is scaled down (not applied
# 1-for-1) and capped well below _MAX_NUDGE -- this is a sanity check on
# the anchor, not a second primary driver alongside transcript_
# specificity_nudge.
_COMPETENCY_DISAGREEMENT_SCALE = 0.3
_MAX_COMPETENCY_DISAGREEMENT_CORRECTION = 10

# Below this many past interviews (with a usable transcript) for this
# user, "the average so far" isn't a meaningful baseline yet -- same
# reasoning as MIN_FEEDBACK_SAMPLES above, just bootstrapping a different
# signal (see transcript_specificity_nudge).
MIN_SPECIFICITY_SAMPLES = 3

# Transcript speaker labels are "[Speaker] text" (see transcriber.py) --
# "You" is always the candidate's own line: dual-channel recording keeps
# the mic and system-audio channels separate and labels them directly,
# not a guess.
_TRANSCRIPT_LINE_RE = re.compile(r"^\[(?P<speaker>[^\]]+)\]\s*(?P<text>.*)$")
_CANDIDATE_SPEAKER_LABEL = "You"
# Any digit -- a percentage, dollar figure, count, or date all contain at
# least one -- a simple, language-agnostic, and crucially non-LLM-judged
# proxy for "this answer included a concrete, checkable detail" rather
# than staying purely qualitative.
_SPECIFICITY_MARKER_RE = re.compile(r"\d")


def _candidate_turns(transcript: str) -> list[str]:
    """The candidate's own speech, with consecutive same-speaker lines
    merged into whole turns first -- faster-whisper often splits one
    continuous utterance across several segments/lines (see dashboard.py's
    own _group_transcript_by_speaker, duplicated here in miniature rather
    than imported, since this lower-level module shouldn't depend on the
    Tk dashboard), so a real answer that happens to start with a one-word
    acknowledgment ("Mm-hmm. So, for that project...") isn't counted as
    two separate, mostly-trivial "turns" instead of one real one --
    reproduced directly: without merging, a real ~1-hour transcript's 279
    raw lines (mostly one-word backchannel) diluted the specificity
    fraction to a meaningless ~7%; merged into 133 real turns it reads
    ~14%, a much more honest reflection of the actual answers given."""
    turns: list[tuple[str, str]] = []
    for line in transcript.splitlines():
        if not line.strip():
            continue
        match = _TRANSCRIPT_LINE_RE.match(line)
        if not match:
            continue
        speaker, text = match.group("speaker").strip(), match.group("text").strip()
        if turns and turns[-1][0] == speaker:
            turns[-1] = (speaker, f"{turns[-1][1]} {text}".strip())
        else:
            turns.append((speaker, text))
    return [text for speaker, text in turns if speaker == _CANDIDATE_SPEAKER_LABEL and text]


def _specificity_fraction(transcript: str) -> Optional[float]:
    """The fraction of the candidate's own turns (see _candidate_turns)
    containing at least one concrete, checkable detail -- None if there's
    no reliable way to isolate candidate-only speech (a mono/no-
    diarization transcript labels everyone "Speaker", not "You" -- see
    transcriber.py's fallback path)."""
    turns = _candidate_turns(transcript)
    if not turns:
        return None
    return sum(1 for t in turns if _SPECIFICITY_MARKER_RE.search(t)) / len(turns)


def transcript_specificity_nudge(
    db, user_id: Optional[int], transcript: str, exclude_interview_id: Optional[int] = None,
) -> Optional[float]:
    """estimate_selection_probability's nudge, computed directly from the
    raw transcript's own text -- genuinely independent of hire_
    recommendation/competency_scores, since it never touches anything the
    model produced (unlike the design this replaced, where the prompt
    explicitly told the model to ground hire_recommendation in the same
    competency scores the old nudge also came from -- see rubric.py's
    build_prompt -- making the two nowhere near independent evidence of
    each other).

    Reads this interview's own _specificity_fraction RELATIVE to this same
    user's past interviews, not against a guessed universal "normal"
    fraction -- how quantitative a "typical" answer is varies enormously by
    role (a Data interview is naturally full of numbers; a Design one
    rarely is), so there's no single fair neutral point across every
    profile. Comparing against this user's OWN history sidesteps needing
    one, and self-calibrates as more interviews accumulate -- same
    bootstrap spirit as calibrated_confidence above.

    Returns 0.0 (no adjustment, not a guess) when there isn't yet enough
    history (fewer than MIN_SPECIFICITY_SAMPLES past interviews with a
    usable transcript) to treat "the average so far" as meaningful.
    Returns None only when THIS transcript has no isolable candidate
    speech at all -- the caller should treat that as "no signal", not as
    a penalty."""
    fraction = _specificity_fraction(transcript)
    if fraction is None:
        return None

    try:
        past_records = db.list_all(user_id=user_id)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Couldn't read past interviews for the specificity nudge baseline; treating as no history yet.",
            exc_info=True,
        )
        past_records = []

    past_fractions = [
        past_fraction
        for record in past_records if record.id != exclude_interview_id and record.transcript
        for past_fraction in [_specificity_fraction(record.transcript)] if past_fraction is not None
    ]
    if len(past_fractions) < MIN_SPECIFICITY_SAMPLES:
        return 0.0

    baseline = sum(past_fractions) / len(past_fractions)
    # a floor on the denominator, not a bare division by baseline -- if
    # this user's past interviews all scored ~0 specificity, a plain
    # (fraction - baseline) / baseline blows up (or is undefined at
    # exactly 0), even though "this one actually included real numbers,
    # unlike any before it" is a perfectly meaningful, clearly positive
    # signal that shouldn't just collapse to "no adjustment".
    relative_change = (fraction - baseline) / max(baseline, 0.1)
    return max(-_MAX_NUDGE, min(_MAX_NUDGE, relative_change * _MAX_NUDGE))


def competency_weight(name: str, profile: AssessmentProfile = GENERIC_PROFILE) -> float:
    """The numeric weight (see _EMPHASIS_WEIGHT) this profile's context
    gives `name` -- a "critical" competency counts far more toward the
    weighted total than a "minor" one.

    Interpolates linearly between _EMPHASIS_WEIGHT's tier weights using
    competency_emphasis_value's CONTINUOUS 0-4 average, rather than looking
    up competency_emphasis_map's already-rounded tier string -- rounding to
    one of only 5 buckets before converting to a number can silently erase
    a real difference when 3 of a profile's 4 dimensions (role/seniority/
    industry/company_type) are held fixed and only one changes, since the
    fixed dimensions can pull two different raw averages onto the exact
    same rounded tier (see competency_emphasis_value's docstring for the
    real case this was fixing: a Senior/Lead vs. Director+ swap that used
    to average out to identical "moderate" weights for some competencies,
    flattening the seniority-comparison gauge's markers to the same
    number)."""
    value = max(0.0, min(len(TIER_ORDER) - 1, competency_emphasis_value(profile, name)))
    lower_index = min(int(value), len(TIER_ORDER) - 2)
    upper_index = lower_index + 1
    fraction = value - lower_index
    lower_weight = _EMPHASIS_WEIGHT[TIER_ORDER[lower_index]]
    upper_weight = _EMPHASIS_WEIGHT[TIER_ORDER[upper_index]]
    return lower_weight + fraction * (upper_weight - lower_weight)


def weighted_competency_total(
    competency_scores: Optional[list[dict]], profile: AssessmentProfile = GENERIC_PROFILE,
) -> Optional[float]:
    """The overall 0-100 competency score: each competency's score weighted
    by how much this profile's context (role/seniority/industry/company --
    see profiles.py) emphasizes it, so a "critical" competency counts far
    more toward the total than a "minor" one. None if there are no usable
    scores to average. Shown directly in report.py/infographic.py as an
    "Overall competency score" -- NOT used by estimate_selection_probability
    below, which nudges its hire-scale anchor with transcript_specificity_
    nudge instead (these scores aren't independent evidence of the model's
    own hire_recommendation verdict, since rubric.py's prompt explicitly
    tells the model to ground that verdict in these same scores)."""
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


# Below this many total labeled outcomes (Hired + Not Hired combined) in a
# given match-specificity group, that group's "average estimate per
# outcome" isn't a meaningful anchor yet -- same bootstrapping reasoning as
# MIN_FEEDBACK_SAMPLES/MIN_SPECIFICITY_SAMPLES above.
MIN_HIRE_OUTCOME_SAMPLES = 4
# The recalibration's own strength (see estimate_selection_probability's
# handling of hire_outcome_calibration_info) grows with sample size but
# saturates well short of fully overriding the model's own reasoning --
# even a large, well-labeled history is one more signal, not the only one.
_HIRE_OUTCOME_CALIBRATION_SATURATION_SAMPLES = 12
_MAX_HIRE_OUTCOME_CALIBRATION_WEIGHT = 0.6


def _profile_field(profile: Optional[AssessmentProfile], name: str):
    return getattr(profile, name, None) if profile is not None else None


# Backoff order for matching a past interview's profile against the
# CURRENT one when picking which labeled outcomes to calibrate against --
# most specific first. Role and seniority are the two dimensions that
# actually distinguish "interview type" in this app's own vocabulary (e.g.
# "Senior PM" vs "Entry-level SWE" -- see profiles.py's SENIORITY_EMPHASIS/
# ROLE_EMPHASIS, which are also built around just these two as the primary
# axes), so they're tried together before falling back to either alone,
# then finally no filter at all. Industry/company_type are deliberately
# NOT part of this hierarchy -- with realistically small numbers of labeled
# outcomes per user, requiring a 4-way match would almost always fall
# straight through to "no data," defeating the point of trying to be
# profile-specific at all.
_HIRE_OUTCOME_MATCH_LEVELS: list[tuple[str, Callable[[Optional[AssessmentProfile], Optional[AssessmentProfile]], bool]]] = [
    (
        "role and seniority",
        lambda past, current: (
            _profile_field(current, "role") is not None
            and _profile_field(current, "seniority") is not None
            and _profile_field(past, "role") == _profile_field(current, "role")
            and _profile_field(past, "seniority") == _profile_field(current, "seniority")
        ),
    ),
    (
        "seniority",
        lambda past, current: (
            _profile_field(current, "seniority") is not None
            and _profile_field(past, "seniority") == _profile_field(current, "seniority")
        ),
    ),
    (
        "role",
        lambda past, current: (
            _profile_field(current, "role") is not None
            and _profile_field(past, "role") == _profile_field(current, "role")
        ),
    ),
    ("all your interviews", lambda past, current: True),
]


def hire_outcome_calibration(db, user_id: Optional[int], profile: Optional[AssessmentProfile]) -> Optional[dict]:
    """Ground-truth calibration from this user's own submitted hire/no-hire
    outcomes (see db.py's FeedbackRecord.hire_outcome -- distinct from
    analysis_score, which rates whether the ASSESSMENT was accurate, not
    what actually happened).

    For each of this user's past interviews with a labeled outcome, looks
    up the selection-probability percent that was actually estimated for
    it at the time, then groups those (percent, outcome) pairs by how
    closely that interview's profile matches the CURRENT one --
    _HIRE_OUTCOME_MATCH_LEVELS, most specific first (role+seniority, then
    seniority alone, then role alone, then everything). Returns the
    MOST SPECIFIC level that has at least MIN_HIRE_OUTCOME_SAMPLES total
    labeled outcomes AND at least one of each ("Hired" and "Not Hired" --
    two anchors are needed to calibrate against, one alone says nothing
    about direction). Returns None if no level qualifies (not enough
    labeled history yet, for interviews like this one or otherwise).

    Return shape: {"hired_anchor": float, "not_hired_anchor": float,
    "sample_size": int, "specificity": str (a label from
    _HIRE_OUTCOME_MATCH_LEVELS, for the basis text)}."""
    try:
        feedback_rows = [
            f for f in db.list_feedback(user_id=user_id)
            if f.hire_outcome in ("Hired", "Not Hired")
        ]
    except Exception:  # noqa: BLE001
        logger.warning("Couldn't read feedback history for hire-outcome calibration.", exc_info=True)
        return None
    if not feedback_rows:
        return None

    samples: list[tuple[float, str, Optional[AssessmentProfile]]] = []
    for fb in feedback_rows:
        try:
            record = db.get(fb.interview_id)
        except Exception:  # noqa: BLE001
            continue
        if record is None:
            continue
        analysis = record.analysis
        if not isinstance(analysis, dict):
            continue
        percent = (analysis.get("selection_probability") or {}).get("percent")
        if not isinstance(percent, (int, float)) or isinstance(percent, bool):
            continue
        samples.append((float(percent), fb.hire_outcome, record.profile))

    for label, matches in _HIRE_OUTCOME_MATCH_LEVELS:
        group = [(pct, outcome) for pct, outcome, past_profile in samples if matches(past_profile, profile)]
        hired = [pct for pct, outcome in group if outcome == "Hired"]
        not_hired = [pct for pct, outcome in group if outcome == "Not Hired"]
        if len(group) >= MIN_HIRE_OUTCOME_SAMPLES and hired and not_hired:
            return {
                "hired_anchor": sum(hired) / len(hired),
                "not_hired_anchor": sum(not_hired) / len(not_hired),
                "sample_size": len(group),
                "specificity": label,
            }
    return None


def estimate_selection_probability(
    hire_recommendation: Optional[dict],
    nudge: Optional[float] = None,
    confidence_info: Optional[dict] = None,
    competency_scores: Optional[list[dict]] = None,
    profile: AssessmentProfile = GENERIC_PROFILE,
    hire_outcome_calibration_info: Optional[dict] = None,
) -> dict:
    """A distinct-from-`calibrated_confidence` estimate of how likely this
    candidate would be selected, expressed as {"percent": int (1-99),
    "label": str, "basis": str, "binary_recommendation": "Recommended"|"Not
    Recommended"}. Never returns 0 or 100 -- this is always an estimate,
    never a certainty. `binary_recommendation` is a simple >=50% reading of
    the same estimate -- shown ALONGSIDE the percentage/label (per explicit
    user request), never in place of it, since collapsing to a single yes/no
    is exactly what the percentage was added to avoid.

    A deliberately pure function -- `nudge` is a precomputed float (see
    transcript_specificity_nudge, which does the actual DB-reading/text-
    analysis work), not a transcript, so this stays as trivially testable as
    it always was: no db, no I/O, plain numbers in, plain numbers out.

    Combines three inputs:
      1. hire_recommendation["level"] (the model's own 7-point hire-scale
         call, see rubric.py) anchors a baseline percentage.
      2. `nudge`, clamped to +-_MAX_NUDGE, shifts that baseline up or down --
         genuinely independent of #1, since it isn't computed from anything
         the model produced (see transcript_specificity_nudge's docstring
         for why the previous design, nudging with the same competency
         scores the model was told to ground hire_recommendation in,
         wasn't really independent evidence at all).
      3. `competency_scores`/`profile` (optional): a small, capped "sanity
         check" correction -- when the profile-weighted competency average
         diverges sharply (more than _COMPETENCY_DISAGREEMENT_THRESHOLD
         points) from the hire-scale anchor, that disagreement is scaled
         down (_COMPETENCY_DISAGREEMENT_SCALE) and capped
         (_MAX_COMPETENCY_DISAGREEMENT_CORRECTION) into a second, smaller
         correction to the baseline. This is deliberately NOT a primary
         driver like #2 -- it only fires on meaningful disagreement, and
         even then moves the number far less than the transcript nudge can,
         because the model was explicitly told to ground the hire-scale
         call in these same competency scores (see rubric.py), so this
         corrects for cases where it visibly failed to do so rather than
         treating the two as independent evidence.

    A fourth input, confidence_info (from calibrated_confidence), pulls the
    whole estimate toward a neutral 50% the less trustworthy the underlying
    assessment is -- a low-confidence assessment must not produce a falsely
    precise-looking selection probability. This pull only ever moves the
    number BETWEEN the corrected baseline and 50%, never past either one --
    so a low hire-scale anchor (e.g. "Lean No Hire" = 30%) can never end up
    at 95+ regardless of the nudge, disagreement correction, or confidence;
    only a high anchor itself (Strong Hire/Exceptional) can.

    A fifth, final input, `hire_outcome_calibration_info` (from
    hire_outcome_calibration -- see its own docstring), is ground truth:
    this user's own submitted real-world hired/not-hired outcomes for past
    interviews of a similar type. Where every other input is a proxy for
    the outcome, this IS the outcome, so it's applied last, rescaling the
    already-computed percent using this user's own historical "hired
    interviews estimated around X%, not-hired ones around Y%" as reference
    points -- blended in proportional to how many labeled outcomes back it
    (see _HIRE_OUTCOME_CALIBRATION_SATURATION_SAMPLES), capped at
    _MAX_HIRE_OUTCOME_CALIBRATION_WEIGHT so even a large labeled history
    stays one input among several rather than fully overriding the rest.
    """
    level = (hire_recommendation or {}).get("level") or ""
    anchor = _HIRE_LEVEL_ANCHOR.get(level, _NEUTRAL_PERCENT)

    clamped_nudge = max(-_MAX_NUDGE, min(_MAX_NUDGE, nudge)) if nudge is not None else 0.0
    after_nudge = max(0, min(100, anchor + clamped_nudge))

    weighted_avg = weighted_competency_total(competency_scores, profile) if competency_scores else None
    disagreement_correction = 0.0
    if weighted_avg is not None:
        disagreement = weighted_avg - anchor
        if abs(disagreement) > _COMPETENCY_DISAGREEMENT_THRESHOLD:
            excess = abs(disagreement) - _COMPETENCY_DISAGREEMENT_THRESHOLD
            magnitude = min(excess * _COMPETENCY_DISAGREEMENT_SCALE, _MAX_COMPETENCY_DISAGREEMENT_CORRECTION)
            disagreement_correction = magnitude if disagreement > 0 else -magnitude
    baseline = max(0, min(100, after_nudge + disagreement_correction))

    confidence_score = (confidence_info or {}).get("score")
    # No usable confidence figure at all -> treat as low-trust (0.5), same
    # spirit as calibrated_confidence's own "unavailable" fallback -- don't
    # let a missing confidence signal accidentally read as full trust.
    confidence_weight = 0.5 if confidence_score is None else max(0.0, min(1.0, confidence_score / 100))
    pulled = _NEUTRAL_PERCENT + (baseline - _NEUTRAL_PERCENT) * confidence_weight
    percent = max(1, min(99, round(pulled)))

    # confidence_weight is how much of the baseline is KEPT (1.0 = fully
    # trust it, 0.0 = ignore it entirely and land exactly on neutral), so
    # the fraction actually pulled toward 50% is its complement -- stating
    # confidence_weight itself as "pulled toward neutral at X% strength"
    # reads backwards (a reader wrote it that way once and asked whether
    # the resulting number was a bug -- it wasn't, the wording was just
    # confusing about which direction X% applied to).
    pulled_fraction = 1 - confidence_weight

    # Ground-truth recalibration -- see hire_outcome_calibration_info's
    # docstring above. Rescales `percent` using this user's own historical
    # "hired interviews estimated around X%, not-hired ones around Y%" as
    # the 0/100 reference points, then blends that rescaled value back in
    # proportional to how much labeled history backs it, rather than
    # applying it at full strength.
    calibration_clause = ""
    if hire_outcome_calibration_info is not None:
        hired_anchor = hire_outcome_calibration_info.get("hired_anchor")
        not_hired_anchor = hire_outcome_calibration_info.get("not_hired_anchor")
        sample_size = hire_outcome_calibration_info.get("sample_size", 0)
        specificity = hire_outcome_calibration_info.get("specificity", "your labeled interviews")
        if (
            isinstance(hired_anchor, (int, float)) and isinstance(not_hired_anchor, (int, float))
            and hired_anchor > not_hired_anchor
        ):
            calibration_weight = (
                min(1.0, sample_size / _HIRE_OUTCOME_CALIBRATION_SATURATION_SAMPLES)
                * _MAX_HIRE_OUTCOME_CALIBRATION_WEIGHT
            )
            scale = 100.0 / (hired_anchor - not_hired_anchor)
            rescaled = max(0.0, min(100.0, (percent - not_hired_anchor) * scale))
            calibrated_float = percent + (rescaled - percent) * calibration_weight
            new_percent = max(1, min(99, round(calibrated_float)))
            if new_percent != percent:
                calibration_clause = (
                    f" Recalibrated against {specificity} ({sample_size} labeled outcomes -- hired "
                    f"interviews you've submitted feedback for were estimated around {hired_anchor:.0f}% on "
                    f"average, not-hired ones around {not_hired_anchor:.0f}%), from {percent}% to "
                    f"{new_percent}%."
                )
            percent = new_percent

    disagreement_clause = ""
    if disagreement_correction != 0.0:
        disagreement_clause = (
            f" competency scores averaged {weighted_avg:.0f}%, disagreeing enough with the "
            f"hire-scale anchor to apply a further {disagreement_correction:+.0f} point sanity-check "
            f"correction to {baseline:.0f}%;"
        )
    basis = (
        f"Hire-scale call: \"{level or 'not given'}\" (anchors {anchor}%); "
        f"answer specificity vs. your own past interviews nudged it by {clamped_nudge:+.0f} points to "
        f"{after_nudge:.0f}%;"
        f"{disagreement_clause} "
        f"{round(confidence_weight * 100)}% assessment confidence kept the estimate close to "
        f"that baseline, pulling only {round(pulled_fraction * 100)}% of the way toward a "
        f"neutral 50%.{calibration_clause}"
    )
    binary_recommendation = "Recommended" if percent >= _RECOMMENDED_THRESHOLD else "Not Recommended"
    return {
        "percent": percent,
        "label": level or None,
        "basis": basis,
        "binary_recommendation": binary_recommendation,
        # The precomputed nudge this estimate used, carried through so
        # callers with only the persisted result (e.g. infographic.py's
        # seniority comparison) can re-run this same function with a
        # different profile/seniority without needing the db/transcript
        # access transcript_specificity_nudge itself requires.
        "nudge": clamped_nudge,
    }
