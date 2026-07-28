"""Tests for confidence.py: the "how much should I trust this assessment"
score shown on each report, and the calibration notes fed back into future
analysis prompts from past feedback. Feedback is a 1-10 quality score (10
highest), not a Yes/No -- see db.py's FeedbackRecord."""
from __future__ import annotations

from unittest.mock import MagicMock

from interview_analyzer.confidence import (
    MIN_FEEDBACK_SAMPLES,
    MIN_HIRE_OUTCOME_SAMPLES,
    MIN_SPECIFICITY_SAMPLES,
    NEGATIVE_SCORE_THRESHOLD,
    _candidate_turns,
    _specificity_fraction,
    calibrated_confidence,
    calibration_notes,
    competency_weight,
    estimate_selection_probability,
    format_confidence,
    hire_outcome_calibration,
    transcript_specificity_nudge,
    weighted_competency_total,
)
from interview_analyzer.db import InterviewDB
from interview_analyzer.profiles import GENERIC_PROFILE, AssessmentProfile


class TestCalibratedConfidence:
    def test_falls_back_to_model_reported_score_when_too_little_feedback(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")  # empty feedback table
        result = calibrated_confidence(db, user_id=1, model_reported=72)
        assert result == {"score": 72, "source": "model", "sample_size": 0}

    def test_returns_unavailable_when_no_feedback_and_no_model_score(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        result = calibrated_confidence(db, user_id=1, model_reported=None)
        assert result["score"] is None
        assert result["source"] == "unavailable"

    def test_uses_average_feedback_score_once_enough_samples_exist(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid_a = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        iid_b = db.start_interview("Zoom", str(tmp_path / "b.wav"), retention_days=3, user_id=1)
        iid_c = db.start_interview("Zoom", str(tmp_path / "c.wav"), retention_days=3, user_id=1)
        db.save_feedback(iid_a, user_id=1, transcript_score=10, analysis_score=10, comment="")
        db.save_feedback(iid_b, user_id=1, transcript_score=10, analysis_score=8, comment="")
        db.save_feedback(iid_c, user_id=1, transcript_score=10, analysis_score=3, comment="")

        assert MIN_FEEDBACK_SAMPLES == 3  # this test assumes exactly the threshold
        result = calibrated_confidence(db, user_id=1, model_reported=99)
        # average analysis_score = (10+8+3)/3 = 7.0 -> 70%
        assert result == {"score": 70, "source": "feedback", "sample_size": 3}

    def test_ignores_ratings_that_only_cover_transcript_not_analysis(self, tmp_path):
        """analysis_score is the signal for analysis confidence -- someone
        who only ever rated transcription accuracy shouldn't count toward
        the analysis-quality sample size."""
        db = InterviewDB(tmp_path / "test.db")
        for i in range(5):
            iid = db.start_interview("Zoom", str(tmp_path / f"{i}.wav"), retention_days=3, user_id=1)
            db.save_feedback(iid, user_id=1, transcript_score=2, analysis_score=None, comment="")

        result = calibrated_confidence(db, user_id=1, model_reported=55)
        assert result == {"score": 55, "source": "model", "sample_size": 0}

    def test_falls_back_gracefully_when_feedback_table_is_unreadable(self):
        broken_db = MagicMock()
        broken_db.list_feedback.side_effect = RuntimeError("db locked")

        result = calibrated_confidence(broken_db, user_id=1, model_reported=80)
        assert result == {"score": 80, "source": "model", "sample_size": 0}

    def test_returns_unavailable_when_feedback_unreadable_and_no_model_score(self):
        broken_db = MagicMock()
        broken_db.list_feedback.side_effect = RuntimeError("db locked")

        result = calibrated_confidence(broken_db, user_id=1, model_reported=None)
        assert result["score"] is None
        assert result["source"] == "unavailable"

    def test_clamps_out_of_range_model_scores(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        assert calibrated_confidence(db, user_id=1, model_reported=150)["score"] == 100
        assert calibrated_confidence(db, user_id=1, model_reported=-10)["score"] == 0

    def test_handles_non_numeric_model_score_gracefully(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        result = calibrated_confidence(db, user_id=1, model_reported="not a number")
        assert result["score"] is None
        assert result["source"] == "unavailable"


class TestFormatConfidence:
    def test_feedback_sourced_score(self):
        text = format_confidence({"score": 82, "source": "feedback", "sample_size": 12})
        assert "82%" in text
        assert "12" in text

    def test_model_sourced_score(self):
        text = format_confidence({"score": 74, "source": "model", "sample_size": 0})
        assert "74%" in text
        assert "self-assessment" in text

    def test_unavailable(self):
        assert format_confidence({"score": None, "source": "unavailable", "sample_size": 0}) == "not available"
        assert format_confidence(None) == "not available"


class TestCalibrationNotes:
    def test_empty_when_no_negative_feedback(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        db.save_feedback(iid, user_id=1, transcript_score=9, analysis_score=9, comment="great")

        assert calibration_notes(db, user_id=1) == ""

    def test_includes_comments_from_low_scoring_feedback(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        db.save_feedback(
            iid, user_id=1, transcript_score=9, analysis_score=NEGATIVE_SCORE_THRESHOLD,
            comment="missed that I gave a metric in my answer",
        )

        notes = calibration_notes(db, user_id=1)
        assert "missed that I gave a metric in my answer" in notes

    def test_excludes_comments_from_above_threshold_feedback(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        db.save_feedback(
            iid, user_id=1, transcript_score=9, analysis_score=NEGATIVE_SCORE_THRESHOLD + 1,
            comment="a decent analysis, minor nitpick",
        )

        assert calibration_notes(db, user_id=1) == ""

    def test_ignores_negative_feedback_without_a_comment(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        db.save_feedback(iid, user_id=1, transcript_score=9, analysis_score=1, comment="")

        assert calibration_notes(db, user_id=1) == ""

    def test_falls_back_to_empty_when_feedback_table_is_unreadable(self):
        broken_db = MagicMock()
        broken_db.list_feedback.side_effect = RuntimeError("db locked")

        assert calibration_notes(broken_db, user_id=1) == ""

    def test_limits_to_the_most_recent_n_comments(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        for i in range(8):
            iid = db.start_interview("Zoom", str(tmp_path / f"{i}.wav"), retention_days=3, user_id=1)
            db.save_feedback(iid, user_id=1, transcript_score=9, analysis_score=1, comment=f"issue {i}")

        notes = calibration_notes(db, user_id=1, limit=3)
        assert notes.count("- issue") == 3
        assert "issue 7" in notes  # most recent kept
        assert "issue 0" not in notes  # oldest dropped


class TestEstimateSelectionProbability:
    """estimate_selection_probability is deliberately NOT the same thing as
    calibrated_confidence -- it's an estimate of the assessed *outcome*
    (would this candidate be selected), not trust in the assessment's
    accuracy. It blends the model's own hire-scale call with a precomputed
    `nudge` float (see transcript_specificity_nudge -- estimate_selection_
    probability itself stays a pure function, no db/transcript involved)
    and the assessment's own confidence."""

    def test_never_returns_a_false_certainty_of_0_or_100(self):
        # even a maximally-positive call at full confidence should not claim
        # absolute certainty
        result = estimate_selection_probability({"level": "Exceptional"}, 15, confidence_info={"score": 100})
        assert 1 <= result["percent"] <= 99

        result = estimate_selection_probability({"level": "Strong No Hire"}, -15, confidence_info={"score": 100})
        assert 1 <= result["percent"] <= 99

    def test_higher_hire_scale_levels_produce_higher_percentages(self):
        low = estimate_selection_probability({"level": "Strong No Hire"}, None, confidence_info={"score": 90})
        mid = estimate_selection_probability({"level": "Lean Hire"}, None, confidence_info={"score": 90})
        high = estimate_selection_probability({"level": "Exceptional"}, None, confidence_info={"score": 90})
        assert low["percent"] < mid["percent"] < high["percent"]

    def test_low_confidence_pulls_the_estimate_toward_a_neutral_midpoint(self):
        """Regression coverage for the explicit requirement: a low-
        confidence assessment must not produce a falsely-precise-looking
        selection probability."""
        confident = estimate_selection_probability({"level": "Exceptional"}, None, confidence_info={"score": 95})
        unsure = estimate_selection_probability({"level": "Exceptional"}, None, confidence_info={"score": 10})
        assert unsure["percent"] < confident["percent"]
        assert abs(unsure["percent"] - 50) < abs(confident["percent"] - 50)

    def test_missing_confidence_info_is_treated_as_low_trust_not_full_trust(self):
        with_none = estimate_selection_probability({"level": "Exceptional"}, None, confidence_info=None)
        with_high = estimate_selection_probability({"level": "Exceptional"}, None, confidence_info={"score": 100})
        assert with_none["percent"] < with_high["percent"]

    def test_unrecognized_or_missing_hire_level_anchors_to_neutral(self):
        result = estimate_selection_probability(None, None, confidence_info={"score": 90})
        assert 40 <= result["percent"] <= 60
        assert result["label"] is None

    def test_positive_nudge_raises_the_estimate_negative_lowers_it(self):
        boosted = estimate_selection_probability({"level": "Lean Hire"}, 15, confidence_info={"score": 90})
        muted = estimate_selection_probability({"level": "Lean Hire"}, -15, confidence_info={"score": 90})
        neutral = estimate_selection_probability({"level": "Lean Hire"}, 0, confidence_info={"score": 90})
        assert muted["percent"] < neutral["percent"] < boosted["percent"]

    def test_nudge_beyond_the_max_is_clamped(self):
        clamped = estimate_selection_probability({"level": "Lean Hire"}, 1000, confidence_info={"score": 100})
        at_max = estimate_selection_probability({"level": "Lean Hire"}, 15, confidence_info={"score": 100})
        assert clamped["percent"] == at_max["percent"]

    def test_none_nudge_is_treated_as_no_adjustment(self):
        """None (e.g. a mono transcript with no isolable candidate speech,
        see transcript_specificity_nudge) means "no signal available", not
        "penalize this interview" -- it should read the same as an
        explicit 0."""
        with_none = estimate_selection_probability({"level": "Lean Hire"}, None, confidence_info={"score": 100})
        with_zero = estimate_selection_probability({"level": "Lean Hire"}, 0, confidence_info={"score": 100})
        assert with_none["percent"] == with_zero["percent"]

    def test_basis_mentions_the_hire_level_and_confidence(self):
        result = estimate_selection_probability({"level": "Hire"}, 5, confidence_info={"score": 85})
        assert "Hire" in result["basis"]
        assert "confidence" in result["basis"].lower()

    def test_basis_states_confidence_as_how_much_it_stayed_near_baseline_not_how_hard_it_was_pulled(self):
        """Regression coverage for a real point of confusion: the basis
        text used to read "pulled toward a neutral 50% at X% strength"
        where X was actually confidence_weight -- the fraction of the
        baseline KEPT, not the fraction pulled toward neutral. A reader
        took that at face value and asked why a 74%-confidence result
        (36%) wasn't close to 50 -- it wasn't a bug, the wording was just
        backwards about which direction the percentage applied to."""
        result = estimate_selection_probability({"level": "Lean No Hire"}, 0.9, confidence_info={"score": 74})
        assert result["percent"] == 36
        assert "74% assessment confidence kept the estimate close to that baseline" in result["basis"]
        assert "pulling only 26% of the way toward a neutral 50%" in result["basis"]

    def test_low_anchor_can_never_cross_neutral_regardless_of_nudge_or_confidence(self):
        """Confirms a real ceiling: from a sub-50 hire-scale anchor, no
        combination of nudge or confidence can push the estimate to 95+ --
        confidence only ever moves the number BETWEEN the anchor+nudge
        baseline and 50%, never past 50%."""
        for confidence_score in (0, 25, 50, 74, 100):
            result = estimate_selection_probability(
                {"level": "Lean No Hire"}, 15,  # anchors at 30%; max possible nudge
                confidence_info={"score": confidence_score},
            )
            assert result["percent"] <= 50

    def test_binary_recommendation_is_recommended_at_or_above_the_pivot(self):
        result = estimate_selection_probability({"level": "Strong Hire"}, None, confidence_info={"score": 90})
        assert result["percent"] >= 50
        assert result["binary_recommendation"] == "Recommended"

    def test_binary_recommendation_is_not_recommended_below_the_pivot(self):
        result = estimate_selection_probability({"level": "Strong No Hire"}, None, confidence_info={"score": 90})
        assert result["percent"] < 50
        assert result["binary_recommendation"] == "Not Recommended"

    def test_binary_recommendation_is_shown_alongside_not_instead_of_the_percentage(self):
        """Regression guard for the explicit user requirement: the output
        must be a probability WITH a recommendation, never a bare binary in
        place of the percentage."""
        result = estimate_selection_probability({"level": "Hire"}, None, confidence_info={"score": 90})
        assert isinstance(result["percent"], int)
        assert result["binary_recommendation"] in ("Recommended", "Not Recommended")

    def test_competency_scores_omitted_behaves_identically_to_before(self):
        """Adding the competency-disagreement sanity check must not change
        behavior for callers that don't pass competency_scores at all."""
        without = estimate_selection_probability({"level": "Lean No Hire"}, 0.9, confidence_info={"score": 74})
        assert without["percent"] == 36
        assert "sanity-check" not in without["basis"]

    def test_competency_scores_agreeing_with_anchor_makes_no_correction(self):
        """A weighted competency average close to the hire-scale anchor is
        not a disagreement -- no correction should be applied."""
        scores = [{"name": "Ownership", "score": 30}]
        result = estimate_selection_probability(
            {"level": "Lean No Hire"}, 0, confidence_info={"score": 90}, competency_scores=scores,
        )
        assert "sanity-check" not in result["basis"]

    def test_competency_scores_disagreeing_sharply_nudges_the_estimate_up(self):
        """Real-data regression: interview #11 had hire level 'Lean No
        Hire' (anchors 30%) but a weighted competency average of ~59%, a
        29-point disagreement -- past the 20-point threshold, this should
        apply a small positive correction rather than leaving the estimate
        pinned to the raw anchor."""
        scores = [{"name": "Ownership", "score": 59}]
        without = estimate_selection_probability(
            {"level": "Lean No Hire"}, 0, confidence_info={"score": 76},
        )
        with_disagreement = estimate_selection_probability(
            {"level": "Lean No Hire"}, 0, confidence_info={"score": 76}, competency_scores=scores,
        )
        assert with_disagreement["percent"] > without["percent"]
        assert "disagreeing enough" in with_disagreement["basis"]

    def test_competency_disagreement_correction_is_capped(self):
        """Even an extreme disagreement (e.g. a 0% anchor vs. a 100%
        competency average) must not blow past
        _MAX_COMPETENCY_DISAGREEMENT_CORRECTION -- this is a sanity check,
        not a second primary driver."""
        mild = estimate_selection_probability(
            {"level": "Strong No Hire"}, 0, confidence_info={"score": 100},
            competency_scores=[{"name": "Ownership", "score": 70}],
        )
        extreme = estimate_selection_probability(
            {"level": "Strong No Hire"}, 0, confidence_info={"score": 100},
            competency_scores=[{"name": "Ownership", "score": 100}],
        )
        # both disagreements exceed the threshold, so both should be capped
        # to the same maximum correction rather than scaling without bound
        assert extreme["percent"] == mild["percent"]

    def test_competency_disagreement_correction_can_pull_the_estimate_down_too(self):
        """A weighted competency average well BELOW the hire-scale anchor
        is just as much a disagreement as one above it, and should pull the
        estimate down, not just up."""
        without = estimate_selection_probability(
            {"level": "Strong Hire"}, 0, confidence_info={"score": 90},
        )
        with_disagreement = estimate_selection_probability(
            {"level": "Strong Hire"}, 0, confidence_info={"score": 90},
            competency_scores=[{"name": "Ownership", "score": 10}],
        )
        assert with_disagreement["percent"] < without["percent"]


class TestCandidateTurns:
    """The candidate's ("You") own speech, read directly off the raw
    transcript -- the whole point of transcript_specificity_nudge's
    independence from hire_recommendation/competency_scores is that this
    never touches anything the model produced."""

    def test_extracts_only_the_candidate_speaker(self):
        transcript = "[Interviewer] Hi\n[You] Hello\n[Interviewer] How are you\n[You] Good"
        assert _candidate_turns(transcript) == ["Hello", "Good"]

    def test_merges_consecutive_candidate_lines_into_one_turn(self):
        """faster-whisper often splits one continuous utterance across
        several segments -- a real answer that starts with a one-word
        acknowledgment shouldn't count as two separate, mostly-trivial
        turns."""
        transcript = "[You] Mm-hmm.\n[You] So, for that project I led a team.\n[Interviewer] Great."
        assert _candidate_turns(transcript) == ["Mm-hmm. So, for that project I led a team."]

    def test_empty_transcript_returns_no_turns(self):
        assert _candidate_turns("") == []

    def test_mono_fallback_speaker_label_yields_no_candidate_turns(self):
        """A no-diarization transcript labels everyone "Speaker", not
        "You" (see transcriber.py's fallback path) -- there's no reliable
        way to isolate candidate-only speech from that, so this correctly
        finds nothing rather than guessing."""
        transcript = "[Speaker] Hello\n[Speaker] there"
        assert _candidate_turns(transcript) == []


class TestSpecificityFraction:
    def test_none_when_no_candidate_turns(self):
        assert _specificity_fraction("[Interviewer] Hi") is None

    def test_fraction_of_turns_containing_a_digit(self):
        transcript = "[You] I grew revenue by 20%.\n[Interviewer] ok\n[You] I think it went well overall."
        assert _specificity_fraction(transcript) == 0.5


class TestTranscriptSpecificityNudge:
    """Reads relative to this SAME user's own past interviews (not a
    guessed universal "normal" fraction, since how quantitative a "typical"
    answer is varies enormously by role) -- self-calibrating the same way
    calibrated_confidence bootstraps from the user's own feedback history."""

    VAGUE = "[Interviewer] How'd it go?\n[You] I think it went pretty well overall, honestly."
    SPECIFIC = "[You] I grew revenue by 20% over 6 months and cut costs by $5000 in Q3."

    def _seed(self, db, transcript, user_id=1):
        iid = db.start_interview("Zoom", "a.wav", retention_days=3, user_id=user_id)
        db.save_transcript(iid, transcript)
        return iid

    def test_none_when_this_transcript_has_no_candidate_speech(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        assert transcript_specificity_nudge(db, 1, "[Interviewer] Hi") is None

    def test_zero_when_not_enough_history_yet(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        for _ in range(MIN_SPECIFICITY_SAMPLES - 1):
            self._seed(db, self.VAGUE)
        assert transcript_specificity_nudge(db, 1, self.SPECIFIC) == 0.0

    def test_positive_when_more_specific_than_this_users_own_history(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        for _ in range(MIN_SPECIFICITY_SAMPLES):
            self._seed(db, self.VAGUE)
        assert transcript_specificity_nudge(db, 1, self.SPECIFIC) > 0

    def test_negative_when_less_specific_than_this_users_own_history(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        for _ in range(MIN_SPECIFICITY_SAMPLES):
            self._seed(db, self.SPECIFIC)
        assert transcript_specificity_nudge(db, 1, self.VAGUE) < 0

    def test_excludes_the_given_interview_id_from_its_own_baseline(self, tmp_path):
        """Reprocessing an already-stored interview must not compare it
        against itself -- its transcript is already saved in the DB by the
        time watcher.py computes this nudge."""
        db = InterviewDB(tmp_path / "test.db")
        this_iid = self._seed(db, self.SPECIFIC)
        for _ in range(MIN_SPECIFICITY_SAMPLES - 1):
            self._seed(db, self.VAGUE)

        # excluding itself leaves too few OTHER samples -> no nudge yet
        excluded = transcript_specificity_nudge(db, 1, self.SPECIFIC, exclude_interview_id=this_iid)
        assert excluded == 0.0

        # not excluding it pads the pool back up, wrongly comparing the
        # interview against itself
        included = transcript_specificity_nudge(db, 1, self.SPECIFIC, exclude_interview_id=None)
        assert included != 0.0

    def test_scoped_per_user(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        for _ in range(MIN_SPECIFICITY_SAMPLES):
            self._seed(db, self.SPECIFIC, user_id=2)
        # user 1 has no history of their own, even though user 2 does
        assert transcript_specificity_nudge(db, 1, self.VAGUE) == 0.0

    def test_db_read_failure_falls_back_to_no_history(self, tmp_path):
        db = MagicMock()
        db.list_all.side_effect = Exception("boom")
        assert transcript_specificity_nudge(db, 1, self.SPECIFIC) == 0.0


class TestWeightedCompetencyTotal:
    """The overall 0-100 competency score shown as a "scorecard" in
    report.py/infographic.py, and reused internally by
    estimate_selection_probability to nudge its hire-scale anchor."""

    def test_none_when_no_scores_given(self):
        assert weighted_competency_total(None) is None
        assert weighted_competency_total([]) is None

    def test_simple_average_under_a_flat_profile(self):
        """GENERIC_PROFILE has no role/seniority/industry/company, so every
        competency gets the same "moderate" weight -- a plain average."""
        scores = [{"name": "Leadership", "score": 60}, {"name": "Execution", "score": 80}]
        assert weighted_competency_total(scores) == 70.0

    def test_critical_competency_counts_more_than_a_minor_one(self):
        """Director+ rates Technical Expertise "low" and Ownership
        "critical" -- a high Ownership score should pull the total up more
        than the same score on Technical Expertise would."""
        profile = AssessmentProfile(competencies=["Technical Expertise", "Ownership"], seniority="Director+")
        ownership_high = weighted_competency_total(
            [{"name": "Technical Expertise", "score": 50}, {"name": "Ownership", "score": 90}], profile,
        )
        technical_high = weighted_competency_total(
            [{"name": "Technical Expertise", "score": 90}, {"name": "Ownership", "score": 50}], profile,
        )
        assert ownership_high > technical_high

    def test_ignores_entries_with_missing_or_non_numeric_scores(self):
        scores = [{"name": "Leadership", "score": 80}, {"name": "Execution", "score": None}, "not a dict"]
        assert weighted_competency_total(scores) == 80.0


class TestCompetencyWeight:
    def test_returns_a_higher_number_for_a_more_emphasized_competency(self):
        profile = AssessmentProfile(competencies=["Leadership"], seniority="Director+")
        assert competency_weight("Leadership", profile) > competency_weight("Learning Agility", profile)

    def test_defaults_to_moderate_for_an_unrecognized_competency_under_a_generic_profile(self):
        assert competency_weight("Leadership") == competency_weight("Execution")

    def test_interpolates_rather_than_snapping_to_the_nearest_rounded_tier(self):
        """Regression coverage for the exact bug that flattened the
        seniority-comparison gauge to identical numbers for adjacent
        seniority levels: Senior/Lead and Director+ both round Technical
        Expertise to "moderate" under a Product/Generic/Generic profile
        (raw averages 2.25 vs 2.0), which used to collapse both to the
        SAME 1.5 weight. The interpolated weight must differ instead,
        since the raw averages genuinely do."""
        senior_lead = AssessmentProfile(
            role="Product", seniority="Senior/Lead", industry="Generic", company_type="Generic",
        )
        director_plus = AssessmentProfile(
            role="Product", seniority="Director+", industry="Generic", company_type="Generic",
        )
        senior_weight = competency_weight("Technical Expertise", senior_lead)
        director_weight = competency_weight("Technical Expertise", director_plus)
        assert senior_weight > director_weight
        assert director_weight == 1.5  # Director+'s raw average lands exactly on "moderate"

    def test_interpolated_weight_still_matches_the_tier_weight_at_exact_boundaries(self):
        """A profile whose average lands exactly on a tier boundary (no
        fractional remainder) must still return that tier's own weight
        exactly -- the interpolation shouldn't introduce drift at the
        boundaries it's meant to preserve."""
        profile = AssessmentProfile(
            competencies=["Business Acumen"], role="Product", seniority="Senior/Lead",
            industry="FinTech", company_type="Consulting",
        )
        assert competency_weight("Business Acumen", profile) == 3.0  # exactly "critical"


class TestHireOutcomeCalibration:
    """Ground-truth calibration from the user's own submitted hired/not-
    hired outcomes (db.py's FeedbackRecord.hire_outcome), scoped to
    interviews of a similar type (role/seniority) via a most-specific-first
    backoff -- see _HIRE_OUTCOME_MATCH_LEVELS."""

    def _seed(self, db, profile, percent, hire_outcome, user_id=1):
        iid = db.start_interview("Zoom", "a.wav", retention_days=3, user_id=user_id)
        db.save_profile_snapshot(iid, profile)
        db.save_analysis(iid, {"selection_probability": {"percent": percent}})
        db.save_feedback(iid, user_id=user_id, transcript_score=None, analysis_score=None, comment="",
                          hire_outcome=hire_outcome)
        return iid

    def test_none_with_no_labeled_outcomes_at_all(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        assert hire_outcome_calibration(db, 1, GENERIC_PROFILE) is None

    def test_none_below_the_minimum_sample_size_even_with_both_classes(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        self._seed(db, GENERIC_PROFILE, 60, "Hired")
        self._seed(db, GENERIC_PROFILE, 40, "Not Hired")
        assert MIN_HIRE_OUTCOME_SAMPLES > 2  # sanity check on the constant this test relies on
        assert hire_outcome_calibration(db, 1, GENERIC_PROFILE) is None

    def test_none_when_only_one_outcome_class_is_represented(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        for percent in (60, 62, 64, 66):
            self._seed(db, GENERIC_PROFILE, percent, "Hired")
        assert hire_outcome_calibration(db, 1, GENERIC_PROFILE) is None

    def test_computes_anchors_from_global_history_with_no_profile_set(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        self._seed(db, GENERIC_PROFILE, 70, "Hired")
        self._seed(db, GENERIC_PROFILE, 74, "Hired")
        self._seed(db, GENERIC_PROFILE, 30, "Not Hired")
        self._seed(db, GENERIC_PROFILE, 34, "Not Hired")

        result = hire_outcome_calibration(db, 1, GENERIC_PROFILE)

        assert result["hired_anchor"] == 72
        assert result["not_hired_anchor"] == 32
        assert result["sample_size"] == 4
        assert result["specificity"] == "all your interviews"

    def test_prefers_the_most_specific_role_and_seniority_match(self, tmp_path):
        """A profile with the SAME role+seniority as the current one must
        be used in preference to a larger but less specific pool, even
        though the broader pool alone would also satisfy the minimum
        sample size."""
        db = InterviewDB(tmp_path / "test.db")
        current = AssessmentProfile(role="Product", seniority="Senior/Lead")
        matching = AssessmentProfile(role="Product", seniority="Senior/Lead")
        other_role = AssessmentProfile(role="Data", seniority="Senior/Lead")

        self._seed(db, matching, 70, "Hired")
        self._seed(db, matching, 74, "Hired")
        self._seed(db, matching, 30, "Not Hired")
        self._seed(db, matching, 34, "Not Hired")
        # a much larger, differently-role'd pool at the same seniority --
        # must NOT be what gets used, even though it alone qualifies
        for _ in range(5):
            self._seed(db, other_role, 95, "Hired")
            self._seed(db, other_role, 5, "Not Hired")

        result = hire_outcome_calibration(db, 1, current)

        assert result["specificity"] == "role and seniority"
        assert result["sample_size"] == 4
        assert result["hired_anchor"] == 72
        assert result["not_hired_anchor"] == 32

    def test_falls_back_to_seniority_alone_when_the_specific_group_is_too_small(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        current = AssessmentProfile(role="Product", seniority="Senior/Lead")
        same_role_and_seniority = AssessmentProfile(role="Product", seniority="Senior/Lead")
        same_seniority_only = AssessmentProfile(role="Data", seniority="Senior/Lead")

        # only 2 in the exact role+seniority group -- below the minimum
        self._seed(db, same_role_and_seniority, 70, "Hired")
        self._seed(db, same_role_and_seniority, 30, "Not Hired")
        # 2 more at the same seniority but a different role, bringing the
        # broader "seniority" group to 4 total
        self._seed(db, same_seniority_only, 74, "Hired")
        self._seed(db, same_seniority_only, 34, "Not Hired")

        result = hire_outcome_calibration(db, 1, current)

        assert result["specificity"] == "seniority"
        assert result["sample_size"] == 4

    def test_falls_back_to_role_alone_when_seniority_never_matches(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        current = AssessmentProfile(role="Product", seniority="Senior/Lead")
        same_role_different_seniority = AssessmentProfile(role="Product", seniority="Entry Level")

        self._seed(db, same_role_different_seniority, 70, "Hired")
        self._seed(db, same_role_different_seniority, 74, "Hired")
        self._seed(db, same_role_different_seniority, 30, "Not Hired")
        self._seed(db, same_role_different_seniority, 34, "Not Hired")

        result = hire_outcome_calibration(db, 1, current)

        assert result["specificity"] == "role"
        assert result["sample_size"] == 4

    def test_falls_back_to_all_interviews_when_nothing_matches_role_or_seniority(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        current = AssessmentProfile(role="Product", seniority="Senior/Lead")
        unrelated = AssessmentProfile(role="Sales", seniority="Entry Level")

        self._seed(db, unrelated, 70, "Hired")
        self._seed(db, unrelated, 74, "Hired")
        self._seed(db, unrelated, 30, "Not Hired")
        self._seed(db, unrelated, 34, "Not Hired")

        result = hire_outcome_calibration(db, 1, current)

        assert result["specificity"] == "all your interviews"
        assert result["sample_size"] == 4

    def test_a_generic_current_profile_goes_straight_to_all_interviews(self, tmp_path):
        """With no role/seniority set on the CURRENT profile, there's
        nothing meaningful to match against -- skip straight to the
        broadest group rather than "matching" on None == None."""
        db = InterviewDB(tmp_path / "test.db")
        specific = AssessmentProfile(role="Product", seniority="Senior/Lead")

        self._seed(db, specific, 70, "Hired")
        self._seed(db, specific, 74, "Hired")
        self._seed(db, specific, 30, "Not Hired")
        self._seed(db, specific, 34, "Not Hired")

        result = hire_outcome_calibration(db, 1, GENERIC_PROFILE)

        assert result["specificity"] == "all your interviews"

    def test_ignores_labeled_interviews_with_no_stored_percent(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", "a.wav", retention_days=3, user_id=1)
        db.save_analysis(iid, {"session_summary": {}})  # no selection_probability at all
        db.save_feedback(iid, user_id=1, transcript_score=None, analysis_score=None, comment="",
                          hire_outcome="Hired")
        self._seed(db, GENERIC_PROFILE, 70, "Hired")
        self._seed(db, GENERIC_PROFILE, 74, "Hired")
        self._seed(db, GENERIC_PROFILE, 30, "Not Hired")
        self._seed(db, GENERIC_PROFILE, 34, "Not Hired")

        result = hire_outcome_calibration(db, 1, GENERIC_PROFILE)

        # the un-scored interview must not have counted toward sample_size
        # or skewed hired_anchor
        assert result["sample_size"] == 4
        assert result["hired_anchor"] == 72

    def test_scoped_to_the_given_user(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        for percent, outcome in ((70, "Hired"), (74, "Hired"), (30, "Not Hired"), (34, "Not Hired")):
            self._seed(db, GENERIC_PROFILE, percent, outcome, user_id=2)

        assert hire_outcome_calibration(db, 1, GENERIC_PROFILE) is None


class TestEstimateSelectionProbabilityHireOutcomeCalibration:
    """estimate_selection_probability's final, ground-truth recalibration
    step -- see its own docstring for why this is applied last, after the
    other (proxy) signals."""

    def test_omitted_behaves_identically_to_before(self):
        without = estimate_selection_probability({"level": "Strong Hire"}, None, confidence_info={"score": 100})
        assert without["percent"] == 90
        assert "Recalibrated" not in without["basis"]

    def test_recalibrates_toward_the_historical_hired_anchor(self):
        """Real numbers, hand-verified: anchor 90 (Strong Hire, full
        confidence, no other corrections) rescaled against a history where
        hired interviews averaged 70% and not-hired ones 30% (a full-weight
        12-sample group) pushes the estimate up to 96%."""
        result = estimate_selection_probability(
            {"level": "Strong Hire"}, None, confidence_info={"score": 100},
            hire_outcome_calibration_info={
                "hired_anchor": 70, "not_hired_anchor": 30, "sample_size": 12, "specificity": "role and seniority",
            },
        )
        assert result["percent"] == 96
        assert "Recalibrated" in result["basis"]
        assert "role and seniority" in result["basis"]

    def test_calibration_strength_scales_with_sample_size(self):
        """Half the saturation sample count should pull about half as
        hard as the full-weight case above."""
        half_weight = estimate_selection_probability(
            {"level": "Strong Hire"}, None, confidence_info={"score": 100},
            hire_outcome_calibration_info={
                "hired_anchor": 70, "not_hired_anchor": 30, "sample_size": 6, "specificity": "role and seniority",
            },
        )
        full_weight = estimate_selection_probability(
            {"level": "Strong Hire"}, None, confidence_info={"score": 100},
            hire_outcome_calibration_info={
                "hired_anchor": 70, "not_hired_anchor": 30, "sample_size": 12, "specificity": "role and seniority",
            },
        )
        uncalibrated = estimate_selection_probability({"level": "Strong Hire"}, None, confidence_info={"score": 100})
        assert uncalibrated["percent"] < half_weight["percent"] < full_weight["percent"]

    def test_no_calibration_when_anchors_are_not_directionally_sane(self):
        """A hired_anchor at or below not_hired_anchor carries no usable
        direction -- must be ignored rather than applied backwards."""
        result = estimate_selection_probability(
            {"level": "Strong Hire"}, None, confidence_info={"score": 100},
            hire_outcome_calibration_info={
                "hired_anchor": 40, "not_hired_anchor": 60, "sample_size": 12, "specificity": "role",
            },
        )
        assert result["percent"] == 90
        assert "Recalibrated" not in result["basis"]

    def test_can_flip_the_binary_recommendation(self):
        """Regression guard: binary_recommendation must reflect the FINAL,
        post-calibration percent, not the pre-calibration one."""
        uncalibrated = estimate_selection_probability({"level": "Lean No Hire"}, None, confidence_info={"score": 100})
        assert uncalibrated["percent"] == 30
        assert uncalibrated["binary_recommendation"] == "Not Recommended"

        calibrated = estimate_selection_probability(
            {"level": "Lean No Hire"}, None, confidence_info={"score": 100},
            hire_outcome_calibration_info={
                "hired_anchor": 20, "not_hired_anchor": 10, "sample_size": 12, "specificity": "role and seniority",
            },
        )
        assert calibrated["percent"] == 72
        assert calibrated["binary_recommendation"] == "Recommended"

    def test_calibrated_percent_stays_within_the_1_to_99_bounds(self):
        result = estimate_selection_probability(
            {"level": "Exceptional"}, 15, confidence_info={"score": 100},
            hire_outcome_calibration_info={
                "hired_anchor": 55, "not_hired_anchor": 54, "sample_size": 50, "specificity": "role",
            },
        )
        assert 1 <= result["percent"] <= 99
