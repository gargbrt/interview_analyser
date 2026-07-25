"""Tests for confidence.py: the "how much should I trust this assessment"
score shown on each report, and the calibration notes fed back into future
analysis prompts from past feedback. Feedback is a 1-10 quality score (10
highest), not a Yes/No -- see db.py's FeedbackRecord."""
from __future__ import annotations

from unittest.mock import MagicMock

from interview_analyzer.confidence import (
    MIN_FEEDBACK_SAMPLES,
    MIN_SPECIFICITY_SAMPLES,
    NEGATIVE_SCORE_THRESHOLD,
    _candidate_turns,
    _specificity_fraction,
    calibrated_confidence,
    calibration_notes,
    competency_weight,
    estimate_selection_probability,
    format_confidence,
    transcript_specificity_nudge,
    weighted_competency_total,
)
from interview_analyzer.db import InterviewDB
from interview_analyzer.profiles import AssessmentProfile


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
