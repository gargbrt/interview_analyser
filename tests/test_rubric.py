from interview_analyzer.rubric import build_prompt


def test_prompt_asks_for_a_confidence_field():
    prompt = build_prompt("[Interviewer] Hi\n[You] Hello")
    assert "confidence" in prompt.lower()


def test_prompt_without_calibration_notes_has_no_notes_section():
    prompt = build_prompt("[Interviewer] Hi\n[You] Hello")
    assert "past feedback" not in prompt.lower()


def test_prompt_includes_calibration_notes_when_given():
    notes = "- previously missed that a metric was actually given"
    prompt = build_prompt("[Interviewer] Hi\n[You] Hello", calibration_notes=notes)
    assert notes in prompt


def test_prompt_pushes_for_thorough_extraction_of_every_question():
    """Regression coverage for a real bug: analysis of a long, real
    interview extracted only 2 qa_pairs when many more questions were
    actually asked -- the prompt now explicitly asks for every distinct
    question, not just a representative few."""
    prompt = build_prompt("[Interviewer] Hi\n[You] Hello")
    assert "every" in prompt.lower() and "distinct question" in prompt.lower()
    assert "do not stop early" in prompt.lower()
