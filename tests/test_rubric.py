from interview_analyzer.rubric import build_prompt, split_transcript_for_chunked_analysis


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


class TestSplitTranscriptForChunkedAnalysis:
    def test_short_transcript_is_a_single_chunk(self):
        transcript = "[Interviewer] Hi\n[You] Hello"
        chunks = split_transcript_for_chunked_analysis(transcript, max_chars=1000)
        assert chunks == [transcript]

    def test_long_transcript_is_split_into_multiple_chunks(self):
        transcript = "\n".join(f"[Interviewer] Question {i}?\n[You] Answer number {i}." for i in range(20))
        chunks = split_transcript_for_chunked_analysis(transcript, max_chars=200)
        assert len(chunks) > 1
        # every chunk (but possibly the last) should respect the budget
        for chunk in chunks[:-1]:
            assert len(chunk) <= 200

    def test_never_splits_a_speaker_turn_mid_line(self):
        """Each line is a whole speaker turn (see transcriber.py) -- a
        chunk boundary must fall between lines, never inside one, or a
        question/answer would be truncated mid-sentence for the model."""
        transcript = "\n".join(f"[Interviewer] Question {i}?\n[You] Answer number {i}." for i in range(20))
        original_lines = transcript.split("\n")
        chunks = split_transcript_for_chunked_analysis(transcript, max_chars=200)

        rejoined_lines = []
        for chunk in chunks:
            rejoined_lines.extend(chunk.split("\n"))
        assert rejoined_lines == original_lines

    def test_no_content_is_lost_or_duplicated(self):
        transcript = "\n".join(f"[Interviewer] Question {i}?\n[You] Answer number {i}." for i in range(50))
        chunks = split_transcript_for_chunked_analysis(transcript, max_chars=150)
        assert "\n".join(chunks) == transcript

    def test_a_single_turn_longer_than_the_budget_still_gets_its_own_chunk(self):
        """Must not infinite-loop or crash if one line alone exceeds
        max_chars -- rare, but a very long single answer is possible."""
        long_line = "[You] " + ("word " * 100)
        transcript = f"[Interviewer] Tell me everything.\n{long_line}"
        chunks = split_transcript_for_chunked_analysis(transcript, max_chars=50)
        assert "\n".join(chunks) == transcript
        assert len(chunks) >= 2
