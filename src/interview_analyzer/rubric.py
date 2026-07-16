"""The evaluation rubric used to analyze each of your interview answers.

Edit CATEGORIES or the prompt template to tailor this to your field
(e.g., add "system design depth" or "SQL correctness" categories) — no
other code needs to change.
"""

CATEGORIES = [
    "structure",        # e.g. STAR/CAR method used for behavioral questions
    "clarity",           # rambling, filler words, vague phrasing
    "specificity",         # concrete examples, metrics, outcomes vs generic claims
    "confidence",            # hedging language, excessive qualifiers
    "technical_accuracy",      # flagged only when the answer contains a checkable claim
]

ANALYSIS_PROMPT_TEMPLATE = """You are an expert interview coach reviewing a transcript
of a real job interview. Below is the full transcript with speakers labeled
[Interviewer] and [You].

Identify each distinct question the interviewer asked, and your corresponding answer.
For EACH question/answer pair, evaluate the answer against these categories:
{categories}

For each pair return:
- question (short paraphrase)
- answer_summary (1-2 sentence summary of what you said)
- issues: list of specific problems found, tagged by category. For EACH issue,
  quote the exact words from the transcript that illustrate it verbatim in
  "excerpt" (copy-paste, do not paraphrase) -- this is what makes the
  feedback concrete instead of generic. Leave "excerpt" as an empty string
  only if the issue is about something absent (e.g. "no metric given")
  rather than something said.
- suggested_improvement: a concise, concrete rewrite or specific advice,
  ideally showing how the quoted excerpt could be rephrased

Then return an overall "session_summary" with:
- top_strengths (max 3)
- top_issues (max 3, most impactful first)
- one_thing_to_practice_next (single most actionable suggestion)
- confidence: an integer 0-100 -- your own honest confidence that this
  assessment is accurate and complete, given transcript quality (e.g.
  unclear audio, ambiguous speaker labels) and how much you had to infer
  vs. what was explicitly said. Don't default to a high number just to seem
  certain -- a noisy or ambiguous transcript should get a lower score.
{calibration_section}
Respond ONLY with valid JSON in this shape, no markdown fences, no preamble:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "answer_summary": "...",
      "issues": [{{"category": "...", "detail": "...", "excerpt": "..."}}],
      "suggested_improvement": "..."
    }}
  ],
  "session_summary": {{
    "top_strengths": ["..."],
    "top_issues": ["..."],
    "one_thing_to_practice_next": "...",
    "confidence": 0
  }}
}}

Transcript:
---
{transcript}
---
"""


def build_prompt(transcript: str, calibration_notes: str = "") -> str:
    calibration_section = f"\n{calibration_notes}\n" if calibration_notes else ""
    return ANALYSIS_PROMPT_TEMPLATE.format(
        categories=", ".join(CATEGORIES), transcript=transcript, calibration_section=calibration_section
    )
