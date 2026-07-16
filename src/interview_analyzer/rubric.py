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
- issues: list of specific problems found, tagged by category
- suggested_improvement: a concise, concrete rewrite or specific advice

Then return an overall "session_summary" with:
- top_strengths (max 3)
- top_issues (max 3, most impactful first)
- one_thing_to_practice_next (single most actionable suggestion)

Respond ONLY with valid JSON in this shape, no markdown fences, no preamble:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "answer_summary": "...",
      "issues": [{{"category": "...", "detail": "..."}}],
      "suggested_improvement": "..."
    }}
  ],
  "session_summary": {{
    "top_strengths": ["..."],
    "top_issues": ["..."],
    "one_thing_to_practice_next": "..."
  }}
}}

Transcript:
---
{transcript}
---
"""


def build_prompt(transcript: str) -> str:
    return ANALYSIS_PROMPT_TEMPLATE.format(
        categories=", ".join(CATEGORIES), transcript=transcript
    )
