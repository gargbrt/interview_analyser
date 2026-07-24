"""The evaluation rubric used to analyze each of your interview answers.

Scoring is driven by an AssessmentProfile (see profiles.py): which of the 12
core competencies apply to this interview, and the role/seniority/industry/
company context used to weight them. Behavioral signals (clarity,
confidence, structure, conciseness, etc.) are deliberately not separate
scored dimensions -- per the reference framework this app's rubric is based
on, they're supporting evidence that shows up inside each competency's
qualitative remark instead of being scored on their own.
"""
from __future__ import annotations

from .profiles import AssessmentProfile, GENERIC_PROFILE, build_profile_guidance

# The reference framework's own hire-recommendation scale (its "Level 6:
# Scoring Rubric") -- used for session_summary's hire_recommendation.level,
# and reused as-is by confidence.py's selection-probability estimate so the
# two never disagree about what the levels mean.
HIRE_RECOMMENDATION_LEVELS = [
    "Strong No Hire", "No Hire", "Lean No Hire", "Lean Hire", "Hire", "Strong Hire", "Exceptional",
]

ANALYSIS_PROMPT_TEMPLATE = """You are an expert interview coach/assessor reviewing a transcript
of a real job interview. Below is the full transcript with speakers labeled
[Interviewer] and [You].

Go through the ENTIRE transcript from beginning to end and identify EVERY
distinct question the interviewer asked, and your corresponding answer to
each one -- not just a few representative examples. A real interview
transcript this length typically contains many separate questions
(follow-ups, sub-questions within a topic, and a change of topic all count
as distinct questions); do not stop early, and do not merge multiple
distinct questions into a single qa_pairs entry just because they're on
the same topic. If the interviewer asked N distinct questions, return N
separate entries in "qa_pairs", in the order they occurred.

For EACH question/answer pair, evaluate the answer against these competencies:
{competencies}

For each pair return:
- question (short paraphrase)
- answer_summary (1-2 sentence summary of what you said)
- issues: list of specific problems found, tagged by competency (use ONLY the
  competency names listed above). For EACH issue, quote the exact words from
  the transcript that illustrate it verbatim in "excerpt" (copy-paste, do not
  paraphrase) -- this is what makes the feedback concrete instead of generic.
  Leave "excerpt" as an empty string only if the issue is about something
  absent (e.g. "no metric given") rather than something said. Behavioral
  signals you notice (e.g. clarity, confidence, structure, conciseness,
  executive presence) are evidence FOR a competency's issue/remark, not
  separate categories of their own.
- suggested_improvement: a concise, concrete rewrite or specific advice,
  ideally showing how the quoted excerpt could be rephrased

{profile_guidance}

Then return an overall "session_summary" with:
- top_strengths (max 3)
- top_issues (max 3, most impactful first)
- one_thing_to_practice_next (single most actionable suggestion)
- confidence: an integer 0-100 -- your own honest confidence that this
  assessment is accurate and complete, given transcript quality (e.g.
  unclear audio, ambiguous speaker labels) and how much you had to infer
  vs. what was explicitly said. Don't default to a high number just to seem
  certain -- a noisy or ambiguous transcript should get a lower score.
- competency_scores: one entry per competency listed above, each
  {{"name": "<competency>", "score": integer 0-100, "remark": "1-2 sentence
  qualitative assessment specifically for this competency, referencing
  concrete evidence from the transcript"}}.
- hire_recommendation: {{"level": one of {hire_levels}, "rationale": "1-2
  sentences explaining the level, grounded in the competency scores above"}}
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
    "confidence": 0,
    "competency_scores": [{{"name": "...", "score": 0, "remark": "..."}}],
    "hire_recommendation": {{"level": "...", "rationale": "..."}}
  }}
}}

Transcript:
---
{transcript}
---
"""


def build_prompt(transcript: str, profile: AssessmentProfile = GENERIC_PROFILE, calibration_notes: str = "") -> str:
    calibration_section = f"\n{calibration_notes}\n" if calibration_notes else ""
    return ANALYSIS_PROMPT_TEMPLATE.format(
        competencies=", ".join(profile.competencies),
        profile_guidance=build_profile_guidance(profile),
        hire_levels=", ".join(f'"{level}"' for level in HIRE_RECOMMENDATION_LEVELS),
        transcript=transcript,
        calibration_section=calibration_section,
    )


def split_transcript_for_chunked_analysis(transcript: str, max_chars: int) -> list[str]:
    """Splits `transcript` into pieces of at most `max_chars`, breaking only
    between speaker-turn lines (never mid-turn) so no single answer gets cut
    in half across two chunks. Used by engines with a per-request token
    budget too small for a full transcript in one call (see GroqEngine's
    max_transcript_chars_per_request in analyzer.py) -- a long interview
    (e.g. a full hour) can easily need more input tokens than Groq's
    free-tier per-minute limit allows in a single request."""
    lines = transcript.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


# JSON Schema matching ANALYSIS_PROMPT_TEMPLATE's requested shape, for
# engines that support constrained/structured output (Ollama's
# /api/generate `format` field accepts a full JSON Schema, not just the
# string "json" -- see OllamaEngine.run() in analyzer.py). Reproduced on a
# real interview: a long transcript against llama3.1:8b -- asking only for
# "format": "json" (valid-JSON-but-any-shape) let the model return
# syntactically valid JSON that ignored the schema entirely (e.g. a
# generic {"title": ..., "topics": [...]} object). Verified empirically
# that passing this schema as `format` instead forces the model's output
# to match it, even when tested against a prompt totally unrelated to
# interview analysis.
#
# "category"/"name"/"level" are left as free-text strings rather than a
# JSON-schema enum -- the specific competency names vary per
# AssessmentProfile (see profiles.py), so a static enum here would need to
# be regenerated per profile; the prompt text itself is what constrains the
# model to the profile's chosen competency names, and
# analyzer._has_the_expected_shape is the actual safety net for a
# non-compliant response either way (same as it already was for categories).
#
# Every object also sets "additionalProperties": false and lists every
# property as "required" -- not needed by Ollama, but required by Groq's
# *strict* structured-output mode (GroqEngine in analyzer.py), which
# rejects a schema that doesn't (optional fields there must be modeled as
# nullable unions, which this rubric doesn't need since the prompt already
# asks for every field unconditionally).
RESULT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer_summary": {"type": "string"},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "detail": {"type": "string"},
                                "excerpt": {"type": "string"},
                            },
                            "required": ["category", "detail", "excerpt"],
                            "additionalProperties": False,
                        },
                    },
                    "suggested_improvement": {"type": "string"},
                },
                "required": ["question", "answer_summary", "issues", "suggested_improvement"],
                "additionalProperties": False,
            },
        },
        "session_summary": {
            "type": "object",
            "properties": {
                "top_strengths": {"type": "array", "items": {"type": "string"}},
                "top_issues": {"type": "array", "items": {"type": "string"}},
                "one_thing_to_practice_next": {"type": "string"},
                "confidence": {"type": "integer"},
                "competency_scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "score": {"type": "integer"},
                            "remark": {"type": "string"},
                        },
                        "required": ["name", "score", "remark"],
                        "additionalProperties": False,
                    },
                },
                "hire_recommendation": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["level", "rationale"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "top_strengths", "top_issues", "one_thing_to_practice_next", "confidence",
                "competency_scores", "hire_recommendation",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["qa_pairs", "session_summary"],
    "additionalProperties": False,
}
