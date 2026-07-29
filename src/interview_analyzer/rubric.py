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
of a real job interview, speakers labeled [Interviewer] and [You].

Go through the ENTIRE transcript and identify EVERY distinct question the
interviewer asked and your answer to each -- follow-ups, sub-questions, and
topic changes all count separately; do not stop early, and do not merge
separate questions into one qa_pairs entry just because they share a topic.

Be fair, not harsh: judge answers as SPOKEN language, not polished essays --
filler words (um, uh, so) and casual phrasing are normal delivery, never a
clarity/competency problem on their own, unless they truly obscure the
substance. Only raise an issue that is a real, meaningful gap; do not invent
minor nitpicks in an answer whose substance was actually clear and complete.
Leave "issues" empty for a pair with nothing genuine to flag.

Only create a qa_pairs entry when the interviewer is genuinely asking YOU
something and expects your answer. Never score these two patterns against
you: (a) the interviewer asks something then keeps talking at length
themselves -- explaining, thinking out loud -- with you only offering brief
acknowledgements ("yeah", "okay", "thank you"); that's them sharing
information, not quizzing you. (b) you are the one asking the interviewer a
question (about the role, company, or their experience) and they respond --
never penalize you for "not answering" a question you were actually the one
who asked. Skip both entirely rather than scoring them; judge by who's
asking vs. explaining across the surrounding lines, not just a question mark.

Evaluate each answer against these competencies: {competencies}

For each pair return:
- question (short paraphrase)
- answer_summary (1-2 sentences)
- issues: specific problems, tagged by competency (ONLY the names above).
  Quote the exact words verbatim in "excerpt" (copy-paste, not paraphrased)
  -- leave it empty only when the issue is about something absent (e.g. "no
  metric given"). Behavioral signals (clarity, confidence, structure,
  executive presence) are evidence FOR one competency's issue, not separate
  categories -- tag each under the single competency it actually reflects,
  not every competency the candidate discussed while unclear.
- suggested_improvement: REQUIRED -- a concrete example, not generic advice
  like "be more specific". Write the actual words/phrasing the candidate
  could have said instead, specific enough to reuse verbatim, e.g. instead
  of "it went well" say "I cut deployment time by 30% by parallelizing the
  build steps". If something should have been said but wasn't, invent a
  plausible concrete example consistent with the rest of the transcript.

{profile_guidance}

Then return an overall "session_summary" with:
- top_strengths (max 3)
- top_issues (max 3, most impactful first)
- one_thing_to_practice_next (single most actionable suggestion)
- confidence: integer 0-100 -- your own honest confidence this assessment
  is accurate/complete, given transcript quality and how much you had to
  infer vs. what was explicit. Don't default high just to seem certain -- a
  noisy or ambiguous transcript should score lower.
- competency_scores: one entry per competency above, each {{"name":
  "<competency>", "score": integer 0-100, "remark": "1-2 sentences, grounded
  in concrete evidence for THAT competency specifically -- not a generic
  "lacked clarity" complaint reused across every competency just because
  the candidate was hard to follow overall. If an answer was too vague to
  judge, say so plainly instead."}}. Score anchor: 80-100 strong (no
  significant gaps), 60-79 solid (minor gaps), 40-59 mixed (noticeable
  gaps), below 40 weak -- reserve scores below 80 for genuine, substantive
  gaps, not stylistic quibbles. Whenever a score is below 80, its remark
  MUST also name a specific word/phrase/example that would have scored
  higher for THAT competency (not just describe the problem), e.g. instead
  of "lacked technical clarity" say "...naming the specific tool used (e.g.
  'I used a hash map for O(1) lookups') would have scored higher".
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
