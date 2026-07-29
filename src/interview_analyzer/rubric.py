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

Be a fair, calibrated assessor, not a harsh one. Judge what was actually said
against what a genuinely effective SPOKEN interview answer looks like, not an
idealized, fully-polished written essay. Spoken answers naturally include
filler words (um, uh, so), false starts, and casual or informal phrasing --
treat these as normal spoken delivery, never as a clarity or competency
problem on their own, unless they genuinely obscure the substance of the
answer. Only raise an issue when it is a real, meaningful gap; do not invent
minor nitpicks in an answer whose substance was actually clear and complete.
If a question's answer has no genuine issues, return an empty "issues" list
for it rather than manufacturing something to criticize just to fill it in.

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
  separate categories of their own -- but tag them under whichever ONE
  competency in the list above they actually reflect, not every
  competency the candidate happened to discuss while being unclear.
- suggested_improvement: REQUIRED, and must be a concrete example, not
  generic advice. Write out the actual words/phrasing/example the candidate
  could have said instead of (or in addition to) the quoted excerpt --
  specific enough that they could use it close to verbatim next time. Never
  settle for vague advice like "be more specific" or "add more detail" --
  show the actual sentence, phrase, or metric that would have scored better,
  e.g. instead of "it went well" say something like "I cut deployment time
  by 30% by parallelizing the build steps". If the issue is about something
  the candidate should have said but didn't, invent a plausible, concrete
  example answer consistent with what they described elsewhere in the
  transcript.

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
  concrete evidence from the transcript"}}. Ground each remark in evidence
  intrinsic to what THAT competency is actually about, not a generic "was
  unclear"/"lacked clarity" complaint copied across every competency's
  remark just because the candidate was hard to follow in general -- a
  communication-style observation belongs under whichever competency in
  the list above is actually about clarity/structure/conciseness, not
  every other one too. If an answer was too vague to judge a given
  competency at all, say that plainly instead of substituting a
  communication complaint as if it were evidence against that competency.
  Score each competency against this anchor so scoring stays fair rather
  than defaulting to a skeptical or nitpicky grading style: 80-100 = strong,
  no significant gaps for this level/role; 60-79 = solid, only minor and
  specific gaps; 40-59 = mixed, noticeable gaps in substance; below 40 =
  weak, largely missing what this competency needs. Reserve scores below 80
  for genuine, substantive gaps, not stylistic quibbles. Whenever a
  competency's score is below 80, its remark MUST also name a specific
  word, phrase, or example the candidate could have said instead to score
  higher for THAT competency -- not just a description of the problem. For
  example, instead of just "lacked technical clarity", say something like
  "lacked technical clarity -- naming the specific tool used (e.g. 'I used
  a hash map for O(1) lookups' instead of 'I used some data structure')
  would have scored higher".
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
