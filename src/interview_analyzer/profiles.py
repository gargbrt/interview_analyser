"""What to assess, independent of how analysis is run (analyzer.py) or how
results are stored (db.py).

Interviews aren't judged the same way regardless of context: a Senior/Lead
candidate is expected to show Leadership and Business Acumen far more than an
Entry-level one; a Software Engineer role weighs Technical Expertise more
heavily than a Sales role does. This module distills that context-dependent
weighting into fixed, versioned Python data -- not something an LLM
improvises per request -- so the *same* profile inputs always produce the
*same* guidance text. Any future change to the weighting itself is a real,
reviewable code change, not something that can drift silently between runs.

The underlying reference is a real-world interview assessment framework
(competency x seniority, x company type, x industry, x role weighting
tables). The tables below are a deliberate *distillation* of that framework
into a small, maintainable emphasis system (five tiers, matching the
framework's own "critical/high/moderate/low/minor" weight scale) rather than
a literal reproduction of every numeric star rating -- the framework itself
describes its regional/company tables as "general trends," not strict
rules, so exact numeric fidelity isn't the goal; consistent, sensible
guidance is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# The 12 core competencies actually scored. Behavioral signals from the
# reference framework (clarity, confidence, structure, conciseness, executive
# presence, etc.) are deliberately NOT separate scored dimensions here -- the
# framework itself says these "are not usually scored independently, but
# heavily influence the interviewer's perception of the core competencies
# above." They show up as supporting detail inside each competency's
# qualitative remark instead (see rubric.py).
CORE_COMPETENCIES: list[str] = [
    "Technical Expertise",
    "Problem Solving",
    "Execution",
    "Ownership",
    "Communication",
    "Leadership",
    "Business Acumen",
    "Stakeholder Management",
    "Collaboration",
    "Learning Agility",
    "Adaptability",
    "Culture & Values Fit",
]

SENIORITIES: list[str] = ["Entry Level", "Mid Level", "Senior/Lead", "Director+"]

ROLES: list[str] = [
    "Software Engineer", "Product", "Data", "Design", "Consultant", "Sales", "Generic",
]

INDUSTRIES: list[str] = [
    "Consumer Tech", "Enterprise SaaS", "FinTech", "Healthcare", "Banking",
    "Consulting", "Manufacturing", "AI / ML", "Generic",
]

COMPANY_TYPES: list[str] = [
    "FAANG / Big Tech", "Growth Startup", "Enterprise SaaS", "Investment Banking",
    "Consulting", "Consumer Internet", "Generic",
]

_TIER_ORDER = ["minor", "low", "moderate", "high", "critical"]


def _tier_at_least(tier: str, floor: str) -> bool:
    return _TIER_ORDER.index(tier) >= _TIER_ORDER.index(floor)


# -- Seniority emphasis -------------------------------------------------
# Distilled directly from the framework's core assessment matrix (star
# ratings mapped 1:1 onto its own five-tier legend: 5 stars=critical, 4=high,
# 3=moderate, 2=low, 1=minor).
SENIORITY_EMPHASIS: dict[str, dict[str, str]] = {
    "Entry Level": {
        "Technical Expertise": "critical", "Problem Solving": "high", "Execution": "moderate",
        "Ownership": "moderate", "Communication": "moderate", "Leadership": "minor",
        "Business Acumen": "low", "Stakeholder Management": "minor", "Collaboration": "high",
        "Learning Agility": "critical", "Adaptability": "high", "Culture & Values Fit": "moderate",
    },
    "Mid Level": {
        "Technical Expertise": "high", "Problem Solving": "critical", "Execution": "high",
        "Ownership": "high", "Communication": "high", "Leadership": "moderate",
        "Business Acumen": "moderate", "Stakeholder Management": "moderate", "Collaboration": "high",
        "Learning Agility": "high", "Adaptability": "high", "Culture & Values Fit": "moderate",
    },
    "Senior/Lead": {
        "Technical Expertise": "moderate", "Problem Solving": "critical", "Execution": "critical",
        "Ownership": "critical", "Communication": "critical", "Leadership": "critical",
        "Business Acumen": "critical", "Stakeholder Management": "critical", "Collaboration": "high",
        "Learning Agility": "moderate", "Adaptability": "high", "Culture & Values Fit": "high",
    },
    "Director+": {
        "Technical Expertise": "low", "Problem Solving": "critical", "Execution": "high",
        "Ownership": "critical", "Communication": "critical", "Leadership": "critical",
        "Business Acumen": "critical", "Stakeholder Management": "critical", "Collaboration": "high",
        "Learning Agility": "low", "Adaptability": "high", "Culture & Values Fit": "high",
    },
}

# -- Role emphasis --------------------------------------------------------
# Software Engineer/Product/Data are lifted directly from the framework's
# core matrix's role columns. Design/Consultant/Sales are distilled from its
# separate role-competency-frameworks table (which uses some different
# competency names -- e.g. "Product Sense"/"Customer Empathy" fold into
# Business Acumen here, "Business Judgment" folds into Business Acumen,
# "Analytics" folds into Problem Solving -- since this app scores against
# the fixed 12-name list above, not the framework's per-role vocabulary).
# "Generic" is deliberately flat (no role-based skew) for the no-selection
# fallback.
ROLE_EMPHASIS: dict[str, dict[str, str]] = {
    "Software Engineer": {
        "Technical Expertise": "critical", "Problem Solving": "high", "Execution": "high",
        "Ownership": "moderate", "Communication": "moderate", "Leadership": "low",
        "Business Acumen": "low", "Stakeholder Management": "low", "Collaboration": "moderate",
        "Learning Agility": "high", "Adaptability": "moderate", "Culture & Values Fit": "moderate",
    },
    "Product": {
        "Technical Expertise": "high", "Problem Solving": "critical", "Execution": "critical",
        "Ownership": "critical", "Communication": "critical", "Leadership": "critical",
        "Business Acumen": "critical", "Stakeholder Management": "critical", "Collaboration": "high",
        "Learning Agility": "high", "Adaptability": "high", "Culture & Values Fit": "moderate",
    },
    "Data": {
        "Technical Expertise": "low", "Problem Solving": "low", "Execution": "low",
        "Ownership": "low", "Communication": "low", "Leadership": "low",
        "Business Acumen": "low", "Stakeholder Management": "low", "Collaboration": "low",
        "Learning Agility": "low", "Adaptability": "low", "Culture & Values Fit": "low",
    },
    "Design": {
        "Technical Expertise": "minor", "Problem Solving": "high", "Execution": "high",
        "Ownership": "moderate", "Communication": "high", "Leadership": "moderate",
        "Business Acumen": "moderate", "Stakeholder Management": "high", "Collaboration": "high",
        "Learning Agility": "moderate", "Adaptability": "moderate", "Culture & Values Fit": "moderate",
    },
    "Consultant": {
        "Technical Expertise": "minor", "Problem Solving": "critical", "Execution": "moderate",
        "Ownership": "moderate", "Communication": "critical", "Leadership": "high",
        "Business Acumen": "critical", "Stakeholder Management": "critical", "Collaboration": "high",
        "Learning Agility": "moderate", "Adaptability": "high", "Culture & Values Fit": "moderate",
    },
    "Sales": {
        "Technical Expertise": "minor", "Problem Solving": "low", "Execution": "high",
        "Ownership": "high", "Communication": "critical", "Leadership": "high",
        "Business Acumen": "high", "Stakeholder Management": "critical", "Collaboration": "high",
        "Learning Agility": "moderate", "Adaptability": "high", "Culture & Values Fit": "moderate",
    },
    "Generic": {c: "moderate" for c in CORE_COMPETENCIES},
}

# -- Industry priority ------------------------------------------------------
# The framework only calls out a short "highest priority" list per industry,
# not a full 12-competency breakdown -- distilled here as which competencies
# get bumped up; everything else for that industry defaults to "moderate"
# (see build_profile_guidance).
INDUSTRY_PRIORITY: dict[str, dict[str, str]] = {
    "Consumer Tech": {"Business Acumen": "critical", "Collaboration": "high", "Adaptability": "high"},
    "Enterprise SaaS": {"Execution": "critical", "Technical Expertise": "high", "Stakeholder Management": "critical"},
    "FinTech": {"Business Acumen": "critical", "Problem Solving": "high", "Culture & Values Fit": "high"},
    "Healthcare": {"Culture & Values Fit": "critical", "Execution": "high"},
    "Banking": {"Problem Solving": "critical", "Technical Expertise": "high", "Execution": "high"},
    "Consulting": {"Problem Solving": "critical", "Communication": "critical"},
    "Manufacturing": {"Execution": "critical", "Adaptability": "high"},
    "AI / ML": {"Technical Expertise": "critical", "Learning Agility": "critical"},
    "Generic": {},
}

# -- Company type emphasis -------------------------------------------------
# Distilled from the framework's company-type weightage table -- "Technical
# Skills"/"Speed"/"Structured Thinking"/"Business Sense" fold into this app's
# Technical Expertise/Execution/Problem Solving/Business Acumen respectively.
COMPANY_TYPE_EMPHASIS: dict[str, dict[str, str]] = {
    "FAANG / Big Tech": {
        "Technical Expertise": "critical", "Problem Solving": "high", "Leadership": "high",
        "Business Acumen": "high",
    },
    "Growth Startup": {
        "Ownership": "critical", "Execution": "critical", "Adaptability": "high",
        "Business Acumen": "critical",
    },
    "Enterprise SaaS": {
        "Technical Expertise": "critical", "Stakeholder Management": "high", "Execution": "high",
    },
    "Investment Banking": {
        "Problem Solving": "critical", "Execution": "high", "Business Acumen": "critical",
    },
    "Consulting": {
        "Problem Solving": "critical", "Communication": "critical", "Business Acumen": "critical",
        "Stakeholder Management": "critical",
    },
    "Consumer Internet": {
        "Business Acumen": "critical", "Execution": "critical", "Collaboration": "high",
    },
    "Generic": {},
}


@dataclass
class AssessmentProfile:
    """A concrete set of assessment settings for one interview: which
    competencies to score, and the role/seniority/industry/company context
    used to weight them. `name` is only set once this has been saved as a
    named template (see db.py's assessment_profiles table); an ad hoc
    (unsaved) profile leaves it None."""

    competencies: list[str] = field(default_factory=lambda: list(CORE_COMPETENCIES))
    role: Optional[str] = None
    seniority: Optional[str] = None
    industry: Optional[str] = None
    company_type: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "competencies": list(self.competencies),
            "role": self.role,
            "seniority": self.seniority,
            "industry": self.industry,
            "company_type": self.company_type,
            "name": self.name,
        }

    @staticmethod
    def from_dict(data: dict) -> "AssessmentProfile":
        return AssessmentProfile(
            competencies=list(data.get("competencies") or CORE_COMPETENCIES),
            role=data.get("role"),
            seniority=data.get("seniority"),
            industry=data.get("industry"),
            company_type=data.get("company_type"),
            name=data.get("name"),
        )


# The "no parameters selected" fallback: every competency, no context skew.
GENERIC_PROFILE = AssessmentProfile(
    competencies=list(CORE_COMPETENCIES), role=None, seniority=None, industry=None, company_type=None,
)


def _competency_emphasis(profile: AssessmentProfile, competency: str) -> str:
    """Combines seniority/role/industry/company_type emphasis for one
    competency into a single tier -- the highest (most critical) tier any
    one dimension assigns wins, since a competency that's critical for even
    one dimension of the context genuinely matters for this interview."""
    tiers = []
    if profile.seniority and profile.seniority in SENIORITY_EMPHASIS:
        tiers.append(SENIORITY_EMPHASIS[profile.seniority].get(competency, "moderate"))
    if profile.role and profile.role in ROLE_EMPHASIS:
        tiers.append(ROLE_EMPHASIS[profile.role].get(competency, "moderate"))
    if profile.industry and profile.industry in INDUSTRY_PRIORITY:
        tiers.append(INDUSTRY_PRIORITY[profile.industry].get(competency, "moderate"))
    if profile.company_type and profile.company_type in COMPANY_TYPE_EMPHASIS:
        tiers.append(COMPANY_TYPE_EMPHASIS[profile.company_type].get(competency, "moderate"))
    if not tiers:
        return "moderate"
    return max(tiers, key=_TIER_ORDER.index)


def competency_emphasis_map(profile: AssessmentProfile) -> dict[str, str]:
    """Public: {competency: tier} for every competency in `profile.competencies`,
    in the same order. Used by both the prompt guidance text and the
    selection-probability weighting (confidence.py) so the two never disagree
    about which competencies matter most for a given profile."""
    return {c: _competency_emphasis(profile, c) for c in profile.competencies}


def build_profile_guidance(profile: AssessmentProfile) -> str:
    """A short, fixed guidance paragraph for the analysis prompt, ranking the
    selected competencies by how much this context emphasizes each one.
    Deterministic: the same profile always produces the same text (no LLM
    involved in producing it), so scoring stays consistent across
    interviews and over time unless this module's own data changes."""
    if not profile.role and not profile.seniority and not profile.industry and not profile.company_type:
        return (
            "No specific role/seniority/industry/company context was given -- weigh all "
            "selected competencies roughly equally."
        )

    emphasis = competency_emphasis_map(profile)
    by_tier: dict[str, list[str]] = {t: [] for t in _TIER_ORDER}
    for competency, tier in emphasis.items():
        by_tier[tier].append(competency)

    context_bits = [
        f"{label}: {value}"
        for label, value in (
            ("Role", profile.role), ("Seniority", profile.seniority),
            ("Industry", profile.industry), ("Company type", profile.company_type),
        )
        if value
    ]
    lines = [f"Context for this interview -- {', '.join(context_bits)}.", "Weight your assessment accordingly:"]
    for tier in reversed(_TIER_ORDER):  # critical first
        if by_tier[tier]:
            lines.append(f"- {tier.capitalize()} emphasis: {', '.join(by_tier[tier])}")
    return "\n".join(lines)
