"""Tests for profiles.py: the fixed, deterministic competency-weighting data
and the AssessmentProfile it's built around. No LLM/DB involved here --
build_profile_guidance must always produce the same text for the same
inputs, since that's the whole point (consistent scoring across interviews
and over time, per the user's explicit requirement)."""
from __future__ import annotations

from interview_analyzer.profiles import (
    COMPANY_TYPES,
    CORE_COMPETENCIES,
    GENERIC_PROFILE,
    INDUSTRIES,
    ROLES,
    SENIORITIES,
    AssessmentProfile,
    build_profile_guidance,
    competency_emphasis_map,
)


def test_core_competencies_are_twelve_unique_names():
    assert len(CORE_COMPETENCIES) == 12
    assert len(set(CORE_COMPETENCIES)) == 12


def test_generic_profile_includes_every_competency_and_no_context():
    assert GENERIC_PROFILE.competencies == CORE_COMPETENCIES
    assert GENERIC_PROFILE.role is None
    assert GENERIC_PROFILE.seniority is None
    assert GENERIC_PROFILE.industry is None
    assert GENERIC_PROFILE.company_type is None


def test_option_lists_are_non_empty_and_include_a_generic_fallback():
    for options in (ROLES, INDUSTRIES, COMPANY_TYPES):
        assert "Generic" in options
    assert len(SENIORITIES) == 4


def test_to_dict_from_dict_round_trip():
    profile = AssessmentProfile(
        competencies=["Technical Expertise", "Leadership"],
        role="Software Engineer", seniority="Senior/Lead",
        industry="FinTech", company_type="FAANG / Big Tech", name="My Template",
    )
    restored = AssessmentProfile.from_dict(profile.to_dict())
    assert restored == profile


def test_from_dict_defaults_missing_competencies_to_core_list():
    restored = AssessmentProfile.from_dict({})
    assert restored.competencies == CORE_COMPETENCIES


def test_build_profile_guidance_is_deterministic():
    profile = AssessmentProfile(
        competencies=CORE_COMPETENCIES, role="Product", seniority="Senior/Lead",
        industry="Consumer Tech", company_type="Growth Startup",
    )
    first = build_profile_guidance(profile)
    second = build_profile_guidance(profile)
    assert first == second
    assert len(first) > 0


def test_build_profile_guidance_with_no_context_says_weigh_equally():
    guidance = build_profile_guidance(GENERIC_PROFILE)
    assert "weigh all selected competencies roughly equally" in guidance.lower()


def test_build_profile_guidance_mentions_the_given_context():
    profile = AssessmentProfile(competencies=CORE_COMPETENCIES, role="Sales", seniority="Entry Level")
    guidance = build_profile_guidance(profile)
    assert "Sales" in guidance
    assert "Entry Level" in guidance


def test_competency_emphasis_map_covers_every_selected_competency():
    profile = AssessmentProfile(
        competencies=["Technical Expertise", "Leadership", "Execution"],
        role="Software Engineer", seniority="Mid Level",
    )
    emphasis = competency_emphasis_map(profile)
    assert set(emphasis.keys()) == {"Technical Expertise", "Leadership", "Execution"}
    assert all(v in ("critical", "high", "moderate", "low", "minor") for v in emphasis.values())


def test_competency_emphasis_takes_the_highest_tier_across_dimensions():
    """Regression coverage for the "any dimension calling it critical wins"
    rule: Data role rates every competency "low", but Director+ seniority
    rates Leadership "critical" -- the combined result must be critical, not
    an average or the role's own low rating."""
    profile = AssessmentProfile(
        competencies=["Leadership"], role="Data", seniority="Director+",
    )
    emphasis = competency_emphasis_map(profile)
    assert emphasis["Leadership"] == "critical"


def test_generic_profile_guidance_has_no_emphasis_bullets():
    """GENERIC_PROFILE has no role/seniority/industry/company at all, so
    build_profile_guidance should short-circuit to the "weigh equally"
    message rather than trying to rank an empty/undefined context."""
    guidance = build_profile_guidance(GENERIC_PROFILE)
    assert "critical emphasis" not in guidance.lower()
