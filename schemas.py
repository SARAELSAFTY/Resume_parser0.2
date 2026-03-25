"""
schemas.py — Pydantic request / response models
------------------------------------------------
Every API endpoint uses these typed models instead of raw Dict[str, Any].
FastAPI validates inputs automatically and generates accurate OpenAPI docs.

Fixes issue #10: No Pydantic schemas — all endpoints returned untyped dicts.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ────────────────────────────────────────────────────────────────────────────
# Shared sub-models
# ────────────────────────────────────────────────────────────────────────────

class NEREntity(BaseModel):
    """A single named entity returned by spaCy."""
    text:  str = Field(..., description="Surface form of the entity")
    label: str = Field(..., description="spaCy entity label, e.g. ORG, GPE, DATE")


class ExtractionResponse(BaseModel):
    """Response from /extract_cv and /extract_job."""
    status:      Literal["success"]
    filename:    str
    file_type:   str
    text_length: int
    # One of these is populated depending on the endpoint:
    cv_text:  Optional[str] = None
    job_text: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# ATS scoring
# ────────────────────────────────────────────────────────────────────────────

class ATSScoreResponse(BaseModel):
    """Response from POST /ats_score."""
    status:           Literal["success"]
    ats_score:        float = Field(..., ge=0, le=100, description="ATS compatibility score (0–100)")
    max_score:        int   = Field(100, description="Always 100")
    keywords_checked: int   = Field(..., description="Number of ATS dimensions evaluated")
    section_scores:   dict  = Field(..., description="Per-section breakdown scores")
    entities:         List[NEREntity] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Job / skill comparison
# ────────────────────────────────────────────────────────────────────────────

class CompareSkillsResponse(BaseModel):
    """Response from POST /compare_skills."""
    status:             Literal["success"]
    similarity_score:   float = Field(..., ge=0, le=100, description="Whole-document semantic similarity")
    skill_match_score:  float = Field(..., ge=0, le=100, description="Skill-vocabulary overlap score")
    composite_score:    float = Field(..., ge=0, le=100,
                                     description="Weighted composite: 0.4*similarity + 0.6*skill_match")
    cv_skills:          List[str]
    job_skills:         List[str]
    matched_skills:     List[str]
    missing_skills:     List[str] = Field(..., description="Job skills absent from CV")
    entities:           List[NEREntity] = Field(default_factory=list)
    max_score:          int = Field(100)


# ────────────────────────────────────────────────────────────────────────────
# Structured match  (match_job)
# ────────────────────────────────────────────────────────────────────────────

class ExperienceLevelLiteral(BaseModel):
    """Allowed experience level values."""
    level: Literal["entry", "junior", "mid", "senior", "lead", ""]


class TitleMatchDetail(BaseModel):
    required: str
    score:    float = Field(..., ge=0, le=100)
    found:    bool


class ExperienceMatchDetail(BaseModel):
    required: str
    score:    float = Field(..., ge=0, le=100)
    found:    bool


class SkillsMatchDetail(BaseModel):
    required:      List[str]
    found:         List[str]
    missing:       List[str]
    score:         float = Field(..., ge=0, le=100)
    matched_count: int
    total_count:   int


class MatchJobMatches(BaseModel):
    job_title:        Optional[TitleMatchDetail]      = None
    experience_level: Optional[ExperienceMatchDetail] = None
    skills:           Optional[SkillsMatchDetail]     = None


class MatchJobResponse(BaseModel):
    """Response from POST /match_job."""
    status:              Literal["success"]
    cv_length:           int
    matches:             MatchJobMatches
    # Overall score is computed from ALL provided dimensions, weighted equally.
    # If only N out of 3 dimensions were supplied, the score is out of 100
    # but labelled with how many dimensions contributed.
    overall_match_score: float = Field(..., ge=0, le=100)
    dimensions_used:     int   = Field(..., description="How many match dimensions contributed to the score")
    # Optional semantic similarity when a job file was provided:
    similarity_score:    Optional[float] = None
    job_text_length:     Optional[int]   = None


# ────────────────────────────────────────────────────────────────────────────
# Error
# ────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    status:  Literal["error"]
    detail:  str
    code:    Optional[int] = None
