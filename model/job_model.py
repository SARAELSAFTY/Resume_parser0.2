"""
models/job_model.py
--------------------
Job-proposal comparison using two complementary models:

1. SentenceTransformer  (all-MiniLM-L6-v2)
   – Whole-document semantic similarity.

2. SentenceTransformer  (paraphrase-MiniLM-L6-v2)
   – Sentence-level skill detection; handles paraphrases
     (e.g. "Postgres" ↔ "PostgreSQL").

3. spaCy NER
   – Extracts key entities (orgs, tools, roles) for UI highlighting.

FIX #4  — unload sets globals to None before deleting names.
FIX #5  — load_job_models() logs each step with context on failure.
FIX #8  — A composite score is computed from BOTH semantic similarity AND
           skill-match score with documented weights (0.4 / 0.6), so the two
           approaches are reconciled into a single authoritative number instead
           of being returned as separate, unrelated floats.
FIX #10 — Returns a typed dict that is validated against CompareSkillsResponse
           in schemas.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import spacy
from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger("cv_analyzer.job_model")

# ── Skill vocabulary ──────────────────────────────────────────────────────────
SKILL_VOCAB: List[str] = [
    # Languages
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Rust",
    "Go", "PHP", "Ruby", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    # Web / Frontend
    "React", "Angular", "Vue", "Next.js", "Svelte", "HTML", "CSS", "SASS",
    # Backend / Frameworks
    "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring Boot",
    "ASP.NET", "Laravel",
    # Data / ML
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "PyTorch", "TensorFlow", "scikit-learn", "Pandas", "NumPy",
    # Databases
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "SQLite", "Oracle", "DynamoDB",
    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins",
    "CI/CD", "Linux", "Git", "GitHub Actions", "Ansible",
    # Methodologies
    "Agile", "Scrum", "Kanban", "REST", "GraphQL", "Microservices",
    "Testing", "TDD", "DevOps", "System Design",
]

# Composite score weights — must sum to 1.0
# FIX #8: documented explicit weights so the blend is transparent and testable
SIMILARITY_WEIGHT = 0.40   # whole-document semantic similarity
SKILL_WEIGHT      = 0.60   # skill-vocabulary overlap

# ── Singletons ────────────────────────────────────────────────────────────────
_semantic_model: Optional[SentenceTransformer] = None
_skill_model:    Optional[SentenceTransformer] = None
_nlp: Any = None


def load_job_models():
    """Load all models at startup. Returns (semantic_model, skill_model, nlp)."""
    global _semantic_model, _skill_model, _nlp

    logger.info("Job: loading semantic similarity model …")
    try:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Job: semantic model loaded.")
    except Exception as exc:
        logger.critical("Job: semantic model load failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Job semantic model load failed: {exc}") from exc

    logger.info("Job: loading skill-matching model …")
    try:
        _skill_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        logger.info("Job: skill-matching model loaded.")
    except Exception as exc:
        logger.critical("Job: skill model load failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Job skill model load failed: {exc}") from exc

    logger.info("Job: loading spaCy NER model …")
    for model_name in ("en_core_web_lg", "en_core_web_sm"):
        try:
            _nlp = spacy.load(model_name)
            logger.info("Job: spaCy model '%s' loaded.", model_name)
            break
        except OSError:
            logger.warning("Job: spaCy model '%s' not found, trying next.", model_name)

    if _nlp is None:
        raise RuntimeError(
            "No spaCy model found. Run: python -m spacy download en_core_web_sm"
        )

    return _semantic_model, _skill_model, _nlp


def unload_job_models() -> None:
    # FIX #4: None before del
    global _semantic_model, _skill_model, _nlp
    _semantic_model = None
    _skill_model    = None
    _nlp            = None
    del _semantic_model, _skill_model, _nlp


# ── NER ────────────────────────────────────────────────────────────────────────

def _extract_entities(text: str) -> List[Dict[str, str]]:
    if _nlp is None:
        return []
    doc  = _nlp(text[:100_000])
    seen: set[tuple[str, str]] = set()
    ents: List[Dict[str, str]] = []
    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if key not in seen and ent.text.strip():
            seen.add(key)
            ents.append({"text": ent.text.strip(), "label": ent.label_})
    return ents


# ── Skill detection ────────────────────────────────────────────────────────────

def _extract_skills_from_text(text: str) -> List[str]:
    """
    For every skill in SKILL_VOCAB find the best-matching sentence in `text`.
    A skill is considered present when the best similarity exceeds 0.45.
    """
    if _skill_model is None:
        return []

    sentences = [s.strip() for s in text.replace("\n", ". ").split(". ") if s.strip()]
    if not sentences:
        return []

    skill_embs    = _skill_model.encode(SKILL_VOCAB,  convert_to_tensor=True)
    sentence_embs = _skill_model.encode(sentences,    convert_to_tensor=True)
    sims          = util.pytorch_cos_sim(skill_embs, sentence_embs)   # (n_skills, n_sents)

    return [
        skill
        for idx, skill in enumerate(SKILL_VOCAB)
        if float(sims[idx].max()) > 0.45
    ]


def _skill_match_score(cv_skills: List[str], job_skills: List[str]) -> float:
    """Percentage of job skills found in the CV skill list."""
    if not job_skills:
        return 0.0
    cv_lower = {s.lower() for s in cv_skills}
    matched  = sum(1 for s in job_skills if s.lower() in cv_lower)
    return round(matched / len(job_skills) * 100, 2)


# ── Public API ─────────────────────────────────────────────────────────────────

def compare_cv_to_job(cv_text: str, job_text: str) -> Dict[str, Any]:
    """
    Returns a dict validated against schemas.CompareSkillsResponse.

    FIX #8: composite_score blends similarity_score and skill_match_score
            with explicit weights so callers have a single authoritative
            score instead of two unrelated floats.
    """
    if _semantic_model is None or _skill_model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Job matching models not loaded",
        )

    # 1. Whole-document semantic similarity
    cv_emb  = _semantic_model.encode(cv_text,  convert_to_tensor=True)
    job_emb = _semantic_model.encode(job_text, convert_to_tensor=True)
    sim     = round(float(util.pytorch_cos_sim(cv_emb, job_emb)) * 100, 2)

    # 2. Skill-vocabulary matching
    cv_skills  = _extract_skills_from_text(cv_text)
    job_skills = _extract_skills_from_text(job_text)
    skill_score = _skill_match_score(cv_skills, job_skills)

    cv_lower       = {s.lower() for s in cv_skills}
    matched_skills = [s for s in job_skills if s.lower() in cv_lower]
    missing_skills = [s for s in job_skills if s.lower() not in cv_lower]

    # 3. Composite  (FIX #8)
    composite = round(sim * SIMILARITY_WEIGHT + skill_score * SKILL_WEIGHT, 2)

    # 4. NER on job text
    entities = _extract_entities(job_text)

    return {
        "status":            "success",
        "similarity_score":  sim,
        "skill_match_score": skill_score,
        "composite_score":   composite,
        "cv_skills":         cv_skills,
        "job_skills":        job_skills,
        "matched_skills":    matched_skills,
        "missing_skills":    missing_skills,
        "entities":          entities,
        "max_score":         100,
    }
