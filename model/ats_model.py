from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import spacy
from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger("cv_analyzer.ats_model")

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "ats_results.json")

# ── ATS section definitions ────────────────────────────────────────────────────
ATS_SECTIONS: List[Dict[str, Any]] = [
    {
        "name":   "professional_summary",
        "weight": 0.10,
        "anchor_phrases": [
            "professional summary career objective",
            "about me profile overview",
            "seeking a position as",
            "experienced professional with background in",
        ],
    },
    {
        "name":   "skills",
        "weight": 0.25,
        "anchor_phrases": [
            "technical skills programming languages",
            "core competencies tools and technologies",
            "software proficiency frameworks libraries",
            "proficient in experienced with",
        ],
    },
    {
        "name":   "work_experience",
        "weight": 0.30,
        "anchor_phrases": [
            "work experience employment history",
            "professional experience job responsibilities",
            "years of experience worked at",
            "responsible for managed developed",
        ],
    },
    {
        "name":   "education",
        "weight": 0.15,
        "anchor_phrases": [
            "education degree university college",
            "bachelor master PhD graduated",
            "academic background GPA",
            "studied majored in",
        ],
    },
    {
        "name":   "certifications",
        "weight": 0.08,
        "anchor_phrases": [
            "certifications professional certificates licensed",
            "certified AWS Google Microsoft",
            "credential accreditation",
        ],
    },
    {
        "name":   "achievements",
        "weight": 0.08,
        "anchor_phrases": [
            "achievements accomplishments awards recognition",
            "led reduced improved increased saved",
            "key contributions delivered results",
            "exceeded targets promoted",
        ],
    },
    {
        "name":   "contact_formatting",
        "weight": 0.04,
        "anchor_phrases": [
            "email phone LinkedIn GitHub",
            "contact information address",
            "portfolio website profile",
        ],
    },
]

# Normalisation range — raw cosine similarity between a CV sentence and an
# anchor phrase realistically falls in [LOW, HIGH]. We map this to [0, 100].
_NORM_LOW  = 0.10
_NORM_HIGH = 0.55

# ── Singletons ─────────────────────────────────────────────────────────────────
_sentence_model: Optional[SentenceTransformer] = None
_nlp: Any = None


def load_ats_models():
    """Load models once at startup. Returns (sentence_model, nlp)."""
    global _sentence_model, _nlp

    logger.info("ATS: loading SentenceTransformer …")
    try:
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("ATS: SentenceTransformer loaded.")
    except Exception as exc:
        logger.critical("ATS: failed to load SentenceTransformer: %s", exc, exc_info=True)
        raise RuntimeError(f"ATS SentenceTransformer load failed: {exc}") from exc

    logger.info("ATS: loading spaCy NER model …")
    for model_name in ("en_core_web_lg", "en_core_web_sm"):
        try:
            _nlp = spacy.load(model_name)
            logger.info("ATS: spaCy model '%s' loaded.", model_name)
            break
        except OSError:
            logger.warning("ATS: spaCy model '%s' not found, trying next.", model_name)

    if _nlp is None:
        raise RuntimeError(
            "No spaCy model found. Run: python -m spacy download en_core_web_sm"
        )

    return _sentence_model, _nlp


def unload_ats_models() -> None:
    global _sentence_model, _nlp
    _sentence_model = None
    _nlp            = None
    del _sentence_model, _nlp


# ── NER ────────────────────────────────────────────────────────────────────────

def extract_entities(text: str) -> List[Dict[str, str]]:
    """Return deduplicated {text, label} from spaCy NER."""
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


# ── Normalisation ──────────────────────────────────────────────────────────────

def _normalise(raw_cosine: float) -> float:
    """
    Map a raw cosine similarity in [_NORM_LOW, _NORM_HIGH] to [0, 100].
    Values outside the range are clamped.
    """
    clamped = max(_NORM_LOW, min(_NORM_HIGH, raw_cosine))
    return round((clamped - _NORM_LOW) / (_NORM_HIGH - _NORM_LOW) * 100, 2)


# ── Section scoring ────────────────────────────────────────────────────────────

def _score_section(sentences: List[str], sentence_embs: Any,
                   anchor_phrases: List[str]) -> float:
    """
    Score one ATS section by:
      1. Encoding each anchor phrase
      2. For each anchor, finding the best-matching CV sentence
      3. Taking the mean of those best-match scores
      4. Normalising to 0-100

    Using sentence-level matching (not whole-CV) and mean (not max)
    gives scores that reflect actual section presence/absence.
    """
    if not sentences:
        return 0.0

    anchor_embs = _sentence_model.encode(anchor_phrases, convert_to_tensor=True)
    # sims shape: (n_anchors, n_sentences)
    sims = util.pytorch_cos_sim(anchor_embs, sentence_embs)

    # Best sentence match per anchor, then mean across anchors
    best_per_anchor = [float(sims[i].max()) for i in range(len(anchor_phrases))]
    mean_raw = sum(best_per_anchor) / len(best_per_anchor)

    return _normalise(mean_raw)


# ── ATS scoring ────────────────────────────────────────────────────────────────

def compute_ats_score(cv_text: str) -> Dict[str, Any]:
    """
    Compute a multi-section ATS score.
    Splits the CV into sentences and scores each section independently.
    """
    if _sentence_model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ATS model not loaded",
        )

    # Split CV into sentences, filter noise
    sentences = [
        s.strip()
        for s in cv_text.replace("\n", ". ").split(". ")
        if len(s.strip()) > 5
    ]

    if not sentences:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "CV text is too short or contains no readable sentences.",
        )

    # Encode all sentences once — reused across all sections
    sentence_embs = _sentence_model.encode(sentences, convert_to_tensor=True)

    section_scores: Dict[str, float] = {}
    composite = 0.0

    for section in ATS_SECTIONS:
        score = _score_section(sentences, sentence_embs, section["anchor_phrases"])
        section_scores[section["name"]] = score
        composite += score * section["weight"]

    composite = round(composite, 2)
    entities  = extract_entities(cv_text)

    result: Dict[str, Any] = {
        "ats_score":        composite,
        "max_score":        100,
        "keywords_checked": len(ATS_SECTIONS),
        "section_scores":   section_scores,
        "entities":         entities,
    }

    _persist(cv_text, result)
    return result


# ── Persistence ────────────────────────────────────────────────────────────────

def _persist(cv_text: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(DATA_FILE)), exist_ok=True)

    records: List[Dict[str, Any]] = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as fh:
                records = json.load(fh)
        except (json.JSONDecodeError, IOError):
            records = []

    records.append({
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "cv_snippet":     cv_text[:300],
        "ats_score":      result["ats_score"],
        "section_scores": result["section_scores"],
        "entities":       result["entities"],
    })

    with open(DATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
