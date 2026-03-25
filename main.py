"""
main.py — CV / ATS Analyzer
=============================
  - Global model variables loaded once at startup
  - No services layer — logic lives in models/
  - HTML served directly from templates/

Run
---
    uvicorn main:app --host 127.0.0.1 --port 8000
or
    python main.py
"""

import os
import logging
import multiprocessing
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Windows: must be set before any torch/transformers import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models.ats_model import load_ats_models, unload_ats_models, compute_ats_score
from models.job_model import load_job_models, unload_job_models, compare_cv_to_job
from models.ocr_model import (
    load_ocr_model, unload_ocr_model,
    read_and_validate_upload, extract_text_from_document,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global model variables ─────────────────────────────────────────────────────
ats_model       = None
ocr_processor   = None
ocr_model       = None
job_model       = None
job_skill_model = None
nlp             = None

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_CV_LEN = 10
MAX_CV_LEN = 50_000

_EXP_KEYWORDS: Dict[str, list] = {
    "entry":  ["entry", "junior", "0-2", "beginner", "fresh"],
    "junior": ["junior", "2-4", "entry", "associate"],
    "mid":    ["mid", "4-7", "intermediate", "senior"],
    "senior": ["senior", "7-10", "lead", "principal", "expert"],
    "lead":   ["lead", "principal", "10+", "senior", "architect", "director"],
}

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE         = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(_BASE, "templates")
STATIC_DIR    = os.path.join(_BASE, "static")


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    global ats_model, ocr_processor, ocr_model, job_model, job_skill_model, nlp

    logger.info("Loading models ...")
    try:
        ats_model, nlp                  = load_ats_models()
        job_model, job_skill_model, nlp = load_job_models()
        ocr_processor, ocr_model        = load_ocr_model()
        logger.info("All models loaded successfully.")
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        raise

    yield

    logger.info("Unloading models ...")
    unload_ats_models()
    unload_job_models()
    unload_ocr_model()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="CV/ATS Analyzer", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(f"HTTP {exc.status_code} at {request.url.path} — {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled error at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Internal server error"},
    )


# ── Helpers ────────────────────────────────────────────────────────────────────
def serve_html(filename: str) -> HTMLResponse:
    """Serve an HTML file directly from the templates/ folder."""
    filepath = os.path.join(TEMPLATES_DIR, filename)
    if not os.path.exists(filepath):
        logger.error(f"HTML file not found: {filepath}")
        raise HTTPException(404, f"{filename} not found")
    with open(filepath, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def validate_cv_text(cv_text: str) -> None:
    n = len(cv_text.strip())
    if n < MIN_CV_LEN:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"CV text must be at least {MIN_CV_LEN} characters.")
    if n > MAX_CV_LEN:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"CV text cannot exceed {MAX_CV_LEN} characters.")


def file_to_text(upload: UploadFile) -> str:
    """Read, validate and extract text from any supported document."""
    data, ct = read_and_validate_upload(upload)
    text = extract_text_from_document(data, ct)
    if not text.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "No text could be extracted from the file.")
    return text


# ── HTML pages ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return serve_html("index.html")


@app.get("/templates/ats.html", response_class=HTMLResponse)
def get_ats():
    return serve_html("ats.html")


@app.get("/templates/job_proposal.html", response_class=HTMLResponse)
def get_job_proposal():
    return serve_html("job_proposal.html")


# ── /ats_score ─────────────────────────────────────────────────────────────────
@app.post("/ats_score")
def ats_score(cv_text: str = Form(...)) -> Dict[str, Any]:
    try:
        validate_cv_text(cv_text)
        result = compute_ats_score(cv_text)
        logger.info(f"ATS score computed: {result['ats_score']}")
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ATS scoring error: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Error processing CV: {str(e)}")


# ── /compare ───────────────────────────────────────────────────────────────────
@app.post("/compare")
def compare(
    cv_text: str        = Form(...),
    image:   UploadFile = File(...),
) -> Dict[str, Any]:
    try:
        validate_cv_text(cv_text)
        job_text = file_to_text(image)
        result   = compare_cv_to_job(cv_text, job_text)
        logger.info(f"Comparison score: {result['similarity_score']}")
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compare error: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Unexpected error: {str(e)}")


# ── /compare_skills ────────────────────────────────────────────────────────────
@app.post("/compare_skills")
def compare_skills(
    cv_text: str        = Form(...),
    image:   UploadFile = File(...),
) -> Dict[str, Any]:
    try:
        validate_cv_text(cv_text)
        job_text = file_to_text(image)
        result   = compare_cv_to_job(cv_text, job_text)
        logger.info(f"Skills comparison score: {result['composite_score']}")
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Skills compare error: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Unexpected error: {str(e)}")


# ── /match_job ─────────────────────────────────────────────────────────────────
@app.post("/match_job")
def match_job(
    cv_text:          str                  = Form(...),
    job_title:        str                  = Form(""),
    experience_level: str                  = Form(""),
    required_skills:  str                  = Form(""),
    image:            Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    try:
        validate_cv_text(cv_text)
        cv_lower = cv_text.lower()
        matches: Dict[str, Any] = {}

        # Job-title match
        if job_title.strip():
            jt    = job_title.lower().strip()
            score = 100.0 if jt in cv_lower else (
                     50.0 if any(w in cv_lower for w in jt.split()) else 0.0)
            matches["job_title"] = {"required": job_title, "score": score, "found": score > 0}

        # Experience-level match
        if experience_level and experience_level in _EXP_KEYWORDS:
            kws   = _EXP_KEYWORDS[experience_level]
            score = min(100.0, sum(1 for k in kws if k in cv_lower) / len(kws) * 100)
            matches["experience_level"] = {
                "required": experience_level,
                "score":    round(score, 1),
                "found":    score > 30,
            }

        # Skills match — semantic via compare_cv_to_job
        if required_skills.strip():
            skills_list = [s.strip().lower() for s in required_skills.split(",") if s.strip()]
            if skills_list:
                result       = compare_cv_to_job(cv_text, " ".join(skills_list))
                cv_skill_set = {x.lower() for x in result["cv_skills"]}
                matched      = [s for s in skills_list if s in cv_skill_set]
                missing      = [s for s in skills_list if s not in cv_skill_set]
                matches["skills"] = {
                    "required":      skills_list,
                    "found":         matched,
                    "missing":       missing,
                    "score":         result["skill_match_score"],
                    "matched_count": len(matched),
                    "total_count":   len(skills_list),
                }

        # Overall score — always /3 so omitted dimensions don't inflate it
        raw     = [m["score"] for m in matches.values() if "score" in m]
        overall = round(sum(raw) / 3, 1)

        result_dict: Dict[str, Any] = {
            "status":              "success",
            "cv_length":           len(cv_text),
            "matches":             matches,
            "overall_match_score": overall,
            "dimensions_used":     len(raw),
        }

        # Optional full-document similarity from uploaded file
        if image and image.filename:
            job_text = file_to_text(image)
            deep     = compare_cv_to_job(cv_text, job_text)
            result_dict["similarity_score"] = deep["composite_score"]
            result_dict["job_text_length"]  = len(job_text)

        logger.info(f"Job match overall score: {overall}")
        return result_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job matching error: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Unexpected error: {str(e)}")


# ── /extract_cv ────────────────────────────────────────────────────────────────
@app.post("/extract_cv")
def extract_cv(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        text = file_to_text(file)
        return {
            "status":      "success",
            "filename":    file.filename,
            "file_type":   file.content_type,
            "cv_text":     text,
            "text_length": len(text),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Error extracting CV: {str(e)}")


# ── /extract_job ───────────────────────────────────────────────────────────────
@app.post("/extract_job")
def extract_job(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        text = file_to_text(file)
        return {
            "status":      "success",
            "filename":    file.filename,
            "file_type":   file.content_type,
            "job_text":    text,
            "text_length": len(text),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Error extracting job description: {str(e)}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)