"""models package."""
from .ats_model import compute_ats_score, load_ats_models, unload_ats_models
from .job_model import compare_cv_to_job, load_job_models, unload_job_models
from .ocr_model import (
    ALLOWED_DOCUMENT_TYPES,
    extract_text_from_document,
    load_ocr_model,
    read_and_validate_upload,
    unload_ocr_model,
)

__all__ = [
    "load_ats_models", "unload_ats_models", "compute_ats_score",
    "load_job_models", "unload_job_models", "compare_cv_to_job",
    "load_ocr_model",  "unload_ocr_model",
    "read_and_validate_upload", "extract_text_from_document",
    "ALLOWED_DOCUMENT_TYPES",
]
