"""
models/ocr_model.py
--------------------
Handles all document-to-text extraction:
  - PDF    → PyPDF2
  - DOCX   → python-docx
  - Image  → Microsoft TrOCR (microsoft/trocr-base-printed)

FIX #1 & #2 — file-pointer bug
    read_and_validate_upload() reads bytes ONCE and returns (bytes, content_type).
    All downstream functions receive bytes — no UploadFile is ever re-read.

FIX #3 — the upload field that accepts any document is now called `document_file`
    everywhere in routes (was: `image`).

FIX #4 — unload sets globals to None before deleting names so the module scope
    no longer holds a reference, letting GC reclaim memory.
"""

from __future__ import annotations

import io
from typing import Set, Tuple

from fastapi import HTTPException, UploadFile, status
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None  # type: ignore

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None  # type: ignore

# ── MIME type constants ────────────────────────────────────────────────────────
ALLOWED_IMAGE_TYPES: Set[str] = {
    "image/jpeg", "image/png", "image/jpg", "image/webp",
}
ALLOWED_PDF_TYPES:   Set[str] = {"application/pdf"}
ALLOWED_WORD_TYPES:  Set[str] = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
ALLOWED_DOCUMENT_TYPES: Set[str] = (
    ALLOWED_IMAGE_TYPES | ALLOWED_PDF_TYPES | ALLOWED_WORD_TYPES
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ── Lazy-loaded model singletons ───────────────────────────────────────────────
_ocr_processor: TrOCRProcessor | None = None
_ocr_model:     VisionEncoderDecoderModel | None = None


def load_ocr_model():
    """Load TrOCR once at startup. Returns (processor, model)."""
    global _ocr_processor, _ocr_model
    _ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    _ocr_model     = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-printed"
    )
    return _ocr_processor, _ocr_model


def unload_ocr_model() -> None:
    # FIX #4: set to None first so module scope drops the reference,
    # then delete the names so subsequent accesses raise NameError instead
    # of silently returning a dead object.
    global _ocr_processor, _ocr_model
    _ocr_processor = None
    _ocr_model     = None
    del _ocr_processor, _ocr_model


# ── Upload reading & validation  (FIX #1 + #2) ────────────────────────────────

def read_and_validate_upload(file: UploadFile) -> Tuple[bytes, str]:
    """
    Read an UploadFile exactly once and return (raw_bytes, content_type).

    FIX #1: file.file.read() is called here and only here.  The bytes are
            returned to the caller; file.file is never touched again.

    FIX #2: content_type is captured into a local variable BEFORE .read()
            so it is always consistent with the data that was just read.
    """
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No file provided")

    content_type: str = file.content_type or ""

    if content_type not in ALLOWED_DOCUMENT_TYPES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type '{content_type}'. "
            "Accepted formats: PDF, DOCX, JPEG, PNG, WebP.",
        )

    # Single read — after this line file.file is at EOF; we never use it again
    data: bytes = file.file.read()

    if len(data) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File size {len(data) / 1_048_576:.1f} MB exceeds the "
            f"{MAX_FILE_SIZE // 1_048_576} MB limit.",
        )

    return data, content_type


# ── Text extractors — all take (bytes) not UploadFile ─────────────────────────

def _from_pdf(data: bytes) -> str:
    if PdfReader is None:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "PDF support not available — run: pip install PyPDF2",
        )
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Could not parse PDF: {exc}",
        ) from exc


def _from_word(data: bytes) -> str:
    if DocxDocument is None:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "Word support not available — run: pip install python-docx",
        )
    try:
        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Could not parse Word document: {exc}",
        ) from exc


def _from_image(data: bytes) -> str:
    if _ocr_processor is None or _ocr_model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "OCR model not loaded — server may still be starting up.",
        )
    try:
        img          = Image.open(io.BytesIO(data)).convert("RGB")
        pixel_values = _ocr_processor(img, return_tensors="pt").pixel_values
        generated    = _ocr_model.generate(pixel_values)
        return _ocr_processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    except Exception as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"OCR failed: {exc}",
        ) from exc


# ── Public dispatcher ──────────────────────────────────────────────────────────

def extract_text_from_document(data: bytes, content_type: str) -> str:
    """Route bytes to the correct extractor. Both args must come from a
    single read_and_validate_upload() call to stay consistent."""
    if content_type in ALLOWED_PDF_TYPES:
        return _from_pdf(data)
    if content_type in ALLOWED_WORD_TYPES:
        return _from_word(data)
    if content_type in ALLOWED_IMAGE_TYPES:
        return _from_image(data)
    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        f"Unsupported file type: {content_type}",
    )
