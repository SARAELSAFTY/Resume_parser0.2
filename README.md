# CV / ATS Analyzer

A locally-running NLP system for resume analysis and job matching. Processes CVs in multiple formats, scores them for ATS compatibility, and compares them against job descriptions or other CVs using semantic similarity.

---

## Technical Category

**Natural Language Processing (NLP)**

### Subcategories

- **Information Extraction** — pulls structured professional data (skills, experience, education, certifications) from unstructured resume text
- **Semantic Similarity** — compares documents by meaning using dense sentence embeddings, not keyword overlap
- **Named Entity Recognition (NER)** — identifies and strips personal identifiers (names, schools, locations) before scoring
- **Document Understanding** — ingests and normalises multi-format documents into plain text
- **Optical Character Recognition (OCR)** — converts image-based and scanned documents into machine-readable text
- **Text Scoring** — assigns weighted section-level scores to resumes based on content relevance
- **Information Retrieval** — ranks CV-to-job or CV-to-CV relevance using cosine similarity over sentence embeddings

---

## Project Structure

```
cv_analyzer/
├── main.py                  — FastAPI app, routing, error handling
├── schemas.py               — Pydantic response models
├── requirements.txt
├── README.md
│
├── models/
│   ├── ats_model.py         — ATS scoring: sentence-level matching, NER stripping, persistence
│   ├── job_model.py         — CV vs job description: dual-model skill matching + NER
│   └── ocr_model.py         — Document extraction: PDF, DOCX, image OCR
│
├── templates/
│   ├── index.html
│   ├── ats.html
│   └── job_proposal.html
│
├── static/
│   ├── css/styles.css
│   └── js/
│       ├── ats.js
│       └── job_proposal.js
│
└── data/
    └── ats_results.json     — scored resume history
```

---

## Models

| Model | Source | Purpose |
|---|---|---|
| all-MiniLM-L6-v2 | sentence-transformers | Semantic similarity for ATS and job matching |
| paraphrase-MiniLM-L6-v2 | sentence-transformers | Skill-level matching, handles paraphrases |
| trocr-base-printed | Microsoft / HuggingFace | OCR for image-based documents |
| en_core_web_sm / en_core_web_lg | spaCy | Named entity recognition |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | / | Home page |
| GET | /ats | ATS scoring page |
| GET | /job_proposal | Job match page |
| POST | /ats_score | Score a CV or compare two CVs section by section |
| POST | /compare | Semantic similarity between CV and job document |
| POST | /compare_skills | Deep skill matching with NER and composite score |
| POST | /match_job | Structured multi-dimension CV vs job match |
| POST | /extract_cv | Extract text from a CV file |
| POST | /extract_job | Extract text from a job description file |

---

## ATS Scoring

The ATS score is a weighted composite across five professional sections. Personal identifiers (names, schools, locations) are removed by spaCy NER before scoring so results reflect professional content only.

| Section | Weight |
|---|---|
| Work Experience | 35% |
| Skills | 30% |
| Achievements | 15% |
| Certifications | 10% |
| Education Level | 10% |

Each section is scored by extracting relevant sentences from both documents and computing mean pairwise cosine similarity. Raw similarity is normalised to a 0-100 scale using a calibrated range so scores spread meaningfully rather than clustering.

---

## Supported File Formats

| Format | MIME Type | Extraction Method |
|---|---|---|
| PDF | application/pdf | PyPDF2 |
| Word | application/vnd.openxmlformats-officedocument.wordprocessingml.document | python-docx |
| JPEG / PNG / WebP | image/* | Microsoft TrOCR |

---

## Setup

```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run
uvicorn main:app --host 127.0.0.1 --port 8000
```

---

## Requirements

```
fastapi
uvicorn[standard]
python-multipart
transformers
sentence-transformers
torch
spacy
Pillow
PyPDF2
python-docx
```

---

## Configuration

| Setting | Value |
|---|---|
| Max file size | 10 MB |
| Min CV text length | 10 characters |
| Max CV text length | 50,000 characters |
| Accepted image types | JPEG, PNG, WebP |
| Accepted document types | PDF, DOCX |
| Data file | data/ats_results.json |

---

## Notes

- All models run locally. No external API calls are made during inference.
- On Windows, set `TOKENIZERS_PARALLELISM=false` and `OMP_NUM_THREADS=1` before starting the server to prevent multiprocessing conflicts with PyTorch. These are set automatically at the top of `main.py`.
- Python 3.11 is required. spaCy 3.8 is not compatible with Python 3.14.
