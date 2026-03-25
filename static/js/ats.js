/* static/js/ats.js — ATS Score page logic */

const uploadArea = document.getElementById('uploadArea');
const fileInput  = document.getElementById('fileInput');
const cvText     = document.getElementById('cvText');
const loading    = document.getElementById('loading');
const error      = document.getElementById('error');
const result     = document.getElementById('result');

/* ── Drag & drop / click-to-upload ── */
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

/* ── Extract text from uploaded file ── */
async function handleFile(file) {
    showLoading(true);
    showError(null);
    try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/extract_cv', { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Error extracting CV');
        }
        const data = await res.json();
        cvText.value = data.cv_text;
    } catch (err) {
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

/* ── Get ATS score ── */
async function getAtsScore() {
    const text = cvText.value.trim();
    if (!text) { showError('Please paste CV text or upload a file'); return; }

    showLoading(true);
    showError(null);

    try {
        const formData = new FormData();
        formData.append('cv_text', text);
        const res = await fetch('/ats_score', { method: 'POST', body: formData });
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || 'Error calculating ATS score');
        }
        const data = await res.json();
        displayResult(data);
    } catch (err) {
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

/* ── Render result card ── */
function displayResult(data) {
    // Build NER block if entities were returned
    let nerHtml = '';
    if (data.entities && data.entities.length > 0) {
        nerHtml = `<div class="ner-section">
            <div class="ner-title">Detected Entities</div>
            ${data.entities.map(e =>
                `<span class="ner-tag ${e.label}" title="${e.label}">${e.text}</span>`
            ).join('')}
        </div>`;
    }

    result.innerHTML = `
        <div class="score-display">
            <div class="score-number">${data.ats_score}</div>
            <div class="score-label">ATS Compatibility Score</div>
        </div>
        <div class="score-details">
            <div class="detail-item">
                <div class="detail-label">Max Score</div>
                <div class="detail-value">${data.max_score}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Keywords Matched</div>
                <div class="detail-value">${data.keywords_matched}/${data.keywords_matched}</div>
            </div>
        </div>
        ${nerHtml}
    `;
    result.classList.add('show');
    result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── Helpers ── */
function showLoading(show) { loading.classList.toggle('show', show); }

function showError(message) {
    if (message) {
        error.textContent = message;
        error.classList.add('show');
    } else {
        error.classList.remove('show');
    }
}

function clearForm() {
    cvText.value = '';
    fileInput.value = '';
    result.classList.remove('show');
    showError(null);
}
