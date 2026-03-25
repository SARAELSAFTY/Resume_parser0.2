/* static/js/job_proposal.js — Job Match page logic */

const cvUploadArea  = document.getElementById('cvUploadArea');
const jobUploadArea = document.getElementById('jobUploadArea');
const cvFileInput   = document.getElementById('cvFileInput');
const jobFileInput  = document.getElementById('jobFileInput');
const compareBtn    = document.getElementById('compareBtn');
const loading       = document.getElementById('loading');
const error         = document.getElementById('error');
const result        = document.getElementById('result');
const jobDetailsCard = document.getElementById('jobDetailsCard');

let cvFile  = null;
let jobFile = null;

/* ── Upload handlers — CV ── */
cvUploadArea.addEventListener('click', () => cvFileInput.click());
cvUploadArea.addEventListener('dragover',  (e) => { e.preventDefault(); cvUploadArea.classList.add('dragover'); });
cvUploadArea.addEventListener('dragleave', () => cvUploadArea.classList.remove('dragover'));
cvUploadArea.addEventListener('drop', (e) => {
    e.preventDefault(); cvUploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) { cvFileInput.files = e.dataTransfer.files; handleCVFile(e.dataTransfer.files[0]); }
});
cvFileInput.addEventListener('change', (e) => { if (e.target.files.length > 0) handleCVFile(e.target.files[0]); });

/* ── Upload handlers — Job ── */
jobUploadArea.addEventListener('click', () => jobFileInput.click());
jobUploadArea.addEventListener('dragover',  (e) => { e.preventDefault(); jobUploadArea.classList.add('dragover'); });
jobUploadArea.addEventListener('dragleave', () => jobUploadArea.classList.remove('dragover'));
jobUploadArea.addEventListener('drop', (e) => {
    e.preventDefault(); jobUploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) { jobFileInput.files = e.dataTransfer.files; handleJobFile(e.dataTransfer.files[0]); }
});
jobFileInput.addEventListener('change', (e) => { if (e.target.files.length > 0) handleJobFile(e.target.files[0]); });

function handleCVFile(file) {
    cvFile = file;
    const el = document.getElementById('cvFileName');
    el.textContent = `✓ ${file.name}`;
    el.style.color = '#00ff88';
    updateCompareButton();
}

function handleJobFile(file) {
    jobFile = file;
    const el = document.getElementById('jobFileName');
    el.textContent = `✓ ${file.name}`;
    el.style.color = '#00ff88';
    updateCompareButton();
}

function updateCompareButton() {
    compareBtn.disabled = !(cvFile && jobFile);
}

/* ── Escape regex special chars ── */
function escapeRegexChars(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/* ── Client-side heuristic skill extractor (fallback) ── */
function extractJobDetails(jobText) {
    const lines    = jobText.split('\n').map(l => l.trim()).filter(l => l);
    const textLower = jobText.toLowerCase();

    let jobTitle = '';
    let experienceLevel = '';
    let skills = [];

    for (let i = 0; i < Math.min(5, lines.length); i++) {
        if (lines[i].length > 5 && lines[i].length < 100 &&
            !lines[i].includes('@') && !lines[i].includes('http') &&
            lines[i].match(/[A-Za-z]/)) {
            jobTitle = lines[i];
            break;
        }
    }

    const expKeywords = ['entry','junior','mid','senior','lead','principal'];
    for (const kw of expKeywords) {
        if (textLower.includes(kw)) {
            experienceLevel = kw.charAt(0).toUpperCase() + kw.slice(1);
            break;
        }
    }

    const commonSkills = [
        'python','javascript','java','c++','c#','typescript','rust','golang','php',
        'react','angular','vue','node','express','django','flask','spring',
        'sql','mysql','postgresql','mongodb','redis','elasticsearch',
        'aws','azure','gcp','docker','kubernetes','jenkins',
        'html','css','sass','webgl','git','rest','graphql',
        'agile','scrum','testing','ci/cd','devops','linux'
    ];
    for (const skill of commonSkills) {
        if (new RegExp('\\b' + escapeRegexChars(skill) + '\\b', 'i').test(textLower)) {
            skills.push(skill.toUpperCase());
        }
    }
    skills = [...new Set(skills)];

    return {
        jobTitle: jobTitle || 'Not specified',
        experienceLevel: experienceLevel || 'Not specified',
        skills: skills.slice(0, 10)
    };
}

/* ── Main compare flow ── */
async function compareCV() {
    if (!cvFile || !jobFile) { showError('Please upload both CV and job proposal'); return; }

    showLoading(true);
    showError(null);
    jobDetailsCard.classList.remove('show');
    result.classList.remove('show');

    try {
        // Extract CV text
        const cvFD = new FormData();
        cvFD.append('file', cvFile);
        const cvRes = await fetch('/extract_cv', { method: 'POST', body: cvFD });
        if (!cvRes.ok) throw new Error('Error extracting CV');
        const { cv_text: cvText } = await cvRes.json();

        // Extract job text
        const jobFD = new FormData();
        jobFD.append('file', jobFile);
        const jobRes = await fetch('/extract_job', { method: 'POST', body: jobFD });
        if (!jobRes.ok) throw new Error('Error extracting job description');
        const { job_text: jobText } = await jobRes.json();

        // Skill-focused comparison via backend
        const matchFD = new FormData();
        matchFD.append('cv_text', cvText);
        matchFD.append('image', jobFile);
        const matchRes = await fetch('/compare_skills', { method: 'POST', body: matchFD });
        if (!matchRes.ok) throw new Error('Error comparing skills');
        const matchData = await matchRes.json();

        // Heuristic parse for display
        const jobDetails = extractJobDetails(jobText);
        const cvDetails  = extractJobDetails(cvText);

        displayJobDetailsCard(jobDetails, cvDetails, matchData);

    } catch (err) {
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

/* ── Render job details card ── */
function displayJobDetailsCard(jobDetails, cvDetails, matchData) {
    const score      = matchData.similarity_score ?? 0;
    const skillScore = matchData.skill_match_score ?? 0;
    const isMatch    = score >= 60;

    const jobSkills     = jobDetails.skills.map(s => s.toLowerCase());
    const cvSkills      = cvDetails.skills.map(s => s.toLowerCase());
    const matchedSkills = jobSkills.filter(s => cvSkills.includes(s));

    // NER block (from backend if present)
    let nerHtml = '';
    if (matchData.entities && matchData.entities.length > 0) {
        nerHtml = `<div class="ner-section">
            <div class="ner-title">Key Entities Detected</div>
            ${matchData.entities.map(e =>
                `<span class="ner-tag ${e.label}" title="${e.label}">${escapeHtml(e.text)}</span>`
            ).join('')}
        </div>`;
    }

    let html = `
        <div class="card-title">📋 Job Proposal Analysis</div>
        <div class="job-info-grid">
            <div class="job-info-item">
                <div class="job-info-label">Job Title Required</div>
                <div class="job-info-value">${escapeHtml(jobDetails.jobTitle)}</div>
            </div>
            <div class="job-info-item">
                <div class="job-info-label">Your CV Title</div>
                <div class="job-info-value">${escapeHtml(cvDetails.jobTitle)}</div>
            </div>
            <div class="job-info-item">
                <div class="job-info-label">Semantic Similarity</div>
                <div class="job-info-value">${score.toFixed(1)}%</div>
            </div>
            <div class="job-info-item">
                <div class="job-info-label">Skill Match Score</div>
                <div class="job-info-value">${skillScore.toFixed(1)}%</div>
            </div>
        </div>`;

    if (jobDetails.skills.length > 0 || cvDetails.skills.length > 0) {
        html += `<div class="skills-comparison">
            <div class="skills-column">
                <div class="skills-column-title">Required Skills (Job)</div>
                <div class="skill-match-stat">${jobDetails.skills.length} Skills Required</div>
                <div>${jobDetails.skills.map(skill => {
                    const matched = cvSkills.includes(skill.toLowerCase());
                    return `<span class="skill-tag ${matched ? 'matched' : ''}">${escapeHtml(skill)}</span>`;
                }).join('')}</div>
            </div>
            <div class="skills-column matched-column">
                <div class="skills-column-title">Your Skills (CV)</div>
                <div class="skill-match-stat high">${matchedSkills.length}/${jobDetails.skills.length} Matched</div>
                <div>${cvDetails.skills.map(skill => {
                    const matched = jobSkills.includes(skill.toLowerCase());
                    return `<span class="skill-tag ${matched ? 'matched' : ''}">${escapeHtml(skill)}</span>`;
                }).join('')}</div>
            </div>
        </div>`;
    }

    html += nerHtml;

    html += `<div class="match-status">
        <p style="color:#a0a0a0;margin-bottom:10px;">
            Skills Match: ${((matchedSkills.length / Math.max(jobDetails.skills.length, 1)) * 100).toFixed(0)}% | Overall CV &amp; Job Match
        </p>
        <div class="match-badge ${isMatch ? 'pass' : 'fail'}">${isMatch ? 'MATCH' : 'MISMATCH'}</div>
        <p style="color:#808080;font-size:12px;margin-top:10px;">
            ${isMatch ? '✓ Your CV matches 60% or more of the job requirements'
                      : '✗ Your CV matches less than 60% of the job requirements'}
        </p>
    </div>`;

    jobDetailsCard.innerHTML = html;
    jobDetailsCard.classList.add('show');
    jobDetailsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── Helpers ── */
function showLoading(show) { loading.classList.toggle('show', show); }

function showError(message) {
    if (message) { error.textContent = message; error.classList.add('show'); }
    else          { error.classList.remove('show'); }
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
