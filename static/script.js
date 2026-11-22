'use strict';
const button = document.getElementById('runBtn');
const input = document.getElementById('imageInput');
const target = document.getElementById('camTarget');
const gridToggle = document.getElementById('rawCamToggle');
const statusBox = document.getElementById('statusMessage');
const keys = {stroke: ['valStroke', 'imgStroke'], ckd: ['valCkd', 'imgCkd'], hyp: ['valHyp', 'imgHyp']};
let latest = null;
function setImage(id, url, message) {
    const image = document.getElementById(id);
    const caption = document.getElementById(id + 'Status');
    image.hidden = !url;
    if (url) { image.src = url; caption.hidden = true; }
    else { image.removeAttribute('src'); caption.textContent = message; caption.hidden = false; }
}
function clearResults() {
    latest = null;
    document.getElementById('gatekeeperStatus').textContent = 'Fundus check runs before both analysis streams.';
    for (const [scoreId, imageId] of Object.values(keys)) {
        document.getElementById(scoreId).textContent = '—';
        setImage(imageId, null, 'No analysis yet');
    }
    for (const id of ['imgRaw', 'imgVesselOverlay', 'imgVesselMask', 'imgVesselProbability']) setImage(id, null, 'No analysis yet');
    for (const id of ['vesselArea', 'vesselLength', 'vesselSize']) document.getElementById(id).textContent = '—';
    document.getElementById('vesselNote').textContent = 'Vessel measurements will appear after analysis.';
    document.getElementById('interpretationResults').hidden = true;
    document.getElementById('interpretationPlaceholder').hidden = false;
    const report = document.getElementById('reportLink'); report.hidden = true; report.removeAttribute('href');
}
function showExplanations() {
    if (!latest) return;
    const raw = gridToggle.checked;
    document.querySelector('.colour-ramp').classList.toggle('gray', raw);
    for (const [key, [, imageId]] of Object.entries(keys)) {
        const result = latest.scores[key];
        setImage(imageId, raw ? result.heatmap_url : result.map_url,
            result.explanation_status === 'no_target_activation' ? 'No activation for this target' : 'Heatmap unavailable');
    }
    document.getElementById('camDescription').textContent =
        `Regions ${latest.cam_target === 'positive' ? 'increasing' : 'decreasing'} each model’s prediction. ` +
        (raw ? 'The activation grid shows the native coarse attribution without smoothing.' : 'Overlay intensity is relative within each image, not comparable across models.');
}
function fillList(id, items) {
    const element = document.getElementById(id); element.replaceChildren();
    for (const text of items) { const li = document.createElement('li'); li.textContent = text; element.append(li); }
}
function renderInterpretation(report) {
    document.getElementById('clinicalSummary').textContent = report.summary;
    document.getElementById('clinicalCaution').textContent = report.caution || 'Experimental model predictions; clinical risk calibration is unavailable.';
    const findings = document.getElementById('clinicalFindings'); findings.replaceChildren();
    for (const finding of report.findings) {
        const article = document.createElement('article'), h3 = document.createElement('h3'), p = document.createElement('p');
        h3.textContent = finding.title; p.textContent = finding.text; article.append(h3, p); findings.append(article);
    }
    fillList('clinicalCorrelation', report.clinical_correlation);
    fillList('clinicalLimitations', report.limitations);
    const references = document.getElementById('clinicalReferences'); references.replaceChildren();
    for (const reference of report.references) {
        const li = document.createElement('li'), a = document.createElement('a');
        a.textContent = reference.label; a.href = reference.url; a.target = '_blank'; a.rel = 'noopener'; li.append(a); references.append(li);
    }
    document.getElementById('interpretationResults').hidden = false;
    document.getElementById('interpretationPlaceholder').hidden = true;
}
input.addEventListener('change', () => { clearResults(); statusBox.textContent = 'Ready to analyse the selected image.'; });
target.addEventListener('change', () => { clearResults(); statusBox.textContent = 'Explanation target changed. Run both streams to update the results.'; });
gridToggle.addEventListener('change', showExplanations);
button.addEventListener('click', async () => {
    if (!input.files[0]) { statusBox.textContent = 'Please select a fundus image first.'; return; }
    if (input.files[0].size > 16 * 1024 * 1024) { statusBox.textContent = 'Upload exceeds the 16 MB limit.'; return; }
    clearResults(); button.disabled = true; input.disabled = true; target.disabled = true;
    document.getElementById('loadingState').hidden = false;
    statusBox.textContent = 'Checking the fundus image, then running both streams. First use may take longer.';
    const body = new FormData(); body.append('file', input.files[0]); body.append('cam_target', target.value);
    try {
        const response = await fetch('/predict', {method: 'POST', body,
            headers: {'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content}});
        let data;
        try { data = await response.json(); } catch (_) { throw new Error('The server returned an unreadable response.'); }
        if (response.status === 401 && data.login_url) { window.location.assign(data.login_url); return; }
        if (!response.ok || data.error) {
            const error = new Error(data.error || 'Analysis failed.'); error.code = data.code; throw error;
        }
        latest = data;
        document.getElementById('gatekeeperStatus').textContent = data.gatekeeper?.accepted ? 'Fundus image check passed.' : 'Fundus check status unavailable.';
        for (const [key, [scoreId]] of Object.entries(keys)) document.getElementById(scoreId).textContent = data.scores[key].display_score;
        setImage('imgRaw', data.raw_url, 'Original image unavailable'); showExplanations();
        const v = data.vessels;
        setImage('imgVesselOverlay', v.overlay_url, 'Segmentation unavailable');
        setImage('imgVesselMask', v.mask_url, 'Segmentation unavailable');
        setImage('imgVesselProbability', v.probability_url, 'Segmentation unavailable');
        if (v.mask_url) {
            document.getElementById('vesselArea').textContent = v.vessel_area_percent.toFixed(2) + '%';
            document.getElementById('vesselLength').textContent = v.skeleton_pixels.toLocaleString();
            document.getElementById('vesselSize').textContent = v.processed_size.join(' × ');
        }
        document.getElementById('vesselNote').textContent = v.note || v.reason || 'Segmentation unavailable.';
        renderInterpretation(data.interpretation);
        const link = document.getElementById('reportLink'); link.href = data.report_url; link.hidden = false;
        statusBox.textContent = v.mask_url ? 'Analysis complete. Review both streams and the final interpretation below.' : 'Classifier analysis complete; U-Net unavailable. See the report for limitations.';
    } catch (error) {
        clearResults();
        const rejected = error.code === 'non_fundus_image';
        if (rejected) document.getElementById('gatekeeperStatus').textContent = 'Rejected by the fundus gatekeeper. Analysis was not run.';
        if (error.code === 'gatekeeper_unavailable') document.getElementById('gatekeeperStatus').textContent = 'Fundus check unavailable. Analysis was not run.';
        statusBox.textContent = (rejected ? 'Image rejected: ' : 'Analysis stopped: ') + error.message;
    } finally {
        button.disabled = false; input.disabled = false; target.disabled = false;
        document.getElementById('loadingState').hidden = true;
    }
});
