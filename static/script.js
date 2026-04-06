document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    const btn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const xaiArea = document.getElementById('xaiArea');
    const xaiPlaceholder = document.getElementById('xaiPlaceholder');
    const insightText = document.getElementById('insightText');

    if (!file) return;

    // UI Feedback
    btn.disabled = true;
    loading.classList.remove('d-none');
    results.classList.add('d-none');
    xaiArea.classList.add('d-none');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();

        // --- TRIAGE LOGIC ---
        // 1. Convert strings "0.6%" to floats 0.6
        const sVal = parseFloat(data.stroke.replace('%', ''));
        const cVal = parseFloat(data.ckd.replace('%', ''));
        const hVal = parseFloat(data.hypertension.replace('%', ''));

        const scores = [
            { id: 'strokeScore', val: sVal, text: data.stroke },
            { id: 'ckdScore', val: cVal, text: data.ckd },
            { id: 'hypScore', val: hVal, text: data.hypertension }
        ];

        const maxRisk = Math.max(sVal, cVal, hVal);

        scores.forEach(item => {
            const el = document.getElementById(item.id);
            el.innerText = item.text;
            
            // CLEAR EVERYTHING: Remove all possible Bootstrap color classes
            el.classList.remove('text-success', 'text-warning', 'text-danger', 'text-primary', 'text-dark', 'fw-bold');
            el.style.fontSize = "1.1rem"; 

            // APPLY TRIAGE
            if (item.val === maxRisk && item.val >= 15) {
                // Highlighting the Primary/Critical concern
                if (item.val > 40) {
                    el.classList.add('text-danger', 'fw-bold');
                } else {
                    el.classList.add('text-warning', 'fw-bold');
                }
                el.style.fontSize = "1.3rem"; // Scale up the most important number
            } else if (item.val < 15) {
                el.classList.add('text-success'); // Green for low risk
            } else {
                el.classList.add('text-primary'); // Neutral blue for others
            }
        });

        // Update Images & Insight
        const ts = new Date().getTime();
        document.getElementById('originalDisplay').src = URL.createObjectURL(file);
        document.getElementById('heatmapMainDisplay').src = data.heatmap_main_url + "?t=" + ts;
        document.getElementById('heatmapHypDisplay').src = data.heatmap_hyp_url + "?t=" + ts;
        insightText.innerText = data.insight;

        // Reveal
        loading.classList.add('d-none');
        results.classList.remove('d-none');
        xaiPlaceholder.classList.add('d-none');
        xaiArea.classList.remove('d-none');

    } catch (error) {
        alert("Inference Error. Check your Flask terminal.");
        loading.classList.add('d-none');
    } finally {
        btn.disabled = false;
    }
});