"""Concise image-specific findings without converting uncalibrated outputs into clinical risk."""
REFERENCES = [
    {'label': 'Blood-pressure assessment — NHLBI', 'url': 'https://www.nhlbi.nih.gov/health/high-blood-pressure/diagnosis'},
    {'label': 'Kidney disease testing — NIDDK', 'url': 'https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/tests-diagnosis'},
    {'label': 'Stroke assessment — NHLBI', 'url': 'https://www.nhlbi.nih.gov/health/stroke/diagnosis'},
]


def build_interpretation(scores, vessels, gatekeeper=None):
    if gatekeeper is not None and not gatekeeper.get('accepted'):
        raise ValueError('Do not generate a clinical interpretation for an image rejected by the fundus gatekeeper.')
    findings, limitations = [], []
    names = [('hyp', 'Hypertension'), ('ckd', 'CKD'), ('stroke', 'Stroke')]
    percentages = ', '.join(f"{name if name == 'CKD' else name.lower()} {scores[key]['score'] * 100:.1f}%" for key, name in names)
    summary = f'For this image, the model predictions are: {percentages}. '
    for key, name in names:
        result = scores[key]
        evidence = 'No usable explanation was generated for the selected target.'
        if result['explanation_status'] == 'available':
            direction = 'increase' if result.get('cam_target') == 'positive' else 'decrease'
            evidence = f'The displayed Grad-CAM highlights regions contributing to an {direction} in this output.'
            if direction == 'decrease':
                evidence = evidence.replace('an decrease', 'a decrease')
            outside = result.get('cam_outside_field_percent')
            if outside is not None and outside > 50:
                evidence += f' {outside:.1f}% of attribution lies outside the estimated retinal field, indicating background sensitivity.'
                limitations.append(f'{name}: {outside:.1f}% of attribution lies outside the estimated retinal field. '
                                   'The model may be responding to borders/background rather than retinal anatomy.')
        else:
            limitations.append(f'{name}: no usable Grad-CAM for this target; do not infer normality from an empty map.')
        findings.append({'title': f'{name} · {result["score"] * 100:.1f}%', 'text': evidence})
    if vessels['status'] in {'available', 'no_vessels_detected'}:
        width, height = vessels['processed_size']
        area = vessels['vessel_area_percent']
        summary += f'U-Net marked {area:.2f}% of the estimated retinal field as vessels. '
        findings.append({'title': 'U-Net vessel findings',
            'text': f'{area:.2f}% segmented vessel area ({vessels["vessel_pixels"]:,} pixels) '
                    f'at {width} × {height} resolution, using a 0.50 segmentation cutoff.'})
        if vessels['status'] == 'no_vessels_detected':
            summary += 'No vessel pixels exceeded the segmentation cutoff. '
            limitations.append('No vessels were segmented above the cutoff. Review image quality and model suitability; this does not establish absent vessels.')
        proxy = vessels.get('width_proxy', {})
        if proxy.get('status') == 'available':
            findings.append({'title': 'Exploratory vessel-width ratio',
                'text': f'Narrow/wide width-group ratio: {proxy["width_ratio_proxy"]:.3f}. '
                        'This is not a clinical AVR; the model does not distinguish arteries from veins.'})
        else:
            limitations.append('A stable vessel-width ratio could not be measured in this image.')
    else:
        summary += 'Vessel segmentation was unavailable for this image. '
        limitations.append('U-Net segmentation was unavailable, so the image analysis is incomplete.')
    limitations.extend([
        'Grad-CAM shows coarse model attribution, not detected lesions. Vessel area and width ratio have no validated normal/abnormal cutoff in this project.',
        'The current CKD and stroke checkpoint label mappings are unverified. Their percentages cannot be interpreted as clinical event probabilities.',
    ])
    return {
        'title': 'Clinical interpretation',
        'status': 'Image-based research summary',
        'summary': summary.strip(),
        'caution': 'Experimental model predictions, not calibrated clinical risk. These findings cannot confirm or exclude disease.',
        'findings': findings,
        'clinical_correlation': [
            'Hypertension assessment uses measured blood pressure and clinical history.',
            'Where kidney assessment is indicated, clinical evaluation uses eGFR and urine albumin testing.',
            'Stroke assessment requires symptoms, examination and appropriate brain imaging; this retinal report is not an acute stroke assessment.',
        ],
        'limitations': limitations,
        'references': REFERENCES,
    }
