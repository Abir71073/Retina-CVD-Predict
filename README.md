---
title: Retina CVD Predict
emoji: 👁️
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Retinal fundus classifiers and vessel segmentation
---

# Retina-CVD Predict

Retina-CVD Predict is a final-year project that examines a colour photograph of the back of the eye in two ways. Three neural networks score the same photograph for hypertension, chronic kidney disease (CKD), and stroke. A separate U-Net traces the visible blood vessels. The dashboard brings the scores, their visual explanations, and the vessel measurements together in one report.

**[Open the live app](https://abirkchakraborty-retina-cvd-predict.hf.space/login)** · **[Browse the Hugging Face Space repository](https://huggingface.co/spaces/abirkchakraborty/Retina-CVD-Predict/tree/main)** · [Space overview](https://huggingface.co/spaces/abirkchakraborty/Retina-CVD-Predict)

## Try a sample photograph

1. Open the live app in its own tab and sign in with an app account supplied by the project team. For a first-time local installation, you can create your own account.
2. Download the [sample fundus photograph](test/normal_fundal_wikimedia.jpg), upload it to the dashboard, and select **Run both streams**.
3. Review the three classifier scores and their heatmaps. Then open the vessel overlay to see which pixels contribute to the vessel-area measurement. The results page offers a printable report and a JSON download.

## How the analysis works

| Stage | What it does |
|---|---|
| Fundus image check | A gate model checks whether the upload resembles a retinal fundus photograph. Rejected uploads stop here. |
| Specialist classification | Three EfficientNet-based classifiers each produce a score for the whole image. Each checkpoint uses its configured input scale. |
| Visual explanation | Grad-CAM highlights regions that contributed to the selected classifier output. The dashboard can show the original coarse activation grid alongside the overlay. |
| Vessel segmentation | U-Net processes overlapping image patches and blends their predictions into a vessel probability map. The app applies a 0.50 cutoff inside the estimated retinal field to produce the binary mask. |

The two types of map answer different questions. A bright region in a Grad-CAM overlay shows where a classifier drew evidence for its score. A white pixel in the vessel mask counts toward the segmented area. **Vessel area (%)** is the number of vessel pixels divided by the number of pixels in the estimated retinal field, multiplied by 100. This keeps the dark margin around the photograph out of the denominator.

The classifier percentages are experimental model outputs. The project reports image-level results and does not establish a clinical diagnosis or a calibrated probability of a future event.

## Run locally on Windows

Use 64-bit Python 3.12, Git, and Git LFS. From PowerShell:

```powershell
git lfs install
git clone https://github.com/Abir71073/Retina-CVD-Predict.git
cd Retina-CVD-Predict
git lfs pull
py -3.12 -m tools.verify_models
```

When all five checkpoints print `OK`, double-click `start_windows.bat`, or start the application manually:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

Open [the local dashboard](http://127.0.0.1:5000). The first prediction can take longer while the app loads the model weights. To run the automated tests from the repository root:

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Explore the repository

| Path | Contents |
|---|---|
| [inference.py](inference.py) | Specialist inference, Grad-CAM, and coordination of both analysis streams |
| [vessel_pipeline.py](vessel_pipeline.py) | Patch blending, field estimation, and vessel measurements |
| [model_config.json](model_config.json) | Specialist checkpoint names and preprocessing settings |
| [test/](test/) and [tests/](tests/) | Sample photographs and automated tests, respectively |
| [reports/](reports/) and [conf_matrix/](conf_matrix/) | Recorded model checks, prediction audits, and archived confusion matrices |
| [tools/](tools/) | Training, evaluation, model verification, and Space publishing utilities |

For implementation details, see the [project notes](docs/project-notes.md). To publish an update to the existing GitHub repository and Hugging Face Docker Space, follow the [release guide](docs/RELEASE.md).
