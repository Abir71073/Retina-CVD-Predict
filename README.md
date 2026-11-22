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

The Flask application screens a fundus photograph, runs three specialist classifiers, and segments retinal vessels using U-Net. Each report includes model scores, Grad-CAM maps, and image-space vessel measurements. Scores are experimental outputs for review.

**Publishing this update?** Follow [the GitHub and Hugging Face release guide](docs/RELEASE.md). The ZIP contains the actual model weights.

## Run locally
Use 64-bit Python 3.12. On Windows, double-click `start_windows.bat`. Or run the following commands from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

Open http://127.0.0.1:5000 and create your local account. In a Linux environment, use `python3.12 -m venv .venv` and `.venv/bin/python` instead. For a Git clone, run `git lfs pull` before starting the app.

## Repository layout

| Path | Contents |
|---|---|
| Root Python modules and JSON configs | Flask entry point, account handling, inference, vessel measurements and report generation |
| Five root-level .keras files | Checkpoints loaded by the application |
| static/ and templates/ | Dashboard scripts, styles and HTML |
| test/ and tests/ | Example fundus photographs and automated tests, respectively |
| tools/ | Training, evaluation, diagnostics and model integrity checks |
| training_fy.ipynb | Historical training notebook |
| reports/ and conf_matrix/ | Recorded evaluations, model checksums and confusion matrices |
| docs/ | Publishing instructions, implementation notes and image-gate audit |
| Dockerfile and space_wsgi.py | Hosted entry point for a Docker Space |

Run `python -m tools.verify_models` to check the weights and `python -m pytest -q` in the application environment to run the tests. The application creates its accounts, sessions and generated analyses in the local instance directory on first use.

[GitHub repository](https://github.com/Abir71073/Retina-CVD-Predict) · [Hugging Face Space](https://huggingface.co/spaces/abirkchakraborty/Retina-CVD-Predict)
