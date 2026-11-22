"""Preview the Space update, then publish with --publish (requires huggingface_hub)."""
import argparse
import json
from pathlib import Path

from .verify_models import ROOT, main as verify_models

SPACE = "abirkchakraborty/Retina-CVD-Predict"
LEGACY = (
    "retina_tri_risk_model.h5",
    "static/heatmaps/analysis_hypertension.jpg",
    "static/heatmaps/analysis_stroke_ckd.jpg",
    "static/heatmaps/last_analysis.jpg",
    "static/heatmaps/latest_xai.jpg",
    "static/heatmaps/temp.jpg",
)

def main():
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--parent", help="Space commit SHA printed by preview")
    parser.add_argument("--source", required=True, help="Merged GitHub commit SHA")
    args = parser.parse_args()
    if verify_models():
        raise SystemExit("Restore the full model weights before publishing.")
    files = [
        "README.md", "Dockerfile", ".dockerignore", "requirements.txt", "requirements-space.txt",
        "app.py", "auth.py", "clinical_report.py", "gatekeeper.py", "inference.py",
        "space_wsgi.py", "vessel_pipeline.py", "gatekeeper_config.json", "model_config.json",
        "tools/__init__.py", "tools/verify_models.py", "reports/model_sha256.json",
        "docs/RELEASE.md", "docs/project-notes.md", "docs/fundus-audit.md",
        "static/report.js", "static/script.js", "static/style.css",
    ]
    files += [item["file"] for item in json.loads((ROOT / "reports/model_sha256.json").read_text())]
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "templates").glob("*.html"))]
    for path in files:
        if not (ROOT / path).is_file():
            raise SystemExit(f"Missing release file: {path}")
    api = HfApi()
    revision = api.repo_info(SPACE, repo_type="space").sha
    existing = set(api.list_repo_files(SPACE, repo_type="space", revision=revision))
    obsolete = [name for name in LEGACY if name in existing]
    print("Space:", SPACE, "\nCurrent Space commit:", revision)
    print("Upload paths:\n" + "\n".join(files))
    print("Remove from current Space version:\n" + "\n".join(obsolete))
    if not args.publish:
        return
    if args.parent != revision:
        raise SystemExit("Space revision changed or --parent missing. Run the preview again.")
    info = api.upload_folder(
        folder_path=ROOT, repo_id=SPACE, repo_type="space", revision="main",
        allow_patterns=files, delete_patterns=obsolete, parent_commit=args.parent,
        commit_message=f"Deploy Retina-CVD-Predict from GitHub {args.source}",
    )
    print(info.commit_url)

if __name__ == "__main__":
    main()
