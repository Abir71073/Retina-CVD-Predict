"""Compare the five shipped checkpoints with the upstream project manifest."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / "reports/model_sha256.json").read_text())
    failed = False
    for entry in manifest:
        path = ROOT / entry["file"]
        try:
            with path.open("rb") as stream:
                matches = hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
        except OSError:
            matches = False
        print(f"{path.name}: {'OK' if matches else 'missing or incorrect'}")
        failed |= not matches
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
