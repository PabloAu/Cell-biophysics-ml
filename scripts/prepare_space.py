from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a self-contained Hugging Face Space")
    parser.add_argument("--model", help="Optional trained joblib artifact to bundle")
    parser.add_argument("--output", default="dist/space")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for filename in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(root / "space" / filename, output / filename)
    shutil.copytree(
        root / "src" / "cell_biophysics_benchmark",
        output / "src" / "cell_biophysics_benchmark",
    )
    if args.model:
        shutil.copy2(args.model, output / "model.joblib")
    print(f"Prepared Space bundle at {output}")


if __name__ == "__main__":
    main()
