from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish versioned dataset, model, and Space artifacts to Hugging Face"
    )
    parser.add_argument("--dataset", required=True, help="Generated .jsonl.gz dataset")
    parser.add_argument("--model", required=True, help="Trained classical .joblib model")
    parser.add_argument("--space-dir", default="dist/space")
    parser.add_argument("--namespace", default="PabloAu")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform Hub writes; without this flag only print and validate the plan",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    model_path = Path(args.model).resolve()
    validation_path = model_path.with_suffix(".validation.json")
    space_path = Path(args.space_dir).resolve()
    required = [dataset_path, manifest_path, model_path, space_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    repositories = {
        "dataset": f"{args.namespace}/cell-biophysics-trajectories",
        "model": f"{args.namespace}/cell-biophysics-trajectory-classifier",
        "space": f"{args.namespace}/cell-biophysics-trajectory-explorer",
    }
    plan = {
        "apply": args.apply,
        "private": args.private,
        "repositories": repositories,
        "dataset_file": str(dataset_path),
        "dataset_manifest": str(manifest_path),
        "model_file": str(model_path),
        "model_validation": str(validation_path) if validation_path.exists() else None,
        "space_directory": str(space_path),
    }
    print(json.dumps(plan, indent=2))
    if not args.apply:
        return

    try:
        from huggingface_hub import HfApi, get_token
    except ImportError as error:
        raise SystemExit("Install the 'hub' optional dependency before publishing") from error
    if not get_token():
        raise SystemExit("No Hugging Face token is configured; run `hf auth login` first")

    api = HfApi()
    for repository_type, repository_id in repositories.items():
        create_options = {
            "repo_id": repository_id,
            "repo_type": repository_type,
            "private": args.private,
            "exist_ok": True,
        }
        if repository_type == "space":
            create_options["space_sdk"] = "gradio"
        api.create_repo(
            **create_options,
        )

    api.upload_file(
        repo_id=repositories["dataset"],
        repo_type="dataset",
        path_or_fileobj=ROOT / "DATASET_CARD.md",
        path_in_repo="README.md",
        commit_message="Add dataset card",
    )
    api.upload_file(
        repo_id=repositories["dataset"],
        repo_type="dataset",
        path_or_fileobj=dataset_path,
        path_in_repo=f"data/{dataset_path.name}",
        commit_message="Add synthetic trajectory benchmark",
    )
    api.upload_file(
        repo_id=repositories["dataset"],
        repo_type="dataset",
        path_or_fileobj=manifest_path,
        path_in_repo=f"data/{manifest_path.name}",
        commit_message="Add reproducibility manifest",
    )

    api.upload_file(
        repo_id=repositories["model"],
        repo_type="model",
        path_or_fileobj=ROOT / "MODEL_CARD.md",
        path_in_repo="README.md",
        commit_message="Add model card",
    )
    api.upload_file(
        repo_id=repositories["model"],
        repo_type="model",
        path_or_fileobj=model_path,
        path_in_repo="model.joblib",
        commit_message="Add physics-feature baseline",
    )
    if validation_path.exists():
        api.upload_file(
            repo_id=repositories["model"],
            repo_type="model",
            path_or_fileobj=validation_path,
            path_in_repo="validation_metrics.json",
            commit_message="Add validation metrics",
        )
    api.upload_folder(
        repo_id=repositories["space"],
        repo_type="space",
        folder_path=space_path,
        commit_message="Deploy trajectory explorer",
    )
    print("Hugging Face publication completed")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
