from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cell_biophysics_benchmark.constants import STATE_LABELS  # noqa: E402
from cell_biophysics_benchmark.data import load_config, read_samples  # noqa: E402
from cell_biophysics_benchmark.evaluation import classification_report  # noqa: E402
from cell_biophysics_benchmark.models.temporal import (  # noqa: E402
    build_temporal_cnn,
    temporal_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the masked temporal CNN")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/temporal_baseline.yaml")
    parser.add_argument("--output", default="artifacts/temporal/model.pt")
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise SystemExit("Install the 'ml' optional dependencies before training") from error

    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    seed = int(training_config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_samples = read_samples(args.data, split="train")
    validation_samples = read_samples(args.data, split="validation")
    classes = [
        label
        for label in STATE_LABELS
        if any(sample.state_label == label for sample in train_samples)
    ]
    class_index = {label: index for index, label in enumerate(classes)}
    train_inputs = [temporal_inputs(sample) for sample in train_samples]
    metadata_stack = np.vstack([item[1] for item in train_inputs])
    metadata_mean = metadata_stack.mean(axis=0)
    metadata_std = metadata_stack.std(axis=0)
    metadata_std[metadata_std < 1e-8] = 1.0

    class TrajectoryDataset(Dataset):
        def __init__(self, samples, cached=None):
            self.samples = samples
            self.cached = cached or [temporal_inputs(sample) for sample in samples]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            sequence, metadata = self.cached[index]
            normalized_metadata = (metadata - metadata_mean) / metadata_std
            return (
                torch.from_numpy(sequence),
                torch.from_numpy(normalized_metadata.astype(np.float32)),
                class_index[self.samples[index].state_label],
            )

    def collate(batch):
        lengths = [item[0].shape[1] for item in batch]
        padded = torch.zeros((len(batch), 3, max(lengths)), dtype=torch.float32)
        metadata = torch.stack([item[1] for item in batch])
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        for index, (sequence, _, _) in enumerate(batch):
            padded[index, :, : sequence.shape[1]] = sequence
        return padded, metadata, labels

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TrajectoryDataset(train_samples, train_inputs),
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        TrajectoryDataset(validation_samples),
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        collate_fn=collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_temporal_cnn(
        n_classes=len(classes),
        metadata_features=5 if model_config["measurement_aware"] else 0,
        hidden_channels=int(model_config["hidden_channels"]),
        kernel_size=int(model_config["kernel_size"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    loss_function = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        for sequence, metadata, labels in train_loader:
            sequence, metadata, labels = sequence.to(device), metadata.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                sequence, metadata if model_config["measurement_aware"] else None
            )
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(labels)

        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        with torch.no_grad():
            for sequence, metadata, labels in validation_loader:
                sequence, metadata, labels = (
                    sequence.to(device),
                    metadata.to(device),
                    labels.to(device),
                )
                logits = model(
                    sequence, metadata if model_config["measurement_aware"] else None
                )
                validation_loss += float(loss_function(logits, labels)) * len(labels)
                validation_correct += int((logits.argmax(dim=1) == labels).sum())
        validation_loss /= max(1, len(validation_samples))
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss / max(1, len(train_samples)),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_correct / max(1, len(validation_samples)),
        }
        history.append(row)
        print(json.dumps(row))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(training_config["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    model.eval()
    probability_rows: list[np.ndarray] = []
    with torch.no_grad():
        for sequence, metadata, _ in validation_loader:
            logits = model(
                sequence.to(device),
                metadata.to(device) if model_config["measurement_aware"] else None,
            )
            probability_rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.vstack(probability_rows)
    report = classification_report(
        [sample.state_label for sample in validation_samples], probabilities, classes
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "classes": classes,
            "model_config": model_config,
            "metadata_mean": metadata_mean,
            "metadata_std": metadata_std,
            "training_config": training_config,
            "training_samples": len(train_samples),
            "history": history,
            "validation": report,
            "torch_version": torch.__version__,
        },
        output,
    )
    output.with_suffix(".validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"model": str(output), "validation": report}, indent=2))


if __name__ == "__main__":
    main()
