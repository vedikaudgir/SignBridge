from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.sign_model import SignSequenceClassifier  # noqa: E402
from ml.wlasl_pipeline import load_processed_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SignBridge sequence model.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "ml/runs/latest")
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "ml/processed/wlasl_all_usable"
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "model_config.json").read_text(encoding="utf-8"))
    x, y = load_processed_split(args.processed_dir, args.split)
    if len(x) == 0:
        raise SystemExit(f"No samples in split: {args.split}")
    if x.ndim != 3 or x.shape[1:] != (config["preprocessing"]["sequence_length"], config["input_dim"]):
        raise SystemExit(f"Processed samples have an incompatible shape: {x.shape}")
    if np.any(y < 0) or np.any(y >= config["num_classes"]):
        raise SystemExit("Processed labels do not match the checkpoint class mapping.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignSequenceClassifier(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config.get("bidirectional", True),
    ).to(device)
    model.load_state_dict(
        torch.load(args.run_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    model.eval()

    preds: list[int] = []
    labels: list[int] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=args.batch_size)
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(y_batch.tolist())

    report = classification_report(
        labels,
        preds,
        labels=list(range(config["num_classes"])),
        target_names=config["class_names"],
        zero_division=0,
        output_dict=True,
    )
    print(f"{args.split}_accuracy={accuracy_score(labels, preds):.4f}")
    print(classification_report(
        labels,
        preds,
        labels=list(range(config["num_classes"])),
        target_names=config["class_names"],
        zero_division=0,
    ))
    matrix = confusion_matrix(labels, preds, labels=list(range(len(config["class_names"]))))
    np.savetxt(args.run_dir / f"{args.split}_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    (args.run_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "accuracy": float(accuracy_score(labels, preds)),
                "macro_f1": float(report["macro avg"]["f1-score"]),
                "weighted_f1": float(report["weighted avg"]["f1-score"]),
                "sample_count": len(labels),
                "class_counts": {
                    name: int(np.count_nonzero(y == index))
                    for index, name in enumerate(config["class_names"])
                },
                "classification_report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
