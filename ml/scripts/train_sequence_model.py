from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.sign_model import SignSequenceClassifier  # noqa: E402
from ml.wlasl_pipeline import load_processed_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SignBridge sequence model.")
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "ml/processed/wlasl_all_usable"
    )
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "ml/runs")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    losses: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        labels.extend(y_batch.detach().cpu().tolist())
    return float(np.mean(losses)), float(accuracy_score(labels, preds))


def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(y_batch.tolist())
    return labels, preds


def write_json(path: Path, value: dict) -> None:
    temporary_path = path.with_name(f"{path.stem}.tmp.json")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def dataset_fingerprint(
    processed_dir: Path, metadata: dict, split_paths: list[Path]
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "class_names": metadata["class_names"],
                "split_counts": metadata["split_counts"],
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for path in split_paths:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    digest.update(str(processed_dir.resolve()).encode("utf-8"))
    return digest.hexdigest()


def preserve_latest_checkpoint(runs_dir: Path) -> Path | None:
    latest = runs_dir / "latest"
    if not latest.exists():
        return None
    backup = runs_dir / f"backup_before_full_wlasl_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copytree(latest, backup)
    return backup


def save_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    epochs_without_improvement: int,
    history: list[dict[str, float]],
    run_dir: Path,
    processed_dir: Path,
    fingerprint: str,
    cumulative_runtime_seconds: float,
) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "run_dir": str(run_dir.resolve()),
        "processed_dir": str(processed_dir.resolve()),
        "dataset_fingerprint": fingerprint,
        "cumulative_runtime_seconds": cumulative_runtime_seconds,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    temporary_path = path.with_name(f"{path.stem}.tmp.pt")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise SystemExit("epochs, batch-size, and patience must be positive.")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = args.processed_dir.resolve()
    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    class_names = metadata["class_names"]
    if len(class_names) != len(set(class_names)) or len(class_names) < 2:
        raise SystemExit("Processed metadata must contain at least two unique classes.")
    for split in ("train", "val", "test"):
        for class_name in class_names:
            if metadata["split_counts"][split].get(class_name, 0) <= 0:
                raise SystemExit(f"Class {class_name} has no {split} examples.")

    train_x, train_y = load_processed_split(processed_dir, "train")
    val_x, val_y = load_processed_split(processed_dir, "val")
    num_classes = len(class_names)
    if not len(train_x) or not len(val_x):
        raise SystemExit("Train and validation splits must both be non-empty.")
    if train_x.ndim != 3 or train_x.shape[1:] != (30, 258):
        raise SystemExit(f"Expected training data shaped (N, 30, 258), got {train_x.shape}.")
    if (
        np.any(train_y < 0)
        or np.any(train_y >= num_classes)
        or np.any(val_y < 0)
        or np.any(val_y >= num_classes)
    ):
        raise SystemExit("Split labels do not match the complete class mapping.")
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise SystemExit("Train/validation splits contain non-finite features.")

    split_paths = [processed_dir / f"{split}.npz" for split in ("train", "val", "test")]
    fingerprint = dataset_fingerprint(processed_dir, metadata, split_paths)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignSequenceClassifier(
        input_dim=train_x.shape[-1],
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    class_counts = np.bincount(train_y, minlength=num_classes)
    if np.any(class_counts == 0):
        absent = [class_names[i] for i in np.flatnonzero(class_counts == 0)]
        raise SystemExit(f"Training samples missing for classes: {absent}")
    # Square-root inverse-frequency weights moderate imbalance without discarding samples.
    class_weights = np.sqrt(len(train_y) / (num_classes * class_counts))
    class_weights /= class_weights.mean()
    train_criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights.astype(np.float32)).to(device)
    )
    val_criterion = nn.CrossEntropyLoss()

    loaders = {
        "train": make_loader(train_x, train_y, args.batch_size, True),
        "val": make_loader(val_x, val_y, args.batch_size, False),
    }
    run_started = time.monotonic()
    first_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    prior_runtime_seconds = 0.0

    if args.resume_from:
        resume_path = args.resume_from.resolve()
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("processed_dir") != str(processed_dir):
            raise SystemExit("Resume checkpoint belongs to a different processed dataset.")
        if state.get("dataset_fingerprint") != fingerprint:
            raise SystemExit("Resume checkpoint dataset fingerprint does not match.")
        run_dir = Path(state["run_dir"])
        if not run_dir.is_dir():
            raise SystemExit(f"Resume run directory is missing: {run_dir}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
        first_epoch = int(state["epoch"]) + 1
        best_val_loss = float(state["best_val_loss"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        history = list(state["history"])
        prior_runtime_seconds = float(state.get("cumulative_runtime_seconds", 0.0))
        print(f"Resuming {run_dir} at epoch {first_epoch}")
    else:
        backup = preserve_latest_checkpoint(args.runs_dir)
        if backup:
            print(f"Preserved existing checkpoint at {backup}")
        run_dir = args.runs_dir / datetime.now().strftime("wlasl_all_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)

    best_path = run_dir / "best_model.pt"
    config = {
        "input_dim": int(train_x.shape[-1]),
        "num_classes": num_classes,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": True,
        "class_names": class_names,
        "class_to_idx": metadata["class_to_idx"],
        "include_face_landmarks": metadata["include_face_landmarks"],
        "preprocessing": metadata["preprocessing"],
        "dataset_fingerprint": fingerprint,
        "processed_dir": str(processed_dir),
    }
    write_json(run_dir / "model_config.json", config)
    write_json(run_dir / "class_mapping.json", metadata["class_to_idx"])
    if args.resume_from is None:
        shutil.copy2(processed_dir / "dataset_audit.json", run_dir / "dataset_audit.json")

    print(
        f"Training {num_classes} classes on {len(train_y)} samples "
        f"(val={len(val_y)}, test=deferred), device={device}, "
        f"batch_size={args.batch_size}, threads={torch.get_num_threads()}",
        flush=True,
    )
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_started = time.monotonic()
        train_loss, train_acc = run_epoch(
            model, loaders["train"], train_criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(model, loaders["val"], val_criterion, None, device)
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(record)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"seconds={record['epoch_seconds']:.1f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            temporary_best = best_path.with_name("best_model.tmp.pt")
            torch.save(model.state_dict(), temporary_best)
            temporary_best.replace(best_path)
        else:
            epochs_without_improvement += 1

        write_json(run_dir / "training_history.json", {"history": history})
        save_training_state(
            run_dir / "training_state.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            run_dir=run_dir,
            processed_dir=processed_dir,
            fingerprint=fingerprint,
            cumulative_runtime_seconds=prior_runtime_seconds
            + time.monotonic()
            - run_started,
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.", flush=True)
            break

    if not best_path.exists():
        raise SystemExit("No validation-selected checkpoint was saved.")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_x, test_y = load_processed_split(processed_dir, "test")
    if not len(test_x) or not len(test_y):
        raise SystemExit("Held-out test split is empty.")
    if (
        np.any(test_y < 0)
        or np.any(test_y >= num_classes)
        or not np.isfinite(test_x).all()
    ):
        raise SystemExit("Held-out test split is invalid.")
    loaders["test"] = make_loader(test_x, test_y, args.batch_size, False)
    labels, preds = predict_all(model, loaders["test"], device)
    report = classification_report(
        labels,
        preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    np.savetxt(run_dir / "test_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    invocation_runtime = time.monotonic() - run_started
    total_runtime = prior_runtime_seconds + invocation_runtime
    metrics = {
        "evaluation_split": "test",
        "test_accuracy": float(accuracy_score(labels, preds)),
        "test_macro_f1": float(report["macro avg"]["f1-score"]),
        "test_weighted_f1": float(report["weighted avg"]["f1-score"]),
        "test_samples": len(labels),
        "test_class_counts": {
            class_names[idx]: int(np.count_nonzero(np.asarray(labels) == idx))
            for idx in range(num_classes)
        },
        "test_classification_report": report,
        "history": history,
        "runtime_seconds": total_runtime,
        "runtime_seconds_this_invocation": invocation_runtime,
        "device": str(device),
        "classes": num_classes,
        "train_samples": len(train_y),
        "validation_samples": len(val_y),
    }
    write_json(run_dir / "metrics.json", metrics)
    latest = args.runs_dir / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)
    print(
        f"Saved run to {run_dir}; test_accuracy={metrics['test_accuracy']:.4f} "
        f"test_macro_f1={metrics['test_macro_f1']:.4f} "
        f"test_weighted_f1={metrics['test_weighted_f1']:.4f} "
        f"test_samples={metrics['test_samples']} "
        f"runtime_seconds={total_runtime:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
