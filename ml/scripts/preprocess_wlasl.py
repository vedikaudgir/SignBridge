from __future__ import annotations

import argparse
import atexit
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from backend.landmarks import create_holistic  # noqa: E402
from backend.preprocessing import PreprocessingConfig, prepare_landmark_sequence  # noqa: E402
from ml.wlasl_pipeline import (  # noqa: E402
    Sample,
    build_manifest,
    compute_train_stats,
    extract_video_sequence,
    index_videos,
    load_wlasl_metadata,
    save_split,
    select_classes,
    write_manifest,
)

logger = logging.getLogger(__name__)
_WORKER_HOLISTIC: object | None = None
_WORKER_CACHE_DIR: Path | None = None
_WORKER_INCLUDE_FACE_LANDMARKS = False
_WORKER_EVERY_N_FRAMES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess all usable WLASL videos.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "ml/dataset/WLASL/start_kit/WLASL_v0.3.json",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=ROOT / "ml/dataset/WLASL/start_kit/videos",
    )
    parser.add_argument(
        "--audit-file",
        type=Path,
        default=ROOT / "ml/audit/wlasl_v0.3/dataset_audit.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "ml/processed/wlasl_all_usable"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "ml/cache/wlasl_holistic"
    )
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--include-face-landmarks", action="store_true")
    parser.add_argument("--every-n-frames", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def close_worker_holistic() -> None:
    global _WORKER_HOLISTIC
    holistic = _WORKER_HOLISTIC
    _WORKER_HOLISTIC = None
    if holistic is not None:
        holistic.close()


def initialize_worker(
    cache_dir: str, include_face_landmarks: bool, every_n_frames: int
) -> None:
    global _WORKER_HOLISTIC
    global _WORKER_CACHE_DIR
    global _WORKER_INCLUDE_FACE_LANDMARKS
    global _WORKER_EVERY_N_FRAMES
    import cv2

    cv2.setNumThreads(1)
    _WORKER_HOLISTIC = create_holistic(
        include_face_landmarks=include_face_landmarks
    )
    _WORKER_CACHE_DIR = Path(cache_dir)
    _WORKER_INCLUDE_FACE_LANDMARKS = include_face_landmarks
    _WORKER_EVERY_N_FRAMES = every_n_frames
    atexit.register(close_worker_holistic)


def process_sample(
    sample: Sample,
) -> tuple[Sample, np.ndarray | None, str | None]:
    if _WORKER_HOLISTIC is None or _WORKER_CACHE_DIR is None:
        return sample, None, "Landmark worker was not initialized"
    try:
        features = load_cached_sequence(
            sample,
            _WORKER_HOLISTIC,
            _WORKER_CACHE_DIR,
            include_face_landmarks=_WORKER_INCLUDE_FACE_LANDMARKS,
            every_n_frames=_WORKER_EVERY_N_FRAMES,
        )
        return sample, features, None
    except Exception as exc:
        return sample, None, f"{type(exc).__name__}: {exc}"


def load_cached_sequence(
    sample: Sample,
    holistic: object,
    cache_dir: Path,
    include_face_landmarks: bool,
    every_n_frames: int,
) -> np.ndarray:
    profile = f"face-{int(include_face_landmarks)}_stride-{every_n_frames}"
    cache_path = cache_dir / profile / f"{sample.video_id}.npz"
    source = Path(sample.path)
    source_stat = source.stat()
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                is_current = (
                    str(cached["source_path"].item()) == str(source.resolve())
                    and int(cached["source_size"].item()) == source_stat.st_size
                    and int(cached["source_mtime_ns"].item())
                    == source_stat.st_mtime_ns
                    and bool(cached["include_face_landmarks"].item())
                    == include_face_landmarks
                    and int(cached["every_n_frames"].item()) == every_n_frames
                )
                if is_current:
                    features = cached["features"].astype(np.float32, copy=False)
                    if (
                        features.ndim == 2
                        and features.shape[0] > 0
                        and features.shape[1] == 258
                        and np.isfinite(features).all()
                    ):
                        return features
        except (OSError, ValueError, KeyError, EOFError) as exc:
            logger.warning("Ignoring invalid landmark cache %s: %s", cache_path, exc)

    features = extract_video_sequence(
        sample,
        holistic,
        include_face_landmarks=include_face_landmarks,
        every_n_frames=every_n_frames,
    )
    if features.ndim != 2 or features.shape[1] != 258 or not np.isfinite(features).all():
        raise ValueError(
            f"Expected finite 258-feature landmarks, got {features.shape} for {sample.video_id}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f"{cache_path.stem}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        features=features,
        source_path=str(source.resolve()),
        source_size=source_stat.st_size,
        source_mtime_ns=source_stat.st_mtime_ns,
        include_face_landmarks=include_face_landmarks,
        every_n_frames=every_n_frames,
    )
    temporary_path.replace(cache_path)
    return features


def main() -> None:
    args = parse_args()
    if not args.audit_file.exists():
        raise SystemExit(
            f"Dataset audit not found at {args.audit_file}; run "
            "python -m ml.scripts.audit_wlasl first."
        )
    if args.sequence_length <= 0 or args.every_n_frames <= 0 or args.workers <= 0:
        raise SystemExit(
            "sequence-length, every-n-frames, and workers must be positive."
        )
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_wlasl_metadata(args.metadata)
    audit = json.loads(args.audit_file.read_text(encoding="utf-8"))
    metadata_stat = args.metadata.stat()
    if (
        audit.get("metadata_path") != str(args.metadata.resolve())
        or audit.get("metadata_source_size") != metadata_stat.st_size
        or audit.get("metadata_source_mtime_ns") != metadata_stat.st_mtime_ns
        or audit.get("videos_dir") != str(args.videos_dir.resolve())
    ):
        raise SystemExit(
            "Dataset metadata/video directory changed since audit; rerun the WLASL audit."
        )
    video_index = index_videos(args.videos_dir)
    if not video_index:
        raise SystemExit(f"No supported videos found in {args.videos_dir}.")

    readable_video_ids: set[str] = set()
    for video_id, status in audit.get("videos", {}).items():
        path = video_index.get(video_id)
        if path is None or not status.get("readable"):
            continue
        stat = path.stat()
        if (
            status.get("path") == str(path.resolve())
            and status.get("source_size") == stat.st_size
            and status.get("source_mtime_ns") == stat.st_mtime_ns
        ):
            readable_video_ids.add(video_id)
    if not readable_video_ids:
        raise SystemExit(
            "No audited readable videos match the current dataset; rerun the WLASL audit."
        )

    class_names = select_classes(
        entries,
        classes=args.classes,
        video_index=video_index,
        usable_video_ids=readable_video_ids,
    )
    manifest = build_manifest(
        entries,
        class_names,
        args.videos_dir,
        video_index,
        readable_video_ids,
    )
    if not manifest:
        raise SystemExit(f"No matching readable videos found in {args.videos_dir}.")

    unique_samples: dict[str, Sample] = {}
    for sample in manifest:
        previous = unique_samples.get(sample.video_id)
        if previous is not None and (
            previous.gloss != sample.gloss or previous.split != sample.split
        ):
            raise SystemExit(
                f"Video {sample.video_id} has conflicting WLASL gloss/split labels."
            )
        unique_samples[sample.video_id] = sample
    manifest = list(unique_samples.values())
    write_manifest(args.output_dir / "manifest.csv", manifest)

    selected_classes = set(class_names)
    selected_ids = {
        sample.video_id for sample in manifest if sample.gloss in selected_classes
    }
    by_split: dict[str, list[tuple[np.ndarray, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    failed: list[dict[str, str]] = []
    failed_ids: set[str] = set()
    cache_profile = (
        args.cache_dir
        / f"face-{int(args.include_face_landmarks)}_stride-{args.every_n_frames}"
    )
    cache_hits_before_run = sum(
        (cache_profile / f"{video_id}.npz").exists() for video_id in unique_samples
    )
    started = time.monotonic()
    processed_count = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(
            str(args.cache_dir),
            args.include_face_landmarks,
            args.every_n_frames,
        ),
    ) as executor:
        results = executor.map(process_sample, manifest, chunksize=1)
        for index, (sample, raw, error) in enumerate(results, start=1):
            if error is not None or raw is None:
                error = error or "Landmark extraction returned no features"
                failed.append(
                    {"video_id": sample.video_id, "gloss": sample.gloss, "error": error}
                )
                failed_ids.add(sample.video_id)
                logger.error("Failed to process WLASL video %s: %s", sample.video_id, error)
            else:
                if sample.video_id in selected_ids:
                    prepared = prepare_landmark_sequence(
                        raw,
                        PreprocessingConfig(
                            sequence_length=args.sequence_length,
                            feature_dim=raw.shape[1],
                            normalize=False,
                        ),
                    )
                    by_split[sample.split].append((prepared, sample.gloss))
                processed_count += 1
            if index % 100 == 0 or index == len(manifest):
                elapsed = time.monotonic() - started
                rate = index / max(elapsed, 1e-6)
                eta_hours = (len(manifest) - index) / max(rate, 1e-6) / 3600
                print(
                    f"[{index}/{len(manifest)}] cached/extracted={processed_count} "
                    f"failed={len(failed)} rate={rate:.2f}/s eta_hours={eta_hours:.1f}",
                    flush=True,
                )

    successful_counts = {
        split: Counter(gloss for _, gloss in rows) for split, rows in by_split.items()
    }
    final_classes = [
        name
        for name in class_names
        if all(successful_counts[split][name] > 0 for split in ("train", "val", "test"))
    ]
    excluded_classes = sorted(set(class_names) - set(final_classes))
    if not final_classes:
        raise SystemExit(
            "No class has successful train, validation, and test landmark extraction."
        )
    class_to_idx = {label: idx for idx, label in enumerate(final_classes)}

    train_rows = [row for row in by_split["train"] if row[1] in class_to_idx]
    if not train_rows:
        raise SystemExit("No training samples remain after extraction.")
    train_x = np.stack([item[0] for item in train_rows])
    mean, std = compute_train_stats(train_x)
    prep_config = PreprocessingConfig(
        sequence_length=args.sequence_length,
        feature_dim=train_x.shape[-1],
        normalize=True,
        mean=mean.tolist(),
        std=std.tolist(),
    )

    final_split_counts: dict[str, dict[str, int]] = {}
    for split, rows in by_split.items():
        retained_rows = [row for row in rows if row[1] in class_to_idx]
        if retained_rows:
            x = np.stack(
                [prepare_landmark_sequence(row[0], prep_config) for row in retained_rows]
            )
            y = np.asarray(
                [class_to_idx[row[1]] for row in retained_rows], dtype=np.int64
            )
        else:
            x = np.zeros(
                (0, args.sequence_length, train_x.shape[-1]), dtype=np.float32
            )
            y = np.zeros((0,), dtype=np.int64)
        save_split(args.output_dir, split, x, y)
        final_split_counts[split] = {
            gloss: successful_counts[split][gloss] for gloss in final_classes
        }

    processed_manifest = [
        sample
        for sample in manifest
        if sample.gloss in class_to_idx and sample.video_id not in failed_ids
    ]
    signers_by_split = {
        split: {
            sample.signer_id
            for sample in processed_manifest
            if sample.split == split and sample.signer_id is not None
        }
        for split in ("train", "val", "test")
    }
    metadata = {
        "class_names": final_classes,
        "class_to_idx": class_to_idx,
        "include_face_landmarks": args.include_face_landmarks,
        "preprocessing": prep_config.__dict__,
        "split_counts": final_split_counts,
        "signer_overlap": {
            "train_val": sorted(signers_by_split["train"] & signers_by_split["val"]),
            "train_test": sorted(signers_by_split["train"] & signers_by_split["test"]),
            "val_test": sorted(signers_by_split["val"] & signers_by_split["test"]),
        },
        "failed": failed,
        "excluded_classes_after_extraction": excluded_classes,
        "dataset_audit": str(args.audit_file.resolve()),
        "readable_video_count": len(readable_video_ids),
        "eligible_video_count": len(unique_samples),
        "unique_videos_processed": processed_count,
        "cache_hits_before_run": cache_hits_before_run,
        "cache_misses_before_run": len(unique_samples) - cache_hits_before_run,
        "preprocessing_runtime_seconds": time.monotonic() - started,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    shutil.copy2(args.audit_file, args.output_dir / "dataset_audit.json")
    print(
        f"Saved {len(final_classes)} classes and {processed_count} usable videos "
        f"to {args.output_dir}; extraction failures={len(failed)}"
    )
    print(
        f"preprocessing_runtime_seconds={metadata['preprocessing_runtime_seconds']:.1f} "
        f"cache_hits_before_run={cache_hits_before_run} "
        f"cache_misses_before_run={metadata['cache_misses_before_run']}"
    )


if __name__ == "__main__":
    main()
