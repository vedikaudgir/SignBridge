from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from landmarks import extract_landmark_vector  # noqa: E402


@dataclass(frozen=True)
class Sample:
    gloss: str
    split: str
    video_id: str
    signer_id: int | None
    frame_start: int
    frame_end: int
    path: str


def load_wlasl_metadata(metadata_path: Path) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def index_videos(videos_dir: Path) -> dict[str, Path]:
    if not videos_dir.exists():
        return {}
    supported_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return {
        path.stem: path
        for path in sorted(videos_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in supported_extensions
    }


def select_classes(
    entries: list[dict],
    classes: list[str] | None = None,
    video_index: dict[str, Path] | None = None,
    usable_video_ids: set[str] | None = None,
) -> list[str]:
    requested = (
        list(dict.fromkeys(label.strip() for label in classes if label.strip()))
        if classes
        else None
    )
    available = {entry["gloss"] for entry in entries}
    if requested is not None:
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"Requested classes not found in WLASL metadata: {missing}")

    eligible: set[str] = set()
    for entry in entries:
        instances = entry.get("instances", [])
        split_counts = Counter()
        for instance in instances:
            video_id = str(instance.get("video_id"))
            if instance.get("split") not in {"train", "val", "test"}:
                continue
            if video_index is not None and video_id not in video_index:
                continue
            if usable_video_ids is not None and video_id not in usable_video_ids:
                continue
            split_counts[instance["split"]] += 1
        if all(split_counts[split] for split in ("train", "val", "test")):
            eligible.add(entry["gloss"])

    selected = requested if requested is not None else sorted(eligible)
    unavailable = sorted(set(selected) - eligible)
    if unavailable:
        raise ValueError(
            "Requested classes lack readable train, validation, and test videos: "
            f"{unavailable}"
        )
    if not selected:
        raise ValueError(
            "No gloss has readable videos in all official train, validation, and test splits."
        )
    return selected


def find_video(video_id: str, videos_dir: Path) -> Path | None:
    for suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        candidate = videos_dir / f"{video_id}{suffix}"
        if candidate.exists():
            return candidate
    matches = list(videos_dir.rglob(f"{video_id}.*"))
    return matches[0] if matches else None


def build_manifest(
    entries: list[dict],
    class_names: list[str],
    videos_dir: Path,
    video_index: dict[str, Path] | None = None,
    usable_video_ids: set[str] | None = None,
) -> list[Sample]:
    selected = set(class_names)
    manifest: list[Sample] = []
    for entry in entries:
        gloss = entry["gloss"]
        if gloss not in selected:
            continue
        for inst in entry.get("instances", []):
            if inst.get("split") not in {"train", "val", "test"}:
                continue
            video_id = str(inst["video_id"])
            if usable_video_ids is not None and video_id not in usable_video_ids:
                continue
            video_path = (
                video_index.get(video_id)
                if video_index is not None
                else find_video(video_id, videos_dir)
            )
            if video_path is None:
                continue
            manifest.append(
                Sample(
                    gloss=gloss,
                    split=inst["split"],
                    video_id=video_id,
                    signer_id=inst.get("signer_id"),
                    frame_start=int(inst.get("frame_start") or 1),
                    frame_end=int(inst.get("frame_end") or -1),
                    path=str(video_path),
                )
            )
    return manifest


def extract_video_sequence(
    sample: Sample,
    holistic,
    include_face_landmarks: bool,
    every_n_frames: int = 1,
) -> np.ndarray:
    cap = cv2.VideoCapture(sample.path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {sample.path}")

    vectors: list[np.ndarray] = []
    frame_number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            if frame_number < sample.frame_start:
                continue
            if sample.frame_end > 0 and frame_number > sample.frame_end:
                break
            if (frame_number - sample.frame_start) % every_n_frames != 0:
                continue
            vectors.append(
                extract_landmark_vector(
                    holistic,
                    frame,
                    include_face_landmarks=include_face_landmarks,
                )
            )
    finally:
        cap.release()

    if not vectors:
        raise ValueError(f"No frames extracted from {sample.path}")
    return np.stack(vectors).astype(np.float32)


def compute_train_stats(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = sequences.reshape(-1, sequences.shape[-1])
    non_zero = np.any(np.abs(flat) > 1e-8, axis=1)
    stat_source = flat[non_zero] if np.any(non_zero) else flat
    mean = stat_source.mean(axis=0).astype(np.float32)
    std = stat_source.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def write_manifest(path: Path, rows: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_split(output_dir: Path, split: str, x: np.ndarray, y: np.ndarray) -> None:
    output_path = output_dir / f"{split}.npz"
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        x=x.astype(np.float32),
        y=y.astype(np.int64),
    )
    temporary_path.replace(output_path)


def load_processed_split(processed_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(processed_dir / f"{split}.npz")
    return data["x"].astype(np.float32), data["y"].astype(np.int64)
