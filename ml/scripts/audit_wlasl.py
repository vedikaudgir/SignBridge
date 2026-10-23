from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.wlasl_pipeline import index_videos, load_wlasl_metadata  # noqa: E402

logger = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local WLASL videos and official metadata splits."
    )
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
        "--output-dir", type=Path, default=ROOT / "ml/audit/wlasl_v0.3"
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def check_video(path: Path) -> dict[str, int | float | str | bool | None]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not open the video")
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise ValueError("Video has no decodable first frame")
        return {
            "readable": True,
            "error": None,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    except (cv2.error, ValueError) as exc:
        return {
            "readable": False,
            "error": str(exc),
            "width": None,
            "height": None,
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()


def write_audit_csv(path: Path, instances: list[dict]) -> None:
    columns = (
        "gloss",
        "video_id",
        "split",
        "signer_id",
        "frame_start",
        "frame_end",
        "has_local_video",
        "readable",
        "path",
        "error",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(instances)


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)
    entries = load_wlasl_metadata(args.metadata)
    video_index = index_videos(args.videos_dir)
    if not video_index:
        raise SystemExit(f"No supported video files found under {args.videos_dir}")

    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "dataset_audit.json"
    previous = {}
    if audit_path.exists() and not args.no_resume:
        previous = json.loads(audit_path.read_text(encoding="utf-8")).get(
            "videos", {}
        )

    videos: dict[str, dict] = {}
    for index, (video_id, path) in enumerate(video_index.items(), start=1):
        stat = path.stat()
        cached = previous.get(video_id, {})
        if (
            cached.get("path") == str(path.resolve())
            and cached.get("source_size") == stat.st_size
            and cached.get("source_mtime_ns") == stat.st_mtime_ns
            and isinstance(cached.get("readable"), bool)
        ):
            videos[video_id] = cached
        else:
            status = check_video(path)
            videos[video_id] = {
                **status,
                "path": str(path.resolve()),
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
            }
        if index % 500 == 0 or index == len(video_index):
            print(f"Audited video files {index}/{len(video_index)}", flush=True)

    instances: list[dict] = []
    counts: dict[str, dict[str, dict[str, int]]] = {}
    glosses_with_instances: set[str] = set()
    for entry in entries:
        gloss = str(entry["gloss"])
        glosses_with_instances.add(gloss)
        counts[gloss] = {
            split: {"metadata": 0, "local": 0, "readable": 0}
            for split in SPLITS
        }
        for instance in entry.get("instances", []):
            video_id = str(instance["video_id"])
            split = str(instance.get("split", "unknown"))
            status = videos.get(video_id)
            has_local = status is not None
            readable = bool(status and status["readable"])
            row = {
                "gloss": gloss,
                "video_id": video_id,
                "split": split,
                "signer_id": instance.get("signer_id"),
                "frame_start": int(instance.get("frame_start") or 1),
                "frame_end": int(instance.get("frame_end") or -1),
                "has_local_video": has_local,
                "readable": readable,
                "path": status["path"] if status else None,
                "error": status.get("error") if status else "video file not found",
            }
            instances.append(row)
            if split not in counts[gloss]:
                continue
            counts[gloss][split]["metadata"] += 1
            if has_local:
                counts[gloss][split]["local"] += 1
            if readable:
                counts[gloss][split]["readable"] += 1

    eligible_classes = sorted(
        gloss
        for gloss, split_counts in counts.items()
        if all(split_counts[split]["readable"] > 0 for split in SPLITS)
    )
    split_totals = {
        split: {
            key: sum(c[split][key] for c in counts.values())
            for key in ("metadata", "local", "readable")
        }
        for split in SPLITS
    }
    audit = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata_path": str(args.metadata.resolve()),
        "metadata_source_size": args.metadata.stat().st_size,
        "metadata_source_mtime_ns": args.metadata.stat().st_mtime_ns,
        "videos_dir": str(args.videos_dir.resolve()),
        "audit_runtime_seconds": time.monotonic() - started,
        "gloss_count": len(entries),
        "metadata_instance_count": len(instances),
        "video_file_count": len(video_index),
        "video_files_unreferenced_by_metadata": sorted(
            set(video_index) - {str(row["video_id"]) for row in instances}
        ),
        "videos": videos,
        "split_totals": split_totals,
        "class_counts": counts,
        "eligible_classes_train_val_test": eligible_classes,
        "per_gloss_split_counts": [
            {
                "gloss": gloss,
                **{
                    f"{split}_{kind}": counts[gloss][split][kind]
                    for split in SPLITS
                    for kind in ("metadata", "local", "readable")
                },
                "eligible": gloss in eligible_classes,
            }
            for gloss in sorted(glosses_with_instances)
        ],
        "instances": instances,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_audit_csv(args.output_dir / "dataset_audit.csv", instances)
    print(
        json.dumps(
            {
                "glosses": len(entries),
                "video_files": len(video_index),
                "readable_videos": sum(
                    bool(video["readable"]) for video in videos.values()
                ),
                "split_totals": split_totals,
                "eligible_classes_train_val_test": len(eligible_classes),
                "unreferenced_video_files": len(
                    audit["video_files_unreferenced_by_metadata"]
                ),
                "audit_json": str(audit_path.resolve()),
                "audit_csv": str((args.output_dir / "dataset_audit.csv").resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
