from __future__ import annotations

import unittest
from pathlib import Path

import torch

from ml.sign_model import SignSequenceClassifier
from ml.wlasl_pipeline import Sample, build_manifest, select_classes


class WlaslPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = []
        self.video_index: dict[str, Path] = {}
        for class_index in range(20):
            gloss = f"gloss_{class_index:02d}"
            instances = []
            for split in ("train", "val", "test"):
                video_id = f"{class_index:02d}_{split}"
                self.video_index[video_id] = Path(video_id + ".mp4")
                instances.append({"video_id": video_id, "split": split})
            self.entries.append({"gloss": gloss, "instances": instances})

    def test_selects_all_eligible_classes_without_a_class_cap(self) -> None:
        selected = select_classes(self.entries, video_index=self.video_index)
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected, [f"gloss_{index:02d}" for index in range(20)])

    def test_manifest_preserves_official_splits(self) -> None:
        selected = select_classes(self.entries, video_index=self.video_index)
        manifest = build_manifest(
            self.entries, selected, Path("."), self.video_index
        )
        self.assertEqual(len(manifest), 60)
        self.assertEqual(
            {sample.split for sample in manifest}, {"train", "val", "test"}
        )

    def test_classifier_output_layer_uses_requested_class_count(self) -> None:
        model = SignSequenceClassifier(
            input_dim=258,
            num_classes=974,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
        )
        self.assertEqual(model.classifier[-1].out_features, 974)
        self.assertEqual(model(torch.zeros((2, 30, 258))).shape, (2, 974))


if __name__ == "__main__":
    unittest.main()
