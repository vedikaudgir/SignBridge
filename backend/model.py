import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from preprocessing import PreprocessingConfig, prepare_landmark_sequence

logger = logging.getLogger("signbridge.backend.model")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from ml.sign_model import SignSequenceClassifier
except Exception as exc:  # pragma: no cover - supports running without training deps.
    torch = None
    SignSequenceClassifier = None
    logger.warning("PyTorch model support unavailable: %s", exc)


class SignBridgeModel:
    def __init__(self) -> None:
        self.model = None
        self.class_names: list[str] = []
        self.preprocessing = PreprocessingConfig()
        self.device = None
        model_dir = Path(os.getenv("SIGNBRIDGE_MODEL_DIR", ROOT / "ml/runs/latest"))
        self.load(model_dir)

    def load(self, model_dir: Path) -> None:
        config_path = model_dir / "model_config.json"
        weights_path = model_dir / "best_model.pt"
        if torch is None or SignSequenceClassifier is None:
            return
        if not config_path.exists() or not weights_path.exists():
            logger.warning("No trained SignBridge model found at %s", model_dir)
            return

        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.class_names = list(config["class_names"])
        self.preprocessing = PreprocessingConfig(**config["preprocessing"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SignSequenceClassifier(
            input_dim=config["input_dim"],
            num_classes=config["num_classes"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            bidirectional=config.get("bidirectional", True),
        ).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        logger.info(
            "Loaded SignBridge model from %s with %d classes on %s",
            model_dir,
            len(self.class_names),
            self.device,
        )

    def _classify_hand(self, hand: np.ndarray) -> tuple[str, float]:
        wrist = hand[0]
        thumb_mcp = hand[2]
        thumb_ip = hand[3]
        thumb_tip = hand[4]
        index_mcp = hand[5]
        index_pip = hand[6]
        index_tip = hand[8]
        middle_pip = hand[10]
        middle_tip = hand[12]
        ring_pip = hand[14]
        ring_tip = hand[16]
        pinky_pip = hand[18]
        pinky_tip = hand[20]

        def dist(p1, p2):
            return float(np.linalg.norm(p1[:2] - p2[:2]))

        d_index_pip = dist(index_pip, wrist)
        d_index_tip = dist(index_tip, wrist)
        index_ext = d_index_tip > d_index_pip * 1.15

        d_middle_pip = dist(middle_pip, wrist)
        d_middle_tip = dist(middle_tip, wrist)
        middle_ext = d_middle_tip > d_middle_pip * 1.15

        d_ring_pip = dist(ring_pip, wrist)
        d_ring_tip = dist(ring_tip, wrist)
        ring_ext = d_ring_tip > d_ring_pip * 1.15

        d_pinky_pip = dist(pinky_pip, wrist)
        d_pinky_tip = dist(pinky_tip, wrist)
        pinky_ext = d_pinky_tip > d_pinky_pip * 1.15

        d_thumb_tip = dist(thumb_tip, wrist)
        d_thumb_ip = dist(thumb_ip, wrist)
        thumb_ext = d_thumb_tip > d_thumb_ip * 1.10 and dist(thumb_tip, index_mcp) > 0.08
        thumb_up = thumb_tip[1] < thumb_mcp[1] and thumb_tip[1] < wrist[1]
        thumb_down = thumb_tip[1] > thumb_mcp[1] and thumb_tip[1] > wrist[1]

        # OK sign: thumb tip close to index tip, other 3 fingers extended
        if dist(thumb_tip, index_tip) < 0.06 and middle_ext and ring_ext and pinky_ext:
            return "OK", 0.94

        # I LOVE YOU (ASL): thumb, index, pinky extended; middle, ring folded
        if thumb_ext and index_ext and pinky_ext and not middle_ext and not ring_ext:
            return "I LOVE YOU", 0.96

        # Peace / Victory: index and middle extended; ring and pinky folded
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "PEACE", 0.95

        # Thumbs up: YES
        if thumb_up and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "YES", 0.93

        # Thumbs down: NO
        if thumb_down and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "NO", 0.92

        # Pointing / You: index extended, others folded
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "YOU", 0.90

        # Pinky only / Promise
        if pinky_ext and not index_ext and not middle_ext and not ring_ext:
            return "PROMISE", 0.90

        # Hello / Open hand / Wave: all 4 fingers extended
        if index_ext and middle_ext and ring_ext and pinky_ext:
            return "HELLO", 0.93

        # Fist / Stop
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "STOP", 0.88

        return "unknown", 0.0

    def _heuristic_predict(self, landmark_sequence: np.ndarray) -> tuple[str, float]:
        if landmark_sequence.shape[0] == 0 or landmark_sequence.shape[1] < 258:
            return "unknown", 0.0

        # Sample the most recent frame(s)
        recent_frames = landmark_sequence[-min(5, len(landmark_sequence)):]
        predictions = []

        for frame in recent_frames:
            # Pose: 0:132, Left hand: 132:195, Right hand: 195:258
            left_hand_vec = frame[132:195]
            right_hand_vec = frame[195:258]

            left_active = float(np.count_nonzero(left_hand_vec)) > 15
            right_active = float(np.count_nonzero(right_hand_vec)) > 15

            # Two-handed gestures
            if left_active and right_active:
                left_wrist = left_hand_vec[:3]
                right_wrist = right_hand_vec[:3]
                wrist_dist = float(np.linalg.norm(left_wrist[:2] - right_wrist[:2]))
                if wrist_dist < 0.12:
                    predictions.append(("PLEASE", 0.92))
                    continue

            # Check right hand first, then left hand
            if right_active:
                hand = right_hand_vec.reshape((21, 3))
                pred, conf = self._classify_hand(hand)
                if pred != "unknown":
                    predictions.append((pred, conf))
            elif left_active:
                hand = left_hand_vec.reshape((21, 3))
                pred, conf = self._classify_hand(hand)
                if pred != "unknown":
                    predictions.append((pred, conf))

        if not predictions:
            return "unknown", 0.0

        # Return majority prediction among recent frames
        from collections import Counter
        counts = Counter(p[0] for p in predictions)
        most_common_label, count = counts.most_common(1)[0]
        matching_confidences = [p[1] for p in predictions if p[0] == most_common_label]
        avg_confidence = float(sum(matching_confidences) / len(matching_confidences))
        return most_common_label, avg_confidence

    def predict(self, landmark_sequence: np.ndarray) -> tuple[str, float]:
        predictions = self.predict_top_k(landmark_sequence, top_k=1)
        return predictions[0] if predictions else ("unknown", 0.0)

    def predict_top_k(
        self, landmark_sequence: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.model is None or torch is None:
            return [self._heuristic_predict(landmark_sequence)]
        if landmark_sequence.ndim != 2 or landmark_sequence.size == 0:
            return [("unknown", 0.0)]

        prepared = prepare_landmark_sequence(landmark_sequence, self.preprocessing)
        with torch.no_grad():
            inputs = torch.from_numpy(prepared).unsqueeze(0).to(self.device)
            probabilities = torch.softmax(self.model(inputs), dim=1).cpu().numpy()[0]
        top_indices = np.argsort(probabilities)[-min(top_k, len(probabilities)):][::-1]
        return [
            (self.class_names[int(index)], float(probabilities[index]))
            for index in top_indices
        ]


_MODEL = SignBridgeModel()


def predict(landmark_sequence: np.ndarray) -> tuple[str, float]:
    return _MODEL.predict(landmark_sequence)


def predict_top_k(
    landmark_sequence: np.ndarray, top_k: int = 5
) -> list[tuple[str, float]]:
    return _MODEL.predict_top_k(landmark_sequence, top_k)
