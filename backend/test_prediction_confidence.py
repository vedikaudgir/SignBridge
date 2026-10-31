from __future__ import annotations

import base64
import sys
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main


class _HolisticStub:
    def close(self) -> None:
        pass


class PredictionConfidenceTests(unittest.TestCase):
    def test_configured_confidence_threshold_is_one_percent(self) -> None:
        self.assertEqual(main.CONFIDENCE_THRESHOLD, 0.01)

    def test_stability_gate_preserves_current_model_confidence(self) -> None:
        state = main.TemporalSmoothingState()

        self.assertEqual(main.smooth_prediction("drink", 0.03, state), (None, 0.03))
        self.assertEqual(main.smooth_prediction("drink", 0.04, state), (None, 0.04))
        self.assertEqual(main.smooth_prediction("drink", 0.05, state), ("drink", 0.05))

    def test_low_confidence_clears_stability_history_and_preserves_score(self) -> None:
        state = main.TemporalSmoothingState()
        main.smooth_prediction("drink", 0.03, state)

        self.assertEqual(main.smooth_prediction("drink", 0.005, state), (None, 0.005))
        self.assertEqual(list(state.recent_predictions), [])

    def test_websocket_returns_stable_label_and_actual_confidence(self) -> None:
        session = main.SessionState(
            session_id="test-session",
            landmark_window=deque(maxlen=main.WINDOW_SIZE),
            holistic=_HolisticStub(),
            smoothing=main.TemporalSmoothingState(),
        )
        scores = iter((0.03, 0.04, 0.05))
        frame_vector = np.zeros(258, dtype=np.float32)

        with (
            patch.object(main, "build_session_state", return_value=session),
            patch.object(main, "decode_base64_frame", return_value=np.zeros((1, 1, 3))),
            patch.object(main, "extract_landmark_vector", return_value=frame_vector),
            patch.object(
                main,
                "predict_top_k",
                side_effect=lambda _sequence, top_k: [("drink", next(scores))],
            ),
            TestClient(main.app) as client,
        ):
            with client.websocket_connect("/ws/sign-stream") as websocket:
                responses = []
                for _ in range(min(main.WINDOW_SIZE, 10) + 2):
                    websocket.send_text(
                        '{"frame":"' + base64.b64encode(b"frame").decode("ascii") + '"}'
                    )
                    responses.append(websocket.receive_json())

        first_inference = min(main.WINDOW_SIZE, 10) - 1
        self.assertEqual(responses[first_inference], {"prediction": None, "confidence": 0.03})
        self.assertEqual(responses[first_inference + 1], {"prediction": None, "confidence": 0.04})
        self.assertEqual(responses[first_inference + 2], {"prediction": "drink", "confidence": 0.05})

    def test_websocket_returns_null_when_inference_fails(self) -> None:
        session = main.SessionState(
            session_id="error-session",
            landmark_window=deque(maxlen=main.WINDOW_SIZE),
            holistic=_HolisticStub(),
            smoothing=main.TemporalSmoothingState(),
        )
        frame_vector = np.zeros(258, dtype=np.float32)

        with (
            patch.object(main, "build_session_state", return_value=session),
            patch.object(main, "decode_base64_frame", return_value=np.zeros((1, 1, 3))),
            patch.object(main, "extract_landmark_vector", return_value=frame_vector),
            patch.object(main, "predict_top_k", side_effect=RuntimeError("inference failed")),
            TestClient(main.app) as client,
        ):
            with client.websocket_connect("/ws/sign-stream") as websocket:
                for _ in range(min(main.WINDOW_SIZE, 10) - 1):
                    websocket.send_text('{"frame":"ZmFrZQ=="}')
                    websocket.receive_json()
                websocket.send_text('{"frame":"ZmFrZQ=="}')
                response = websocket.receive_json()

        self.assertEqual(response, {"prediction": None, "confidence": 0.0})


if __name__ == "__main__":
    unittest.main()
