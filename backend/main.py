import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("signbridge.backend")
logging.basicConfig(level=logging.INFO)

from landmarks import create_holistic, decode_base64_frame, extract_landmark_vector
from model import predict_top_k


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


BASE_DIR = Path(__file__).resolve().parent
load_env_file(BASE_DIR / ".env")

WINDOW_SIZE = int(os.getenv("SIGNBRIDGE_WINDOW_SIZE", "30"))
CONFIDENCE_THRESHOLD = float(os.getenv("SIGNBRIDGE_CONFIDENCE_THRESHOLD", "0.01"))
CONSECUTIVE_PREDICTION_COUNT = int(
    os.getenv("SIGNBRIDGE_CONSECUTIVE_PREDICTIONS", "3")
)
INCLUDE_FACE_LANDMARKS = (
    os.getenv("SIGNBRIDGE_INCLUDE_FACE_LANDMARKS", "false").lower() == "true"
)
HOST = os.getenv("SIGNBRIDGE_HOST", "0.0.0.0")
PORT = int(os.getenv("SIGNBRIDGE_PORT", "8000"))
RELOAD = os.getenv("SIGNBRIDGE_RELOAD", "true").lower() == "true"

app = FastAPI(title="SignBridge Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class TemporalSmoothingState:
    recent_predictions: Deque[str] = field(
        default_factory=lambda: deque(maxlen=CONSECUTIVE_PREDICTION_COUNT)
    )
    last_emitted_prediction: str | None = None


@dataclass
class SessionState:
    session_id: str
    landmark_window: Deque[np.ndarray]
    holistic: object
    smoothing: TemporalSmoothingState


active_sessions: dict[str, SessionState] = {}
# TODO: If you later need cross-endpoint session lookups, move this into a small session manager.


def build_session_state() -> SessionState:
    session_id = str(uuid4())
    return SessionState(
        session_id=session_id,
        landmark_window=deque(maxlen=WINDOW_SIZE),
        holistic=create_holistic(include_face_landmarks=INCLUDE_FACE_LANDMARKS),
        smoothing=TemporalSmoothingState(),
    )


def smooth_prediction(
    prediction: str,
    confidence: float,
    smoothing_state: TemporalSmoothingState,
) -> tuple[str | None, float]:
    if confidence < CONFIDENCE_THRESHOLD:
        smoothing_state.recent_predictions.clear()
        return None, confidence

    smoothing_state.recent_predictions.append(prediction)

    if len(smoothing_state.recent_predictions) < CONSECUTIVE_PREDICTION_COUNT:
        return None, confidence

    if len(set(smoothing_state.recent_predictions)) != 1:
        return None, confidence

    smoothing_state.last_emitted_prediction = prediction
    return prediction, confidence


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/sign-stream")
async def sign_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session = build_session_state()
    active_sessions[session.session_id] = session
    logger.info("WebSocket connected: %s", session.session_id)

    try:
        while True:
            raw_message = await websocket.receive_text()
            frames_received = len(session.landmark_window)

            try:
                payload = json.loads(raw_message)
                frame_b64 = payload["frame"]

                frame = decode_base64_frame(frame_b64)
                landmark_vector = extract_landmark_vector(
                    session.holistic,
                    frame,
                    include_face_landmarks=INCLUDE_FACE_LANDMARKS,
                )
                session.landmark_window.append(landmark_vector)
                frames_received += 1

                prediction_payload = {"prediction": None, "confidence": 0.0}

                if len(session.landmark_window) >= min(WINDOW_SIZE, 10):
                    landmark_sequence = np.array(
                        session.landmark_window, dtype=np.float32
                    )
                    top_predictions = predict_top_k(landmark_sequence, top_k=5)
                    logger.info(
                        "WebSocket %s raw top-5 model predictions: %s",
                        session.session_id,
                        [(label, round(confidence, 4)) for label, confidence in top_predictions],
                    )
                    predicted_label, confidence = top_predictions[0]
                    if confidence < CONFIDENCE_THRESHOLD:
                        logger.info(
                            "WebSocket %s prediction rejected: low confidence "
                            "(%s=%.4f, threshold=%.4f)",
                            session.session_id,
                            predicted_label,
                            confidence,
                            CONFIDENCE_THRESHOLD,
                        )
                    smoothed_prediction, smoothed_confidence = smooth_prediction(
                        predicted_label,
                        confidence,
                        session.smoothing,
                    )
                    if confidence >= CONFIDENCE_THRESHOLD and not smoothed_prediction:
                        recent = list(session.smoothing.recent_predictions)
                        if len(recent) < CONSECUTIVE_PREDICTION_COUNT:
                            logger.info(
                                "WebSocket %s prediction rejected: unstable prediction "
                                "(stability pending %d/%d, labels=%s)",
                                session.session_id,
                                len(recent),
                                CONSECUTIVE_PREDICTION_COUNT,
                                recent,
                            )
                        else:
                            logger.info(
                                "WebSocket %s prediction rejected: unstable prediction "
                                "(recent labels=%s)",
                                session.session_id,
                                recent,
                            )
                    elif smoothed_prediction:
                        logger.info(
                            "WebSocket %s prediction accepted: %s confidence=%.4f",
                            session.session_id,
                            smoothed_prediction,
                            smoothed_confidence,
                        )
                    prediction_payload = {
                        "prediction": smoothed_prediction,
                        "confidence": smoothed_confidence,
                    }
                else:
                    logger.info(
                        "WebSocket %s prediction rejected: insufficient frames "
                        "(%d/%d)",
                        session.session_id,
                        frames_received,
                        min(WINDOW_SIZE, 10),
                    )

                await websocket.send_json(prediction_payload)
            except KeyError:
                logger.warning(
                    "WebSocket %s sent a payload without a frame key",
                    session.session_id,
                )
                try:
                    await websocket.send_json({"prediction": None, "confidence": 0.0})
                except Exception:
                    break
            except Exception as exc:
                logger.exception(
                    "WebSocket %s prediction rejected: inference/frame processing "
                    "error: %s",
                    session.session_id,
                    exc,
                )
                try:
                    await websocket.send_json({"prediction": None, "confidence": 0.0})
                except Exception:
                    break
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", session.session_id)
    finally:
        session.landmark_window.clear()
        session.smoothing.recent_predictions.clear()
        session.holistic.close()
        active_sessions.pop(session.session_id, None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=RELOAD)
