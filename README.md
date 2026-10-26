# SignBridge

**Bridging Communication Through Sign Language Recognition**

SignBridge is a sign language recognition system designed to facilitate communication between sign language users and individuals who may not understand sign language. The project combines computer vision, machine learning, and real-time web technologies to recognize sign language gestures from webcam input and translate them into corresponding text predictions.

The system aims to explore accessible, technology-assisted communication through a modular architecture that integrates a React frontend, a FastAPI backend, MediaPipe-based landmark extraction, and a Gated Recurrent Unit (GRU) sequence classification model.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Communication Flow](#communication-flow)
- [Current Scope and Limitations](#current-scope-and-limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Communication barriers can arise when sign language users interact with people who are unfamiliar with sign language. SignBridge investigates how computer vision and sequence-based machine learning can support more accessible interactions.

The application captures webcam frames through a web interface and transmits them to a Python backend using WebSockets. The backend extracts body and hand landmarks, constructs temporal feature sequences, and passes them to a trained GRU-based classifier to predict the corresponding sign.

The predicted output is then processed by the backend and made available to the frontend for presentation.

SignBridge currently focuses on isolated sign recognition rather than unrestricted continuous sign language translation.

## Objectives

- Develop a real-time sign recognition pipeline using webcam input.
- Extract meaningful body and hand movement features using MediaPipe.
- Model temporal dependencies between consecutive frames using a GRU network.
- Establish low-latency communication between the frontend and backend.
- Explore the practical application of machine learning in accessible communication systems.
- Build a modular foundation for future improvements in recognition accuracy, vocabulary coverage, and usability.

## Key Features

- **Real-time webcam input:** Captures video frames through the browser.
- **Landmark-based feature extraction:** Uses MediaPipe Holistic to extract pose and hand landmarks.
- **Temporal sequence modeling:** Uses a GRU-based neural network to classify sequences of landmark features.
- **WebSocket communication:** Streams frame data between the frontend and backend.
- **Prediction smoothing:** Applies temporal smoothing to reduce unstable predictions.
- **Confidence-based filtering:** Uses a configurable confidence threshold to filter uncertain predictions.
- **Session management:** Maintains session-specific state and supports cleanup when a WebSocket session ends.
- **Modular architecture:** Separates the user interface, inference service, preprocessing pipeline, and model implementation.

## System Architecture

SignBridge follows a client-server architecture with a dedicated machine learning inference pipeline.

```text
+------------------------------+
|       React Frontend         |
|                              |
|  Webcam Capture              |
|  Frame Processing            |
|  Prediction Display          |
+---------------+--------------+
                |
                | WebSocket
                | JPEG / Base64 Frames
                v
+------------------------------+
|       FastAPI Backend        |
|                              |
|  WebSocket Endpoint          |
|  Frame Decoding              |
|  Session Management          |
+---------------+--------------+
                |
                v
+------------------------------+
|    MediaPipe Processing      |
|                              |
|  Pose Landmark Extraction    |
|  Left Hand Landmarks         |
|  Right Hand Landmarks        |
+---------------+--------------+
                |
                v
+------------------------------+
|     Feature Processing       |
|                              |
|  Feature Vector: 258 Values  |
|  Temporal Window: 30 Frames  |
+---------------+--------------+
                |
                v
+------------------------------+
|    GRU Classification Model  |
|                              |
|  Sequence Inference          |
|  Class Prediction            |
|  Confidence Estimation       |
+---------------+--------------+
                |
                v
+------------------------------+
| Prediction Post-processing   |
|                              |
|  Temporal Smoothing          |
|  Confidence Filtering        |
+---------------+--------------+
                |
                v
+------------------------------+
|       React Frontend         |
|       Text Prediction        |
+------------------------------+
```

### Frontend

The frontend is built with React and Vite. It accesses the user's webcam, renders the video stream, and uses a hidden canvas to convert captured frames into JPEG images. The encoded frames are transmitted to the backend through a WebSocket connection.

### Backend

The backend is implemented using FastAPI. It exposes a WebSocket endpoint at `/ws/sign-stream` to receive frames, manage inference sessions, and coordinate the prediction pipeline.

### Machine Learning Component

The machine learning component extracts pose and hand landmarks, processes them into temporal sequences, and uses a trained GRU classifier to predict the sign class.

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React |
| Frontend tooling | Vite |
| Backend framework | FastAPI |
| Communication | WebSockets |
| Computer vision | OpenCV |
| Landmark extraction | MediaPipe Holistic |
| Machine learning | PyTorch |
| Sequence classification | GRU |
| Dataset | WLASL |
| Programming languages | JavaScript, Python |

---

## Machine Learning Pipeline

The recognition pipeline consists of the following stages.

### 1. Video Frame Acquisition

The frontend captures frames from the webcam and encodes them as JPEG images before transmitting them to the backend.

### 2. Landmark Extraction

MediaPipe Holistic extracts body pose and hand landmarks. Face landmarks are disabled in the current feature configuration.

The feature representation contains:

- Pose landmarks: 33 landmarks with 4 values each, including visibility.
- Left hand landmarks: 21 landmarks with 3 coordinates each.
- Right hand landmarks: 21 landmarks with 3 coordinates each.

The combined feature dimension is:

\[
(33 \times 4) + (21 \times 3) + (21 \times 3) = 258
\]

Each processed frame is represented as a 258-dimensional feature vector.

### 3. Temporal Sequence Construction

Individual frames are organized into temporal sequences to preserve movement information. The current configuration uses a window of 30 frames.

Inference begins once the configured minimum number of frames has been collected.

### 4. GRU-Based Classification

A Gated Recurrent Unit (GRU) network processes the temporal sequence and predicts a class from the model's trained vocabulary.

GRUs are recurrent neural network architectures designed to learn temporal dependencies in sequential data.

### 5. Prediction Post-processing

The backend applies temporal smoothing and confidence-based filtering to reduce unstable or low-confidence outputs before presenting predictions to the frontend.

---

## Dataset

SignBridge uses the **WLASL (Word-Level American Sign Language)** dataset for model development.

WLASL provides video samples of isolated American Sign Language signs and associated class metadata.

The project uses dataset metadata to identify sign classes and their video instances. Preprocessing includes selecting a subset of classes and preserving the official training, validation, and test assignments where applicable.

The current implementation has explored a 15-class subset of WLASL for model training and evaluation.

Dataset preparation involves:

- Reading class labels and video metadata.
- Identifying available and missing video files.
- Extracting pose and hand landmarks from available videos.
- Converting videos into temporal feature sequences.
- Preparing training, validation, and test splits.
- Training and evaluating the sequence classification model.

**Dataset attribution:** WLASL is a research dataset. Refer to the original dataset publication and its distribution terms before redistributing dataset files or derived assets.

---

## Project Structure

The following is a representative layout of the project. Directory names may need to be adjusted to match the current repository.

```text
SignBridge/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── main.py
│   ├── landmarks.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── requirements.txt
│   │
│   └── ml/
│       ├── dataset/
│       ├── checkpoints/
│       └── preprocessing/
│
├── README.md
└── .gitignore
```

The structure above is illustrative. Update it to reflect the exact files and directories present in the repository.

---

## Installation and Setup

### Prerequisites

Ensure that the following tools are installed:

- Python 3.10 or a compatible Python version supported by the project's dependencies.
- Node.js and npm.
- Git.
- A webcam.
- A supported operating system with the required computer vision dependencies.

Use the Python version specified by the project's dependency configuration if it differs from the version listed above.

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd SignBridge
```

Replace `<YOUR_REPOSITORY_URL>` with the repository's actual Git URL.

### 2. Set Up the Backend

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the backend dependencies:

```bash
pip install -r requirements.txt
```

Ensure that the trained model checkpoint and its associated configuration and class-mapping files are available at the paths expected by the backend.

### 3. Set Up the Frontend

Open a separate terminal and navigate to the frontend directory:

```bash
cd frontend
npm install
```

If the frontend uses a configurable WebSocket URL, verify that it points to the backend instance you intend to run.

---

## Running the Application

### Start the Backend

From the `backend/` directory, activate the virtual environment and run:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

The WebSocket endpoint is:

```text
ws://127.0.0.1:8000/ws/sign-stream
```

The reload option is intended for development and should not be used as a production deployment configuration.

### Start the Frontend

From the `frontend/` directory, run:

```bash
npm run dev
```

Open the local URL printed by Vite in your browser. Grant webcam permission when prompted.

Ensure that the backend is running, the model files are accessible, and the configured WebSocket URL matches the backend address.

---

## Communication Flow

The typical application workflow is as follows:

1. The user opens the SignBridge web interface.
2. The browser requests permission to access the webcam.
3. Video frames are captured and encoded as JPEG images.
4. The frontend transmits the encoded frames through a WebSocket connection.
5. The backend decodes the incoming frames and extracts pose and hand landmarks.
6. Landmark vectors are collected into a temporal sequence.
7. The trained GRU model generates a class prediction and confidence score.
8. The backend applies prediction smoothing and confidence filtering.
9. The resulting prediction is returned to the frontend for display.

---

## Current Scope and Limitations

SignBridge is an ongoing project, and its current implementation has several limitations:

- **Limited vocabulary:** Recognition is restricted to the classes represented in the trained model.
- **Dataset availability:** Missing or unavailable source videos can reduce the number of usable training samples.
- **Recognition accuracy:** Performance depends on dataset coverage, class balance, preprocessing quality, and model generalization.
- **Environmental sensitivity:** Lighting, camera position, occlusion, and signing speed may affect landmark extraction.
- **Temporal constraints:** A fixed-length temporal window may not capture every sign equally well.
- **Isolated sign recognition:** The current system is not a complete continuous sign language translation solution.
- **Language coverage:** The model is trained for a subset of American Sign Language signs and should not be assumed to support other sign languages.
- **Deployment constraints:** Local development URLs and webcam access settings must be adapted for remote or production deployment.

The system is intended as an assistive technology research prototype. Its predictions should not be treated as a substitute for qualified sign language interpretation in critical situations.

---

## Future Improvements

Potential directions for future development include:

- Expanding the training vocabulary and improving class balance.
- Increasing the availability and diversity of training samples.
- Improving landmark normalization and temporal preprocessing.
- Evaluating alternative sequence architectures and training strategies.
- Improving robustness across different users, backgrounds, and recording conditions.
- Calibrating prediction confidence using validation data.
- Supporting configurable inference thresholds and more informative feedback for uncertain predictions.
- Optimizing inference latency and resource utilization.
- Adding sentence-level recognition and continuous signing support.
- Improving accessibility, interface usability, and deployment reliability.
- Conducting more extensive evaluation using independent test data and established classification metrics.

---

## Contributing

Contributions that improve recognition performance, system reliability, documentation, accessibility, or maintainability are welcome.

A typical contribution workflow is:

1. Fork the repository.
2. Create a feature branch.
3. Implement and test the proposed changes.
4. Commit the changes with clear messages.
5. Open a pull request describing the changes and their impact.

For changes to the machine learning pipeline, document the dataset split, preprocessing configuration, model configuration, and evaluation methodology to support reproducibility.

## License

No license is specified in this README. Add a license after deciding how the source code may be used, modified, and redistributed. Dataset licensing and attribution requirements should be reviewed separately.

---

## Acknowledgements

SignBridge builds on open-source technologies and research in computer vision, recurrent neural networks, and sign language recognition.

We acknowledge the developers and researchers behind React, Vite, FastAPI, OpenCV, MediaPipe, PyTorch, and the WLASL dataset for making these technologies and resources available to the research and development community.

---

**SignBridge — Exploring accessible communication through computer vision and machine learning. Yet to be deployed.**
