# SignBridge ML Pipeline

This pipeline trains a lightweight sequence classifier on WLASL videos using the
same MediaPipe Holistic feature vector as the live FastAPI backend:

`pose 33x(x,y,z,visibility) + left hand 21x(x,y,z) + right hand 21x(x,y,z)`.

Face landmarks are optional and must match `SIGNBRIDGE_INCLUDE_FACE_LANDMARKS`
in the backend.

## 1. Install dependencies

The repository environment is configured for Python 3.12. From the repo root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
```

Training uses CUDA when available and otherwise runs on CPU.

## 2. Put WLASL videos in place

The metadata is expected at:

```text
ml/dataset/WLASL/start_kit/WLASL_v0.3.json
```

The WLASL start-kit videos are stored under:

```text
ml/dataset/WLASL/start_kit/videos
```

The audit checks every metadata instance against local video files and decodes
the first frame of every found video. It records per-gloss/per-split metadata,
local, and readable counts. A class is eligible when it has at least one
readable example in each official split; there is no class-count cap.

## 3. Audit and preprocess all eligible classes

The audit resumes unchanged video checks from its prior report. Landmark
features for all videos in eligible classes are cached independently and reused
across interrupted preprocessing runs; readable videos in classes missing an
official split remain in the audit but are not used for supervised training.

```powershell
.\.venv\Scripts\python.exe -m ml.scripts.audit_wlasl
.\.venv\Scripts\python.exe -m ml.scripts.preprocess_wlasl --workers 6
```

Audit JSON and CSV (all metadata instances and per-gloss/split counts) are
written to `ml/audit/wlasl_v0.3`. Processed official splits and a copy of the
audit are written to `ml/processed/wlasl_all_usable`.

## 4. Train and resume

```powershell
.\.venv\Scripts\python.exe -m ml.scripts.train_sequence_model
```

The GRU uses square-root inverse-frequency class weights. A full optimizer/model
checkpoint and training history are saved after every epoch. Resume an
interrupted run with:

```powershell
.\.venv\Scripts\python.exe -m ml.scripts.train_sequence_model `
  --resume-from ml/runs/<run-name>/training_state.pt
```

The previous `ml/runs/latest` checkpoint is backed up before a new run. Outputs
include:

- `best_model.pt`
- `model_config.json`
- `class_mapping.json`
- `dataset_audit.json`
- `training_history.json`
- `training_state.pt`
- `metrics.json`
- `test_confusion_matrix.csv`

On completion, `ml/runs/latest` points to the new run, including when training
was resumed from an epoch checkpoint.

## 5. Evaluate

```powershell
.\.venv\Scripts\python.exe -m ml.scripts.evaluate_sequence_model `
  --run-dir ml/runs/latest `
  --processed-dir ml/processed/wlasl_all_usable
```

## 6. Run Real Inference In The Backend

Point the backend at the trained artifacts:

```powershell
$env:SIGNBRIDGE_MODEL_DIR = "E:\SignBridge\ml\runs\latest"
$env:SIGNBRIDGE_WINDOW_SIZE = "30"
$env:SIGNBRIDGE_INCLUDE_FACE_LANDMARKS = "false"
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --reload
```

`backend/model.py` returns `unknown` until a valid checkpoint is configured.
The existing WebSocket shape and smoothing behavior are unchanged.
