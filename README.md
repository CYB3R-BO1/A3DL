# Autonomous Adversarial Attack & Defense Lab (A3DL)

A3DL is an MVP AI red-teaming platform for adversarial robustness testing of image classifiers.

## Current Implementation Status

- Backend: FastAPI service with attack, detection, defense, report, and agent recommendation APIs.
- Training: Checkpoint training endpoint for SimpleCNN on CIFAR-10 or MNIST.
- Experiment History: SQLite-backed experiment listing endpoint.
- Attack Engine: FGSM and PGD attacks with artifact generation.
- Detection Engine: Statistical confidence-drop detector.
- Defense Engine: Gaussian noise + bit-depth reduction evaluation and adversarial training utility.
- Report Generator: JSON report with vulnerability explanation and recommendations.
- Frontend: React + Tailwind dashboard to run workflow end-to-end.
- Persistence: SQLite metadata + local filesystem artifacts.

## Architecture

Frontend (React + Tailwind)
-> Backend (FastAPI)
-> Core modules (Attack, Detection, Defense, Report, Agent)
-> ML stack (PyTorch + torchvision)
-> Storage (SQLite + artifacts folder)

## Backend Setup

```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API Endpoints

### Standard Endpoints

- `GET /api/health`
- `POST /api/train`
- `GET /api/train/checkpoints`
- `POST /api/attack/run`
- `POST /api/detect`
- `POST /api/defend`
- `POST /api/report`
- `GET /api/report/{run_id}`
- `GET /api/report/{run_id}/download?format=json|txt`
- `POST /api/agent/recommend`
- `GET /api/experiments`

### Async Job Submission Endpoints (v0.3.0+)

- `POST /api/jobs/train` – Submit training job
- `POST /api/jobs/attack` – Submit attack job
- `POST /api/jobs/defend` – Submit defense job
- `GET /api/jobs` – List recent jobs (query param: `limit=50`)
- `GET /api/jobs/{job_id}` – Poll job status and progress
- `GET /api/jobs/{job_id}/result` – Retrieve job result

### Model Registry Endpoints (v0.4.0+)

- `POST /api/models/upload` – Upload and register a PyTorch model
- `GET /api/models` – List registered models
- `GET /api/models/{model_id}` – Get model metadata
- `DELETE /api/models/{model_id}` – Delete a model

## Example Train Payload

```json
{
  "dataset": "cifar10",
  "epochs": 1,
  "batch_size": 64,
  "learning_rate": 0.001,
  "max_batches_per_epoch": 100
}
```

## Async Job Submission Examples (v0.3.0+)

### Submit a training job

```bash
curl -X POST http://localhost:8000/api/jobs/train \
  -H "Content-Type: application/json" \
  -d '{"dataset": "cifar10", "epochs": 2, "batch_size": 64, "learning_rate": 0.001}'
```

Response: `{"job_id": "job_abc123", "status": "pending"}`

### Poll job status

```bash
curl http://localhost:8000/api/jobs/job_abc123
```

Response: `{"job_id": "job_abc123", "job_type": "train", "status": "running", "progress": 45, "error": null}`

### Get job result (after completion)

```bash
curl http://localhost:8000/api/jobs/job_abc123/result
```

Response: `{"job_id": "job_abc123", "status": "completed", "result": {...}, "error": null}`

## Example Attack Payload

```json
{
  "attack_type": "pgd",
  "dataset": "cifar10",
  "model_name": "simple_cnn",
  "epsilon": 0.031372549,
  "alpha": 0.007843137,
  "steps": 10,
  "sample_limit": 64,
  "batch_size": 32,
  "checkpoint_path": "../artifacts/models/cifar10_simple_cnn_YYYYMMDDTHHMMSS.pt"
}
```

## Model Upload Examples (v0.4.0+)

### Upload a model

```bash
curl -X POST http://localhost:8000/api/models/upload \
  -F "file=@/path/to/model.pt" \
  -F "model_name=my_cifar10_resnet" \
  -F "architecture=resnet18" \
  -F "dataset=cifar10"
```

Response: `{"model_id": "abc123-uuid", "name": "my_cifar10_resnet", "architecture": "resnet18", "dataset": "cifar10", "uploaded_at": "2025-01-15T10:30:00"}`

### List models

```bash
curl http://localhost:8000/api/models
```

### Use uploaded model in attack

```json
{
  "attack_type": "pgd",
  "dataset": "cifar10",
  "model_id": "abc123-uuid",
  "epsilon": 0.031372549,
  "alpha": 0.007843137,
  "steps": 10,
  "sample_limit": 64,
  "batch_size": 32
}
```

## Typical Flow

1. Call `POST /api/train` to create a checkpoint.
2. Use returned `checkpoint_path` in `POST /api/attack/run`.
3. Call `POST /api/detect`, `POST /api/defend`, and `POST /api/report` using the `run_id`.
4. Call `GET /api/experiments` to retrieve recent run history.

## Git Workflow (Frequent Pushes)

```bash
git status
git add .
git commit -m "feat: <short summary>"
git push
```

Notes:

- Local artifacts under `artifacts/images`, `artifacts/reports`, and `artifacts/models` are git-ignored.
- `.env*`, local venv folders, and frontend build/cache folders are git-ignored.
- `.gitattributes` marks model/image files as binary for safer diffs.

Dashboard additions:

- Load available checkpoints directly from backend and select one for attack execution.
- Download generated reports as JSON or TXT from the UI.

## Notes

- CIFAR-10 is default; MNIST fallback is available.
- The model currently initializes from architecture defaults unless external weights are added.
- Artifacts are saved under `artifacts/images` and run metadata under `artifacts/reports`.
