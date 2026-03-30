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

- `GET /api/health`
- `POST /api/train`
- `POST /api/attack/run`
- `POST /api/detect`
- `POST /api/defend`
- `POST /api/report`
- `POST /api/agent/recommend`
- `GET /api/experiments`

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

## Notes

- CIFAR-10 is default; MNIST fallback is available.
- The model currently initializes from architecture defaults unless external weights are added.
- Artifacts are saved under `artifacts/images` and run metadata under `artifacts/reports`.
