# A3DL Architecture (MVP)

## System Diagram (Text)

Client Dashboard
-> API Gateway (FastAPI)
-> Job Queue System (async dispatch v0.3.0+)
-> Attack Service (FGSM/PGD)
-> Training Service (checkpoint generation)
-> Detection Service (confidence-drop statistical detector)
-> Defense Service (input transforms + adversarial training utility)
-> Report Service (JSON + summary)
-> Agent Controller (rule-based decisions)
-> Experiment Service (history listing from SQLite)

Storage layer
-> SQLite (`artifacts/a3dl.db`) for experiment metadata
-> SQLite job tracking (in-memory CRUD + persistence)
-> Filesystem (`artifacts/images`, `artifacts/reports`, `artifacts/results`) for images, reports, and job results

ML layer
-> PyTorch model registry
-> Dataset loader for CIFAR-10 and MNIST

## Module Responsibilities

- `backend/app/services/attack_service.py`: orchestrates adversarial generation and metric computation. **v0.3.0+**: supports `on_progress` callback. **v0.4.0+**: accepts `model_id` parameter.
- `backend/app/services/training_service.py`: trains SimpleCNN and saves checkpoints. **v0.3.0+**: supports `on_progress` callback.
- `backend/app/services/detection_service.py`: labels samples as clean/adversarial with probability. **v0.3.0+**: supports `on_progress` callback.
- `backend/app/services/defense_service.py`: evaluates simple defenses and computes robustness score. **v0.3.0+**: supports `on_progress` callback.
- `backend/app/services/report_service.py`: compiles attack/detection/defense into security report.
- `backend/app/services/agent_controller.py`: recommends attack and defense strategy.
- `backend/app/services/experiment_store.py`: persists and serves run metadata through SQLite.
- `backend/app/services/model_registry.py`: **v0.4.0+** manages uploaded model registration, loading, and listing.

## Model Registry System (v0.4.0+)

### Model Infrastructure

- `backend/app/services/model_registry.py`: Singleton registry managing model lifecycle. Thread-safe CRUD operations.
  - `register_model()`: validates and stores model file, creates SQLite entry.
  - `load_model()`: instantiates model architecture and loads weights from disk.
  - `get_model()`, `list_models()`, `delete_model()`: metadata operations.
  - File validation: verifies PyTorch state_dict format before registration.

- `backend/app/api/routes/models.py`: REST endpoints for model management.
  - Upload endpoint validates file extension (.pt, .pth), size (max 1GB), and PyTorch format.
  - Multipart form-data: file, model_name, architecture, dataset.

### Database Schema

```sql
CREATE TABLE models (
  id TEXT PRIMARY KEY,           -- UUID
  name TEXT NOT NULL UNIQUE,     -- user-friendly name
  path TEXT NOT NULL,            -- absolute path to .pt file
  architecture TEXT NOT NULL,    -- simple_cnn, resnet18, etc.
  dataset TEXT NOT NULL,         -- cifar10, mnist, etc.
  uploaded_at TEXT NOT NULL      -- ISO datetime
)
```

### API Surface

- `POST /api/models/upload`: upload and register a PyTorch model.
- `GET /api/models`: list registered models (pagination supported).
- `GET /api/models/{model_id}`: retrieve model metadata by ID.
- `DELETE /api/models/{model_id}`: delete model file and registry entry.

### Integration with Attack Pipeline

1. Client uploads model file via `POST /api/models/upload`, receives model_id.
2. Client calls `POST /api/attack/run` with `model_id` parameter (or legacy `checkpoint_path`).
3. Attack service calls `registry.load_model(model_id)` to instantiate and load weights.
4. Model used in FGSM/PGD attack pipeline as-is; rest of attack logic unchanged.
5. Backward compatible: `checkpoint_path` parameter still functional for legacy checkpoints.

### Storage

- Models stored in `artifacts/models/{uuid}.pt` (UUID collision-free).
- Metadata persisted in SQLite table `models`.
- Supported architectures: simple_cnn, resnet18 (extensible).
- Supported datasets: cifar10, mnist (guides num_classes instantiation).

## Job Queue System (v0.3.0+)

## Job Queue System (v0.3.0+)

### Job Infrastructure

- `backend/app/jobs/job_models.py`: Pydantic models for Job, JobStatus (pending/running/completed/failed), JobType (train/attack/defend).
- `backend/app/jobs/job_manager.py`: Singleton thread-safe registry managing job lifecycle. Stores in-memory + persists to `artifacts/results/{job_id}.job.json`.
- `backend/app/jobs/job_worker.py`: Async executors for train/attack/defend. Wraps sync services with `asyncio.to_thread.run_in_executor()` to avoid blocking event loop. Sends progress updates via callback.

### API Surface

- `POST /api/jobs/train`: submit training job, returns job_id + "pending" status (non-blocking).
- `POST /api/jobs/attack`: submit attack job (non-blocking).
- `POST /api/jobs/defend`: submit defense job (non-blocking).
- `GET /api/jobs/{job_id}`: poll job status and progress (0-100%).
- `GET /api/jobs/{job_id}/result`: retrieve job result (202 if running, result dict if done, 400 if failed).
- `GET /api/jobs`: list recent jobs with pagination (limit=50 default, max 500).

### Execution Flow

1. Client calls `POST /api/jobs/train` with payload.
2. Server creates Job(status=pending) and returns job_id immediately.
3. Server calls `asyncio.create_task()` to launch background executor.
4. Executor runs service in thread pool via `loop.run_in_executor()`.
5. Service calls `on_progress(percent)` callback throughout execution.
6. Progress updates stored in job.progress (0-100).
7. Job status => running while executing, => completed/failed on finish.
8. Result persisted to disk and available via GET endpoint.

## New API Surface

- `POST /api/train`: **v0.2.x** synchronous training endpoint (blocking).
- `GET /api/experiments`: list recent experiment runs.
- **v0.3.0+** Async Job endpoints (see Job Queue System above).
- **v0.4.0+** Model Registry endpoints (see Model Registry System above).

## Planned Next Enhancements

- Add pretrained model support (hub integration).
- Expand model registry to support more architectures (VGG, EfficientNet).
- Expand job queue to multi-worker task distribution (Celery/RQ).
- Frontend model management dashboard with upload UI.
- Real-time progress streaming via Server-Sent Events (SSE).
- Model versioning and rollback support.
