# A3DL Architecture (MVP)

## System Diagram (Text)

Client Dashboard
-> API Gateway (FastAPI)
-> Attack Service (FGSM/PGD)
-> Training Service (checkpoint generation)
-> Detection Service (confidence-drop statistical detector)
-> Defense Service (input transforms + adversarial training utility)
-> Report Service (JSON + summary)
-> Agent Controller (rule-based decisions)
-> Experiment Service (history listing from SQLite)

Storage layer
-> SQLite (`artifacts/a3dl.db`) for experiment metadata
-> Filesystem (`artifacts/images`, `artifacts/reports`) for images and run payloads

ML layer
-> PyTorch model registry
-> Dataset loader for CIFAR-10 and MNIST

## Module Responsibilities

- `backend/app/services/attack_service.py`: orchestrates adversarial generation and metric computation.
- `backend/app/services/training_service.py`: trains SimpleCNN and saves checkpoints.
- `backend/app/services/detection_service.py`: labels samples as clean/adversarial with probability.
- `backend/app/services/defense_service.py`: evaluates simple defenses and computes robustness score.
- `backend/app/services/report_service.py`: compiles attack/detection/defense into security report.
- `backend/app/services/agent_controller.py`: recommends attack and defense strategy.
- `backend/app/services/experiment_store.py`: persists and serves run metadata through SQLite.

## New API Surface

- `POST /api/train`: train a model checkpoint.
- `GET /api/experiments`: list recent experiment runs.

## Planned Next Enhancements

- Add pretrained model support and model upload endpoint.
- Add asynchronous jobs for long-running adversarial training.
- Add richer visualizations (true image rendering in UI instead of path list).
