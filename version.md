# A3DL Version History

## v0.4.0 – Model Upload & Registry

- **Model Registry**: Upload and register PyTorch models (.pt, .pth).
- **Upload Endpoint**: `POST /api/models/upload` with multipart form-data (file, model_name, architecture, dataset).
- **List Models**: `GET /api/models` to view registered models.
- **Model Details**: `GET /api/models/{model_id}` for metadata.
- **Delete Models**: `DELETE /api/models/{model_id}` to remove uploaded models.
- **Attack Integration**: Pass `model_id` to attack endpoints instead of `checkpoint_path`.
- **Database**: Models table in SQLite with id, name, path, architecture, dataset, uploaded_at.
- **Storage**: Models saved to `artifacts/models/{uuid}.pt`.
- **Backward Compatible**: Existing `checkpoint_path` parameter still works.

## v0.3.0 – Async Job Execution

- **Async Job Queue**: `POST /api/jobs/{train|attack|defend}` for non-blocking jobs.
- **Progress Tracking**: Real-time progress (0-100%) via `GET /api/jobs/{job_id}`.
- **Job Results**: `GET /api/jobs/{job_id}/result` to retrieve completed job outputs.
- **Job Persistence**: Results stored as JSON in `artifacts/results/`.
- **Service Callbacks**: Train/attack/defend services support `on_progress` callbacks.
- **Backward Compatible**: Synchronous endpoints unchanged.

## v0.2.x (Previous)
