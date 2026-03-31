"""
Model registry API endpoints for uploading and managing models.

Endpoints:
- POST /api/models/upload - Upload and register a model
- GET /api/models - List registered models  
- GET /api/models/{model_id} - Get model metadata
- DELETE /api/models/{model_id} - Delete a model (optional)
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.api.schemas import ModelListResponse, ModelMetadata, ModelUploadResponse
from app.services.model_registry import get_model_registry

router = APIRouter()

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pt", ".pth"}
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB


@router.post("/models/upload", response_model=ModelUploadResponse)
async def upload_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    architecture: str = Form(...),
    dataset: str = Form(...),
) -> ModelUploadResponse:
    """
    Upload and register a PyTorch model.

    Args:
        file: Model file (.pt or .pth)
        model_name: User-friendly name for the model
        architecture: Architecture type (simple_cnn, resnet18, vgg16)
        dataset: Dataset model was trained on (cifar10, mnist)

    Returns:
        ModelUploadResponse with model ID and metadata
    """
    # Validate file extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    tmp_path: str | None = None

    # Save to temp file first
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large (max 1GB)")
            tmp.write(content)
            tmp_path = tmp.name

        # Register model
        registry = get_model_registry()
        model_id = registry.register_model(
            file_path=tmp_path,
            model_name=model_name,
            architecture=architecture,
            dataset=dataset,
        )

        # Get metadata to return
        metadata = registry.get_model(model_id)
        if not metadata:
            raise HTTPException(status_code=500, detail="Failed to retrieve registered model")

        return ModelUploadResponse(
            model_id=model_id,
            name=metadata["name"],
            architecture=metadata["architecture"],
            dataset=metadata["dataset"],
            path=metadata["path"],
            uploaded_at=metadata["uploaded_at"],
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@router.get("/models", response_model=ModelListResponse)
def list_models(limit: int = Query(default=50, ge=1, le=500)) -> ModelListResponse:
    """List all registered models."""
    registry = get_model_registry()
    models_data = registry.list_models(limit=limit)
    models = [
        ModelMetadata(
            id=m["id"],
            name=m["name"],
            path=m["path"],
            architecture=m["architecture"],
            dataset=m["dataset"],
            uploaded_at=m["uploaded_at"],
        )
        for m in models_data
    ]
    return ModelListResponse(models=models)


@router.get("/models/{model_id}", response_model=ModelMetadata)
def get_model_metadata(model_id: str) -> ModelMetadata:
    """Get metadata for a specific model."""
    registry = get_model_registry()
    metadata = registry.get_model(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return ModelMetadata(
        id=metadata["id"],
        name=metadata["name"],
        path=metadata["path"],
        architecture=metadata["architecture"],
        dataset=metadata["dataset"],
        uploaded_at=metadata["uploaded_at"],
    )


@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    """Delete a registered model."""
    registry = get_model_registry()
    if not registry.delete_model(model_id):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return {"status": "deleted", "model_id": model_id}
