from fastapi import APIRouter, HTTPException

from app.api.schemas import DetectRequest, DetectResponse
from app.services.detection_service import detect_adversarial

router = APIRouter()


@router.post("/detect", response_model=DetectResponse)
def run_detection(payload: DetectRequest) -> DetectResponse:
    try:
        data = detect_adversarial(
            run_id=payload.run_id,
            confidence_drop_threshold=payload.confidence_drop_threshold,
        )
        return DetectResponse(**data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run_id not found") from exc
