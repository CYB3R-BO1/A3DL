from fastapi import APIRouter, HTTPException

from app.api.schemas import ReportRequest, ReportResponse
from app.services.defense_service import evaluate_defense
from app.services.detection_service import detect_adversarial
from app.services.report_service import generate_report

router = APIRouter()


@router.post("/report", response_model=ReportResponse)
def create_report(payload: ReportRequest) -> ReportResponse:
    try:
        detection = detect_adversarial(payload.run_id) if payload.include_detection else None
        defense = evaluate_defense(payload.run_id, dataset="cifar10") if payload.include_defense else None
        report = generate_report(payload.run_id, defense_payload=defense, detection_payload=detection)
        return ReportResponse(**report)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run_id not found") from exc
