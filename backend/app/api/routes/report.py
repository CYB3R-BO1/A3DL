from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse

from app.api.schemas import ReportRequest, ReportResponse
from app.services.defense_service import evaluate_defense
from app.services.detection_service import detect_adversarial
from app.services.report_service import generate_report, render_report_text
from app.storage.run_store import RunStore

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


@router.get("/report/{run_id}", response_model=ReportResponse)
def get_report(run_id: str) -> ReportResponse:
    store = RunStore()
    report_key = f"report_{run_id}"

    try:
        report = store.load(report_key)
    except FileNotFoundError:
        try:
            detection = detect_adversarial(run_id)
            defense = evaluate_defense(run_id, dataset="cifar10")
            report = generate_report(run_id, defense_payload=defense, detection_payload=detection)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run_id not found") from exc

    return ReportResponse(**report)


@router.get("/report/{run_id}/download")
def download_report(run_id: str, format: Literal["json", "txt"] = Query(default="json")):
    store = RunStore()
    report_key = f"report_{run_id}"

    try:
        report = store.load(report_key)
    except FileNotFoundError:
        try:
            detection = detect_adversarial(run_id)
            defense = evaluate_defense(run_id, dataset="cifar10")
            report = generate_report(run_id, defense_payload=defense, detection_payload=detection)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run_id not found") from exc

    if format == "json":
        report_path = Path("../artifacts/reports") / f"{report_key}.json"
        if not report_path.exists():
            store.save(report_key, report)
        return FileResponse(
            path=str(report_path),
            filename=f"report_{run_id}.json",
            media_type="application/json",
        )

    text_report = render_report_text(report)
    return PlainTextResponse(
        content=text_report,
        headers={"Content-Disposition": f"attachment; filename=report_{run_id}.txt"},
    )
