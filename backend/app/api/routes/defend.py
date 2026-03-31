from fastapi import APIRouter, HTTPException

from app.api.schemas import DefendRequest, DefendResponse
from app.config import settings
from app.services.defense_service import evaluate_defense

router = APIRouter()


@router.post("/defend", response_model=DefendResponse)
def run_defense(payload: DefendRequest) -> DefendResponse:
    try:
        data = evaluate_defense(
            run_id=payload.run_id,
            dataset=payload.dataset,
            gaussian_sigma=payload.gaussian_sigma,
            bit_depth_bits=payload.bit_depth_bits,
            model_id=payload.model_id,
            device=settings.device,
        )
        return DefendResponse(**data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run_id not found") from exc
