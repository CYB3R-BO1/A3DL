from fastapi import APIRouter, HTTPException

from app.api.schemas import AttackRequest, AttackResponse
from app.config import settings
from app.services.attack_service import run_attack

router = APIRouter()


@router.post("/attack/run", response_model=AttackResponse)
def execute_attack(payload: AttackRequest) -> AttackResponse:
    try:
        return run_attack(
            attack_type=payload.attack_type,
            dataset=payload.dataset,
            epsilon=payload.epsilon,
            alpha=payload.alpha,
            steps=payload.steps,
            sample_limit=payload.sample_limit,
            batch_size=payload.batch_size,
            checkpoint_path=payload.checkpoint_path,
            model_id=payload.model_id,
            device=settings.device,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
