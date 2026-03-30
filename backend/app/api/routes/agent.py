from fastapi import APIRouter

from app.api.schemas import RecommendationRequest, RecommendationResponse
from app.services.agent_controller import recommend_attack_and_defense

router = APIRouter()


@router.post("/agent/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest) -> RecommendationResponse:
    return RecommendationResponse(**recommend_attack_and_defense(payload.attack_success_rate))
