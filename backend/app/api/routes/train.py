from fastapi import APIRouter

from app.api.schemas import TrainRequest, TrainResponse
from app.config import settings
from app.services.training_service import train_and_save_checkpoint

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
def train_model(payload: TrainRequest) -> TrainResponse:
    result = train_and_save_checkpoint(
        dataset=payload.dataset,
        epochs=payload.epochs,
        batch_size=payload.batch_size,
        learning_rate=payload.learning_rate,
        max_batches_per_epoch=payload.max_batches_per_epoch,
        device=settings.device,
    )
    return TrainResponse(**result)
