from fastapi import APIRouter, Query

from app.api.schemas import ExperimentListResponse, ExperimentSummary
from app.services.experiment_store import ExperimentStore

router = APIRouter()


@router.get("/experiments", response_model=ExperimentListResponse)
def list_experiments(limit: int = Query(default=50, ge=1, le=500)) -> ExperimentListResponse:
    store = ExperimentStore()
    rows = store.list_experiment_dicts(limit=limit)
    return ExperimentListResponse(experiments=[ExperimentSummary(**row) for row in rows])
