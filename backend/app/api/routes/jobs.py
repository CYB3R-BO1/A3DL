from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import JobResponse, JobResultWrapper, JobStatusResponse
from app.config import settings
from app.jobs.job_manager import get_job_manager
from app.jobs.job_models import JobStatus, JobType
from app.jobs.job_worker import (
    _execute_attack_job_async,
    _execute_defend_job_async,
    _execute_train_job_async,
)

router = APIRouter()


@router.post("/jobs/train", response_model=JobResponse)
def start_train_job(payload: dict) -> JobResponse:
    """Start a training job asynchronously"""
    job_manager = get_job_manager()
    job_id = job_manager.create_job(JobType.TRAIN, payload)
    asyncio.create_task(_execute_train_job_async(job_id, payload, settings.device))
    return JobResponse(job_id=job_id, status=JobStatus.PENDING)


@router.post("/jobs/attack", response_model=JobResponse)
def start_attack_job(payload: dict) -> JobResponse:
    """Start an attack job asynchronously"""
    job_manager = get_job_manager()
    job_id = job_manager.create_job(JobType.ATTACK, payload)
    asyncio.create_task(_execute_attack_job_async(job_id, payload, settings.device))
    return JobResponse(job_id=job_id, status=JobStatus.PENDING)


@router.post("/jobs/defend", response_model=JobResponse)
def start_defend_job(payload: dict) -> JobResponse:
    """Start a defense job asynchronously"""
    job_manager = get_job_manager()
    job_id = job_manager.create_job(JobType.DEFEND, payload)
    asyncio.create_task(_execute_defend_job_async(job_id, payload, settings.device))
    return JobResponse(job_id=job_id, status=JobStatus.PENDING)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll for job status and progress"""
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        error=job.error,
    )


@router.get("/jobs/{job_id}/result", response_model=JobResultWrapper)
def get_job_result(job_id: str):
    """Get job result (or 202 if still running)"""
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status == JobStatus.RUNNING or job.status == JobStatus.PENDING:
        return {"code": 202, "message": "Job still processing"}
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=400, detail=f"Job failed: {job.error}")
    if job.result_path:
        try:
            result_file = Path(job.result_path)
            with open(result_file, "r") as f:
                result_data = json.load(f)
            return JobResultWrapper(
                job_id=job_id,
                status=job.status,
                result=result_data,
                error=None,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Result file not found")
    raise HTTPException(status_code=500, detail="Job completed but no result stored")


@router.get("/jobs", response_model=list[JobStatusResponse])
def list_jobs_endpoint(limit: int = Query(default=50, ge=1, le=500)) -> list[JobStatusResponse]:
    """List recent jobs"""
    job_manager = get_job_manager()
    jobs = job_manager.list_jobs(limit=limit)
    return [
        JobStatusResponse(
            job_id=job.job_id,
            job_type=job.job_type,
            status=job.status,
            progress=job.progress,
            error=job.error,
        )
        for job in jobs
    ]
