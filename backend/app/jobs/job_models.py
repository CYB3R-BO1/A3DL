from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    TRAIN = "train"
    ATTACK = "attack"
    DEFEND = "defend"


class Job(BaseModel):
    """Job model for internal storage"""
    job_id: str
    job_type: JobType
    status: JobStatus
    progress: int = Field(default=0, ge=0, le=100)
    result_path: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class JobResponse(BaseModel):
    """Response when job is created"""
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    """Response for status polling"""
    job_id: str
    job_type: JobType
    status: JobStatus
    progress: int
    error: str | None = None


class JobResultWrapper(BaseModel):
    """Response for result retrieval"""
    job_id: str
    status: JobStatus
    result: dict[str, Any] | None = None
    error: str | None = None
