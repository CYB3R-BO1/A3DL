from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from app.jobs.job_models import Job, JobStatus, JobType


class JobManager:
    """Singleton thread-safe job registry with persistence"""

    _instance: Optional[JobManager] = None
    _creation_lock = threading.Lock()

    def __new__(cls) -> JobManager:
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.jobs: dict[str, Job] = {}
        self.jobs_lock = threading.Lock()
        self.results_dir = Path("../artifacts/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def create_job(self, job_type: JobType, params: dict) -> str:
        """Create new job and return job_id"""
        job_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            progress=0,
            result_path=None,
            error=None,
            created_at=now,
            updated_at=now,
        )

        with self.jobs_lock:
            self.jobs[job_id] = job

        self._persist_job(job)
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID from memory"""
        with self.jobs_lock:
            return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[Job]:
        """List all jobs sorted by created_at DESC"""
        with self.jobs_lock:
            jobs_list = list(self.jobs.values())

        jobs_list.sort(key=lambda j: j.created_at, reverse=True)
        return jobs_list[:limit]

    def update_status(
        self, job_id: str, status: JobStatus, progress: int, error: str | None = None
    ) -> bool:
        """Update job status and progress"""
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if not job:
                return False

            job.status = status
            job.progress = progress
            job.error = error
            job.updated_at = datetime.now(timezone.utc).isoformat()

        self._persist_job(job)
        return True

    def update_result(self, job_id: str, result_path: str | None, error: str | None = None) -> bool:
        """Mark job as completed/failed and store result path"""
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if not job:
                return False

            if error:
                job.status = JobStatus.FAILED
                job.error = error
                job.progress = 0
            else:
                job.status = JobStatus.COMPLETED
                job.result_path = result_path
                job.progress = 100

            job.updated_at = datetime.now(timezone.utc).isoformat()

        self._persist_job(job)
        return True

    def _persist_job(self, job: Job) -> None:
        """Persist job to JSON file"""
        job_file = self.results_dir / f"{job.job_id}.job.json"
        try:
            with open(job_file, "w") as f:
                json.dump(job.model_dump(), f)
        except Exception:
            pass


_job_manager_instance: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get singleton job manager instance"""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = JobManager()
    return _job_manager_instance
