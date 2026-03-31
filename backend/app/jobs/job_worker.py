from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from app.jobs.job_manager import get_job_manager
from app.jobs.job_models import JobStatus, JobType
from app.services.attack_service import run_attack
from app.services.defense_service import evaluate_defense
from app.services.training_service import train_and_save_checkpoint


async def _execute_train_job_async(
    job_id: str,
    params: dict,
    device: str,
) -> None:
    """Execute training job asynchronously"""
    job_manager = get_job_manager()

    def on_progress(progress: int) -> None:
        job_manager.update_status(job_id, JobStatus.RUNNING, progress)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _sync_train,
            params,
            device,
            on_progress,
        )

        result_dir = Path("../artifacts/results")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{job_id}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        job_manager.update_result(job_id, str(result_file), error=None)

    except Exception as e:
        job_manager.update_result(job_id, None, error=str(e))


async def _execute_attack_job_async(
    job_id: str,
    params: dict,
    device: str,
) -> None:
    """Execute attack job asynchronously"""
    job_manager = get_job_manager()

    def on_progress(progress: int) -> None:
        job_manager.update_status(job_id, JobStatus.RUNNING, progress)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _sync_attack,
            params,
            device,
            on_progress,
        )

        result_dir = Path("../artifacts/results")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{job_id}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        job_manager.update_result(job_id, str(result_file), error=None)

    except Exception as e:
        job_manager.update_result(job_id, None, error=str(e))


async def _execute_defend_job_async(
    job_id: str,
    params: dict,
    device: str,
) -> None:
    """Execute defense job asynchronously"""
    job_manager = get_job_manager()

    def on_progress(progress: int) -> None:
        job_manager.update_status(job_id, JobStatus.RUNNING, progress)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _sync_defend,
            params,
            device,
            on_progress,
        )

        result_dir = Path("../artifacts/results")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{job_id}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        job_manager.update_result(job_id, str(result_file), error=None)

    except Exception as e:
        job_manager.update_result(job_id, None, error=str(e))


def _sync_train(
    params: dict,
    device: str,
    on_progress: Callable[[int], None],
) -> dict[str, Any]:
    """Sync wrapper for training service"""
    return train_and_save_checkpoint(
        dataset=params.get("dataset", "cifar10"),
        epochs=params.get("epochs", 5),
        batch_size=params.get("batch_size", 32),
        learning_rate=params.get("learning_rate", 0.001),
        max_batches_per_epoch=params.get("max_batches_per_epoch", 100),
        device=device,
        on_progress=on_progress,
    )


def _sync_attack(
    params: dict,
    device: str,
    on_progress: Callable[[int], None],
) -> dict[str, Any]:
    """Sync wrapper for attack service"""
    result = run_attack(
        attack_type=params.get("attack_type", "fgsm"),
        dataset=params.get("dataset", "cifar10"),
        epsilon=params.get("epsilon", 8.0 / 255.0),
        alpha=params.get("alpha", 2.0 / 255.0),
        steps=params.get("steps", 10),
        sample_limit=params.get("sample_limit", 64),
        batch_size=params.get("batch_size", 32),
        checkpoint_path=params.get("checkpoint_path"),
        device=device,
        on_progress=on_progress,
    )
    return result.model_dump()


def _sync_defend(
    params: dict,
    device: str,
    on_progress: Callable[[int], None],
) -> dict[str, Any]:
    """Sync wrapper for defense service"""
    return evaluate_defense(
        run_id=params.get("run_id"),
        dataset=params.get("dataset", "cifar10"),
        gaussian_sigma=params.get("gaussian_sigma", 0.03),
        bit_depth_bits=params.get("bit_depth_bits", 4),
        device=device,
        on_progress=on_progress,
    )
