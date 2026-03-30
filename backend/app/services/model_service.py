from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def list_checkpoints(limit: int = 100) -> list[dict]:
    models_dir = Path("../artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    checkpoints: list[dict] = []
    for checkpoint_path in files[:limit]:
        stem = checkpoint_path.stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue

        dataset = parts[0]
        model_name = parts[1]
        checkpoint_id = parts[-1]

        stat = checkpoint_path.stat()
        created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

        checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "path": str(checkpoint_path),
                "dataset": dataset,
                "model_name": model_name,
                "created_at": created_at,
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
            }
        )

    return checkpoints
