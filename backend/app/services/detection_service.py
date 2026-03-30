from __future__ import annotations

import math

from app.storage.run_store import RunStore


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def detect_adversarial(run_id: str, confidence_drop_threshold: float = 0.15):
    store = RunStore()
    payload = store.load(run_id)
    results = []

    for sample in payload.get("samples", []):
        drop = sample["original"]["confidence"] - sample["adversarial"]["confidence"]
        score = _sigmoid((drop - confidence_drop_threshold) * 12.0)
        is_adv = score >= 0.5 or sample.get("attack_success", False)
        results.append(
            {
                "sample_index": sample["sample_index"],
                "detection_probability": float(score),
                "label": "adversarial" if is_adv else "clean",
            }
        )

    return {
        "run_id": run_id,
        "threshold": confidence_drop_threshold,
        "results": results,
    }
