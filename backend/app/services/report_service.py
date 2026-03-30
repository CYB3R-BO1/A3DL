from __future__ import annotations

from datetime import datetime, timezone

from app.services.agent_controller import recommend_attack_and_defense
from app.storage.run_store import RunStore


def generate_report(run_id: str, defense_payload: dict | None = None, detection_payload: dict | None = None):
    store = RunStore()
    attack = store.load(run_id)

    metrics = attack.get("metrics", {})
    success_rate = float(metrics.get("attack_success_rate", 0.0))
    recommendation = recommend_attack_and_defense(success_rate)

    vulnerability = (
        "Model predictions are sensitive to bounded perturbations, indicating adversarial fragility "
        "in gradient-aligned directions."
    )

    report = {
        "report_id": f"report_{run_id}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_details": {
            "model_name": attack.get("model_name"),
            "dataset": attack.get("dataset"),
        },
        "attack": {
            "type": attack.get("attack_type"),
            "parameters": {
                "epsilon": attack.get("epsilon"),
                "alpha": attack.get("alpha"),
                "steps": attack.get("steps"),
            },
            "metrics": metrics,
        },
        "detection": detection_payload,
        "defense": defense_payload,
        "vulnerability_explanation": vulnerability,
        "agent_recommendation": recommendation,
        "summary": (
            f"Attack success rate: {success_rate:.2%}. "
            f"Recommended defense: {recommendation['recommended_defense']}."
        ),
    }

    store.save(f"report_{run_id}", report)
    return report
