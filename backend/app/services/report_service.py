from __future__ import annotations

from datetime import datetime, timezone

from app.services.agent_controller import recommend_attack_and_defense
from app.storage.run_store import RunStore


def render_report_text(report: dict) -> str:
    model = report.get("model_details", {})
    attack = report.get("attack", {})
    params = attack.get("parameters", {})
    metrics = attack.get("metrics", {})
    recommendation = report.get("agent_recommendation", {})

    lines = [
        "A3DL Security Report",
        "",
        f"Report ID: {report.get('report_id', 'unknown')}",
        f"Generated At: {report.get('generated_at', 'unknown')}",
        "",
        "[Model Details]",
        f"Model: {model.get('model_name', 'unknown')}",
        f"Dataset: {model.get('dataset', 'unknown')}",
        "",
        "[Attack Configuration]",
        f"Type: {attack.get('type', 'unknown')}",
        f"Epsilon: {params.get('epsilon', 'unknown')}",
        f"Alpha: {params.get('alpha', 'unknown')}",
        f"Steps: {params.get('steps', 'unknown')}",
        "",
        "[Attack Metrics]",
        f"Clean Accuracy: {metrics.get('clean_accuracy', 0.0)}",
        f"Adversarial Accuracy: {metrics.get('adversarial_accuracy', 0.0)}",
        f"Attack Success Rate: {metrics.get('attack_success_rate', 0.0)}",
        f"Average Confidence Drop: {metrics.get('avg_confidence_drop', 0.0)}",
        "",
        "[Vulnerability Explanation]",
        str(report.get("vulnerability_explanation", "n/a")),
        "",
        "[Recommended Action]",
        f"Attack: {recommendation.get('recommended_attack', 'n/a')}",
        f"Epsilon: {recommendation.get('recommended_epsilon', 'n/a')}",
        f"Defense: {recommendation.get('recommended_defense', 'n/a')}",
        f"Reason: {recommendation.get('reason', 'n/a')}",
        "",
        "[Summary]",
        str(report.get("summary", "n/a")),
        "",
    ]

    return "\n".join(lines)


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
