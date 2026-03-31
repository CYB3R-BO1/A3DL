from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import torch

from app.api.schemas import AttackMetrics, AttackResponse, AttackSampleResult, PredictionInfo
from app.core.attacks import AttackConfig, fgsm_attack, pgd_attack
from app.core.data.dataset_loader import get_loader
from app.core.models.model_registry import get_model, maybe_load_weights
from app.core.utils.metrics import confidence_scores
from app.services.experiment_store import ExperimentStore
from app.services.model_registry import get_model_registry
from app.storage.artifact_store import ArtifactStore
from app.storage.run_store import RunStore


def _predict(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(x)
    pred, conf = confidence_scores(logits)
    return pred, conf


def run_attack(
    attack_type: str,
    dataset: str,
    epsilon: float,
    alpha: float,
    steps: int,
    sample_limit: int,
    batch_size: int,
    checkpoint_path: str | None = None,
    model_id: str | None = None,
    device: str = "cpu",
    on_progress: callable = None,
) -> AttackResponse:
    """
    Execute an adversarial attack.
    
    Args:
        model_id: UUID of uploaded model (takes precedence over checkpoint_path)
        checkpoint_path: Path to model weights (legacy support)
    """
    # Load model: prefer uploaded model_id, fall back to checkpoint_path
    if model_id:
        registry = get_model_registry()
        model = registry.load_model(model_id, device=device)
    else:
        model = get_model(dataset=dataset, device=device)
        model = maybe_load_weights(model=model, weights_path=checkpoint_path, device=device)
    loader, sample_indices = get_loader(dataset=dataset, sample_limit=sample_limit, batch_size=batch_size)
    attack_config = AttackConfig(epsilon=epsilon, alpha=alpha, steps=steps)
    image_store = ArtifactStore()
    run_store = RunStore()
    experiment_store = ExperimentStore()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    samples: list[AttackSampleResult] = []

    clean_correct = 0
    adv_correct = 0
    successes = 0
    confidence_drop = 0.0

    model.eval()
    idx_offset = 0
    num_batches = len(loader)
    
    if on_progress:
        on_progress(10)
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_pred, clean_conf = _predict(model, images)

        if attack_type == "fgsm":
            adv = fgsm_attack(model, images, labels, attack_config)
        elif attack_type == "pgd":
            adv = pgd_attack(model, images, labels, attack_config)
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")

        with torch.no_grad():
            adv_pred, adv_conf = _predict(model, adv)

        clean_correct += int((clean_pred == labels).sum().item())
        adv_correct += int((adv_pred == labels).sum().item())

        for i in range(images.size(0)):
            global_idx = sample_indices[idx_offset + i]
            is_success = int(adv_pred[i].item() != labels[i].item())
            successes += is_success
            confidence_drop += max(0.0, float(clean_conf[i].item() - adv_conf[i].item()))

            original_np = images[i].detach().cpu().numpy()
            adv_np = adv[i].detach().cpu().numpy()
            perturb_np = ((adv_np - original_np) + epsilon) / max(2 * epsilon, 1e-8)

            orig_path = image_store.save_image(run_id, f"sample_{global_idx}_original", original_np)
            adv_path = image_store.save_image(run_id, f"sample_{global_idx}_adversarial", adv_np)
            pert_path = image_store.save_image(run_id, f"sample_{global_idx}_perturbation", perturb_np)

            samples.append(
                AttackSampleResult(
                    sample_index=global_idx,
                    true_label=int(labels[i].item()),
                    original=PredictionInfo(label=int(clean_pred[i].item()), confidence=float(clean_conf[i].item())),
                    adversarial=PredictionInfo(label=int(adv_pred[i].item()), confidence=float(adv_conf[i].item())),
                    attack_success=bool(is_success),
                    original_image_path=orig_path,
                    adversarial_image_path=adv_path,
                    perturbation_image_path=pert_path,
                )
            )

        idx_offset += images.size(0)
        
        if on_progress:
            progress = int(10 + (90 * batch_idx / max(num_batches, 1)))
            on_progress(progress)

    total = len(sample_indices)
    metrics = AttackMetrics(
        clean_accuracy=clean_correct / total if total else 0.0,
        adversarial_accuracy=adv_correct / total if total else 0.0,
        attack_success_rate=successes / total if total else 0.0,
        avg_confidence_drop=confidence_drop / total if total else 0.0,
    )

    response = AttackResponse(
        run_id=run_id,
        attack_type=attack_type,
        dataset=dataset,
        model_name="simple_cnn",
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        metrics=metrics,
        samples=samples,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    run_store.save(run_id, response.model_dump())
    experiment_store.upsert_experiment(
        (
            run_id,
            attack_type,
            dataset,
            "simple_cnn",
            epsilon,
            alpha,
            steps,
            metrics.clean_accuracy,
            metrics.adversarial_accuracy,
            metrics.attack_success_rate,
            created_at,
        )
    )

    return response
