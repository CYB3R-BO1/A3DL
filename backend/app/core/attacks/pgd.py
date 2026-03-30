import torch
import torch.nn.functional as F

from app.core.attacks.base import AttackConfig


def pgd_attack(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, config: AttackConfig) -> torch.Tensor:
    model.eval()
    base = images.detach()
    adv = base + torch.empty_like(base).uniform_(-config.epsilon, config.epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)

    for _ in range(config.steps):
        adv.requires_grad_(True)
        logits = model(adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, adv)[0]

        adv = adv + config.alpha * grad.sign()
        delta = torch.clamp(adv - base, min=-config.epsilon, max=config.epsilon)
        adv = torch.clamp(base + delta, 0.0, 1.0).detach()

    return adv
