import torch
import torch.nn.functional as F

from app.core.attacks.base import AttackConfig


def fgsm_attack(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, config: AttackConfig) -> torch.Tensor:
    model.eval()
    x = images.detach().clone().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, x)[0]
    adv = x + config.epsilon * grad.sign()
    adv = torch.clamp(adv, 0.0, 1.0)
    return adv.detach()
