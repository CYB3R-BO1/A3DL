import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.core.attacks import AttackConfig, fgsm_attack


def adversarial_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epsilon: float,
    device: str,
) -> float:
    model.train()
    config = AttackConfig(epsilon=epsilon)
    total_loss = 0.0
    total_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        adv = fgsm_attack(model, images, labels, config)
        mixed_x = torch.cat([images, adv], dim=0)
        mixed_y = torch.cat([labels, labels], dim=0)

        optimizer.zero_grad()
        logits = model(mixed_x)
        loss = F.cross_entropy(logits, mixed_y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    if total_batches == 0:
        return 0.0
    return total_loss / total_batches
