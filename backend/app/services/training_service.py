from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.core.models.model_registry import get_model


def _build_train_loader(dataset: str, batch_size: int) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "mnist":
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    else:
        ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_and_save_checkpoint(
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_batches_per_epoch: int,
    device: str,
) -> dict:
    model = get_model(dataset=dataset, device=device)
    loader = _build_train_loader(dataset=dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    epoch_losses: list[float] = []

    for _ in range(epochs):
        total_loss = 0.0
        seen_batches = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            seen_batches += 1
            if seen_batches >= max_batches_per_epoch:
                break

        epoch_losses.append(total_loss / max(seen_batches, 1))

    out_dir = Path("../artifacts/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    checkpoint_path = out_dir / f"{dataset}_simple_cnn_{checkpoint_id}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "dataset": dataset,
        "model_name": "simple_cnn",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epoch_losses": epoch_losses,
    }
