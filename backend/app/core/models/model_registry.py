from pathlib import Path

import torch

from app.core.models.simple_cnn import SimpleCNN


def get_model(dataset: str, device: str) -> torch.nn.Module:
    if dataset == "mnist":
        model = SimpleCNN(in_channels=1, num_classes=10)
    else:
        model = SimpleCNN(in_channels=3, num_classes=10)
    return model.to(device)


def maybe_load_weights(model: torch.nn.Module, weights_path: str | None, device: str) -> torch.nn.Module:
    if not weights_path:
        return model
    path = Path(weights_path)
    if not path.exists():
        return model
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model
