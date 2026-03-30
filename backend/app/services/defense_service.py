from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from app.core.defense.input_transforms import bit_depth_reduction, gaussian_noise
from app.core.models.model_registry import get_model
from app.storage.run_store import RunStore


def _load_image(path: str, grayscale: bool = False) -> torch.Tensor:
    image = Image.open(path)
    if grayscale:
        image = image.convert("L")
        arr = np.array(image, dtype=np.float32)[None, :, :] / 255.0
    else:
        image = image.convert("RGB")
        arr = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


def _predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x).argmax(dim=1)


def evaluate_defense(run_id: str, dataset: str, gaussian_sigma: float = 0.03, bit_depth_bits: int = 4, device: str = "cpu"):
    store = RunStore()
    payload = store.load(run_id)
    model = get_model(dataset=dataset, device=device)
    is_mnist = dataset == "mnist"

    total = 0
    adv_correct = 0
    defended_correct = 0

    for sample in payload.get("samples", []):
        label = int(sample["true_label"])
        adv_x = _load_image(sample["adversarial_image_path"], grayscale=is_mnist).to(device)
        pred_adv = int(_predict(model, adv_x).item())
        adv_correct += int(pred_adv == label)

        defended = gaussian_noise(adv_x, sigma=gaussian_sigma)
        defended = bit_depth_reduction(defended, bits=bit_depth_bits)
        pred_def = int(_predict(model, defended).item())
        defended_correct += int(pred_def == label)
        total += 1

    adversarial_accuracy = adv_correct / total if total else 0.0
    defended_accuracy = defended_correct / total if total else 0.0
    robustness_score = max(0.0, defended_accuracy - adversarial_accuracy)

    return {
        "run_id": run_id,
        "dataset": dataset,
        "gaussian_sigma": gaussian_sigma,
        "bit_depth_bits": bit_depth_bits,
        "adversarial_accuracy": adversarial_accuracy,
        "defended_accuracy": defended_accuracy,
        "robustness_score": robustness_score,
    }
