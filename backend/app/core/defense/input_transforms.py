import torch


def gaussian_noise(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    noisy = x + torch.randn_like(x) * sigma
    return torch.clamp(noisy, 0.0, 1.0)


def bit_depth_reduction(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    levels = float(2**bits - 1)
    return torch.round(x * levels) / levels
