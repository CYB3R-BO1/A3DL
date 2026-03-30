import torch


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def confidence_scores(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf
