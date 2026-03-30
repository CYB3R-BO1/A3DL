from dataclasses import dataclass


@dataclass(slots=True)
class AttackConfig:
    epsilon: float
    alpha: float = 2.0 / 255.0
    steps: int = 10
