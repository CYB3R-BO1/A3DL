from __future__ import annotations


def recommend_attack_and_defense(attack_success_rate: float) -> dict[str, str | float]:
    if attack_success_rate >= 0.6:
        return {
            "recommended_attack": "pgd",
            "recommended_epsilon": 8.0 / 255.0,
            "recommended_defense": "adversarial_training + gaussian_noise + bit_depth",
            "reason": "High attack success rate indicates weak robustness; stronger iterative attack and training defense advised.",
        }
    if attack_success_rate >= 0.3:
        return {
            "recommended_attack": "pgd",
            "recommended_epsilon": 4.0 / 255.0,
            "recommended_defense": "gaussian_noise + bit_depth",
            "reason": "Moderate vulnerability detected; apply lightweight preprocessing defenses and retest.",
        }
    return {
        "recommended_attack": "fgsm",
        "recommended_epsilon": 2.0 / 255.0,
        "recommended_defense": "monitor only",
        "reason": "Current robustness is acceptable in this test band; continue periodic red-team scans.",
    }
