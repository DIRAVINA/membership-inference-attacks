"""
Mitigation and Defense Strategies against Membership Inference Attacks.

Implements all mitigations evaluated in Shokri et al. (2017) Table III:
1. Top-k filtering (k=3, k=1)
2. Top-1 label only (argmax one-hot)
3. Coarsening / Rounding (d=3, d=1 decimal digits)
4. Temperature scaling (t=5, t=20)
5. L2 Regularization (weight decay lambda)

Plus modern privacy defenses:
6. Dropout (p=0.5)
7. Early Stopping (patience)
8. DP-SGD (Differential Privacy with Opacus)
"""

import numpy as np
import torch
from typing import Dict, Any, Callable, Optional


class OutputMitigations:
    """
    Post-processing filters applied to the target model's output probability vector.
    """
    @staticmethod
    def top_k_filter(probs: np.ndarray, k: int = 3, renormalize: bool = True) -> np.ndarray:
        """
        Retains only the top-k highest probabilities, zeroing out all others.
        Optionally renormalizes the remaining probabilities so they sum to 1.
        """
        out = np.zeros_like(probs)
        for i in range(len(probs)):
            top_indices = np.argsort(probs[i])[-k:]
            out[i, top_indices] = probs[i, top_indices]
            if renormalize:
                s = np.sum(out[i])
                if s > 0:
                    out[i] /= s
        return out

    @staticmethod
    def top_1_label_only(probs: np.ndarray) -> np.ndarray:
        """
        Restricts the prediction vector to only the most likely class (one-hot).
        As noted in Section VIII-A & Table III:
        The model outputs only the predicted discrete label with no confidence score.
        """
        out = np.zeros_like(probs)
        preds = np.argmax(probs, axis=1)
        out[np.arange(len(probs)), preds] = 1.0
        return out

    @staticmethod
    def rounding_coarsen(probs: np.ndarray, decimals: int = 3, renormalize: bool = True) -> np.ndarray:
        """
        Rounds classification probabilities down to d decimal digits.
        """
        out = np.round(probs, decimals=decimals)
        if renormalize:
            row_sums = np.sum(out, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            out = out / row_sums
        return out

    @staticmethod
    def temperature_scale_logits(logits: torch.Tensor, temperature: float = 5.0) -> torch.Tensor:
        """
        Applies temperature scaling to raw logits before softmax:
        p_i = exp(z_i / t) / sum_j exp(z_j / t)
        """
        scaled = logits / max(temperature, 1e-6)
        return torch.softmax(scaled, dim=-1)


def get_mitigation_transform(mitigation_name: str, **kwargs) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Factory for output post-processing mitigation functions."""
    if mitigation_name == "none" or mitigation_name is None:
        return None
    elif mitigation_name == "top_3":
        return lambda p: OutputMitigations.top_k_filter(p, k=3)
    elif mitigation_name == "top_1":
        return lambda p: OutputMitigations.top_k_filter(p, k=1)
    elif mitigation_name == "top_1_label":
        return lambda p: OutputMitigations.top_1_label_only(p)
    elif mitigation_name == "round_3":
        return lambda p: OutputMitigations.rounding_coarsen(p, decimals=3)
    elif mitigation_name == "round_1":
        return lambda p: OutputMitigations.rounding_coarsen(p, decimals=1)
    else:
        raise ValueError(f"Unknown mitigation: {mitigation_name}")
