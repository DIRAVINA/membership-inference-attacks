"""
Evaluation metrics for Membership Inference Attacks.

Supports:
- Standard metrics: Precision, Recall, Accuracy, F1 Score
- Modern MI metrics: ROC-AUC, TPR at low FPR (e.g., TPR @ 1% FPR, TPR @ 0.1% FPR)
- Normalized prediction entropy: -1 / log(C) * sum(p_i * log(p_i))
- Distribution extraction for Figure 11 (per-class precision vs train-test gap)
- Distribution extraction for Figure 12 (confidence and entropy histograms for members vs non-members)
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, accuracy_score, f1_score
from typing import Dict, Any, Tuple, List, Optional


def compute_normalized_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes normalized entropy as defined in Section VII / Figure 12:
    Uncertainty = -1 / log(C) * sum(p_i * log(p_i))
    where C is the number of classes.
    """
    n_classes = probs.shape[1]
    if n_classes <= 1:
        return np.zeros(len(probs), dtype=np.float32)

    p_safe = np.clip(probs, eps, 1.0)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
    norm_entropy = entropy / np.log(n_classes)
    return norm_entropy


def compute_tpr_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """
    Computes True Positive Rate (TPR) at a fixed False Positive Rate (FPR)
    (modern MIA evaluation standard from Carlini et al.).
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # Find the maximum TPR where FPR <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0
    return float(tpr[valid_idx[-1]])


def evaluate_attack_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluates binary membership inference predictions.
    y_true: 1 for training set members, 0 for non-members
    y_pred_proba: continuous membership probability / score in [0, 1]
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = float(auc(fpr, tpr))
    tpr_at_1fpr = compute_tpr_at_fpr(y_true, y_pred_proba, target_fpr=0.01)
    tpr_at_01fpr = compute_tpr_at_fpr(y_true, y_pred_proba, target_fpr=0.001)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "tpr_at_1%_fpr": tpr_at_1fpr,
        "tpr_at_0.1%_fpr": tpr_at_01fpr
    }


def compute_per_class_leakage(
    y_true_class: np.ndarray,
    is_member: np.ndarray,
    membership_scores: np.ndarray,
    target_train_acc_per_class: Dict[int, float],
    target_test_acc_per_class: Dict[int, float]
) -> List[Dict[str, Any]]:
    """
    Computes attack precision and generalization gap for each class,
    reproducing Figure 11.
    """
    classes = np.unique(y_true_class)
    class_stats = []

    for c in classes:
        mask = (y_true_class == c)
        if np.sum(mask) == 0:
            continue

        c_is_member = is_member[mask]
        c_scores = membership_scores[mask]
        c_pred = (c_scores >= 0.5).astype(int)

        prec = float(precision_score(c_is_member, c_pred, zero_division=0))
        rec = float(recall_score(c_is_member, c_pred, zero_division=0))

        train_acc = target_train_acc_per_class.get(int(c), 0.0)
        test_acc = target_test_acc_per_class.get(int(c), 0.0)
        acc_gap = train_acc - test_acc

        class_stats.append({
            "class": int(c),
            "precision": prec,
            "recall": rec,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "acc_gap": acc_gap,
            "sample_count": int(np.sum(mask)),
            "member_count": int(np.sum(c_is_member))
        })

    return class_stats
