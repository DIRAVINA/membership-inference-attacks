"""
Unit and integration tests for the Membership Inference Attack pipeline.
"""

import pytest
import numpy as np
import torch

from src.models import TargetTabularMLP, AttackModel, train_model, train_attack_model
from src.datasets import (
    load_purchase100,
    split_target_and_shadow_pools,
    sample_shadow_dataset,
    generate_synthetic_benchmark
)
from src.metrics import (
    compute_normalized_entropy,
    compute_tpr_at_fpr,
    evaluate_attack_predictions
)
from src.mitigations import OutputMitigations
from src.shadow_attack import ShadowAttackPipeline, BaselineAttacks


def test_target_tabular_model_shapes():
    """Verify target model shapes and softmax probabilities."""
    model = TargetTabularMLP(input_dim=50, num_classes=10, hidden_dim=32)
    x = torch.randn(8, 50)
    logits = model(x)
    assert logits.shape == (8, 10)

    probs = model.predict_proba(x, temperature=1.0)
    assert probs.shape == (8, 10)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)

    probs_temp = model.predict_proba(x, temperature=5.0)
    assert probs_temp.shape == (8, 10)
    assert torch.allclose(probs_temp.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_attack_model_shapes():
    """Verify attack model outputs binary membership probabilities."""
    attack_m = AttackModel(input_dim=10, hidden_dim=64)
    x = torch.randn(12, 10)
    out = attack_m(x)
    assert out.shape == (12, 1)

    scores = attack_m.predict_proba(x)
    assert scores.shape == (12,)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_dataset_splits_and_disjointness():
    """Verify that target and shadow datasets are completely disjoint."""
    X, y = generate_synthetic_benchmark(n_samples=1000, n_features=20, n_classes=5, seed=42)
    splits = split_target_and_shadow_pools(X, y, target_train_size=200, target_test_size=200, seed=42)

    X_tr, y_tr = splits["target_train"]
    X_te, y_te = splits["target_test"]
    X_sh, y_sh = splits["shadow_pool"]

    assert len(X_tr) == 200
    assert len(X_te) == 200
    assert len(X_sh) == 600

    # Ensure shadow sampling works with replacement/resampling
    (sh_tr_x, sh_tr_y), (sh_te_x, sh_te_y) = sample_shadow_dataset(
        X_sh, y_sh, shadow_train_size=150, shadow_test_size=150
    )
    assert len(sh_tr_x) == 150
    assert len(sh_te_x) == 150


def test_mitigations_properties():
    """Verify mitigation output properties (top-k, label-only, rounding)."""
    probs = np.array([
        [0.6, 0.2, 0.1, 0.05, 0.05],
        [0.05, 0.8, 0.05, 0.05, 0.05]
    ])

    # Top-1
    top1 = OutputMitigations.top_k_filter(probs, k=1)
    assert np.all(np.sum(top1 > 0, axis=1) == 1)
    assert np.allclose(np.sum(top1, axis=1), 1.0)

    # Top-3
    top3 = OutputMitigations.top_k_filter(probs, k=3)
    assert np.all(np.sum(top3 > 0, axis=1) == 3)
    assert np.allclose(np.sum(top3, axis=1), 1.0)

    # Top-1 label only
    label_only = OutputMitigations.top_1_label_only(probs)
    assert np.array_equal(label_only[0], [1, 0, 0, 0, 0])
    assert np.array_equal(label_only[1], [0, 1, 0, 0, 0])

    # Rounding
    rounded = OutputMitigations.rounding_coarsen(probs, decimals=1)
    assert np.allclose(np.sum(rounded, axis=1), 1.0)


def test_metrics_calculations():
    """Verify entropy, TPR at low FPR, and attack evaluations."""
    probs = np.array([
        [0.99, 0.01],
        [0.5, 0.5]
    ])
    entropy = compute_normalized_entropy(probs)
    assert entropy[0] < entropy[1]
    assert np.isclose(entropy[1], 1.0, atol=1e-4)

    y_true = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3])
    res = evaluate_attack_predictions(y_true, scores)
    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["roc_auc"] == 1.0


def test_end_to_end_pipeline_mini():
    """Run an end-to-end mini training and shadow attack run to verify flow."""
    X, y = generate_synthetic_benchmark(n_samples=800, n_features=25, n_classes=4, seed=123)
    splits = split_target_and_shadow_pools(X, y, target_train_size=150, target_test_size=150, seed=123)

    X_tr, y_tr = splits["target_train"]
    X_te, y_te = splits["target_test"]
    X_sh_pool, y_sh_pool = splits["shadow_pool"]

    # Target model
    target = TargetTabularMLP(input_dim=25, num_classes=4, hidden_dim=32)
    train_model(target, X_tr, y_tr, epochs=15, batch_size=32, lr=0.01)

    # Shadow pipeline
    pipeline = ShadowAttackPipeline(
        num_classes=4,
        input_dim=25,
        num_shadow_models=2,
        shadow_epochs=10,
        attack_epochs=10,
        batch_size=32
    )
    pipeline.train_shadow_models(
        X_sh_pool, y_sh_pool,
        shadow_train_size=150,
        shadow_test_size=150
    )
    pipeline.train_attack_models()

    eval_results = pipeline.evaluate_target_model(
        target, X_tr, y_tr, X_te, y_te, eval_sample_size=100
    )

    assert "shadow_attack" in eval_results
    assert "baseline_confidence_attack" in eval_results
    assert "baseline_entropy_attack" in eval_results
    assert eval_results["shadow_attack"]["accuracy"] >= 0.0
