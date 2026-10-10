"""
Automated Experiment Runner for Membership Inference Attack Paper Replication.
Runs:
1. Primary Benchmark (Purchase-100): Overfitted Target Model, Shadow Pipeline,
   Modern Metrics (AUC, TPR@1% FPR), Baselines Comparison, Fig 11 & Fig 12 data.
2. Class Scaling (C in {2, 10, 20, 50, 100}): Reproducing Fig 10 & Table II.
3. Low-Leakage Contrast Case (UCI Adult Census): Reproducing Table II.
4. Mitigations & Defenses Benchmark: Table III + Dropout + Early Stopping + DP-SGD.
Saves all outputs into results/ directory as JSON and CSV for immediate loading by Streamlit.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

from src.datasets import (
    load_purchase100,
    load_texas100,
    load_adult_census,
    split_target_and_shadow_pools,
    sample_shadow_dataset
)
from src.models import TargetTabularMLP, train_model
from src.shadow_attack import ShadowAttackPipeline, BaselineAttacks
from src.metrics import compute_normalized_entropy, compute_per_class_leakage, evaluate_attack_predictions
from src.mitigations import OutputMitigations


def run_purchase100_experiment(
    data_dir: str = "data",
    results_dir: str = "results",
    num_classes: int = 100,
    target_train_size: int = 5000,
    target_test_size: int = 5000,
    num_shadow_models: int = 5,
    epochs: int = 35,
    seed: int = 42
):
    print(f"\n==========================================")
    print(f"Running Experiment 1: Purchase-{num_classes} Primary Benchmark")
    print(f"==========================================")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load data
    print("Loading Purchase dataset...")
    X, y = load_purchase100(data_dir=data_dir, num_classes=num_classes, max_records=40000)
    splits = split_target_and_shadow_pools(
        X, y,
        target_train_size=target_train_size,
        target_test_size=target_test_size,
        seed=seed
    )
    X_tr, y_tr = splits["target_train"]
    X_te, y_te = splits["target_test"]
    X_sh, y_sh = splits["shadow_pool"]

    # 2. Train Target Model (deliberately overfit with no reg to match Table I)
    print("Training Target Model (MLP with Tanh)...")
    target_model = TargetTabularMLP(input_dim=X.shape[1], num_classes=num_classes, hidden_dim=128)
    history = train_model(target_model, X_tr, y_tr, X_val=X_te, y_val=y_te, epochs=epochs, batch_size=64, lr=0.001)

    # 3. Train Shadow Attack Pipeline
    print(f"Training {num_shadow_models} Shadow Models and per-class Attack Classifiers...")
    pipeline = ShadowAttackPipeline(
        num_classes=num_classes,
        input_dim=X.shape[1],
        num_shadow_models=num_shadow_models,
        shadow_epochs=epochs,
        attack_epochs=30,
        batch_size=64
    )
    pipeline.train_shadow_models(X_sh, y_sh, shadow_train_size=target_train_size, shadow_test_size=target_test_size, seed=seed)
    pipeline.train_attack_models()

    # 4. Evaluate against Target Model
    print("Evaluating Attack against Target Model...")
    eval_res = pipeline.evaluate_target_model(
        target_model, X_tr, y_tr, X_te, y_te, eval_sample_size=3000, seed=seed
    )

    t_stats = eval_res["target_model_stats"]
    s_attack = eval_res["shadow_attack"]
    b_conf = eval_res["baseline_confidence_attack"]
    b_ent = eval_res["baseline_entropy_attack"]

    print("\n--- Summary Results ---")
    print(f"Target Train Acc: {t_stats['train_accuracy']:.3f} | Test Acc: {t_stats['test_accuracy']:.3f} | Gen Gap: {t_stats['generalization_gap']:.3f}")
    print(f"Shadow Attack -> Prec: {s_attack['precision']:.3f} | Rec: {s_attack['recall']:.3f} | Acc: {s_attack['accuracy']:.3f} | AUC: {s_attack['roc_auc']:.3f} | TPR@1%FPR: {s_attack['tpr_at_1%_fpr']:.3f}")
    print(f"Baseline Conf -> Prec: {b_conf['precision']:.3f} | Rec: {b_conf['recall']:.3f} | Acc: {b_conf['accuracy']:.3f} | AUC: {b_conf['roc_auc']:.3f}")
    print(f"Baseline Ent  -> Prec: {b_ent['precision']:.3f} | Rec: {b_ent['recall']:.3f} | Acc: {b_ent['accuracy']:.3f} | AUC: {b_ent['roc_auc']:.3f}")

    # 5. Extract Figure 11 data (per-class precision vs train-test gap)
    eval_data = eval_res["eval_data"]
    fig11_stats = compute_per_class_leakage(
        eval_data["y_eval_class"],
        eval_data["is_member_eval"],
        eval_data["shadow_scores"],
        t_stats["train_acc_per_class"],
        t_stats["test_acc_per_class"]
    )

    # 6. Extract Figure 12 data (Normalized Entropy & Confidence histograms for members vs non-members)
    probs_m = eval_data["probs_m"]
    probs_nm = eval_data["probs_nm"]
    ent_m = compute_normalized_entropy(probs_m).tolist()
    ent_nm = compute_normalized_entropy(probs_nm).tolist()
    conf_m = np.max(probs_m, axis=1).tolist()
    conf_nm = np.max(probs_nm, axis=1).tolist()

    # Save to disk
    with open(os.path.join(results_dir, "purchase100_core_results.json"), "w") as f:
        json.dump({
            "target_model_stats": {
                "train_accuracy": t_stats["train_accuracy"],
                "test_accuracy": t_stats["test_accuracy"],
                "generalization_gap": t_stats["generalization_gap"]
            },
            "shadow_attack": s_attack,
            "baseline_confidence_attack": b_conf,
            "baseline_entropy_attack": b_ent
        }, f, indent=2)

    with open(os.path.join(results_dir, "figure11_data.json"), "w") as f:
        json.dump(fig11_stats, f, indent=2)

    with open(os.path.join(results_dir, "figure12_data.json"), "w") as f:
        json.dump({
            "entropy_members": ent_m[:1000],
            "entropy_non_members": ent_nm[:1000],
            "confidence_members": conf_m[:1000],
            "confidence_non_members": conf_nm[:1000]
        }, f, indent=2)

    print("Experiment 1 complete and results saved.")
    return target_model, pipeline, splits


def run_class_scaling_experiment(
    data_dir: str = "data",
    results_dir: str = "results",
    seed: int = 42
):
    print(f"\n==========================================")
    print(f"Running Experiment 2: Effect of Number of Classes (Fig 10 & Table II)")
    print(f"==========================================")
    class_configs = [2, 10, 20, 50, 100]
    results = []

    for c in class_configs:
        print(f"\n--- Testing C = {c} classes ---")
        X, y = load_purchase100(data_dir=data_dir, num_classes=c, max_records=25000)
        splits = split_target_and_shadow_pools(X, y, target_train_size=3000, target_test_size=3000, seed=seed)
        X_tr, y_tr = splits["target_train"]
        X_te, y_te = splits["target_test"]
        X_sh, y_sh = splits["shadow_pool"]

        target_model = TargetTabularMLP(input_dim=X.shape[1], num_classes=c, hidden_dim=128)
        train_model(target_model, X_tr, y_tr, epochs=25, batch_size=64, lr=0.001)

        pipeline = ShadowAttackPipeline(
            num_classes=c, input_dim=X.shape[1], num_shadow_models=3,
            shadow_epochs=20, attack_epochs=20, batch_size=64
        )
        pipeline.train_shadow_models(X_sh, y_sh, shadow_train_size=3000, shadow_test_size=3000, seed=seed)
        pipeline.train_attack_models()

        eval_res = pipeline.evaluate_target_model(target_model, X_tr, y_tr, X_te, y_te, eval_sample_size=1500, seed=seed)
        s_att = eval_res["shadow_attack"]
        t_stat = eval_res["target_model_stats"]

        res_entry = {
            "num_classes": c,
            "target_train_acc": t_stat["train_accuracy"],
            "target_test_acc": t_stat["test_accuracy"],
            "generalization_gap": t_stat["generalization_gap"],
            "attack_precision": s_att["precision"],
            "attack_recall": s_att["recall"],
            "attack_accuracy": s_att["accuracy"],
            "attack_roc_auc": s_att["roc_auc"]
        }
        print(f"C={c}: Target Gap={res_entry['generalization_gap']:.3f} | Attack Prec={res_entry['attack_precision']:.3f} | AUC={res_entry['attack_roc_auc']:.3f}")
        results.append(res_entry)

    with open(os.path.join(results_dir, "class_scaling_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Experiment 2 complete and saved.")


def run_adult_contrast_experiment(
    data_dir: str = "data",
    results_dir: str = "results",
    seed: int = 42
):
    print(f"\n==========================================")
    print(f"Running Experiment 3: UCI Adult Census Contrast (Table II)")
    print(f"==========================================")
    X, y = load_adult_census(data_dir=data_dir, max_records=25000)
    splits = split_target_and_shadow_pools(X, y, target_train_size=4000, target_test_size=4000, seed=seed)
    X_tr, y_tr = splits["target_train"]
    X_te, y_te = splits["target_test"]
    X_sh, y_sh = splits["shadow_pool"]

    # Target model
    target_model = TargetTabularMLP(input_dim=X.shape[1], num_classes=2, hidden_dim=64)
    train_model(target_model, X_tr, y_tr, epochs=20, batch_size=64, lr=0.001)

    # Shadow pipeline
    pipeline = ShadowAttackPipeline(
        num_classes=2, input_dim=X.shape[1], num_shadow_models=3,
        shadow_epochs=15, attack_epochs=20, batch_size=64
    )
    pipeline.train_shadow_models(X_sh, y_sh, shadow_train_size=4000, shadow_test_size=4000, seed=seed)
    pipeline.train_attack_models()

    eval_res = pipeline.evaluate_target_model(target_model, X_tr, y_tr, X_te, y_te, eval_sample_size=2000, seed=seed)
    s_att = eval_res["shadow_attack"]
    t_stat = eval_res["target_model_stats"]

    res_entry = {
        "dataset": "UCI Adult Census",
        "num_classes": 2,
        "target_train_acc": t_stat["train_accuracy"],
        "target_test_acc": t_stat["test_accuracy"],
        "generalization_gap": t_stat["generalization_gap"],
        "attack_precision": s_att["precision"],
        "attack_recall": s_att["recall"],
        "attack_accuracy": s_att["accuracy"],
        "attack_roc_auc": s_att["roc_auc"]
    }
    print(f"Adult Census: Target Gap={res_entry['generalization_gap']:.3f} | Attack Prec={res_entry['attack_precision']:.3f} | AUC={res_entry['attack_roc_auc']:.3f}")

    with open(os.path.join(results_dir, "adult_contrast_results.json"), "w") as f:
        json.dump(res_entry, f, indent=2)
    print("Experiment 3 complete and saved.")


def run_mitigations_benchmark(
    target_model: TargetTabularMLP,
    pipeline: ShadowAttackPipeline,
    splits: dict,
    results_dir: str = "results",
    seed: int = 42
):
    print(f"\n==========================================")
    print(f"Running Experiment 4: Mitigations Benchmark (Table III + Modern Defenses)")
    print(f"==========================================")
    X_tr, y_tr = splits["target_train"]
    X_te, y_te = splits["target_test"]

    records = []

    # Helper evaluator
    def evaluate_defense(name: str, t_model, transform=None):
        res = pipeline.evaluate_target_model(
            t_model, X_tr, y_tr, X_te, y_te, eval_sample_size=2000, output_transform=transform, seed=seed
        )
        t_acc = res["target_model_stats"]["test_accuracy"]
        att = res["shadow_attack"]
        rec = {
            "Defense": name,
            "Testing Accuracy": round(t_acc, 3),
            "Attack Total Accuracy": round(att["accuracy"], 3),
            "Attack Precision": round(att["precision"], 3),
            "Attack Recall": round(att["recall"], 3),
            "Attack ROC-AUC": round(att["roc_auc"], 3),
            "Attack TPR @ 1% FPR": round(att["tpr_at_1%_fpr"], 3)
        }
        print(f"-> {name:<22} | TestAcc: {rec['Testing Accuracy']:.3f} | AttPrec: {rec['Attack Precision']:.3f} | AttAcc: {rec['Attack Total Accuracy']:.3f} | AUC: {rec['Attack ROC-AUC']:.3f}")
        records.append(rec)

    # 1. Baseline: No Mitigation
    evaluate_defense("No Mitigation", target_model)

    # 2. Output Mitigations (Post-processing on target model)
    evaluate_defense("Top-k (k=3)", target_model, lambda p: OutputMitigations.top_k_filter(p, k=3))
    evaluate_defense("Top-k (k=1)", target_model, lambda p: OutputMitigations.top_k_filter(p, k=1))
    evaluate_defense("Top-1 Label Only", target_model, lambda p: OutputMitigations.top_1_label_only(p))
    evaluate_defense("Rounding (d=3)", target_model, lambda p: OutputMitigations.rounding_coarsen(p, decimals=3))
    evaluate_defense("Rounding (d=1)", target_model, lambda p: OutputMitigations.rounding_coarsen(p, decimals=1))

    # Temperature scaling wrapper
    class TempModelWrapper:
        def __init__(self, base_model, temp):
            self.base_model = base_model
            self.temp = temp
        def predict_proba(self, x):
            return self.base_model.predict_proba(x, temperature=self.temp)

    evaluate_defense("Temperature (t=5)", TempModelWrapper(target_model, temp=5.0))
    evaluate_defense("Temperature (t=20)", TempModelWrapper(target_model, temp=20.0))

    # 3. In-Training Mitigations (L2 Regularization)
    for l2_val, l2_name in [(1e-4, "L2 (lambda=1e-4)"), (1e-3, "L2 (lambda=1e-3)"), (1e-2, "L2 (lambda=1e-2)")]:
        m_l2 = TargetTabularMLP(input_dim=X_tr.shape[1], num_classes=pipeline.num_classes, hidden_dim=128)
        train_model(m_l2, X_tr, y_tr, epochs=25, batch_size=64, lr=0.001, weight_decay=l2_val)
        evaluate_defense(l2_name, m_l2)

    # 4. Modern Defenses: Dropout (p=0.5)
    m_drop = TargetTabularMLP(input_dim=X_tr.shape[1], num_classes=pipeline.num_classes, hidden_dim=128, dropout=0.5)
    train_model(m_drop, X_tr, y_tr, epochs=30, batch_size=64, lr=0.001)
    evaluate_defense("Dropout (p=0.5)", m_drop)

    # 5. Modern Defenses: Early Stopping
    m_es = TargetTabularMLP(input_dim=X_tr.shape[1], num_classes=pipeline.num_classes, hidden_dim=128)
    train_model(m_es, X_tr, y_tr, X_val=X_te, y_val=y_te, epochs=35, batch_size=64, lr=0.001, patience=3)
    evaluate_defense("Early Stopping", m_es)

    # 6. Modern Defenses: DP-SGD with Opacus
    for eps in [10.0, 5.0, 1.0]:
        m_dp = TargetTabularMLP(input_dim=X_tr.shape[1], num_classes=pipeline.num_classes, hidden_dim=128)
        train_model(m_dp, X_tr, y_tr, epochs=20, batch_size=64, lr=0.001, use_dp=True, dp_target_epsilon=eps)
        evaluate_defense(f"DP-SGD (eps={eps})", m_dp)

    df_table = pd.DataFrame(records)
    csv_path = os.path.join(results_dir, "mitigations_table.csv")
    json_path = os.path.join(results_dir, "mitigations_table.json")
    df_table.to_csv(csv_path, index=False)
    df_table.to_json(json_path, orient="records", indent=2)
    print(f"\nMitigations Benchmark completed and saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Membership Inference Attack Experiments")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--shadow_models", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    # Experiment 1: Primary Purchase-100 Benchmark
    target_model, pipeline, splits = run_purchase100_experiment(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_classes=100,
        num_shadow_models=args.shadow_models,
        epochs=args.epochs
    )

    # Experiment 2: Number of classes scaling
    run_class_scaling_experiment(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )

    # Experiment 3: Adult Census Contrast
    run_adult_contrast_experiment(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )

    # Experiment 4: Mitigations Benchmark
    run_mitigations_benchmark(
        target_model, pipeline, splits,
        results_dir=args.results_dir
    )

    print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
