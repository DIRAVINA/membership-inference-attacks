"""
Shadow Model Training and Membership Inference Attack Pipeline.
Implements:
- Training of k shadow models on overlapping resampled subsets of the shadow pool
- Supervised attack dataset generation: (prediction_vector, true_class, member_label)
- Per-class binary attack model training (Section V-D & Section VI-D)
- Simple baseline attacks (Confidence thresholding, Entropy thresholding) for comparison
- Defense-aware and defense-unaware evaluation
"""

import copy
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable

from src.models import TargetTabularMLP, AttackModel, train_model, train_attack_model
from src.datasets import sample_shadow_dataset
from src.metrics import compute_normalized_entropy, evaluate_attack_predictions


class BaselineAttacks:
    """
    Simple threshold-based baseline membership inference attacks:
    - Maximum confidence / True-class confidence thresholding
    - Negative normalized entropy thresholding
    Used to demonstrate whether shadow models provide superior signal.
    """
    @staticmethod
    def true_class_confidence(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Returns the model's confidence on the true class."""
        n = len(y_true)
        conf = probs[np.arange(n), y_true]
        return conf

    @staticmethod
    def max_confidence(probs: np.ndarray) -> np.ndarray:
        """Returns the maximum output probability."""
        return np.max(probs, axis=1)

    @staticmethod
    def negative_entropy(probs: np.ndarray) -> np.ndarray:
        """Lower entropy implies higher model confidence -> likely a member."""
        norm_entropy = compute_normalized_entropy(probs)
        return 1.0 - norm_entropy


class ShadowAttackPipeline:
    """
    Shokri et al. (2017) Shadow Modeling Attack System.
    """
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        num_shadow_models: int = 5,
        shadow_epochs: int = 30,
        attack_epochs: int = 40,
        batch_size: int = 64,
        lr: float = 0.001,
        device: str = "cpu"
    ):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.num_shadow_models = num_shadow_models
        self.shadow_epochs = shadow_epochs
        self.attack_epochs = attack_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        self.shadow_models: List[TargetTabularMLP] = []
        self.attack_models: Dict[int, AttackModel] = {}
        self.attack_training_data: Dict[str, np.ndarray] = {}

    def train_shadow_models(
        self,
        X_shadow_pool: np.ndarray,
        y_shadow_pool: np.ndarray,
        shadow_train_size: int = 5000,
        shadow_test_size: int = 5000,
        output_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        seed: int = 42
    ):
        """
        Trains k shadow models on resampled subsets of the shadow pool,
        queries them on their training (in) and test (out) sets,
        and constructs the attack training set.
        """
        rng = np.random.RandomState(seed)
        self.shadow_models = []

        all_probs = []
        all_labels = []
        all_memberships = []

        for i in range(self.num_shadow_models):
            shadow_seed = rng.randint(0, 1000000)
            shadow_rng = np.random.RandomState(shadow_seed)
            (X_sh_train, y_sh_train), (X_sh_test, y_sh_test) = sample_shadow_dataset(
                X_shadow_pool, y_shadow_pool,
                shadow_train_size=shadow_train_size,
                shadow_test_size=shadow_test_size,
                rng=shadow_rng
            )

            # Build and train shadow model
            shadow_model = TargetTabularMLP(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                hidden_dim=128
            )
            train_model(
                shadow_model,
                X_sh_train, y_sh_train,
                epochs=self.shadow_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device
            )
            self.shadow_models.append(shadow_model)

            # Query shadow model on training set (members: 1)
            t_x_tr = torch.tensor(X_sh_train, dtype=torch.float32)
            probs_train = shadow_model.predict_proba(t_x_tr).cpu().numpy()

            # Query shadow model on test set (non-members: 0)
            t_x_te = torch.tensor(X_sh_test, dtype=torch.float32)
            probs_test = shadow_model.predict_proba(t_x_te).cpu().numpy()

            if output_transform is not None:
                probs_train = output_transform(probs_train)
                probs_test = output_transform(probs_test)

            all_probs.append(probs_train)
            all_labels.append(y_sh_train)
            all_memberships.append(np.ones(len(y_sh_train), dtype=np.int64))

            all_probs.append(probs_test)
            all_labels.append(y_sh_test)
            all_memberships.append(np.zeros(len(y_sh_test), dtype=np.int64))

        X_attack_all = np.concatenate(all_probs, axis=0)
        y_class_all = np.concatenate(all_labels, axis=0)
        membership_all = np.concatenate(all_memberships, axis=0)

        self.attack_training_data = {
            "probs": X_attack_all,
            "classes": y_class_all,
            "membership": membership_all
        }

    def train_attack_models(self):
        """
        Trains per-class binary attack models as defined in Section V-D & Section VI-D.
        For each class c, a neural network is trained on the shadow model predictions
        for instances with true class c to predict membership (in vs out).
        """
        if not self.attack_training_data:
            raise ValueError("Must train shadow models and generate attack data first.")

        probs_all = self.attack_training_data["probs"]
        classes_all = self.attack_training_data["classes"]
        membership_all = self.attack_training_data["membership"]

        self.attack_models = {}

        # Fallback global attack model for classes with very few samples
        global_model = AttackModel(input_dim=self.num_classes, hidden_dim=64)
        # Train global model on subset of all data
        sub_idx = np.random.permutation(len(membership_all))[:min(len(membership_all), 10000)]
        train_attack_model(
            global_model,
            probs_all[sub_idx], membership_all[sub_idx],
            epochs=self.attack_epochs,
            batch_size=self.batch_size,
            device=self.device
        )
        self.attack_models[-1] = global_model

        for c in range(self.num_classes):
            mask = (classes_all == c)
            c_count = np.sum(mask)

            # If class has sufficient samples and both members and non-members
            if c_count >= 20 and len(np.unique(membership_all[mask])) > 1:
                c_attack_model = AttackModel(input_dim=self.num_classes, hidden_dim=64)
                train_attack_model(
                    c_attack_model,
                    probs_all[mask], membership_all[mask],
                    epochs=self.attack_epochs,
                    batch_size=self.batch_size,
                    device=self.device
                )
                self.attack_models[c] = c_attack_model
            else:
                self.attack_models[c] = global_model

    def predict_membership(
        self,
        probs: np.ndarray,
        y_true_class: np.ndarray
    ) -> np.ndarray:
        """
        Computes membership inference probabilities for target records.
        probs: prediction probability vectors from target model
        y_true_class: ground truth classes of the records
        """
        n = len(y_true_class)
        membership_scores = np.zeros(n, dtype=np.float32)

        for c in range(self.num_classes):
            mask = (y_true_class == c)
            if np.sum(mask) == 0:
                continue
            model = self.attack_models.get(c, self.attack_models.get(-1))
            c_probs = torch.tensor(probs[mask], dtype=torch.float32)
            c_scores = model.predict_proba(c_probs).cpu().numpy()
            membership_scores[mask] = c_scores

        return membership_scores

    def evaluate_target_model(
        self,
        target_model: TargetTabularMLP,
        X_target_train: np.ndarray,
        y_target_train: np.ndarray,
        X_target_test: np.ndarray,
        y_target_test: np.ndarray,
        eval_sample_size: Optional[int] = 2000,
        output_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Evaluates the attack against the target model using a balanced evaluation set
        (equal number of target training members and target test non-members).
        Returns comprehensive metrics for:
        - Shadow Model Attack
        - Baseline True-Class Confidence Attack
        - Baseline Negative Entropy Attack
        - Target model train/test accuracy and per-class stats
        """
        rng = np.random.RandomState(seed)

        n_members = len(y_target_train)
        n_non_members = len(y_target_test)
        sample_size = min(n_members, n_non_members)
        if eval_sample_size is not None and eval_sample_size < sample_size:
            sample_size = eval_sample_size

        idx_m = rng.choice(n_members, size=sample_size, replace=False)
        idx_nm = rng.choice(n_non_members, size=sample_size, replace=False)

        X_m, y_m = X_target_train[idx_m], y_target_train[idx_m]
        X_nm, y_nm = X_target_test[idx_nm], y_target_test[idx_nm]

        # Get target model predictions
        t_m = torch.tensor(X_m, dtype=torch.float32)
        t_nm = torch.tensor(X_nm, dtype=torch.float32)

        probs_m = target_model.predict_proba(t_m).cpu().numpy()
        probs_nm = target_model.predict_proba(t_nm).cpu().numpy()

        # Compute target model train/test classification accuracy
        target_train_acc = float(np.mean(np.argmax(probs_m, axis=1) == y_m))
        target_test_acc = float(np.mean(np.argmax(probs_nm, axis=1) == y_nm))
        gen_gap = target_train_acc - target_test_acc

        # Per-class target accuracy
        train_acc_per_class = {}
        test_acc_per_class = {}
        for c in range(self.num_classes):
            m_c = (y_m == c)
            nm_c = (y_nm == c)
            if np.sum(m_c) > 0:
                train_acc_per_class[c] = float(np.mean(np.argmax(probs_m[m_c], axis=1) == y_m[m_c]))
            if np.sum(nm_c) > 0:
                test_acc_per_class[c] = float(np.mean(np.argmax(probs_nm[nm_c], axis=1) == y_nm[nm_c]))

        # Apply output mitigation if specified
        if output_transform is not None:
            probs_m_eval = output_transform(probs_m)
            probs_nm_eval = output_transform(probs_nm)
        else:
            probs_m_eval = probs_m
            probs_nm_eval = probs_nm

        # Assemble balanced evaluation dataset (baseline = 50%)
        probs_eval = np.concatenate([probs_m_eval, probs_nm_eval], axis=0)
        y_eval_class = np.concatenate([y_m, y_nm], axis=0)
        is_member_eval = np.concatenate([
            np.ones(sample_size, dtype=np.int64),
            np.zeros(sample_size, dtype=np.int64)
        ], axis=0)

        # 1. Shadow Model Attack Predictions
        shadow_scores = self.predict_membership(probs_eval, y_eval_class)
        shadow_metrics = evaluate_attack_predictions(is_member_eval, shadow_scores)

        # 2. Baseline Attacks
        base_conf_scores = BaselineAttacks.true_class_confidence(probs_eval, y_eval_class)
        base_conf_metrics = evaluate_attack_predictions(is_member_eval, base_conf_scores)

        base_ent_scores = BaselineAttacks.negative_entropy(probs_eval)
        base_ent_metrics = evaluate_attack_predictions(is_member_eval, base_ent_scores)

        return {
            "target_model_stats": {
                "train_accuracy": target_train_acc,
                "test_accuracy": target_test_acc,
                "generalization_gap": gen_gap,
                "train_acc_per_class": train_acc_per_class,
                "test_acc_per_class": test_acc_per_class
            },
            "shadow_attack": shadow_metrics,
            "baseline_confidence_attack": base_conf_metrics,
            "baseline_entropy_attack": base_ent_metrics,
            "eval_data": {
                "probs_m": probs_m,
                "probs_nm": probs_nm,
                "y_eval_class": y_eval_class,
                "is_member_eval": is_member_eval,
                "shadow_scores": shadow_scores
            }
        }
