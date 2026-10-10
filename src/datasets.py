"""
Dataset loaders and samplers for Membership Inference Attacks.

Supports:
- Purchase-100 (real preprocessed dataset from Shokri et al., 197,324 records, 600 features, 100 classes)
- Texas-100 (real hospital discharge dataset from Shokri et al., 67,330 records, 6,169 features, 100 classes)
- Adult Census (UCI Adult dataset, 14 features, binary classification)
- CIFAR-10 (image classification benchmark)
- Synthetic Purchase-style / Texas-style generators for rapid experimentation
- Shadow data synthesis: marginal statistics and hill-climbing (Algorithm 1)
- Clean target / shadow pool disjoint partitioning with resampling
"""

import os
import urllib.request
import tarfile
import io
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any


def ensure_purchase_and_texas(data_dir: str = "data"):
    """Ensures that purchase100.npz and texas100.npz are present."""
    os.makedirs(data_dir, exist_ok=True)
    p_npz = os.path.join(data_dir, "purchase100.npz")
    t_npz = os.path.join(data_dir, "texas100.npz")

    if not os.path.exists(p_npz):
        url = "https://raw.githubusercontent.com/privacytrustlab/datasets/master/dataset_purchase.tgz"
        print("Downloading original Purchase-100 dataset...")
        with urllib.request.urlopen(url) as resp:
            with tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz") as tar:
                f = tar.extractfile("dataset_purchase")
                raw = np.loadtxt(f, dtype=np.float32, delimiter=",")
                labels = (raw[:, 0] - 1).astype(np.int64)
                features = raw[:, 1:].astype(np.float32)
                np.savez_compressed(p_npz, features=features, labels=labels)

    if not os.path.exists(t_npz):
        url = "https://raw.githubusercontent.com/privacytrustlab/datasets/master/dataset_texas.tgz"
        print("Downloading original Texas-100 dataset...")
        with urllib.request.urlopen(url) as resp:
            with tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz") as tar:
                f_feats = tar.extractfile("texas/100/feats")
                f_labels = tar.extractfile("texas/100/labels")
                features = np.loadtxt(f_feats, dtype=np.float32, delimiter=",")
                labels = (np.loadtxt(f_labels, dtype=np.int64) - 1).astype(np.int64)
                np.savez_compressed(t_npz, features=features, labels=labels)


def load_purchase100(
    data_dir: str = "data",
    num_classes: int = 100,
    max_records: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads Purchase-100 dataset.
    Can filter to {2, 10, 20, 50, 100} classes to reproduce Section VI-F & Figure 10.
    """
    ensure_purchase_and_texas(data_dir)
    p_npz = os.path.join(data_dir, "purchase100.npz")
    data = np.load(p_npz)
    X = data["features"]
    y = data["labels"]

    if num_classes < 100:
        mask = y < num_classes
        X = X[mask]
        y = y[mask]

    if max_records is not None and max_records < len(y):
        indices = np.random.RandomState(42).permutation(len(y))[:max_records]
        X = X[indices]
        y = y[indices]

    return X, y


def load_texas100(
    data_dir: str = "data",
    num_classes: int = 100,
    max_records: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads Texas-100 hospital discharge dataset."""
    ensure_purchase_and_texas(data_dir)
    t_npz = os.path.join(data_dir, "texas100.npz")
    data = np.load(t_npz)
    X = data["features"]
    y = data["labels"]

    if num_classes < 100:
        mask = y < num_classes
        X = X[mask]
        y = y[mask]

    if max_records is not None and max_records < len(y):
        indices = np.random.RandomState(42).permutation(len(y))[:max_records]
        X = X[indices]
        y = y[indices]

    return X, y


def load_adult_census(
    data_dir: str = "data",
    max_records: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads UCI Adult Census dataset (binary classification: >$50k income).
    Cached to data/adult.npz.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, "adult.npz")
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        X, y = data["features"], data["labels"]
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        cols = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]
        df = pd.read_csv(url, header=None, names=cols, skipinitialspace=True)
        # Drop rows with missing values
        df = df.replace('?', np.nan).dropna()
        y = (df["income"].str.contains(">50K")).astype(int).values
        df = df.drop(columns=["income", "fnlwgt"])
        df = pd.get_dummies(df, drop_first=True)
        X = df.astype(np.float32).values
        np.savez_compressed(cache_file, features=X, labels=y)

    if max_records is not None and max_records < len(y):
        idx = np.random.RandomState(42).permutation(len(y))[:max_records]
        X, y = X[idx], y[idx]

    return X, y


def generate_synthetic_benchmark(
    n_samples: int = 20000,
    n_features: int = 600,
    n_classes: int = 10,
    sparsity: float = 0.9,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic shopping baskets ('Purchase-style') for testing,
    offline runs, or quick demonstrations.
    """
    rng = np.random.RandomState(seed)
    class_probs = rng.uniform(0.01, 1.0 - sparsity, size=(n_classes, n_features))
    y = rng.randint(0, n_classes, size=n_samples)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for c in range(n_classes):
        c_mask = (y == c)
        c_count = np.sum(c_mask)
        if c_count > 0:
            X[c_mask] = (rng.uniform(size=(c_count, n_features)) < class_probs[c]).astype(np.float32)
    return X, y


def load_cifar10(
    data_dir: str = "data",
    flatten: bool = False,
    max_records: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads CIFAR-10 dataset using torchvision."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)

    X_tr, y_tr = next(iter(train_loader))
    X_te, y_te = next(iter(test_loader))

    X = np.concatenate([X_tr.numpy(), X_te.numpy()], axis=0)
    y = np.concatenate([y_tr.numpy(), y_te.numpy()], axis=0)

    if flatten:
        X = X.reshape(len(X), -1)

    if max_records is not None and max_records < len(y):
        idx = np.random.RandomState(42).permutation(len(y))[:max_records]
        X, y = X[idx], y[idx]

    return X, y


def split_target_and_shadow_pools(
    X: np.ndarray,
    y: np.ndarray,
    target_train_size: int = 10000,
    target_test_size: int = 10000,
    seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Partitions the overall dataset into:
    1. Target training set D_target^train (size: target_train_size)
    2. Target test set D_target^test (size: target_test_size)
    3. Shadow pool D_shadow (disjoint from target dataset, used to sample shadow datasets)
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    perm = rng.permutation(n)

    target_total = target_train_size + target_test_size
    if target_total >= n:
        # Scale proportionally if dataset is smaller
        target_train_size = int(n * 0.25)
        target_test_size = int(n * 0.25)
        target_total = target_train_size + target_test_size

    idx_target_train = perm[:target_train_size]
    idx_target_test = perm[target_train_size:target_total]
    idx_shadow_pool = perm[target_total:]

    return {
        "target_train": (X[idx_target_train], y[idx_target_train]),
        "target_test": (X[idx_target_test], y[idx_target_test]),
        "shadow_pool": (X[idx_shadow_pool], y[idx_shadow_pool])
    }


def sample_shadow_dataset(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    shadow_train_size: int = 10000,
    shadow_test_size: int = 10000,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Samples a training and test set for one shadow model from the shadow pool.
    Shadow models are permitted to overlap with each other, but each individual
    shadow model has disjoint train and test sets.
    """
    if rng is None:
        rng = np.random.RandomState()

    n = len(y_pool)
    needed = shadow_train_size + shadow_test_size
    if needed > n:
        # Sample with replacement if pool is smaller than needed
        idx = rng.choice(n, size=needed, replace=True)
    else:
        idx = rng.choice(n, size=needed, replace=False)

    train_idx = idx[:shadow_train_size]
    test_idx = idx[shadow_train_size:]

    return (X_pool[train_idx], y_pool[train_idx]), (X_pool[test_idx], y_pool[test_idx])


def synthesize_marginal_dataset(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    n_samples: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Statistics-based synthesis (Section V-C):
    Samples feature values independently from marginal distributions per class.
    """
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y_ref, return_counts=True)
    class_priors = counts / len(y_ref)
    y_synth = rng.choice(classes, size=n_samples, p=class_priors)

    X_synth = np.zeros((n_samples, X_ref.shape[1]), dtype=np.float32)
    for c in classes:
        c_mask = (y_synth == c)
        c_count = np.sum(c_mask)
        if c_count == 0:
            continue
        c_ref = X_ref[y_ref == c]
        if len(c_ref) == 0:
            c_ref = X_ref
        marginals = np.mean(c_ref, axis=0)
        # Sample Bernoulli based on marginal probabilities
        X_synth[c_mask] = (rng.uniform(size=(c_count, X_ref.shape[1])) < marginals).astype(np.float32)

    return X_synth, y_synth
