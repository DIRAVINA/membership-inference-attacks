# Membership Inference Attacks Against Machine Learning Models
### Replication and Defense Analysis for Data Privacy and Security Course

**Research Paper:**
> **"Membership Inference Attacks Against Machine Learning Models"**  
> *Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov*  
> IEEE Symposium on Security and Privacy (S&P 2017) / [arXiv:1610.05820](https://arxiv.org/abs/1610.05820)

---

## 📌 Executive Summary

This project implements an end-to-end, scientifically validated reproduction of the seminal research paper by Shokri et al. (2017). In this paper, the authors pioneered the **Membership Inference Attack (MIA)** against black-box machine learning models:
> *Given a black-box API query access to a machine learning model and a specific data record, determine whether that individual's record was used in the model's training dataset.*

The project features:
1. **Faithful Paper Replication:**
   - Multi-class target models with `Tanh` activation and `SoftMax` output.
   - Exact benchmark datasets downloaded directly from the authors' official distribution (`Purchase-100` with 600 binary features and `Texas-100` with 6,169 binary features).
   - Shadow training technique using disjoint data partitions and random resampling.
   - Per-class binary attack neural networks ($f_{attack}^c$ for each class $c$).
   - Exact replication of **Figure 11** (Class Attack Precision vs Generalization Gap) and **Figure 12** (Prediction Uncertainty & Confidence Histograms for Members vs Non-Members).
2. **Modern Privacy Additions:**
   - Modern evaluation metrics: **ROC-AUC** and **TPR at low FPR** (TPR @ 1% FPR and 0.1% FPR, per Carlini et al.).
   - Baseline attack comparisons: Maximum Confidence Threshold and Negative Prediction Entropy Threshold.
   - Comprehensive mitigations: Table III paper defenses (Top-$k$, Rounding, Temperature Scaling, $L_2$ Regularization) plus **Dropout**, **Early Stopping**, and **Differentially Private SGD (DP-SGD)** with Opacus.
3. **Interactive Streamlit Dashboard:**
   - Full graphical user interface (`app.py`) for exploring results, viewing interactive Plotly diagrams, running live mini-experiments, and conducting individual record membership audits.

---

## 🏗️ Architecture & Attack Workflow

```
[Private Training Set] ---> Train Target Model f_target ---> [Black-box API: x -> y]
                                                                        |
                                                                        v
[Disjoint Shadow Pool] ---> Sample k Shadow Sets ---> Train k Shadow Models
                                                              |
                                                              v
                                             Query on Shadow Train (IN) & Test (OUT)
                                                              |
                                                              v
                                            Train per-class Attack Classifiers f_attack^c
                                                              |
                                                              v
                                              Evaluate Membership on Target Model!
```

---

## 📊 Key Experimental Findings

### 1. Primary Benchmark (Purchase-100)
- **Target Generalization Gap:** Train Acc: ~100%, Test Acc: ~65.7% (Generalization Gap = 34.3%, exactly matching Table I's ~34% gap).
- **Shadow Attack Performance:**
  - **Attack Precision:** **>75%–85%** (baseline random guessing is 50%).
  - **Attack Recall:** **~1.00** (matching Section VI-D & Figure 5).
  - **ROC-AUC:** **>0.91**.
  - **TPR @ 1% FPR:** **~7.8%** (compared to only 2.6% for naive baseline confidence thresholding), proving shadow models learn significantly sharper membership signals.

### 2. Generalization Gap vs Leakage (Figure 11)
- Classes with larger train-test accuracy gaps suffer substantially higher attack precision (up to 95%+). Overfitting is the primary engine of membership leakage.

### 3. Prediction Uncertainty & Confidence (Figure 12)
- Training set **members** produce a sharp spike at **zero uncertainty (normalized entropy $\to$ 0)** and **maximum confidence ($p \to 1.0$)**.
- Held-out **non-members** have broadly distributed entropy and lower peak confidence.

### 4. Low-Leakage Contrast (UCI Adult Census - Table II)
- On the UCI Adult Census binary classification dataset ($Acc_{train} \approx Acc_{test} \approx 84\%$), attack precision drops to **~50%** (random guessing). When generalizability is high and classes are binary complements, the model leaks negligible membership information.

### 5. Mitigations Benchmark (Table III + Modern Defenses)
| Defense Strategy | Target Test Acc | Attack Prec | Attack Acc | Attack ROC-AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **No Mitigation** | 65.7% | **75.2%** | 83.5% | 0.911 | Baseline overfitted target |
| **Top-$k$ ($k=3$)** | 65.7% | 75.1% | 83.4% | 0.910 | Ineffective against shadow attack |
| **Top-1 Label Only** | 65.7% | 61.2% | 66.8% | 0.720 | Mislabeling behavior still leaks data |
| **Rounding ($d=1$)** | 65.7% | 74.8% | 82.9% | 0.908 | Minimal privacy benefit |
| **Temperature ($t=20$)**| 65.7% | 71.5% | 78.2% | 0.865 | Flattens probability vector |
| **$L_2$ ($\lambda=10^{-2}$)**| 63.1% | 54.2% | 55.0% | 0.582 | Drastically reduces overfitting |
| **Dropout ($p=0.5$)** | 66.2% | 67.4% | 72.1% | 0.785 | Strong heuristic defense |
| **Early Stopping** | 65.9% | 68.1% | 73.0% | 0.792 | Prevents memorization |
| **DP-SGD ($\epsilon=1.0$)**| 58.4% | **51.2%** | **51.0%** | **0.518** | **Provable Defense (Collapses to random guess)** |

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
python -m pip install -r requirements.txt
```

### 2. Run Automated Experiments
To regenerate all benchmark results, figures, and mitigation tables:
```bash
python run_experiments.py --shadow_models 3 --epochs 25
```
Results will be exported to `results/` as JSON and CSV files.

### 3. Run the Streamlit Dashboard
Launch the interactive web UI:
```bash
streamlit run app.py
```
or
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Test Suite
Verify that all unit tests pass:
```bash
python -m pytest tests/test_pipeline.py -v
```

---

## 📁 Repository Structure
```
├── app.py                      # Interactive Streamlit Web Application
├── run_experiments.py          # Automated paper replication experiment runner
├── requirements.txt            # Python dependencies
├── data/                       # Cached Purchase-100, Texas-100, Adult datasets
│   ├── purchase100.npz
│   └── texas100.npz
├── results/                    # Precomputed experiment metrics & figures
│   ├── purchase100_core_results.json
│   ├── figure11_data.json
│   ├── figure12_data.json
│   ├── class_scaling_results.json
│   ├── adult_contrast_results.json
│   └── mitigations_table.csv
├── src/                        # Core algorithmic modules
│   ├── models.py               # Target MLP, CNN, Attack MLP, training loops
│   ├── datasets.py             # Loaders for Purchase, Texas, Adult, CIFAR, Synthetic
│   ├── shadow_attack.py        # Shadow model training, attack classifiers, baselines
│   ├── mitigations.py          # Post-processing filters, L2 reg, DP-SGD
│   └── metrics.py              # Precision, Recall, AUC, TPR@low FPR, entropy
└── tests/
    └── test_pipeline.py        # Pytest test suite
```
