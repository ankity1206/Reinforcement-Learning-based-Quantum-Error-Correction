# Reinforcement Learning for Quantum Error Correction

An experiment on using Supervised Deep Learning (MLP) and Reinforcement Learning (DQN) for decoding rotated surface codes under circuit-level Pauli noise, benchmarked against Minimum-Weight Perfect Matching (MWPM).
---

## 1. Summary

This project is a comparison between:
- **PyMatching (MWPM)** algorithmic baseline.
- **Supervised MLP Decoder** (24 $\to$ 128 $\to$ 128 $\to$ 1).
- **Deep Q-Network (DQN) RL Agent** operating within a custom vectorized QEC Markov Decision Process (MDP).

**Result:**
1. The **Supervised MLP** achieves competitive logical decoding accuracy (**96.58%** at $p=0.005$ and **89.13%** at $p=0.01$), closely approaching PyMatching (**98.11%** and **93.61%** respectively).
2. The **RL agent (DQN)** achieves **76.66% $\pm$ 1.12% logical success** and **55.48% $\pm$ 1.18% joint success** across 8 independent training replicates under restricted detector observability (20 historical detectors, blind to final-round measurement syndrome).
3. Exact non-parametric paired sign-flip testing with Holm-Bonferroni correction ($M=24, \alpha=0.05$) revealed **no statistically significant advantage** from artificial class-balanced sampling strategies over the natural empirical distribution for RL training.

---

## 2. Quantum Error Correction Background

### The Rotated Surface Code
The rotated surface code encodes a single logical qubit into a 2D checkerboard arrangement of data and ancilla qubits. For distance $d=3$:
- **Data qubits:** $d^2 = 9$ physical qubits (indices 1, 3, 5, 8, 10, 12, 15, 17, 19).
- **Syndrome measurements:** Interleaved rounds of $X$- and $Z$-type stabilizer measurements executed by ancilla qubits.
- **Code distance ($d=3$):** Minimum weight of an undetectable logical operator is 3, allowing correction of any arbitrary weight-1 physical error.
- **Detection Events:** Parity changes between successive rounds of stabilizer checks form binary detector events ($D \in \{0, 1\}$).

---

## 3. Experimental Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Quantum Circuit (Stim)                          │
│               rotated_memory_x, d=3, rounds=3, p in [0.005, 0.02]      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 20,000 Monte Carlo Shots
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Dataset: [X: 24 detectors, y: Logical Target]      │
└─────────────┬─────────────────────┬────────────────────┬───────────────┘
              │                     │                    │
              ▼                     ▼                    ▼
     ┌─────────────────┐   ┌─────────────────┐   ┌───────────────────────┐
     │ PyMatching      │   │ Supervised MLP  │   │ RL Environment (MDP)  │
     │ Full MWPM graph │   │ 24 -> 128 -> 1  │   │ Vectorized Gym Env    │
     │ 24 detectors    │   │ 24 detectors    │   │ 20 historical det.    │
     └────────┬────────┘   └────────┬────────┘   └───────────┬───────────┘
              │                     │                        │
              ▼                     ▼                        ▼
     Logical Observable     Logical Observable       Joint Success
     Prediction             Prediction               (Syndrome + Observable)
```

### Code Family & Noise Model
- **Circuit Generator:** Stim `rotated_memory_x` with depolarizing circuit-level Pauli noise $p \in \{0.005, 0.010, 0.020\}$.
- **Rounds ($r$):** 3 extraction rounds.
- **Detector Count:** 24 detectors total (20 historical round detectors + 4 final boundary measurement detectors).
- **Logical Observables:** 1 logical $X$ operator.

### Dataset Construction & Splits
- **Total shots:** 20,000 Monte Carlo samples generated via Stim.
- **Data partition:** Stratified split on logical target observable ($L_{\text{target}}$):
  - **Train:** 13,999 shots (70.0%)
  - **Validation:** 2,999 shots (15.0%) — 2,433 Class-0 (81.1%), 566 Class-1 (18.9%)
  - **Test (Held-out):** 3,002 shots (15.0%)

### Decoders Implemented

#### 1. PyMatching Baseline (MWPM)
Constructed directly from the Stim detector error model (`circuit.detector_error_model()`). Applies Edmond's blossom algorithm to find the minimum-weight matching of defect pairs on the syndrome graph using full 24-bit detector observations.

#### 2. Supervised Neural Network (MLP v2)
- **Architecture:** `Linear(24, 128)` $\to$ `ReLU` $\to$ `Dropout(0.1)` $\to$ `Linear(128, 128)` $\to$ `ReLU` $\to$ `Dropout(0.1)` $\to$ `Linear(128, 1)`.
- **Loss:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
- **Optimization:** Adam optimizer, learning rate $10^{-3}$, batch size 64, early stopping based on validation loss (best checkpoint at epoch 33).

#### 3. Reinforcement Learning Decoder (DQN)
- **State Formulation ($s_t$):** Restricted 20-bit historical detector syndrome vector + internal physical correction tracking state + normalized time step. *(Excluded the 4 final-round detector bits).*
- **Action Space ($a_t$):** Discrete terminal correction selections on candidate data qubits.
- **Environment:** Custom batch-vectorized environment implementing GF(2) stabilizer tracking.
- **Agent Architecture:** Deep Q-Network with experience replay buffer, target network updates, and $\epsilon$-greedy exploration.
- **Training Budget:** 7,000 gradient steps per replicate across 7 sampling distributions (`natural`, `80_20`, `75_25`, `70_30`, `65_35`, `60_40`, `50_50`), replicated over 8 distinct seeds (56 trained policies total; 392,000 gradient updates).

---

## 4. Empirical Findings & Benchmarks

### Noise Sensitivity Benchmarks ($p \in \{0.005, 0.01, 0.02\}$)

Evaluated across varying physical error rates on held-out test data:

| Noise Rate ($p$) | Decoder | Accuracy | Balanced Acc. | Precision | Recall | F1 Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0.005** | Majority Class Baseline | 89.48% | 50.00% | 0.000 | 0.000 | 0.000 |
| | **Supervised MLP v2** | **96.58%** | **95.80%** | 0.776 | **0.948** | 0.854 |
| | **PyMatching (MWPM)** | **98.11%** | **94.61%** | **0.917** | 0.902 | **0.910** |
| **0.010** | Majority Class Baseline | 81.36% | 50.00% | 0.000 | 0.000 | 0.000 |
| | **Supervised MLP v2** | **89.13%** | **89.74%** | 0.649 | **0.907** | 0.757 |
| | **PyMatching (MWPM)** | **93.61%** | **88.92%** | **0.838** | 0.815 | **0.826** |
| **0.020** | Majority Class Baseline | 69.30% | 50.00% | 0.000 | 0.000 | 0.000 |
| | **Supervised MLP v2** | **73.06%** | **76.81%** | 0.538 | **0.865** | 0.664 |
| | **PyMatching (MWPM)** | **81.07%** | **77.20%** | **0.700** | 0.672 | **0.685** |

> **Observation:** MLP v2 exhibits superior recall across all noise regimes, sacrificing precision compared to MWPM. As physical noise scales to $p=0.02$, both decoders degrade as multi-qubit error chains exceed the code distance threshold.

---

### Benchmark ($d=3, r=3, p=0.01$)

The benchmark (evaluated on 2,999 stratified validation samples):

| Decoder | Input Feature Dimension | Joint Success (Mean $\pm$ SD) | Logical Success (Mean $\pm$ SD) | Class 0 Acc. | Class 1 Acc. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyMatching** | 24 bits (Full) | — | **92.63%** | 95.60% | 79.86% |
| **Supervised MLP** | 24 bits (Full) | — | **90.36%** | 94.86% | 71.03% |
| **RL DQN (Natural)** | 20 bits (Partial) + State | **55.48% $\pm$ 1.18%** | **76.66% $\pm$ 1.12%** | 78.41% | 69.13% |

*Note: PyMatching and Supervised MLP report logical success from a single deterministic model baseline. RL metrics represent empirical mean and sample standard deviation across 8 independent training seeds.*

---

### RL Sampling Strategy Analysis

Seven training-distribution rebalancing strategies evaluated across 8 seeds on 2,999 validation samples:

| Strategy (Class 0 : Class 1) | Joint Success (Mean $\pm$ SD) | Logical Success (Mean $\pm$ SD) |
| :--- | :---: | :---: |
| **Natural (~81:19)** | **55.48% $\pm$ 1.18%** | **76.66% $\pm$ 1.12%** |
| **80 : 20** | 54.39% $\pm$ 3.33% | 76.93% $\pm$ 2.43% |
| **75 : 25** | 53.52% $\pm$ 2.51% | 74.67% $\pm$ 2.39% |
| **70 : 30** | 51.30% $\pm$ 2.84% | 73.15% $\pm$ 2.54% |
| **65 : 35** | 50.98% $\pm$ 3.20% | 71.05% $\pm$ 4.22% |
| **60 : 40** | 48.53% $\pm$ 2.46% | 67.85% $\pm$ 2.29% |
| **50 : 50 (Balanced)** | 42.34% $\pm$ 3.39% | 60.43% $\pm$ 4.80% |

---

## 5. Statistical Analysis

To evaluate whether deliberate oversampling of minority-class error configurations improved decoding performance:
- **Hypothesis Test:** Exact two-sided paired sign-flip permutation test comparing each alternative sampling strategy against `natural` across the $N=8$ paired training replicates.
- **Familywise Error Rate Control:** Holm step-down procedure ($\alpha = 0.05$) across $M = 24$ primary comparisons (overall, class 0, class 1, macro).
- **Permutation Space:** $2^8 = 256$ sign permutations per test.
- **Outcome:** **0 of 24 comparisons** demonstrated statistically significant improvement over the natural sampling distribution after Holm correction.
- **Conclusion:** Artificially shifting training distribution toward balanced sampling (50:50) systematically degrades overall and joint success by distorting the agent's prior over the true physical error distribution.

---

## 6. Limitations

1. **Information Asymmetry (Observability Gap):**
   - PyMatching and Supervised MLP observed the complete **24-bit detector vector** (including the 4 final-round data-qubit measurement detectors).
   - The RL DQN agent operated under **partial observability**, receiving only the **20 historical detectors** plus internal state. The lower logical success (76.7% vs 92.6%) is primarily driven by this missing round of terminal syndrome information.
2. **Endpoint Divergence (Joint vs. Logical Success):**
   - Standard benchmark metrics report *logical success* (whether the logical observable $L$ was preserved).
   - The primary RL objective was *joint success* (simultaneously resolving all residual detector syndromes and preserving $L$). Achieving correct logical parity while leaving residual syndromes constitutes a joint failure.
3. **Unit of Replication:**
   - Multi-seed evaluations treat the **8 training replicates** as the unit of replication, avoiding pseudo-replication from individual validation samples.

---

## 7. Repository Structure

```text
├── data/
│   ├── train/                         # Training datasets at p=0.005, 0.01, 0.02 (.npz)
│   └── test/                          # Held-out test datasets (.npz)
├── datasets/
│   ├── surface_code_d3_r3_p001.npz    # Standard frozen distance-3 dataset (20,000 shots)
│   ├── surface_code_d3_r3_p001_splits.npz # Stratified train/val/test split indices
│   └── *_metadata.json                # JSON schemas and split provenance
├── models/
│   └── mlp_v2_best.pt                 # Optimized PyTorch weights for MLP v2
├── notebooks/
│   ├── data_generation_first_mlp.ipynb   # Circuit design in Stim & initial MLP
│   ├── mlp_v2_vs_pymatching.ipynb       # Supervised MLP vs PyMatching comparison
│   ├── RL_agent_.ipynb                  # Initial RL environment exploration
│   ├── RL_decoder_v2.ipynb              # QEC feedback environment definition
│   ├── RL_decoder_vectorized.ipynb      # Vectorized GPU environment & DQN implementation
│   ├── final_test_evaluation.ipynb      # Final held-out test evaluation suite
│   └── phase2_freeze_20261104/          # Frozen validation benchmark artifacts
│       ├── PHASE2_FINAL_REPORT.md       # Comprehensive frozen benchmark report
│       ├── manifest_sha256.csv          # SHA-256 data integrity hashes
│       └── phase2_comparison_summary.csv# Tabulated replicate statistics
└── results/
    └── d3_decoder_comparison.csv       # Summary metrics across noise rates
```


## 8. Setup & Reproducibility

### Install dependencies
```bash pip install stim pymatching torch numpy scipy pandas scikit-learn matplotlib```

### Reproducing Benchmark Tables
1. **Supervised & Baseline Results:**
   Execute [notebooks/mlp_v2_vs_pymatching.ipynb](notebooks/mlp_v2_vs_pymatching.ipynb) to reproduce the $p \in \{0.005, 0.01, 0.02\}$ comparison table.
2. **RL Agent Training & Vectorized Environment:**
   Run [notebooks/RL_decoder_vectorized.ipynb](notebooks/RL_decoder_vectorized.ipynb) to inspect environment step dynamics and DQN convergence curves.
3. **Audit Frozen Benchmark Integrity:**
   Verify files in `notebooks/phase2_freeze_20261104/` against `manifest_sha256.csv`:
   ```bash
   cd notebooks/phase2_freeze_20261104
   sha256sum -c manifest_sha256.csv
   ```
