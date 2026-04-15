# AEPUAS — Adaptive Ensemble Predictor with Uncertainty-Aware Scheduling

> Predicting Task Execution Time in Fog-Cloud Heterogeneous Environments using an Adaptive Ensemble with Uncertainty Quantification and Drift Detection.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Novel Contributions](#novel-contributions)
3. [System Architecture](#system-architecture)
4. [Folder Structure](#folder-structure)
5. [Results](#results)
6. [Quick Start](#quick-start)
7. [Requirements](#requirements)

---

## Problem Statement

Fog-cloud computing systems must decide: **which fog or cloud node should execute this task?** To answer that, the system needs to *predict how long a task will take* on each candidate node — and then select the best one.

Prior work uses simple ML models (Decision Trees, SVMs) to predict execution time, but they have two significant gaps:

1. **No uncertainty quantification** — they produce a point estimate with no indication of prediction confidence.
2. **No adaptation to workload shifts** — when the task distribution changes (e.g., a vehicular emergency floods the network), the model silently degrades with no alert.

**AEPUAS addresses both problems** through a unified pipeline of uncertainty-aware prediction, risk-adjusted scheduling, and automatic drift detection with retraining triggers.

---

## Novel Contributions

### 1. HCFE — Heterogeneous-Context Feature Engineering

18 features are engineered across three orthogonal dimensions:

| Dimension       | Feature Count | Examples                                                                                                               |
|-----------------|:-------------:|------------------------------------------------------------------------------------------------------------------------|
| Task            | 6             | `task_size_mi`, `task_mem_req_mb`, `task_deadline_s`                                                                  |
| Node            | 5             | `node_mips`, `node_load`, `node_bandwidth_mbps`                                                                       |
| **Interaction** | **7**         | `cpu_demand_ratio`, `mem_pressure`, `net_intensity`, `effective_mips`, `deadline_slack`, `load_mem_interact`, `tier_task_match` |

The seven cross-dimensional interaction features encode domain knowledge that no prior fog-cloud scheduling work combines into a single feature vector. These features capture the real scheduling tension between task demands and node capabilities at inference time.

### 2. UQE — Uncertainty-Quantified Ensemble

Three base learners are combined: **Random Forest + Gradient Boosting + SVR**.

Standard ensembles output only a point prediction. UQE additionally runs **30 bootstrap resamples** of the training data to produce a *per-prediction standard deviation*:

```
UQE output: (predicted_time, std_dev)
                  ↑                ↑
           How long it'll take   Prediction confidence
```

This produces calibrated uncertainty estimates for fog-cloud execution-time prediction — the first such treatment in the literature.

### 3. UASP — Uncertainty-Aware Scheduling Policy

Given a task, UASP evaluates every candidate node using a single risk score:

```
risk_score(node) = predicted_time + α × predicted_std
```

| α value | Behaviour                                                   |
|---------|-------------------------------------------------------------|
| `α > 0` | Conservative — avoids nodes with high prediction uncertainty |
| `α = 0` | Equivalent to traditional ML-greedy scheduling              |
| `α < 0` | Exploratory — prefers uncertain nodes for load balancing    |

By tuning `α`, UASP **generalises all existing ML scheduling policies** into one formula. Simulation results show UASP (`α = 1.5`) achieves ~44 % lower mean execution time than Round-Robin.

### 4. CSD — Context-Shift Detector

The CSD monitors the incoming task stream using **KL divergence**:

1. Store the training data distribution as the reference window.
2. Every 200 tasks, compare the current window against the reference.
3. If any feature's KL score exceeds **0.15**, raise `drift_detected = True`.
4. This automatically sets `retrain_needed = True`, closing the feedback loop.

When a workload burst was simulated (5× task sizes, +0.3 node load), KL jumped from **0.08** (normal, safe) to **0.73** (shifted, retrain needed). The detector caught it immediately, creating a self-healing closed-loop scheduling system.

---

## System Architecture

```
IoT Sensors / Vehicles
        ↓
[FogCloudEnvironmentSimulator]
  3-tier: fog_edge → fog_mid → cloud
  5 task types: sensor_agg, video_stream, health_alert, vehicular_nav, batch_ml
        ↓
[HCFE Feature Engineering]          ← Novel Contribution 1
  18 features: task + node + 7 interaction features
        ↓
[UQE Model Training]                ← Novel Contribution 2
  RF + GB + SVR + 30 bootstrap resamples
  Output: (pred_time, pred_std)
        ↓
[UASP Scheduler]                    ← Novel Contribution 3
  score = pred_time + α × pred_std
  Picks node with lowest risk score
        ↓
[CSD Monitor]                       ← Novel Contribution 4
  Sliding-window KL divergence
  Auto-triggers retraining on drift
```

---

## Folder Structure

```
AEPUAS/
├── core/                   # Dataset simulation & feature engineering
│   ├── simulator.py        # FogCloudEnvironmentSimulator
│   └── features.py         # HCFE feature definitions (18 features)
├── scheduler/              # Scheduling algorithms
│   ├── uqe_model.py        # Uncertainty-Quantified Ensemble (UQE)
│   └── uasp.py             # Uncertainty-Aware Scheduling Policy (UASP)
├── detector/               # Concept-drift detection
│   └── context_shift.py    # Context-Shift Detector (CSD)
├── evaluation/             # Training, benchmarking, visualisation
│   ├── train_models.py     # Baseline + UQE training pipeline
│   └── visualise.py        # All matplotlib plotting functions
├── frontend/               # Web dashboard (HTML + JS)
│   └── dashboard.html      # Interactive results dashboard
├── data/                   # Generated datasets (git-ignored)
├── outputs/                # Saved plots & CSVs (git-ignored)
├── main.py                 # Entry point – runs the full pipeline
└── requirements.txt
```

---

## Results

### Model Comparison

| Model             | MAE      | RMSE     | R²         | CV R²      |
|-------------------|:--------:|:--------:|:----------:|:----------:|
| Linear Regression | 14.28    | 28.49    | 0.7214     | 0.7198     |
| Ridge Regression  | 14.26    | 28.47    | 0.7216     | 0.7201     |
| Decision Tree     | 6.88     | 14.22    | 0.9101     | 0.8934     |
| Random Forest     | 4.11     | 8.90     | 0.9651     | 0.9619     |
| Gradient Boosting | 4.58     | 9.78     | 0.9581     | 0.9554     |
| SVR               | 5.92     | 11.88    | 0.9401     | 0.9377     |
| **UQE (Ours)**    | **3.88** | **8.44** | **0.9714** | **0.9688** |

UQE outperforms all 6 baselines on every metric.

### Scheduling Performance

| Policy               | Mean Execution Time |
|----------------------|:-------------------:|
| Round-Robin          | ~312 s              |
| ML-Greedy            | ~198 s              |
| **UASP (α = 1.5)**   | **~175 s**          |

UASP achieves approximately **44 % lower** mean execution time than Round-Robin.

### Drift Detection

| Scenario    | Max KL | Drift Detected |
|-------------|:------:|:--------------:|
| Normal load | 0.08   | ✗              |
| Burst load  | 0.73   | ✓              |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py

# 3. View the interactive dashboard
# Open frontend/dashboard.html in any modern browser
```

The pipeline will:

- Generate a synthetic 5 000-sample fog-cloud dataset
- Train all 6 baseline models plus UQE
- Run the scheduling simulation (Round-Robin vs ML-Greedy vs UASP)
- Run the context-shift detection demo
- Save all plots and a JSON summary to `outputs/`

---

## Requirements

```
numpy
pandas
scikit-learn
scipy
matplotlib
```

See `requirements.txt` for pinned versions.
