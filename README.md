# SHADE: Scalable Heterogeneity-Aware Deep Reinforcement Learning for Edge-Cloud Task Scheduling

<p align="center">
  <img src="docs/figures/architecture.png" alt="SHADE Architecture" width="700"/>
</p>

<p align="center">
  <a href="https://doi.org/10.XXXX/XXXX"><img src="https://img.shields.io/badge/Paper-Published-blue" alt="Paper"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"/></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Status-Reproducible-brightgreen" alt="Reproducible"/>
</p>

> **Scalable Heterogeneity-Aware Deep Reinforcement Learning for Task Scheduling in Heterogeneous Edge-Cloud Environments: Design, Architecture, Benchmarking, and Real-World Validation**
>
> *[Padala sravan] 

---

## Overview

SHADE is a hierarchical Deep Reinforcement Learning framework for task scheduling across heterogeneous IoT–Edge–Cloud deployments spanning four qualitatively distinct device classes (Class A microcontrollers through Class D cloud VMs). It is the first DRL scheduler to encode device-class heterogeneity as a first-class MDP state component.

### Key Results

| Metric | SHADE | Best Baseline | Improvement |
|--------|-------|--------------|-------------|
| Average Task Latency | 18.4 ± 0.5 ms | 22.6 ms (A3C) | **−41.2% vs FIFO** |
| Energy Consumption | 3.44 ± 0.09 kWh | 3.52 kWh (A3C) | **−29.8% vs FIFO** |
| Task Completion Rate | 94.7% | 92.8% (A3C) | **+12.6 pp vs FIFO** |
| SLA Violation Rate | 3.2% | 4.8% (A3C) | **−66% vs FIFO** |
| Task-Node Affinity | 0.89 | 0.71 (A3C) | **+0.18 over A3C** |
| Overhead Scaling (β) | **0.21** (near-linear) | 1.48 (A3C) | **15.2× less at 2,000 nodes** |
| Transfer Speedup | **5.12×** faster | — | vs full retraining |

All results: 30 independent trials, 95% bootstrapped CI, Bonferroni-corrected paired t-tests, Cohen's d effect sizes.

---

## Architecture

SHADE has three core contributions:

1. **Device-Class-Aware State Encoding** — class-relative normalization + learned 8-dim class embeddings + affinity-sorted PCA-compressed node representation
2. **Heterogeneity-Weighted Multi-Objective Reward** — five-term reward including novel cosine-distance Mismatch Penalty + dynamic Pareto weight adjustment
3. **Layer-Selective Transfer Learning** — freeze lower layers, fine-tune class embeddings + output head, EMA weight merge (α = 0.1)

```
IoT Tasks → [Meta-Agent: FC(128)×2] → cluster k* → [Sub-Agent: FC(256-512-256)+LSTM(128)] → node j*
                 ↕ sparse gradient sharing (top-32)
            [Class Embedding (8-dim)] + [PCA State (N×d_pca)] + [Affinity Sort]
```

---

## Repository Structure

```
shade/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── configs/
│   └── default.yaml             # All hyperparameters (Table 5 from paper)
├── shade/
│   ├── __init__.py
│   ├── agents/
│   │   ├── meta_agent.py        # Global meta-agent (FC 128×2)
│   │   ├── sub_agent.py         # Local sub-agent (FC 256-512-256 + LSTM 128)
│   │   └── base_agent.py        # Shared DQN logic, PER, target network
│   ├── encoding/
│   │   ├── state_encoder.py     # Device-class-aware state encoding (Section 4.2)
│   │   ├── class_embedding.py   # Learned 8-dim class embeddings
│   │   └── pca_compressor.py    # PCA compression for N > 50 nodes
│   ├── reward/
│   │   ├── reward_function.py   # Five-term reward (Eq. 11–15)
│   │   └── pareto_weights.py    # Dynamic Pareto weight update (Eq. 16)
│   ├── transfer/
│   │   ├── transfer_agent.py    # Layer-selective fine-tuning (Algorithm 3)
│   │   └── ema_merge.py         # Exponential Moving Average weight merge
│   ├── communication/
│   │   └── gradient_sharing.py  # Top-k sparse gradient sharing (Section 4.5)
│   └── utils/
│       ├── replay_buffer.py     # Prioritized Experience Replay
│       ├── statistics.py        # Bootstrap CI, Cohen's d, Bonferroni correction
│       └── metrics.py           # All 7 evaluation metrics (Table 2)
├── envs/
│   ├── edge_cloud_env.py        # iFogSim 2.0-compatible Gym environment
│   ├── node.py                  # Hardware node simulation (Classes A–D)
│   ├── task_generator.py        # Poisson task generator (λ = 5 tasks/s)
│   └── workload/
│       ├── synthetic_het.py     # Synthetic-HET benchmark
│       ├── google_cluster.py    # Google Cluster 2011 trace adapter
│       └── azure_functions.py   # Azure Functions 2023 trace adapter
├── baselines/
│   ├── classical/
│   │   ├── fifo.py
│   │   ├── round_robin.py
│   │   ├── min_min.py
│   │   └── greedy_energy.py
│   └── drl/
│       ├── ddqn.py              # DDQN [Amini & Kalbasi, 2024]
│       ├── ppo.py               # PPO [Zhan et al., 2020]
│       └── a3c.py               # A3C+R2N2 [Tuli et al., 2022]
├── datasets/
│   ├── README.md                # Dataset descriptions and download instructions
│   ├── synthetic_het/
│   │   └── generate.py          # Generator for Synthetic-HET benchmark
│   ├── google_cluster/
│   │   └── preprocess.py        # Google Cluster 2011 → τᵢ mapping
│   └── azure_functions/
│       └── preprocess.py        # Azure Functions 2023 → τᵢ mapping
├── experiments/
│   ├── train_shade.py           # Main training script
│   ├── evaluate.py              # Evaluation against all baselines
│   ├── ablation.py              # Ablation study (Section 7.6)
│   ├── scalability.py           # Scalability analysis (Section 7.4)
│   └── transfer_learning.py     # Transfer learning experiment (Section 7.5)
├── scripts/
│   ├── run_all_experiments.sh   # Reproduce all paper results
│   ├── run_ablation.sh
│   └── download_datasets.sh     # Download Google Cluster + Azure traces
├── results/
│   ├── README.md
│   └── pretrained/
│       ├── shade_synthetic_het.pt    # Pretrained weights (Synthetic-HET)
│       ├── shade_google_cluster.pt   # Pretrained weights (Google Cluster)
│       └── shade_azure_functions.pt  # Pretrained weights (Azure Functions)
├── tests/
│   ├── test_state_encoder.py
│   ├── test_reward.py
│   ├── test_transfer.py
│   └── test_baselines.py
└── docs/
    ├── figures/                 # All paper figures (reproducible)
    ├── REPRODUCIBILITY.md       # Step-by-step reproduction guide
    └── HARDWARE.md              # Physical testbed setup
```

---

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)

```bash
# Clone repository
git clone https://github.com/[username]/SHADE.git
cd SHADE

# Create virtual environment
python -m venv shade_env
source shade_env/bin/activate   # Linux/Mac
# shade_env\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install SHADE package
pip install -e .
```

---

## Quick Start

### 1. Train SHADE from scratch

```python
from experiments.train_shade import train_shade
from configs import load_config

config = load_config('configs/default.yaml')
model = train_shade(config, dataset='synthetic_het')
```

Or via command line:

```bash
python experiments/train_shade.py \
    --dataset synthetic_het \
    --n_nodes 500 \
    --n_trials 30 \
    --seed 42
```

### 2. Evaluate against all baselines

```bash
python experiments/evaluate.py \
    --checkpoint results/pretrained/shade_synthetic_het.pt \
    --dataset synthetic_het \
    --baselines fifo round_robin min_min greedy_energy ddqn ppo a3c
```

### 3. Run transfer learning experiment

```bash
python experiments/transfer_learning.py \
    --pretrained results/pretrained/shade_synthetic_het.pt \
    --new_class fpga \
    --n_trials 10
```

### 4. Reproduce all paper results

```bash
bash scripts/run_all_experiments.sh
```

---

## Hyperparameters

All hyperparameters are defined in `configs/default.yaml` and match Table 5 of the paper exactly. Key values:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gamma` | 0.99 | Discount factor |
| `lr` | 1e-4 | Adam learning rate |
| `buffer_size` | 100,000 | Replay buffer capacity |
| `batch_size` | 256 | Training batch size |
| `alpha_per` | 0.6 | PER prioritization exponent |
| `tau` | 0.005 | Polyak target network update |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.05 | Final exploration rate |
| `epsilon_decay` | 0.995 | Per-step decay multiplier |
| `t_max` | 50,000 | Training steps |
| `lambda_aux` | 0.1 | Auxiliary affinity loss weight |
| `pca_variance` | 0.95 | PCA variance retention threshold |
| `top_k_gradients` | 32 | Sparse gradient sharing k |
| `embed_dim` | 8 | Class embedding dimension |
| `t_soft` | 1.0 | Inference softmax temperature |

---

## Datasets

### Synthetic-HET (generated locally)
```bash
python datasets/synthetic_het/generate.py --n_nodes 500 --n_tasks 10000 --seed 42
```

### Google Cluster 2011 (download required)
```bash
bash scripts/download_datasets.sh --dataset google_cluster
python datasets/google_cluster/preprocess.py --input data/raw/google_cluster/ --output data/processed/
```

The preprocessing maps Google Cluster fields to SHADE task descriptor τᵢ = ⟨cpu, mem, deadline, priority, size, type⟩. Deadlines are synthetically assigned as d_i = 1.5 × E[T_exec(τᵢ, assigned class)].

### Azure Functions 2023 (download required)
```bash
bash scripts/download_datasets.sh --dataset azure_functions
python datasets/azure_functions/preprocess.py --input data/raw/azure/ --output data/processed/
```

See `datasets/README.md` for full preprocessing details.

---

## Reproducing Paper Results

See `docs/REPRODUCIBILITY.md` for step-by-step instructions. Summary:

| Experiment | Script | Expected Runtime |
|-----------|--------|-----------------|
| Table 5–7 (main results) | `experiments/evaluate.py` | ~4 hours (GPU) |
| Table 8 (scalability) | `experiments/scalability.py` | ~2 hours |
| Table 9 (ablation) | `experiments/ablation.py` | ~8 hours |
| Figure 9 (transfer learning) | `experiments/transfer_learning.py` | ~1 hour |

All scripts use `--seed` for reproducibility and save results to `results/`.

---

## Physical Testbed

The paper validates SHADE on a 4-node physical testbed:

| Class | Device | CPU | RAM / GPU | TDP |
|-------|--------|-----|-----------|-----|
| A | Raspberry Pi 4 Model B | ARM Cortex-A72, 1.8 GHz | 4 GB / None | ~3.4 W |
| B | NVIDIA Jetson Orin NX 16GB | ARM Cortex-A78AE, 12-core | 16 GB / 1024-core Ampere | ~15 W |
| C | Dell PowerEdge R750 | Intel Xeon Gold 5318Y, 24-core | 128 GB / NVIDIA A30 | ~210 W |
| D | AWS c5.4xlarge | Intel Xeon Platinum 8275CL, 16 vCPU | 32 GB / None | Cloud-managed |

Power measured via TP-Link Tapo P115 smart plugs (Classes A, B) and Dell iDRAC embedded sensors (Class C). See `docs/HARDWARE.md` for testbed setup instructions.

---


---

## License

This project is licensed under the MIT License — see [MIT](LICENSE) for details.

---

## Acknowledgements

Simulation platform built on [iFogSim 2.0](https://github.com/Cloudsim-DEWS/ifogsim), [CloudSim++](https://cloudsimplus.org/), and [EdgeDroid](https://github.com/cmusatyalab/EdgeDroid2). Google Cluster trace from [Google Research](https://github.com/google/cluster-data). Azure Functions trace from [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset).
