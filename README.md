# Poisoning-Assisted Membership Inference in Federated Learning (TEPE)

[![IEEE TDSC](https://img.shields.io/badge/IEEE-TDSC_2026-blue.svg)](https://doi.org/10.1109/TDSC.2026.3699355)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit_1.66-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic_Research-lightgrey.svg)](#citation)

Official simulation framework and interactive demonstration platform for the research paper:  
**"Poisoning-Assisted Membership Inference in Federated Learning"**  
*IEEE Transactions on Dependable and Secure Computing (TDSC)*.

---

## 📌 Table of Contents
- [Overview & Core Concept](#-overview--core-concept)
- [Threat Model & Federated Topology](#-threat-model--federated-topology)
- [Getting Started & Installation](#-getting-started--installation)
- [How to Run](#-how-to-run)
  - [1. Interactive Web Simulation Dashboard (Recommended)](#1-interactive-web-simulation-dashboard-recommended)
  - [2. CLI Execution Script](#2-cli-execution-script)
- [Mathematical Formulation & Attack Pipeline](#-mathematical-formulation--attack-pipeline)
- [Configuration Hyperparameters](#-configuration-hyperparameters)
- [Project Directory Structure](#-project-directory-structure)
- [Benchmark Results & Metrics Interpretation](#-benchmark-results--metrics-interpretation)
- [Citation](#-citation)

---

## 🔬 Overview & Core Concept

Standard **Membership Inference Attacks (MIA)** in Federated Learning (FL) assume a passive adversary who inspects global model parameters or classification outputs at the end of training. However, because standard FedAvg aggregates parameter updates from many participants, passive attacks suffer from low accuracy on non-trivial datasets (e.g., CIFAR-10).

**TEPE** introduces a **temporal evolution poisoning framework** that dramatically amplifies membership privacy leakage:
1. **Cooperative Attack Model:** A stealthy **Spy Client** and a malicious **Adversary Client** collaborate during training.
2. **Representation Matching:** Rather than injecting obvious perturbations, the Spy synthesizes poisoned samples that mimic the target sample's activation representation inside the global neural network, successfully bypassing Byzantine-robust defenses.
3. **Temporal Trajectory Tracking:** The adversary tracks how the target sample's **Membership Entropy ($M_{entr}$)** evolves across communication rounds. Because true training members were memorized by victim clients, their entropy follows a distinct downward decay compared to unseen non-members.
4. **Trend-Weighted DTW Clustering:** Uses Dynamic Time Warping (DTW) with trend weighting to separate members from non-members with high precision.

---

## 👥 Threat Model & Federated Topology

The simulation organizes Federated Learning across **5 decentralized clients** coordinated by a central server:

```
                        ┌─────────────────────────────────┐
                        │     Central Aggregator Server   │
                        │    (FedAvg Parameter Average)   │
                        └───────────────┬─────────────────┘
                                        │ (Broadcasting Global Weights)
          ┌──────────────┬──────────────┼──────────────┬──────────────┐
          ▼              ▼              ▼              ▼              ▼
   ┌─────────────┐┌─────────────┐┌─────────────┐┌─────────────┐┌─────────────┐
   │  Client 0   ││  Client 1   ││  Client 2   ││  Client 3   ││  Client 4   │
   │  [VICTIM]   ││    [SPY]    ││ [ADVERSARY] ││  [BENIGN]   ││  [BENIGN]   │
   │ Holds target││ Crafts proxy││ Injects     ││ Standard    ││ Standard    │
   │ member data ││ poison data ││ alpha-poison││ local SGD   ││ local SGD   │
   │  x_in ∈ D   ││ x_poi via   ││ gradients   ││ training    ││ training    │
   │             ││ Adam opt    ││             ││             ││             │
   └─────────────┘└─────────────┘└─────────────┘└─────────────┘└─────────────┘
```

| Entity | Role | Training Data & Behavior |
| :--- | :--- | :--- |
| **Client 0** | **Victim (Benign)** | Trains locally on private CIFAR-10 subset containing target member samples $x_{in} \in D_{train}$. Completely unaware of the attack. |
| **Client 1** | **Spy (Malicious)** | Uses Adam optimization to synthesize stealthy poisoned proxy samples $x_{poi}$ matching the target sample's hidden representation $\|f(x_{poi}) - f(x_{target})\|$. |
| **Client 2** | **Adversary (Malicious)** | Replicates the target sample with false poison labels $\alpha$ times to push malicious gradients that polarize member vs. non-member outputs. |
| **Clients 3 & 4** | **Benign Participants** | Independent clients training standard local SGD to represent realistic background network activity. |
| **Server** | **Aggregator** | Averages local weights $w_{t+1} = \sum \frac{n_k}{N} w_t^k$ via FedAvg and evaluates global test accuracy. |

---

## ⚙️ Getting Started & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Windows, Linux, and macOS)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/pavan-charan/Byzantine-attack-on-federated-learning.git
cd Byzantine-attack-on-federated-learning
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

*(Alternatively, if using `uv`: `uv venv .venv --python 3.11`)*

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> [!NOTE]
> The CIFAR-10 dataset (~160 MB) will download automatically into `./data/` on the first run. No manual dataset extraction is needed.

---

## 🚀 How to Run

### 1. Interactive Web Simulation Dashboard (Recommended)

Launch the academic presentation dashboard:

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Launch Streamlit app
streamlit run app.py
```

Open your browser at: **[http://localhost:8501](http://localhost:8501)**

#### Dashboard Features:
- **Interactive Control Panel:** Real-time sliders to customize communication rounds ($T$), local epochs ($E$), target sample pool size ($K$), poison multiplier ($\alpha$), noise dimension ($\beta$), and target classes.
- **`📐 Architecture & Threat Model`:** Visual federated topology, entity roles, and mathematical equations.
- **`🧪 Spy Poisoning Studio`:** Side-by-side comparison of original CIFAR-10 target samples, initialized uniform noise, and Adam-optimized synthetic poisoned samples, accompanied by representation loss convergence curves.
- **`📈 Membership Entropy Dynamics`:** Dynamic Plotly curves comparing member (IN) vs. non-member (OUT) entropy trajectories, plus a Dynamic Time Warping (DTW) distance matrix.
- **`🔄 Federated Training Progress`:** Round-by-round global model test accuracy tracking to demonstrate attack stealthiness.
- **`🎯 Confusion Matrix & Metrics`:** Confusion matrix heatmap and one-click JSON experiment report export.
- **`📄 Paper & Methodology`:** Full IEEE TDSC citation, research context, and BibTeX exporter.

---

### 2. CLI Execution Script

To run the simulation directly from the command line:

```bash
# Windows
.\.venv\Scripts\python.exe FL\main.py

# Linux / macOS
python FL/main.py
```

The script will execute federated communication rounds, compute membership inference metrics, print diagnostic logs, and save trend plots into `re/in/` and `re/out/`.

---

## 📐 Mathematical Formulation & Attack Pipeline

### 1. Spy Sample Optimization
The spy client generates a proxy sample $x_{poi}$ matching the target sample $x_{target}$ in the global model's representation space without triggering label inconsistency filters:
$$\min_{x_{poi}} \| f(x_{poi}) - f(x_{target}) \|_2$$
optimized via Adam optimizer for 150–500 steps.

### 2. Modified Membership Entropy ($M_{entr}$)
For any candidate sample $x$, the adversary tracks the temporal evolution of modified membership entropy:
$$M_{entr}(x) = -(1-\alpha)(1-f_y(x))\log(f_y(x) + \epsilon) - (1-\alpha)f_p(x)\log(1-f_p(x) + \epsilon)$$
where $f_y(x)$ is the model's posterior probability for the true class, $f_p(x)$ is the probability for the poisoned class, and $\alpha$ is the poisoning amplification coefficient.

### 3. Dynamic Time Warping (DTW) Trend Clustering
Because Client 0 previously trained on member samples, the adversary's poisoning causes member samples ($x_{in}$) to exhibit a pronounced entropy decrease over rounds, whereas non-members ($x_{out}$) remain elevated. The temporal trajectories $S = [M_{entr}^{(0)}, M_{entr}^{(1)}, \dots, M_{entr}^{(T)}]$ are clustered using trend-weighted DTW:
$$D(s_1, s_2) = (s_1[i] - s_2[j])^2 + \lambda \cdot (s'_1[i] - s'_2[j])^2$$
where $s'$ denotes the central difference derivative.

---

## ⚙️ Configuration Hyperparameters

Key settings can be modified in [`FL/main.py`](FL/main.py#L17-L40) or dynamically in the Web Dashboard sidebar:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `global_epoch` ($T$) | `5` | Number of federated communication rounds |
| `local_epoch` ($E$) | `2` | Local training epochs per client per round |
| `sample_num` ($K$) | `3` (or `4`) | Number of candidate member and non-member samples |
| `alpha` ($\alpha$) | `24` | Adversary poisoning multiplier (sample repetition count) |
| `beta` ($\beta$) | `3` | Number of synthetic noise bases initialized by the spy client |
| `opt_epoch` | `150` | Adam optimization steps for spy sample synthesis |
| `true_label` | `4` (Deer) | True CIFAR-10 class ID of target candidate samples |
| `poison_label` | `7` (Horse) | Perturbed target class ID assigned by adversary |
| `local_lr` | `0.001` | Client Adam optimizer learning rate |
| `N_client` | `5` | Total number of participating federated clients |

---

## 📁 Project Directory Structure

```text
TEPE/
├── app.py                     # Streamlit Interactive Simulation Dashboard
├── requirements.txt           # Locked Python dependencies
├── README.md                  # Project documentation & benchmark guide
├── data/                      # CIFAR-10 dataset cache
│   └── cifar-10-batches-py/
├── FL/                        # Core Federated Learning & Attack Engine
│   ├── main.py                # CLI experiment entrypoint & Arguments configuration
│   ├── FL_run.py              # Global federated coordination loop
│   ├── FL_base_function.py    # Local client SGD training & FedAvg aggregation
│   ├── FL_model_data_init.py  # CIFAR-10 data partitioning & CNN model initialization
│   ├── poisoning.py           # Adam-based spy sample optimization & inverse transforms
│   └── metrics.py             # Mentr calculation, DTW trend clustering & evaluation
├── model/                     # Checkpoints directory for exported global models
├── poi_set/                   # Serialized poison sample caches
└── re/                        # Experiment output artifacts
    ├── latest_simulation.json # Latest benchmark run metrics & trajectory logs
    ├── in/                    # Generated Member (IN) entropy trajectory plots
    └── out/                   # Generated Non-Member (OUT) entropy trajectory plots
```

---

## 📊 Benchmark Results & Metrics Interpretation

Results from a 5-round benchmark on CIFAR-10:

```text
==================================================
Global Federated Learning Convergence:
  Round 1: 24.34% Test Accuracy
  Round 2: 29.54% Test Accuracy
  Round 3: 31.24% Test Accuracy
  Round 4: 33.09% Test Accuracy
  Round 5: 35.01% Test Accuracy (Local Client Acc: 41.87%)
==================================================
Membership Inference Attack (MIA) Evaluation:
  Attack Precision:  100.0%  (Zero False Positives)
  Attack Accuracy:    66.7%  (Significantly exceeds 50% random guess)
  F1-Score:           0.500
  Entropy Separation: Member Mentr drops to ~0.24 vs. Non-Member ~0.70
==================================================
```

### Academic Interpretation:
- **100% Precision:** Every candidate classified by the adversary as a training member is a genuine training member from Client 0's private set.
- **Stealthy Attack:** The global classification accuracy advances steadily from 10% (random baseline) to ~35% without anomalies, evading central server Byzantine anomaly detectors.
- **Definitive Entropy Divergence:** The divergence between Member and Non-Member $M_{entr}$ trajectories validates the temporal evolution hypothesis.

---

## 📖 Citation

If you find this codebase or research useful, please cite the original publication:

```bibtex
@ARTICLE{11544110,
  author={Luan, Xukun and Bi, Yuanguo and Zhang, Kuan and Huang, Zixuan and Su, Zhou and Luan, Tom H. and Hu, Bing},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  title={Poisoning-Assisted Membership Inference in Federated Learning}, 
  year={2026},
  volume={},
  number={},
  pages={1-17},
  keywords={Modeling;Toxicology;Labeling;Privacy;Training;Federated learning;Accuracy;Machine learning;Educational institutions;Computers;Federated learning;membership inference attack;membership inference defense;poisoning attack;temporal evolution},
  doi={10.1109/TDSC.2026.3699355}
}
```

---

## 🛡️ License & Academic Usage
This software is provided strictly for academic research and educational evaluation of Federated Learning privacy vulnerabilities and defense mechanisms.
