# Dynamic Cost-Sensitive Adaptive Random Forest (Dynamic CS-ARF) for Data Streams

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The **Dynamic Cost-Sensitive Adaptive Random Forest (Dynamic CS-ARF)** is an online machine learning ensemble designed specifically to tackle the dual challenges of extreme class imbalance and concept drift in continuous financial data streams.

By integrating an Exponential Moving Average (EMA) to track real-time Imbalance Ratio ($IR_t$) and an ADWIN detector to track Drift Confidence ($D_t$), Dynamic CS-ARF dynamically modulates an asymmetric Poisson bagging penalty for minority class instances. 

**v2 Update (Hybrid Online Oversampling):** The algorithm has been updated to include a conservative $\lambda_t$ cap ($\Lambda_{max}$), synthetic diversification via a small minority window queue, and self-regulating rolling-Precision feedback to prevent over-penalization. This allows Dynamic CS-ARF to remain strictly competitive with full memory-intensive SMOTE implementations while retaining a lightweight, single-pass streaming architecture.

## Features
- **Dynamic Hybrid Online Oversampling:** Mathematically oversamples rare fraud instances in real-time via dynamic Poisson weight ($\lambda_t$) and synthetic diversification.
- **Precision Feedback Damping:** Self-regulates the bagging penalty if rolling Precision drops below a tuned threshold.
- **Strict Single-Pass Execution:** Operates efficiently over streaming data without requiring massive historical buffers.
- **Robust against Concept Drift:** Replaces obsolete background trees when prequential error spikes.

## Repository Structure

```text
CS-ARF-StreamFraud/
│
├── data/                   # Directory for raw datasets and processed metrics (ignored in git)
├── src/
│   ├── preprocess_datasets.py        # Scripts to clean and format raw datasets
│   ├── stream_utils.py               # Utility functions for streaming evaluations
│   ├── run_benchmark.py              # Main benchmarking script vs. SMOTE-Window, CSARF-MCC, UOB, ARF
│   ├── run_synth_benchmark.py        # Evaluation on synthetic streams (Agrawal, SEA, Hyperplane)
│   ├── run_ablation.py               # Ablation study on lambda cap and diversification
│   ├── run_drift_analysis.py         # Analysis of recall recovery during explicit concept drift
│   ├── run_delay_experiment.py       # Prequential evaluation with verification latency
│   ├── run_throughput_experiment.py  # TPS processing speed benchmarks
│   ├── run_pr_curve.py               # Precision-Recall trajectory plotting
│   ├── statistical_test.py           # Friedman tests and Nemenyi post-hoc analysis
│   └── pipeline.py                   # Central execution pipeline
│
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

## Prerequisites & Installation

We recommend using a virtual environment. The codebase heavily relies on the [`River`](https://riverml.xyz/) framework for online machine learning.

```bash
# Clone the repository
git clone https://github.com/muhammadratsanjani/Dynamic-CS-ARF-StreamFraud.git
cd Dynamic-CS-ARF-StreamFraud

# Create and activate virtual environment (Windows)
python -m venv env
.\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Access

Due to size constraints and licensing, the raw datasets are not included in this repository. To fully reproduce the experiments across our 20 tested streams, please download the primary datasets and place them in the `data/raw/` directory:

1. **IEEE-CIS Fraud Detection (2019):** Available on [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection).
2. **PaySim (Mobile Money Simulation):** Available on [Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1).
3. **ULB Credit Card Fraud (2013):** Available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
4. **BankSim:** Available on Kaggle.

After downloading, run the preprocessing script to format them for streaming input:
```bash
python src/preprocess_datasets.py
```

## How to Run the Experiments

To run the main Interleaved Test-Then-Train (Prequential) benchmarking against standard algorithms and modern cost-sensitive baselines:

```bash
python src/run_benchmark.py
```

To run the ablation study evaluating the isolated impact of synthetic diversification and the lambda cap:
```bash
python src/run_ablation.py
```

To test the algorithmic response to verification delay (latency):
```bash
python src/run_delay_experiment.py
```

To generate the non-parametric statistical significance tests (Friedman Test & Nemenyi CD):
```bash
python src/statistical_test.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
