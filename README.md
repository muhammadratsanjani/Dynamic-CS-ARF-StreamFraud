# Cost-Sensitive Adaptive Random Forest (CS-ARF) for Data Streams

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The **Dynamic Cost-Sensitive Adaptive Random Forest (Dynamic CS-ARF)** is an online machine learning ensemble designed specifically to tackle the dual challenges of extreme class imbalance and concept drift in continuous financial data streams.

By integrating an Exponential Moving Average (EMA) to track real-time Imbalance Ratio ($IR_t$) and an ADWIN detector to track Drift Confidence ($D_t$), Dynamic CS-ARF dynamically modulates an asymmetric Poisson bagging penalty for minority class instances.

The full theoretical formulation, implementation details, and empirical benchmarking results against state-of-the-art baselines on six industrial datasets can be found in our comprehensive manuscript: [**main_manuscript.pdf**](./main_manuscript.pdf).

## Features
- **Dynamic Implicit Online Oversampling:** Mathematically oversamples rare fraud instances in real-time via dynamic Poisson weight ($\lambda_t$).
- **Strict Single-Pass Execution:** Requires no data buffering or physical minority class duplication, operating in strict $O(1)$ memory per instance.
- **Robust against Concept Drift:** Replaces obsolete background trees when prequential error spikes.

## Repository Structure

```text
CS-ARF-StreamFraud/
│
├── data/                   # Directory to place downloaded datasets (ignored in git)
├── notebooks/              # Jupyter notebooks for exploratory data analysis
├── src/
│   ├── preprocess_datasets.py    # Scripts to clean and format raw datasets
│   ├── stream_utils.py           # Utility functions for streaming evaluations
│   ├── run_benchmark.py          # Main script to run Prequential evaluation across algorithms
│   ├── run_ablation.py           # Script to run sensitivity analysis on beta (cost) parameters
│   ├── statistical_test.py       # Script to compute Friedman tests and post-hoc analysis
│   ├── pipeline.py               # Central execution pipeline
│   └── recalculate_metrics.py    # Auxiliary script to re-evaluate specific logs
│
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

## Prerequisites & Installation

We recommend using a virtual environment. The codebase heavily relies on the [`River`](https://riverml.xyz/) framework for online machine learning.

```bash
# Clone the repository
git clone https://github.com/muhammadratsanjani/CS-ARF-StreamFraud.git
cd CS-ARF-StreamFraud

# Create and activate virtual environment (Windows)
python -m venv env
.\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Access

Due to size constraints and licensing, the raw datasets are not included in this repository. To fully reproduce the experiments, please download the datasets and place them in the `data/raw/` directory:

1. **IEEE-CIS Fraud Detection (2019):** Available on [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection).
2. **PaySim (Mobile Money Simulation):** Available on [Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1).
3. **ULB Credit Card Fraud (2013):** Available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

After downloading, run the preprocessing script to format them for streaming input:
```bash
python src/preprocess_datasets.py
```

## How to Run the Experiments

To run the main Interleaved Test-Then-Train (Prequential) benchmarking against standard algorithms (ARF, OzaBag, UOB, SRP, etc.):

```bash
python src/run_benchmark.py
```

To run the ablation study on the asymmetric bagging parameter ($\beta$):
```bash
python src/run_ablation.py
```

To generate the non-parametric statistical significance tests (Friedman Test):
```bash
python src/statistical_test.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
