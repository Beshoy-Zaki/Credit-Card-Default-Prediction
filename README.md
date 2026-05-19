# Credit Card Default Prediction (Taiwan) — ML Classification Pipeline

Supervised **binary classification** project to predict whether a credit card client in Taiwan will **default on their payment in the following month** using the “Default of Credit Card Clients (Taiwan)” dataset.

- **Target:** `default payment next month` (0 = no default, 1 = default)
- **Primary focus metric:** F1 for class **Default (1)** (plus Recall(1))

## Repository contents

- `credit_card_default_prediction.ipynb` — full notebook (EDA + feature engineering + modeling + evaluation)
- `credit_card_default_prediction.py` — runnable script export
- `data/` — dataset folder (see instructions below)

## Dataset

- **Name:** Default of Credit Card Clients (Taiwan)
- **Source (UCI ML Repository):** https://archive.uci.edu/dataset/350/default+of+credit+card+clients
- The raw XLS contains a descriptive header row at index 0, followed by the actual column names at index 1. The code uses `header=1`.

### Download the dataset

1) Download the dataset from UCI (link above). 
2) Place the file here:

- `data/default of credit card clients.xls`

> Note: The dataset file is intentionally **not committed** to this repository.

## Setup

### 1) Create and activate a virtual environment (recommended)

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.[0m\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

## Run

### Option A — Run the notebook
Open `credit_card_default_prediction.ipynb` in Jupyter / VS Code / Colab.

### Option B — Run as a script
```bash
python credit_card_default_prediction.py
```

## Reproducibility
- Random seed: `RANDOM_STATE = 42`
- Train/test split: 80% train+val / 20% held-out test (stratified)
- Cross-validation: Stratified 5-fold
- Leakage prevention: scaling + SMOTE applied **inside each CV fold**

## Notes / Common issues

### Excel reader error
If `pd.read_excel(..., engine='xlrd')` fails, use the versions in `requirements.txt`.
