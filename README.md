# Credit Card Default Prediction (Taiwan) — ML Classification Pipeline

Supervised **binary classification** project to predict whether a credit card client will **default on payment next month** using the “Default of Credit Card Clients (Taiwan)” dataset.

- **Target:** `default payment next month` (0 = no default, 1 = default)
- **Primary focus metric:** F1 for class **Default (1)** (plus Recall(1))

## Repository contents

- `Phase2_Final.ipynb` — full notebook (EDA + feature engineering + modeling + evaluation)
- `phase2_final.py` — script export of the notebook (runnable end-to-end)
- `assets/` — saved figures (ROC, PR curves, confusion matrices, etc.)

## Dataset

Source: UCI Machine Learning Repository — *Default of Credit Card Clients (Taiwan)*

This repo **does not include the dataset file** by default.

### Download
1. Download the dataset from UCI.
2. Place the raw file into:

```
./data/default of credit card clients.xls
```

> Note: The original XLS has an extra descriptive header row; the code uses `header=1` to read the real column names.

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
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

## Run

### Option A — Run the notebook
Open `Phase2_Final.ipynb` in Jupyter / VS Code / Colab.

### Option B — Run as a script
```bash
python phase2_final.py
```

The script will generate and save figures such as:
- `confusion_matrices.png`
- `roc_curves.png`
- `pr_curves.png`
- `metrics_comparison.png`
- `error_analysis_probs.png`

(You can move them into `assets/` if you prefer.)

## Reproducibility
- Random seed: `RANDOM_STATE = 42`
- Train/test split: 80% train+val / 20% held-out test (stratified)
- Cross-validation: Stratified 5-fold
- Leakage prevention: scaling + SMOTE applied **inside each CV fold**

## Notes / Common issues

### Excel reader error
If `pd.read_excel(..., engine='xlrd')` fails, install the pinned versions in `requirements.txt`, or convert the dataset to `.xlsx` / `.csv` and update the loading line.

## License
Add a license if you plan to make this publicly reusable (e.g., MIT).
