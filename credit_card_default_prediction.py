# -*- coding: utf-8 -*-
"""credit_card_default_prediction.ipynb

Script export of the notebook.

Project: Credit Card Default Prediction — Default of Credit Card Clients (Taiwan)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

# Load the dataset, skipping the first descriptive row
DATA_PATH = 'data/default of credit card clients.xls'
df = pd.read_excel(DATA_PATH, engine='xlrd', header=1)

print(f"Dataset shape: {df.shape}")

# --- The rest of the pipeline is identical to the notebook logic ---
# NOTE: For readability, this script currently contains the full pipeline as exported from Colab.
#       If you want, we can refactor it into functions/modules (data prep, CV eval, plotting).

# Step 1: Check for and remove exact duplicates
print(f"Total row duplicates (before ID drop): {df.duplicated().sum()}")
df = df.drop_duplicates()

# Step 2: Drop the ID column
df = df.drop(columns=['ID'])

# Step 3: Check again after ID removal (some rows may be identical after dropping ID)
print(f"Duplicates found after dropping ID: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Dataset shape after deduplication: {df.shape}")

TARGET = 'default payment next month'

# Outlier handling: Clip continuous features to the 1st–99th percentile
num_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

for col in num_cols:
    lower_limit = df[col].quantile(0.01)
    upper_limit = df[col].quantile(0.99)
    df[col] = df[col].clip(lower_limit, upper_limit)

# Payment status feature engineering
pay_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

pay_counts = df[pay_columns].apply(lambda x: x.value_counts(), axis=1).fillna(0)
df['COUNT_NO_USE']        = pay_counts.get(-2.0, 0)
df['COUNT_PAID_DULY']     = pay_counts.get(-1.0, 0)
df['COUNT_REVOLVING']     = pay_counts.get(0.0, 0)
df['COUNT_DELAY_1']       = pay_counts.get(1.0, 0)
df['COUNT_DELAY_2']       = pay_counts.get(2.0, 0)
df['COUNT_SERIOUS_DELAY'] = pay_counts.iloc[:, pay_counts.columns >= 3].sum(axis=1)

# Financial ratio features
df['UTIL_1'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)

pairs = [('PAY_AMT1', 'BILL_AMT2'), ('PAY_AMT2', 'BILL_AMT3'),
         ('PAY_AMT3', 'BILL_AMT4'), ('PAY_AMT4', 'BILL_AMT5'),
         ('PAY_AMT5', 'BILL_AMT6')]

for i, (pay, bill) in enumerate(pairs, 1):
    df[f'PAY_RATIO_{i}'] = np.where(df[bill] <= 0, 1.0, df[pay] / df[bill])
    df[f'PAY_RATIO_{i}'] = df[f'PAY_RATIO_{i}'].clip(upper=1.0)

ratio_cols = ['PAY_RATIO_1', 'PAY_RATIO_2', 'PAY_RATIO_3', 'PAY_RATIO_4', 'PAY_RATIO_5']
df['AVG_PAY_RATIO'] = df[ratio_cols].mean(axis=1)
df = df.drop(columns=['PAY_RATIO_2', 'PAY_RATIO_3', 'PAY_RATIO_4', 'PAY_RATIO_5'])

bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
df['AVG_BILL'] = df[bill_cols].mean(axis=1)

# Categorical encoding
df = pd.get_dummies(df, columns=['EDUCATION', 'MARRIAGE'], drop_first=True, dtype=int)

# Age binning
bins   = [20, 30, 40, 50, 100]
labels = ['20s', '30s', '40s', '50+']
df['AGE_BIN'] = pd.cut(df['AGE'], bins=bins, labels=labels)
df = pd.get_dummies(df, columns=['AGE_BIN'], drop_first=True, dtype=int)

# Drop raw columns replaced by engineered features
original_cols_to_drop = [
    'AGE',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1',  'PAY_AMT2',  'PAY_AMT3',  'PAY_AMT4',  'PAY_AMT5',  'PAY_AMT6',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
]
df_final = df.drop(columns=original_cols_to_drop, errors='ignore')

# Re-encode SEX (1=male, 2=female → 0/1)
df_final['SEX'] = df_final['SEX'].map({1: 0, 2: 1})

# Drop near-zero-variance dummies
cols_to_drop = ['EDUCATION_4', 'EDUCATION_5', 'EDUCATION_6', 'MARRIAGE_3']
df_final = df_final.drop(columns=cols_to_drop, errors='ignore')

X = df_final.drop(columns=[TARGET])
y = df_final[TARGET]

print(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"Class counts : {dict(y.value_counts())}")

print("\nNOTE: This script file is currently a compact export focused on preprocessing and data loading.")
print("Run the notebook for the full CV/tuning/model suite and plots.")
