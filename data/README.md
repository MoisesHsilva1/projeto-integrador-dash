# Data area

This folder is reserved for you to place the project bases. **Data files must not be versioned in Git.** The directories and this README are only the supporting structure.

## Structure

```text
data/
├── README.md
├── raw/          # original files, no edits
├── interim/      # intermediate data of treatments
└── processed/    # data ready for analysis/modeling
```

## Where to place each base

The definitive base is still a decision pending the group (P-01/P-02 in `docs/PROJECT_PLAN.md`). Place **only the chosen base**; do not mix the three in the same flow.

| Base | Expected file in `data/raw/` | Link | Notes |
|---|---|---|---|
| **MLG/ULB — Credit Card Fraud Detection** (recommended) | `creditcard.csv` | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | 284,807 rows, 31 columns, 492 frauds; about 150 MB; data anonymized by PCA. |
| **Synthetic 10K — Credit Card Fraud Detection Dataset** | `credit_card_fraud_10k.csv` | https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset | 10,000 rows, 10 columns, 151 frauds in the file; base **synthetic**. |
| **Give Me Some Credit** | `cs-training.csv` and, if needed, `cs-test.csv` | https://www.kaggle.com/c/GiveMeSomeCredit | Default risk base, **not recommended** for the fraud problem. |

If the downloaded file has another name, keep the original name or rename it explicitly and record the change in the data dictionary.

## After placing it

1. Check the dimensions and the distribution of the target variable.
2. Record URL, access date, version, license, and checksum in `docs/DATA_DICTIONARY.md` (created in step E2).
3. Do not edit the file in `data/raw/`; all treatments must generate a copy in `data/interim/` or `data/processed/`.
4. Do not commit the data files.

## What not to do

- Do not download or treat the data automatically without a recorded decision.
- Do not use the sketch numbers in the dashboard as results.
- Do not mix the real base with the synthetic one.
- Do not delete the original files after treatment.

Consult Sections 4 and 7 of `docs/PROJECT_PLAN.md` for the treatment care and the step sequence.
