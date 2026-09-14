# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.03

# 6. Current score

0.00055

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00372) has done: 'I fix the NaN/TypeError issues by separating numeric vs categorical handling (mean/mode imputation) and by ensuring all remaining categorical columns (e.g., `density`, `BIRADS`, `machine_id`) are either one-hot encoded or safely dropped before modeling. I also make train/test feature alignment deterministic (same columns/order) so LogisticRegression can fit and then predict without shape/NaN errors, which allow the submission to be generated. To keep core logic intact, I keep your upsampling + LogisticRegression approach, but ensure `prediction_id` is excluded from features and then correctly used to aggregate image-level predictions into the required submission rows. Finally, I write `submission.csv` to `/kaggle/working/submission.csv` and validate it against `sample_submission.csv`’s required `prediction_id,cancer` format.'
- What this solution (achieved 0.0) has done: 'Your current pipeline is valid but is likely under-scoring because it trains on a balanced (upsampled) dataset while Kaggle’s pF1 is highly sensitive to probability calibration under the real (very imbalanced) prevalence. I keep the same core approach (tabular preprocessing + upsampling + LogisticRegression) and add a minimal probability calibration step using the existing validation split, then use the calibrated probabilities for the submission. I also one-hot encode `density` in test (mirroring train) to reduce train/test mismatch and keep feature alignment deterministic; this should improve generalization without changing the modeling approach. Finally, I keep the same submission aggregation by `prediction_id` and ensure the output CSV remains identical in schema and path.'
- What this solution (achieved 0.0) has done: 'I fix the `KeyError: ['density'] not in index` by only one-hot encoding columns that actually exist in each dataframe (since `density` is train-only), then align train/test dummy columns deterministically. I also ensure `machine_id` is treated consistently as categorical (one-hot) rather than being incorrectly standardized as numeric IDs, which tends to hurt generalization and probability calibration. The rest of your approach (mixed imputation → upsampling → LogisticRegression → isotonic calibration → groupby prediction_id mean → submission merge) is kept intact, just made robust so it runs end-to-end and produces a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current submission likely scores 0.0 because the model trained on an artificially balanced (upsampled) distribution and then additionally used strong `class_weight`, and isotonic calibration on that shifted distribution can collapse predicted probabilities near 0 for the real (highly imbalanced) test prevalence—pF1 then becomes ~0. I keep your exact core approach (mixed imputation → one-hot → upsampling → LogisticRegression → isotonic calibration → groupby prediction_id mean), but make two minimal, score-relevant fixes: remove `class_weight` (since upsampling already balances) and calibrate isotonic on a validation split from the original (non-upsamped) training data so probabilities reflect the real base rate. Finally, I fit the final LR on the upsampled data as you do, then apply the isotonic calibrator trained on the original distribution to produce a better-calibrated submission (still identical semantics and format).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most plausibly coming from probability collapse/miscalibration for pF1, where predictions end up too close to 0 for almost all rows (yielding near-zero pTP and thus pF1≈0). Keeping your exact core pipeline (mixed imputation → one-hot → upsampling → LogisticRegression → isotonic calibration → groupby prediction_id mean), I make one minimal score-directed change: add a prevalence (prior) correction to map the upsampled-trained LR probabilities back to the original training prevalence before isotonic calibration. This is a standard, lightweight correction that doesn’t change the model/feature logic, but typically prevents the “all-near-zero” or “all-near-one” behavior that can tank pF1. I also add a tiny safety clip before isotonic (strictly numerical stability) and print quick probability summaries to confirm we’re no longer degenerate.'
- What this solution (achieved 0.00203) has done: 'I fix the runtime error by applying the fitted calibrator to the same feature matrix shape it was trained to expect (the original tabular features), rather than incorrectly passing a 2-column probability matrix. I keep your core pipeline intact (mixed imputation → one-hot → upsampling → LogisticRegression → prior correction → probability calibration → groupby prediction_id aggregation), but move the calibration step to operate on the LR model’s probabilities for each row via a lightweight wrapper that exposes a `predict_proba` API. Finally, I ensure the submission file is always written to `/kaggle/working/submission.csv` with the exact `prediction_id,cancer` schema so the last validation cell can read it successfully.'
- What this solution (achieved 0.00207) has done: 'You’re currently far below the 0.03 target (0.00203 vs 0.03), so we should gently increase pF1 without changing the core pipeline. The biggest score leak here is probability calibration: you trained a `CalibratedClassifierCV` but then didn’t use it; instead you fit a second 1D logistic calibrator on the *base* model probabilities and applied it to the *final* model probabilities, which is a miscalibration and tends to suppress useful signal for pF1. I make the smallest score-directed fix by calibrating the final upsampled LR with a sigmoid calibrator trained on out-of-fold probabilities from that same final model (after prior correction), then use that mapping for test. I also switch the `prediction_id` aggregation from `max` to `mean`, which is usually better aligned with pF1’s probabilistic nature while keeping the same “group by prediction_id” semantics.'
- What this solution (achieved 0.00205) has done: 'Your score (0.00207) is far below the 0.03 target, so we should improve pF1 without changing your pipeline (mixed imputation → one-hot → upsampling → LogisticRegression → prior correction → calibration → groupby prediction_id mean). The most score-relevant bug is that your sigmoid calibrator is trained using probabilities from `lr` evaluated on `X_orig` rows it was not fit/aligned to (it was fit on a split of the upsampled data), which makes the calibration mapping noisy/misaligned and can suppress signal. I make a minimal fix by fitting the final LR on *all* upsampled data (same core model), then training the 1D sigmoid calibrator on out-of-fold probabilities from that same upsampled-trained LR but evaluated on the original-distribution fold (so calibration reflects the real prevalence after prior-correction). Finally, I keep the same prediction/aggregation/submission logic and ensure the CSV schema stays identical.'
- What this solution (achieved 0.00055) has done: 'Your current pF1 is far below the 0.03 target, so we should cautiously increase it without changing the core pipeline (mixed imputation → one-hot → upsampling → LogisticRegression → prior correction → calibration → groupby mean). The most score-relevant issue is that the 1D “sigmoid” calibrator is being trained only on a single holdout’s probabilities, which is noisy and can suppress signal; switching to out-of-fold (OOF) probabilities makes the calibration mapping much more stable while keeping the same semantics. I replace the single split calibrator-fit with a stratified K-fold OOF procedure on the original distribution (still using the same final upsampled LR for test inference) and keep the same prior-correction and final aggregation. This is a minimal change localized to calibration, and it should move pF1 upward toward the target without altering the model class or feature logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"

train_data = pd.read_csv(f"{DATA_DIR}/train.csv")
test_data = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")



## === cell 1
print(train_data.info())



## === cell 2
print(test_data.info())



## === cell 3
train_data = train_data.drop(["patient_id", "image_id"], axis=1)
test_data = test_data.drop(["patient_id", "image_id"], axis=1)




## === cell 4
def impute_mixed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean(numeric_only=True))
    for c in cat_cols:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            fill_val = mode.iloc[0] if len(mode) else "missing"
            df[c] = df[c].fillna(fill_val)
    return df


train_data = impute_mixed(train_data)
test_data = impute_mixed(test_data)



## === cell 5
print(test_data.info())



## === cell 6
print(train_data.info())



## === cell 7
import matplotlib.pyplot as plt
import seaborn as sns

print(train_data.describe(include="all"))

if "age" in train_data.columns:
    sns.histplot(train_data["age"], kde=False)
    plt.show()



## === cell 8
from sklearn.preprocessing import StandardScaler

train_cat_cols = [
    c
    for c in ["laterality", "view", "implant", "density", "machine_id", "site_id"]
    if c in train_data.columns
]
test_cat_cols = [
    c
    for c in ["laterality", "view", "implant", "density", "machine_id", "site_id"]
    if c in test_data.columns
]

train_data = pd.get_dummies(train_data, columns=train_cat_cols, dummy_na=False)
test_data = pd.get_dummies(test_data, columns=test_cat_cols, dummy_na=False)

if "cancer" in train_data.columns:
    y_tmp = train_data["cancer"]
    X_tmp = train_data.drop(columns=["cancer"])
else:
    y_tmp = None
    X_tmp = train_data.copy()

X_tmp, test_data = X_tmp.align(test_data, join="left", axis=1, fill_value=0)

train_data = X_tmp.copy()
if y_tmp is not None:
    train_data["cancer"] = y_tmp.values



## === cell 9
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols_for_corr = train_data.select_dtypes(include=[np.number]).columns
corr = train_data[numeric_cols_for_corr].corr()

fig, ax = plt.subplots(figsize=(21, 21))
sns.heatmap(corr, annot=False, fmt=".2f", cmap="coolwarm", ax=ax)

ax.set_title("Correlation Matrix (Numeric Columns)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment="right")
plt.show()



## === cell 10
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df_majority = train_data[train_data["cancer"] == 0]
df_minority = train_data[train_data["cancer"] == 1]

df_minority_upsampled = resample(
    df_minority, replace=True, n_samples=len(df_majority), random_state=42
)

train_data_upsampled = pd.concat([df_majority, df_minority_upsampled], axis=0).sample(
    frac=1.0, random_state=42
)

print(train_data_upsampled["cancer"].value_counts())

X_up = train_data_upsampled.drop("cancer", axis=1)
y_up = train_data_upsampled["cancer"]

X_orig = train_data.drop("cancer", axis=1)
y_orig = train_data["cancer"]

if "prediction_id" in X_up.columns:
    X_up = X_up.drop(columns=["prediction_id"])
if "prediction_id" in X_orig.columns:
    X_orig = X_orig.drop(columns=["prediction_id"])

cols_to_scale = [c for c in ["age"] if c in X_up.columns]
scaler = StandardScaler()
if cols_to_scale:
    X_up[cols_to_scale] = scaler.fit_transform(X_up[cols_to_scale])
    if all(c in X_orig.columns for c in cols_to_scale):
        X_orig[cols_to_scale] = scaler.transform(X_orig[cols_to_scale])

test_features = test_data.copy()
if "prediction_id" in test_features.columns:
    test_features = test_features.drop(columns=["prediction_id"])

for c in X_up.columns:
    if c not in test_features.columns:
        test_features[c] = 0
extra_cols = [c for c in test_features.columns if c not in X_up.columns]
if extra_cols:
    test_features = test_features.drop(columns=extra_cols)
test_features = test_features[X_up.columns]

if cols_to_scale:
    test_features[cols_to_scale] = scaler.transform(test_features[cols_to_scale])

X_up = X_up.replace([np.inf, -np.inf], np.nan).fillna(0)
X_orig = X_orig.replace([np.inf, -np.inf], np.nan).fillna(0)
test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0)



## === cell 11
from sklearn.calibration import CalibratedClassifierCV

Xo_train, Xo_val, yo_train, yo_val = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)

print("Original target distribution (non-upsamped):")
print(y_orig.value_counts())

lr_cal_base = LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")
lr_cal_base.fit(Xo_train, yo_train)

yo_val_pred = lr_cal_base.predict(Xo_val)
print("Calibration base model (orig-dist) metrics on holdout:")
print("Accuracy:", accuracy_score(yo_val, yo_val_pred))
print("Precision:", precision_score(yo_val, yo_val_pred, zero_division=0))
print("Recall:", recall_score(yo_val, yo_val_pred, zero_division=0))
print(
    "F1-score:", fbeta_score(yo_val, yo_val_pred, beta=1, average="binary", pos_label=1)
)

calibrator = CalibratedClassifierCV(lr_cal_base, method="sigmoid", cv="prefit")
calibrator.fit(Xo_val, yo_val)

X_train_up, X_val_up, y_train_up, y_val_up = train_test_split(
    X_up, y_up, test_size=0.2, random_state=42, stratify=y_up
)

lr = LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")
lr.fit(X_train_up, y_train_up)

y_pred_up = lr.predict(X_val_up)
print("Final model (upsampled) metrics on upsampled holdout:")
print("Accuracy:", accuracy_score(y_val_up, y_pred_up))
print("Precision:", precision_score(y_val_up, y_pred_up, zero_division=0))
print("Recall:", recall_score(y_val_up, y_pred_up, zero_division=0))
print(
    "F1-score:", fbeta_score(y_val_up, y_pred_up, beta=1, average="binary", pos_label=1)
)

pi_true = float(y_orig.mean())  # original prevalence
pi_train = float(y_up.mean())  # upsampled prevalence (~0.5)
print("pi_true (orig prevalence):", pi_true)
print("pi_train (upsampled prevalence):", pi_train)


def prior_correct_proba(p: np.ndarray, pi_true: float, pi_train: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    adj = np.log(pi_true / (1 - pi_true)) - np.log(pi_train / (1 - pi_train))
    logit_adj = logit + adj
    p_adj = 1 / (1 + np.exp(-logit_adj))
    return np.clip(p_adj, 0.0, 1.0)




## === cell 12
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as _LR

lr.fit(X_up, y_up)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X_orig), dtype=np.float64)

for tr_idx, va_idx in skf.split(X_orig, y_orig):
    lr_fold = LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")
    lr_fold.fit(X_up, y_up)

    p_va = lr_fold.predict_proba(X_orig.iloc[va_idx])[:, 1]
    p_va = prior_correct_proba(p_va, pi_true=pi_true, pi_train=pi_train)
    oof_proba[va_idx] = np.clip(p_va, 1e-6, 1 - 1e-6)

sig_cal = _LR(random_state=42, max_iter=1000, solver="lbfgs")
sig_cal.fit(oof_proba.reshape(-1, 1), y_orig.values)

oof_cal = sig_cal.predict_proba(oof_proba.reshape(-1, 1))[:, 1]
print(
    "OOF proba summary (final+prior-correct, before cal):",
    float(np.min(oof_proba)),
    float(np.mean(oof_proba)),
    float(np.max(oof_proba)),
)
print(
    "OOF proba summary (after sigmoid cal):",
    float(np.min(oof_cal)),
    float(np.mean(oof_cal)),
    float(np.max(oof_cal)),
)



## === cell 13
test_data_orig = pd.read_csv(f"{DATA_DIR}/test.csv")
prediction_ids = test_data_orig["prediction_id"].copy()

test_proba = lr.predict_proba(test_features)[:, 1]
test_proba = prior_correct_proba(test_proba, pi_true=pi_true, pi_train=pi_train)
test_proba = np.clip(test_proba, 1e-6, 1 - 1e-6)

y_pred_proba = sig_cal.predict_proba(test_proba.reshape(-1, 1))[:, 1]
y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)

print(
    "Test proba summary (after prior correction, before calibration):",
    float(np.min(test_proba)),
    float(np.mean(test_proba)),
    float(np.max(test_proba)),
)
print(
    "Submission proba summary (after sigmoid calibration):",
    float(np.min(y_pred_proba)),
    float(np.mean(y_pred_proba)),
    float(np.max(y_pred_proba)),
)

submission = pd.DataFrame({"prediction_id": prediction_ids, "cancer": y_pred_proba})

submission = submission.groupby("prediction_id", as_index=False)["cancer"].mean()

submission = sample_submission[["prediction_id"]].merge(
    submission, on="prediction_id", how="left"
)
submission["cancer"] = submission["cancer"].fillna(0.0).clip(0.0, 1.0)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)



## === cell 14
sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.head())
print(sub.info())
print(
    "Matches sample_submission ids:",
    sub["prediction_id"].equals(sample_submission["prediction_id"]),
)
print("Cancer range:", (sub["cancer"].min(), sub["cancer"].max()))
