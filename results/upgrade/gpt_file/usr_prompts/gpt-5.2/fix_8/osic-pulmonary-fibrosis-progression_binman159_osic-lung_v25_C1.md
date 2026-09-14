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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-8.5999

# 6. Current score

-16.64826

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.09357) has done: 'I fix the crash in the confidence model by ensuring there are no NaNs in both the training features/targets and the test features passed to `HuberRegressor` (your current imputation only handled the test matrix). I also make the train/test preprocessing consistent by applying the same categorical mappings to the test-side merged frames so columns like `Sex`/`SmokingStatus` don’t become NaN after reindexing. Finally, I make the submission creation robust (always has `Patient_Week,FVC,Confidence`, no missing values, numeric types), so a valid `submission.csv` is written end-to-end without changing the core modeling approach.'
- What this solution (achieved -16.94287) has done: 'Your current score is far below the target (gap = -24.09357 − (-8.5999) = -15.49; higher is better), so we should make small, metric-aligned fixes that improve predictions without changing the core model choices. The biggest issue is inconsistent normalization: you re-normalize test rows independently with `add_norm(test_...)`, which breaks the feature scale the models learned on; we instead normalize test using the *training* mean/std for the same numeric columns. Second, the Confidence you output is incorrectly modeled as predicted `Percent` (unitless) rather than an FVC uncertainty in ml; we keep the same HuberRegressor but train it on an absolute residual target in ml (derived from the existing FVC model’s out-of-fold style residuals on train), then clip to the metric’s required minimum of 70. These are minimal, directly metric-relevant changes and should move the score substantially upward toward the -8.6 band while keeping architecture/training loops intact.'
- What this solution (achieved -16.89929) has done: 'Your current score is far below the target (gap = -16.94 − (-8.60) ≈ -8.34; higher is better), so we make small, metric-aligned improvements without changing the core modeling choices (XGB for FVC + Huber for Confidence). The biggest score drag now is Confidence calibration: training sigma on in-sample residuals makes it too optimistic and unstable; we instead train sigma on out-of-fold residuals (same XGB params, same model family) so Confidence better matches true uncertainty and the Laplace metric improves. We also make the validation split non-leaky by splitting by Patient (your current “last 5 rows” split is arbitrary and can mislead GridSearch), and we fix the scorer to use the model’s prediction as `predFVC` (via `greater_is_better=True` and a wrapper), while keeping the same competition metric definition. All changes are directly tied to producing better-calibrated (FVC, Confidence) pairs for the metric and keep the overall pipeline intact.'
- What this solution (achieved -16.55549) has done: 'Your current score (-16.899) is far below the target (-8.600), so we should improve (higher is better) with minimal, metric-aligned changes. The biggest remaining drag is that you’re predicting FVC for *all* Patient_Week rows using only baseline features but not explicitly modeling the strong linear “Weeks” effect per patient; we can keep the same XGBRegressor core logic and simply add a single, leak-free “baseline_FVC_at_Week0” feature computed from each patient’s earliest (or Week==0 if present) training record, then reuse it for test. Second, your Confidence model is trained on OOF residuals, but the feature set still includes target-proxy columns (like FVC history-derived Height) that can be unstable when merged into test; we keep the same HuberRegressor and target (OOF |residual| in ml) but add one stabilizing feature “delta_weeks_from_baseline” and ensure the confidence features are aligned to the same columns used in training. These are small changes that preserve your modeling approach while usually moving the Laplace log-likelihood upward by improving both mean prediction and uncertainty calibration.'
- What this solution (achieved -16.64826) has done: 'We make two metric-aligned, minimal adjustments that preserve your current modeling choices (XGB for FVC, Huber for sigma) but reduce the gap to the target by improving calibration and removing a subtle preprocessing inconsistency. First, we stop using the normalized `Weeks` inside the merged test matrices by recomputing `delta_weeks_from_baseline` from the *true* submission week and the patient’s baseline week, then re-normalize it with the same train mu/sd; this keeps the strong time effect coherent at inference. Second, we calibrate the predicted Confidence by a single global multiplier computed on out-of-fold residuals (so it’s leak-free) to better match the Laplace metric’s optimal sigma scale, without changing the confidence model or adding new training loops beyond what you already do. These changes are small, directly tied to the competition metric, and should improve score (higher is better) toward the -8.6 target without altering the core approach.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import typing as tp
import pydicom
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import make_scorer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor

RANDOM_STATE = 42



## === cell 1
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

df_all = pd.concat([train_df, test_df], ignore_index=True)
df_all["Patient_Week"] = (
    df_all["Patient"].astype(str) + "_" + df_all["Weeks"].astype(str)
)



## === cell 2
print("Shape of Training data: ", train_df.shape)
print("Shape of Test data: ", test_df.shape)




## === cell 3
def add_height(data: pd.DataFrame) -> None:
    if "Sex" not in data.columns:
        raise KeyError("Expected column 'Sex' to exist before add_height().")
    denom_f = 21.78 - (0.101 * data["Age"])
    denom_m = 27.63 - (0.112 * data["Age"])
    is_female = (data["Sex"] == 0) | (data["Sex"] == "Female")
    data["Height"] = np.where(is_female, data["FVC"] / denom_f, data["FVC"] / denom_m)


def add_norm(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return out
    std = out[num_cols].std(ddof=0).replace(0, 1.0)
    out[num_cols] = (out[num_cols] - out[num_cols].mean()) / std
    return out


def fit_norm_params(train_numeric_df: pd.DataFrame) -> tp.Tuple[pd.Series, pd.Series]:
    mu = train_numeric_df.mean()
    sd = train_numeric_df.std(ddof=0).replace(0, 1.0)
    return mu, sd


def apply_norm_params(
    df_numeric: pd.DataFrame, mu: pd.Series, sd: pd.Series
) -> pd.DataFrame:
    mu = mu.reindex(df_numeric.columns)
    sd = sd.reindex(df_numeric.columns).replace(0, 1.0)
    return (df_numeric - mu) / sd




## === cell 4
def compute_patient_baseline_fvc(train_raw: pd.DataFrame) -> pd.Series:
    t = train_raw.copy()
    t = t.sort_values(["Patient", "Weeks"])
    wk0 = t[t["Weeks"] == 0].groupby("Patient")["FVC"].first()
    earliest = t.groupby("Patient")["FVC"].first()
    baseline = earliest.copy()
    baseline.loc[wk0.index] = wk0
    return baseline


patient_baseline_fvc = compute_patient_baseline_fvc(train_df)


def compute_patient_baseline_week(train_raw: pd.DataFrame) -> pd.Series:
    t = train_raw.copy().sort_values(["Patient", "Weeks"])
    wk0 = t[t["Weeks"] == 0].groupby("Patient")["Weeks"].first()
    earliest = t.groupby("Patient")["Weeks"].first()
    base_w = earliest.copy()
    base_w.loc[wk0.index] = wk0
    return base_w


patient_baseline_week = compute_patient_baseline_week(train_df)



## === cell 5
df = train_df.copy()

df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1})
df["SmokingStatus"] = df["SmokingStatus"].map(
    {"Currently smokes": 0, "Never smoked": 1, "Ex-smoker": 2}
)

df["Patient_Week"] = df["Patient"].astype(str) + "_" + df["Weeks"].astype(str)
df = df.drop("Patient_Week", axis=1)
df = df.set_index("Patient")

df["baseline_FVC"] = df.index.map(patient_baseline_fvc).astype(float)
df["baseline_Week"] = df.index.map(patient_baseline_week).astype(float)
df["delta_weeks_from_baseline"] = (
    df["Weeks"].astype(float) - df["baseline_Week"]
).astype(float)

add_height(df)

feature_cols_to_norm = df.columns[~df.columns.isin(["FVC", "Percent"])]

_train_mu, _train_sd = fit_norm_params(
    df[feature_cols_to_norm].select_dtypes(include=[np.number])
)
df_num = df[feature_cols_to_norm].select_dtypes(include=[np.number])
df.loc[:, df_num.columns] = apply_norm_params(df_num, _train_mu, _train_sd)

df.head()



## === cell 6
sub_df = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
sub_df.drop(["FVC", "Confidence"], axis=1, inplace=True)

sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.rsplit("_", 1)[0])
sub_df["pred_Weeks"] = (
    sub_df["Patient_Week"].apply(lambda x: x.rsplit("_", 1)[1]).astype(int)
)

sub_df.head()



## === cell 7
test_FVC = pd.merge(
    sub_df[["Patient_Week", "Patient", "pred_Weeks"]],
    df.reset_index(),
    how="left",
    on="Patient",
)

test_FVC = test_FVC.rename(columns={"pred_Weeks": "Weeks"})
if "FVC" in test_FVC.columns:
    test_FVC = test_FVC.drop(["FVC"], axis=1)

test_FVC = test_FVC.set_index("Patient_Week")
test_FVC = test_FVC.drop(columns=["Patient"], errors="ignore")
test_FVC = test_FVC.loc[:, ~test_FVC.columns.duplicated()]

if "baseline_Week" in test_FVC.columns:
    test_FVC["delta_weeks_from_baseline"] = (
        test_FVC["Weeks"].astype(float) - test_FVC["baseline_Week"].astype(float)
    ).astype(float)

test_FVC_num_cols = (
    test_FVC[feature_cols_to_norm].select_dtypes(include=[np.number]).columns
)
test_FVC.loc[:, test_FVC_num_cols] = apply_norm_params(
    test_FVC.loc[:, test_FVC_num_cols],
    _train_mu.reindex(test_FVC_num_cols),
    _train_sd.reindex(test_FVC_num_cols),
)

test_FVC.head()



## === cell 8
test_conf = pd.merge(
    sub_df[["Patient_Week", "Patient", "pred_Weeks"]],
    df.reset_index(),
    how="left",
    on="Patient",
)
test_conf = test_conf.rename(columns={"pred_Weeks": "Weeks"})
if "Percent" in test_conf.columns:
    test_conf = test_conf.drop(["Percent"], axis=1)

test_conf = test_conf.set_index("Patient_Week")
test_conf = test_conf.drop(columns=["Patient"], errors="ignore")
test_conf = test_conf.loc[:, ~test_conf.columns.duplicated()]

if "baseline_Week" in test_conf.columns:
    test_conf["delta_weeks_from_baseline"] = (
        test_conf["Weeks"].astype(float) - test_conf["baseline_Week"].astype(float)
    ).astype(float)

test_conf_num_cols = (
    test_conf[feature_cols_to_norm].select_dtypes(include=[np.number]).columns
)
test_conf.loc[:, test_conf_num_cols] = apply_norm_params(
    test_conf.loc[:, test_conf_num_cols],
    _train_mu.reindex(test_conf_num_cols),
    _train_sd.reindex(test_conf_num_cols),
)

test_conf.head()



## === cell 9
print(test_FVC.shape)
print(test_conf.shape)



## === cell 10
X = df.iloc[:, df.columns != "FVC"]
y = df["FVC"]

groups = X.index.astype(str)
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)



## === cell 11
X_conf = df.iloc[:, df.columns != "Percent"]
y_conf = df["Percent"]

X_train_conf = X_conf.iloc[train_idx]
y_train_conf = y_conf.iloc[train_idx]
X_val_conf = X_conf.iloc[val_idx]
y_val_conf = y_conf.iloc[val_idx]

print(X_conf.shape)
print(y_conf.shape)
print(X_train_conf.shape)
print(y_train_conf.shape)
print(X_val_conf.shape)
print(y_val_conf.shape)



## === cell 12
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
train_df.boxplot("FVC", by="SmokingStatus", ax=axs[0, 0])
train_df.boxplot("Percent", by="SmokingStatus", ax=axs[0, 1])
train_df.boxplot("FVC", by="Sex", ax=axs[1, 0])
train_df.boxplot("Percent", by="Sex", ax=axs[1, 1])

plt.show()



## === cell 13
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177411956430/100.dcm"
if os.path.exists(img):
    ds = pydicom.dcmread(img)
    plt.figure(figsize=(7, 7))
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.show()



## === cell 14
img_1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177411956430/100.dcm"
img_2 = "../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/10.dcm"

if os.path.exists(img_1) and os.path.exists(img_2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ds = pydicom.dcmread(img_1)
    ax[0].set_title("Patient 1: Ex-Smoker")
    ax[0].imshow(ds.pixel_array, cmap=plt.cm.bone)

    ds = pydicom.dcmread(img_2)
    ax[1].set_title("Patient 2: Never smoked")
    ax[1].imshow(ds.pixel_array, cmap=plt.cm.bone)

    plt.show()




## === cell 15
def competition_metric(trueFVC, predFVC, predSTD=100):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    error = np.mean(
        -1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD)
    )
    return error


def competition_scorer(y_true, y_pred):
    return competition_metric(np.asarray(y_true), np.asarray(y_pred), predSTD=100)




## === cell 16
class model_selection:
    def __init__(self):
        self.y_pred_FVC = pd.DataFrame()
        self.my_scorer = make_scorer(competition_scorer, greater_is_better=True)
        self.best_param = None
        self.scoring = 0

    def xgboost(self, X, y, X_val, y_val, test, groups=None):
        parameters = {
            "learning_rate": [0.002],
            "n_estimators": [4000],
            "max_depth": [4],
            "reg_alpha": [0.005],
        }
        base = XGBRegressor(
            min_child_weight=0,
            gamma=0,
            colsample_bytree=0.7,
            objective="reg:squarederror",
            nthread=-1,
            scale_pos_weight=1,
            subsample=0.7,
            seed=27,
            random_state=RANDOM_STATE,
        )

        cv = 3
        fit_params = {}
        if groups is not None:
            cv = GroupKFold(n_splits=3)
            fit_params = {"groups": groups}

        clf = GridSearchCV(
            base, param_grid=parameters, scoring=self.my_scorer, cv=cv, refit=True
        )
        clf.fit(X, y, **fit_params)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_xgb_FVC = clf.predict(test)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test.index, name="Patient_Week"),
                pd.Series(y_pred_xgb_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC, self.best_param, self.scoring

    def lightgbm(self, X, y, X_val, y_val, test, groups=None):
        parameters = {
            "learning_rate": [0.00005, 0.0001, 0.0005],
            "n_estimators": [2500, 3000, 4000],
            "num_leaves": [1, 2, 3],
        }

        cv = 3
        fit_params = {}
        if groups is not None:
            cv = GroupKFold(n_splits=3)
            fit_params = {"groups": groups}

        clf = GridSearchCV(
            LGBMRegressor(
                objective="regression",
                max_bin=200,
                bagging_fraction=0.75,
                bagging_freq=5,
                bagging_seed=7,
                feature_fraction=0.2,
                feature_fraction_seed=7,
                verbose=-1,
                random_state=RANDOM_STATE,
            ),
            param_grid=parameters,
            scoring=self.my_scorer,
            cv=cv,
            refit=True,
        )
        clf.fit(X, y, **fit_params)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_lgb_FVC = clf.predict(test)

        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test.index, name="Patient_Week"),
                pd.Series(y_pred_lgb_FVC, name="FVC"),
            ],
            axis=1,
        )
        return self.y_pred_FVC, self.best_param, self.scoring

    def HuberRegressor(self, X, y, test):
        hbr = HuberRegressor(max_iter=200)
        hbr.fit(X, y)
        y_pred = hbr.predict(test)
        self.y_pred_FVC = pd.concat(
            [
                pd.Series(test.index, name="Patient_Week"),
                pd.Series(y_pred, name="Confidence"),
            ],
            axis=1,
        )
        return self.y_pred_FVC




## === cell 17
test_FVC_aligned = test_FVC.loc[:, ~test_FVC.columns.duplicated()].reindex(
    columns=X.columns
)
train_medians_X = X.median(numeric_only=True)
test_FVC_aligned = test_FVC_aligned.fillna(train_medians_X)

model = model_selection()
output = model.xgboost(
    X_train, y_train, X_val, y_val, test_FVC_aligned, groups=X_train.index.astype(str)
)



## === cell 18
print(output[1])  # best params
print(output[2])  # score on validation (per competition_metric scorer)



## === cell 19
y_pred_FVC = output[0].copy()
y_pred_FVC.head()



## === cell 20
test_conf_aligned = test_conf.loc[:, ~test_conf.columns.duplicated()].reindex(
    columns=X_conf.columns
)
train_medians_Xconf = X_conf.median(numeric_only=True)
test_conf_aligned = test_conf_aligned.fillna(train_medians_Xconf)
test_conf_aligned.head()



## === cell 21
X_imp = X.fillna(train_medians_X)
best_params = output[1]

oof_pred = np.zeros(len(X_imp), dtype=float)
gkf_oof = GroupKFold(n_splits=5)
for tr, va in gkf_oof.split(X_imp, y, groups=X_imp.index.astype(str)):
    est = XGBRegressor(
        min_child_weight=0,
        gamma=0,
        colsample_bytree=0.7,
        objective="reg:squarederror",
        nthread=-1,
        scale_pos_weight=1,
        subsample=0.7,
        seed=27,
        random_state=RANDOM_STATE,
        **best_params,
    )
    est.fit(X_imp.iloc[tr], y.iloc[tr])
    oof_pred[va] = est.predict(X_imp.iloc[va])

y_sigma = np.abs(y.values - oof_pred).astype(float)

X_conf_imp = X_conf.fillna(train_medians_Xconf)

model = model_selection()
y_pred_conf = model.HuberRegressor(X_conf_imp, y_sigma, test_conf_aligned)

sigma_pred_train = (
    HuberRegressor(max_iter=200).fit(X_conf_imp, y_sigma).predict(X_conf_imp)
)
sigma_pred_train = np.clip(np.abs(sigma_pred_train.astype(float)), 1e-6, 1e9)

target_sigma_train = np.sqrt(2.0) * y_sigma
calib_mult = np.median(target_sigma_train) / np.median(sigma_pred_train)
calib_mult = float(np.clip(calib_mult, 0.5, 3.0))  # keep calibration conservative

y_pred_conf["Confidence"] = np.abs(y_pred_conf["Confidence"].astype(float)) * calib_mult

y_pred_conf.head()



## === cell 22
submission = sub_df[["Patient_Week"]].merge(y_pred_FVC, how="left", on="Patient_Week")
submission = submission.merge(y_pred_conf, how="left", on="Patient_Week")

submission = submission[["Patient_Week", "FVC", "Confidence"]]

submission["FVC"] = pd.to_numeric(submission["FVC"], errors="coerce").fillna(
    train_df["FVC"].median()
)

submission["Confidence"] = pd.to_numeric(
    submission["Confidence"], errors="coerce"
).fillna(200.0)
submission["Confidence"] = submission["Confidence"].abs()
submission["Confidence"] = submission["Confidence"].clip(lower=70.0)

submission.head()



## === cell 23
submission.to_csv("./submission.csv", index=False)
print("Wrote submission to ./submission.csv")
print(submission.shape)
print(submission.columns.tolist())
print(submission.head())
