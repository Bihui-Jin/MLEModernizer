# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor

SEED = 42
np.random.seed(SEED)



## === cell 1
EPOCHS = 5
BATCH_SIZE = 32
FOLDS = 5

COMP_DIR = "../input/osic-pulmonary-fibrosis-progression/"

TRAIN_CSV = os.path.join(COMP_DIR, "train.csv")
TEST_CSV = os.path.join(COMP_DIR, "test.csv")
SUB_PATH = os.path.join(COMP_DIR, "sample_submission.csv")



## === cell 2
train_data = pd.read_csv(TRAIN_CSV)
test_data = pd.read_csv(TEST_CSV)
sub = pd.read_csv(SUB_PATH)

train_data = train_data.drop_duplicates(
    keep=False, subset=["Patient", "Weeks"]
).reset_index(drop=True)

train_data.head(), test_data.head(), sub.head()



## === cell 3
td = train_data.copy()
td["_abs_week"] = td["Weeks"].abs()
train_base_idx = (
    td.sort_values(["Patient", "_abs_week", "Weeks"])
    .groupby("Patient", sort=False)
    .head(1)
    .index
)
train_data_u = td.loc[train_base_idx].drop(columns=["_abs_week"]).copy()

train_data_u = train_data_u.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
train_data_u["Typical_FVC"] = (
    train_data_u["Base_FVC"].values / train_data_u["Base_Percent"].values
) * 100.0

train_data = train_data.merge(
    train_data_u.drop(["Age", "Sex", "SmokingStatus"], axis=1), on="Patient", how="left"
)

train_data.head()



## === cell 4
pw = sub["Patient_Week"].astype(str)
sub["Patient"] = pw.str.split("_", n=1, expand=True)[0]
sub["Weeks"] = pw.str.rsplit("_", n=1, expand=True)[1].astype(np.int32)
sub = sub[["Patient", "Weeks", "Patient_Week"]].copy()

test_data_b = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
).copy()
test_data_b["Typical_FVC"] = (
    test_data_b["Base_FVC"].values / test_data_b["Base_Percent"].values
) * 100.0

sub = sub.merge(test_data_b, how="left", on="Patient")
sub.head()



## === cell 5
train_data = train_data.copy()
train_data["Type"] = "train"
sub["Type"] = "test"

data = pd.concat([train_data, sub], axis=0, ignore_index=True)
data.shape, data["Type"].value_counts()



## === cell 6
prediction_col = ["FVC"]
Continuos_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Age",
    "Percent",
    "Base_Percent",
]

missing_cols = [c for c in Continuos_cols if c not in data.columns]
missing_cols



## === cell 7
data["Weeks_raw"] = data["Weeks"].astype(np.int32)
data["Base_Week_raw"] = data["Base_Week"].astype(np.int32)

scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols])

data[Continuos_cols].describe().T



## === cell 8
data["sex_m"] = (data["Sex"] == "Male").astype(np.float32)
data["sex_f"] = (data["Sex"] == "Female").astype(np.float32)

data["sm_es"] = (data["SmokingStatus"] == "Ex-smoker").astype(np.float32)
data["sm_ns"] = (data["SmokingStatus"] == "Never smoked").astype(np.float32)
data["sm_cs"] = (data["SmokingStatus"] == "Currently smokes").astype(np.float32)

unknown_smoke = data[["sm_es", "sm_ns", "sm_cs"]].sum(axis=1) == 0
data.loc[unknown_smoke, "sm_cs"] = 1.0

x_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Age",
    "sex_m",
    "sex_f",
    "sm_es",
    "sm_ns",
    "sm_cs",
]



## === cell 9
train_mask = data["Type"] == "train"
test_mask = data["Type"] == "test"

x_train = data.loc[train_mask, x_cols].values.astype(np.float32)
y_train = data.loc[train_mask, prediction_col].values.astype(np.float32).reshape(-1)

x_test = data.loc[test_mask, x_cols].values.astype(np.float32)

test_patient_week = data.loc[test_mask, "Patient_Week"].values
test_patients = data.loc[test_mask, "Patient"].values.astype(str)

test_weeks_raw = data.loc[test_mask, "Weeks_raw"].values.astype(np.int32)
train_patients = data.loc[train_mask, "Patient"].values.astype(str)
train_weeks_raw = data.loc[train_mask, "Weeks_raw"].values.astype(np.int32)

train_base_weeks_raw = data.loc[train_mask, "Base_Week_raw"].values.astype(np.int32)
test_base_weeks_raw = data.loc[test_mask, "Base_Week_raw"].values.astype(np.int32)

x_train.shape, y_train.shape, x_test.shape, len(test_patient_week)



## === cell 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_pred = np.zeros(x_train.shape[0], dtype=np.float32)
pred_test = np.zeros(x_test.shape[0], dtype=np.float32)

fold_sigmas = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train), 1):
    model = GradientBoostingRegressor(
        random_state=SEED + fold,
        loss="squared_error",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
    )
    model.fit(x_train[tr_idx], y_train[tr_idx])

    va_pred = model.predict(x_train[va_idx]).astype(np.float32)
    oof_pred[va_idx] = va_pred

    resid = (y_train[va_idx] - va_pred).astype(np.float32)
    resid = np.clip(resid, -1000.0, 1000.0)
    sigma = float(np.std(resid))
    fold_sigmas.append(max(sigma, 70.0))

    pred_test += model.predict(x_test).astype(np.float32) / FOLDS

float(np.mean(fold_sigmas)), np.min(fold_sigmas), np.max(fold_sigmas)




## === cell 11
def competition_metric(y_true, y_pred, sigma):
    sigma_c = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return (-np.sqrt(2.0) * delta / sigma_c) - np.log(np.sqrt(2.0) * sigma_c)


oof_resid = (y_train.astype(np.float32) - oof_pred.astype(np.float32)).astype(
    np.float32
)
oof_resid = np.clip(oof_resid, -1000.0, 1000.0)

oof_df = pd.DataFrame(
    {
        "Patient": train_patients,
        "Weeks_raw": train_weeks_raw,
        "Base_Week_raw": train_base_weeks_raw,
        "resid": oof_resid,
    }
)

_LN2 = float(np.log(2.0))
_SQRT2 = float(np.sqrt(2.0))

_SIGMA_EPS = 1e-3


def resid_to_sigma_laplace_robust(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    mad = max(mad, 0.0)
    b = mad / _LN2 if mad > 0 else 0.0  # Laplace scale
    sigma = _SQRT2 * b  # convert to "sigma" used in metric
    return float(max(sigma, _SIGMA_EPS))


global_sigma_robust = float(resid_to_sigma_laplace_robust(oof_resid))
global_sigma_robust = max(global_sigma_robust, 70.0)
global_sigma_std = max(float(np.nanstd(oof_resid.astype(np.float32), ddof=0)), 70.0)
global_sigma = float(0.65 * global_sigma_robust + 0.35 * global_sigma_std)
global_sigma = max(global_sigma, 70.0)

pat_grp = oof_df.groupby("Patient")["resid"]
pat_cnt = pat_grp.size().astype(np.float32)

pat_sigma_rob = pat_grp.apply(resid_to_sigma_laplace_robust).astype(np.float32)
pat_sigma_std = pat_grp.std(ddof=0).astype(np.float32).fillna(0.0)
pat_sigma_raw = np.maximum(pat_sigma_rob.values, pat_sigma_std.values).astype(
    np.float32
)
pat_sigma_raw = pd.Series(pat_sigma_raw, index=pat_sigma_rob.index).astype(np.float32)
pat_sigma_raw = pat_sigma_raw.fillna(global_sigma).astype(np.float32)

y_true = y_train.astype(np.float32)
y_pred = oof_pred.astype(np.float32)

abs_w_test = np.abs(test_weeks_raw.astype(np.float32))
abs_w_train = np.abs(train_weeks_raw.astype(np.float32))

dist_train = np.abs(
    train_weeks_raw.astype(np.float32) - train_base_weeks_raw.astype(np.float32)
)
dist_test = np.abs(
    test_weeks_raw.astype(np.float32) - test_base_weeks_raw.astype(np.float32)
)

dist0_train = np.abs(train_weeks_raw.astype(np.float32) - 0.0)
dist0_test = np.abs(test_weeks_raw.astype(np.float32) - 0.0)

alpha_grid = np.array(
    [0.00, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.22, 0.26], dtype=np.float32
)
scales = np.linspace(0.10, 1.90, 361).astype(np.float32)

k_grid = np.array([3.0, 6.0, 8.0, 12.0, 20.0, 30.0], dtype=np.float32)
week_cap_grid = np.array([1.25, 1.35, 1.45, 1.60, 1.80, 2.05], dtype=np.float32)

beta_grid = np.array([0.00, 0.03, 0.06, 0.09, 0.12], dtype=np.float32)
dist_cap_grid = np.array([1.00, 1.15, 1.30, 1.45, 1.60], dtype=np.float32)

gamma_grid = np.array([0.00, 0.02, 0.04, 0.06], dtype=np.float32)
dist0_cap_grid = np.array([1.00, 1.10, 1.20, 1.30], dtype=np.float32)

best_score = -1e18
best_scale = 1.0
best_alpha = 0.10
best_k = 8.0
best_week_cap = 1.60
best_beta = 0.00
best_dist_cap = 1.00
best_gamma = 0.00
best_dist0_cap = 1.00

patient_sigma_by_k = {}
train_base_sigma_by_k = {}
for k in k_grid:
    w = (pat_cnt / (pat_cnt + k)).astype(np.float32)
    patient_sigma_series = (w * pat_sigma_raw + (1.0 - w) * global_sigma).astype(
        np.float32
    )
    patient_sigma_series = np.clip(patient_sigma_series, 70.0, np.inf).astype(
        np.float32
    )
    patient_sigma_by_k[float(k)] = patient_sigma_series

    s_aligned = patient_sigma_series.reindex(
        pd.Index(train_patients), fill_value=global_sigma
    )
    train_base_sigma_by_k[float(k)] = s_aligned.values.astype(np.float32)

delta = np.minimum(np.abs(y_true - y_pred), 1000.0).astype(np.float32)
A = (np.sqrt(2.0) * delta).astype(np.float32)
LOG_SQRT2 = float(np.log(np.sqrt(2.0)))

scales_f32 = scales.astype(np.float32)
log_scales = np.log(scales_f32).astype(np.float32)

abs_w_train_over100 = (abs_w_train / 100.0).astype(np.float32)
dist_train_over100 = (dist_train / 100.0).astype(np.float32)
dist0_train_over100 = (dist0_train / 100.0).astype(np.float32)

for k in k_grid:
    kf = float(k)
    train_base_sigma = train_base_sigma_by_k[kf]  # (n_train,)

    for week_cap in week_cap_grid:
        wc = float(week_cap)
        for alpha in alpha_grid:
            a = float(alpha)
            week_factor_train = 1.0 + (a * abs_w_train_over100)
            week_factor_train = np.clip(week_factor_train, 1.0, wc).astype(np.float32)

            for dist_cap in dist_cap_grid:
                dc = float(dist_cap)
                for beta in beta_grid:
                    b = float(beta)
                    dist_factor_train = 1.0 + (b * dist_train_over100)
                    dist_factor_train = np.clip(dist_factor_train, 1.0, dc).astype(
                        np.float32
                    )

                    for dist0_cap in dist0_cap_grid:
                        d0c = float(dist0_cap)
                        for gamma in gamma_grid:
                            g = float(gamma)
                            dist0_factor_train = 1.0 + (g * dist0_train_over100)
                            dist0_factor_train = np.clip(
                                dist0_factor_train, 1.0, d0c
                            ).astype(np.float32)

                            sigma_nom = (
                                train_base_sigma
                                * week_factor_train
                                * dist_factor_train
                                * dist0_factor_train
                            ).astype(np.float32)

                            sig_mat = (sigma_nom[:, None] * scales_f32[None, :]).astype(
                                np.float32
                            )
                            sig_mat = np.maximum(sig_mat, 70.0, dtype=np.float32)

                            score_mat = (
                                (-A[:, None] / sig_mat) - np.log(sig_mat) - LOG_SQRT2
                            )
                            scores = score_mat.mean(axis=0).astype(np.float32)

                            j = int(np.argmax(scores))
                            score_best_here = float(scores[j])
                            if score_best_here > best_score:
                                best_score = score_best_here
                                best_scale = float(scales_f32[j])
                                best_alpha = a
                                best_k = kf
                                best_week_cap = wc
                                best_beta = b
                                best_dist_cap = dc
                                best_gamma = g
                                best_dist0_cap = d0c

best_patient_sigma_series = patient_sigma_by_k[best_k]
base_conf = best_patient_sigma_series.reindex(
    pd.Index(test_patients), fill_value=global_sigma
).values.astype(np.float32)
base_conf = np.clip(base_conf, 70.0, np.inf).astype(np.float32)

week_factor_test = 1.0 + best_alpha * (abs_w_test / 100.0)
week_factor_test = np.clip(week_factor_test, 1.0, best_week_cap).astype(np.float32)

dist_factor_test = 1.0 + best_beta * (dist_test / 100.0)
dist_factor_test = np.clip(dist_factor_test, 1.0, best_dist_cap).astype(np.float32)

dist0_factor_test = 1.0 + best_gamma * (dist0_test / 100.0)
dist0_factor_test = np.clip(dist0_factor_test, 1.0, best_dist0_cap).astype(np.float32)

conf = np.clip(
    base_conf * week_factor_test * dist_factor_test * dist0_factor_test * best_scale,
    70.0,
    np.inf,
).astype(np.float32)

fvc_pred = pred_test.astype(np.float32)

subm = pd.DataFrame(
    {"Patient_Week": test_patient_week, "FVC": fvc_pred, "Confidence": conf}
)
subm = sub[["Patient_Week"]].merge(subm, on="Patient_Week", how="left")

subm["FVC"] = subm["FVC"].fillna(np.nanmedian(subm["FVC"].values)).astype(np.float32)
subm["Confidence"] = subm["Confidence"].fillna(global_sigma).astype(np.float32)
subm["Confidence"] = np.maximum(subm["Confidence"].values.astype(np.float32), 70.0)

print("OOF metric (using calibrated sigma):", best_score)
print("Chosen global Confidence scale:", best_scale)
print("Chosen week alpha:", best_alpha)
print("Chosen shrinkage k:", best_k)
print("Chosen week cap:", best_week_cap)
print("Chosen baseline-distance beta:", best_beta)
print("Chosen distance cap:", best_dist_cap)
print("Chosen week0-distance gamma:", best_gamma)
print("Chosen week0-distance cap:", best_dist0_cap)
subm.head(), subm.shape



## === cell 12
subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print("Confidence summary:", pd.Series(subm["Confidence"]).describe())
print(subm.head(10))
