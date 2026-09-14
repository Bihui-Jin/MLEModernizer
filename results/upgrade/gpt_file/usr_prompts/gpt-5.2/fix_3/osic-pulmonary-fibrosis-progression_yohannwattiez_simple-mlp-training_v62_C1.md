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

# 5. Target score

-7.021343122877855

# 6. Current score

-7.98693

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.75147) has done: 'I fix the environment/runtime blockers first: TensorFlow import crash (protobuf incompat) by switching to a pure scikit-learn baseline that doesn’t require TF, and I correct the broken Kaggle input paths to the provided `/kaggle/input/...` dataset. Then I preserve the “clinical-only” core intent (no CT) by fitting a simple per-patient linear trend on `Weeks` using the patient’s training history, falling back to a global regression when the patient is unseen. Finally, I generate `Patient_Week,FVC,Confidence` exactly matching `sample_submission.csv`, ensuring it writes `submission.csv` end-to-end.'
- What this solution (achieved -7.98693) has done: 'Your current score (-7.75147) is worse than the target (-7.02134), so we should improve (increase) the score with minimal risk while keeping the same “clinical-only + per-patient trend” core logic. The biggest low-risk gain here is calibrating the predicted `Confidence`: using MAE directly as sigma is miscalibrated for Laplace log-likelihood, and it’s better to use a robust sigma estimated from residuals and a single global scaling factor fitted on train. I keep the same per-patient linear/constant models and the same global fallback for unseen patients, but compute per-patient sigma from median absolute residuals (Laplace-consistent) with a small-sample fallback, then tune one scalar multiplier `k` on the training set to maximize the competition metric (with sigma clipping at 70). I also remove submission rounding (unnecessary and slightly harmful) while preserving exact submission schema and writing `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import random
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error




## === cell 1
def seed_all(seed: int = 20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_all(20)



## === cell 2
DATA_ROOT = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)




## === cell 3
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Sex"] = out["Sex"].fillna("Unknown")
    out["SmokingStatus"] = out["SmokingStatus"].fillna("Unknown")
    out["sex_male"] = (out["Sex"] == "Male").astype(int)
    out["sex_female"] = (out["Sex"] == "Female").astype(int)

    out["smoke_current"] = (out["SmokingStatus"] == "Currently smokes").astype(int)
    out["smoke_ex"] = (out["SmokingStatus"] == "Ex-smoker").astype(int)
    out["smoke_never"] = (out["SmokingStatus"] == "Never smoked").astype(int)

    for c in ["Percent", "Age", "Weeks", "FVC"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Percent"] = out["Percent"].fillna(out["Percent"].median())
    out["Age"] = out["Age"].fillna(out["Age"].median())
    return out


train_fe = add_basic_features(train)
test_fe = add_basic_features(test)

FEATURES_GLOBAL = [
    "Weeks",
    "Percent",
    "Age",
    "sex_male",
    "sex_female",
    "smoke_current",
    "smoke_ex",
    "smoke_never",
]



## === cell 4
global_model = LinearRegression()
global_model.fit(train_fe[FEATURES_GLOBAL], train_fe["FVC"].values)

global_pred_train = global_model.predict(train_fe[FEATURES_GLOBAL])
global_mae = mean_absolute_error(train_fe["FVC"].values, global_pred_train)

patient_models = {}
patient_sigmas = {}

LN2 = np.log(2.0)
for pid, g in train_fe.groupby("Patient"):
    if len(g) >= 2:
        m = LinearRegression()
        m.fit(g[["Weeks"]].values, g["FVC"].values)
        pred = m.predict(g[["Weeks"]].values)
        abs_err = np.abs(g["FVC"].values - pred)

        med_abs = float(np.median(abs_err))
        if np.isfinite(med_abs) and med_abs > 0:
            sigma_hat = float((np.sqrt(2.0) * med_abs) / LN2)
        else:
            sigma_hat = float(global_mae)

        patient_models[pid] = ("linear", m)
        patient_sigmas[pid] = sigma_hat
    else:
        const_fvc = float(g["FVC"].iloc[0])
        patient_models[pid] = ("const", const_fvc)
        patient_sigmas[pid] = float(global_mae)




## === cell 5
def laplace_metric_np(y_true, y_pred, sigma):
    sigma_clip = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return -(np.sqrt(2.0) * delta) / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)


train_patients = train_fe["Patient"].values
train_weeks = train_fe["Weeks"].values.astype(np.float32)
y_true = train_fe["FVC"].values.astype(np.float32)

y_pred = np.zeros_like(y_true, dtype=np.float32)
sigma_base = np.zeros_like(y_true, dtype=np.float32)

for i, (pid, wk) in enumerate(zip(train_patients, train_weeks)):
    kind, model = patient_models[pid]
    if kind == "linear":
        y_pred[i] = float(model.predict(np.array([[wk]], dtype=np.float32))[0])
    else:
        y_pred[i] = float(model)
    sigma_base[i] = float(patient_sigmas.get(pid, global_mae))

k_grid = np.linspace(0.6, 1.8, 61).astype(np.float32)
best_k = 1.0
best_score = -1e18
for k in k_grid:
    score = float(np.mean(laplace_metric_np(y_true, y_pred, sigma_base * k)))
    if score > best_score:
        best_score = score
        best_k = float(k)

print(
    f"Calibrated confidence scale best_k={best_k:.3f} (train mean metric {best_score:.6f})"
)



## === cell 6
sub = sample_sub.copy()

sub["Patient"] = sub["Patient_Week"].str.extract(r"^(.*)_")[0]
sub["Weeks"] = sub["Patient_Week"].str.extract(r"_(\-?\d+)$")[0].astype(int)

sub = sub.merge(
    test_fe[
        [
            "Patient",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "sex_male",
            "sex_female",
            "smoke_current",
            "smoke_ex",
            "smoke_never",
        ]
    ],
    on="Patient",
    how="left",
)

fvc_pred = np.zeros(len(sub), dtype=np.float32)
conf_pred = np.zeros(len(sub), dtype=np.float32)

for i, row in sub.iterrows():
    pid = row["Patient"]
    week = float(row["Weeks"])

    if pid in patient_models:
        kind, model = patient_models[pid]
        if kind == "linear":
            fvc_hat = float(model.predict(np.array([[week]], dtype=np.float32))[0])
        else:
            fvc_hat = float(model)
        conf = float(patient_sigmas.get(pid, global_mae))
    else:
        xg = row[FEATURES_GLOBAL].values.astype(np.float32).reshape(1, -1)
        fvc_hat = float(global_model.predict(xg)[0])
        conf = float(global_mae)

    fvc_pred[i] = fvc_hat
    conf_pred[i] = conf

sub["FVC"] = fvc_pred
sub["Confidence"] = conf_pred



## === cell 7
sub["Confidence"] = (sub["Confidence"].astype(float) * best_k).clip(lower=70.0)

test_pw = test.copy()
test_pw["Patient_Week"] = (
    test_pw["Patient"].astype(str) + "_" + test_pw["Weeks"].astype(str)
)

known_map_fvc = dict(zip(test_pw["Patient_Week"], test_pw["FVC"]))
mask_known = sub["Patient_Week"].isin(known_map_fvc.keys())
sub.loc[mask_known, "FVC"] = (
    sub.loc[mask_known, "Patient_Week"].map(known_map_fvc).astype(float)
)
sub.loc[mask_known, "Confidence"] = 70.0



## === cell 8
submission = sub[["Patient_Week", "FVC", "Confidence"]].copy()

submission["FVC"] = submission["FVC"].astype(float)
submission["Confidence"] = submission["Confidence"].astype(float)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["Patient_Week", "FVC", "Confidence"]
assert submission["Patient_Week"].isnull().sum() == 0

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
