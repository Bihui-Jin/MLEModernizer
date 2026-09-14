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

-6.982957922028886

# 6. Current score

-9.58282

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved -9.58282) has done: 'I fix the environment-breaking TensorFlow import issue by removing the TensorFlow/Keras dependency entirely (it is currently unusable in this Kaggle runtime due to a protobuf/Keras incompatibility), while keeping the same “predict FVC + calibrated uncertainty” evaluation semantics required by the competition. I also correct all broken input paths to use the provided `/kaggle/input/osic-pulmonary-fibrosis-progression/` dataset, and remove references to missing external pickles (`data_prep`, `list_patient_score`). To keep the core idea intact (tabular clinical model predicting FVC over weeks with a confidence), I replace the unavailable neural net with a simple per-patient linear trend model fitted from the training history, then generate predictions for the sample submission weeks and set a reasonable confidence based on training residuals. Finally, the script always write a valid `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os
import random
import time
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error



## === cell 2
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_all(20)



## === cell 3
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_path), f"Missing: {train_path}"
assert os.path.exists(test_path), f"Missing: {test_path}"
assert os.path.exists(sample_path), f"Missing: {sample_path}"



## === cell 4
train = pd.read_csv(train_path)
raw_test = pd.read_csv(test_path)
X_prediction = pd.read_csv(sample_path)

req_cols = ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
for c in req_cols:
    assert c in train.columns, f"train missing {c}"
    assert c in raw_test.columns, f"test missing {c}"

assert set(X_prediction.columns) >= {"Patient_Week", "FVC", "Confidence"}



## === cell 5
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.8
EPOCH = 250
BATCH_SIZE = 128



## === cell 6
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")[0]
X_prediction["Weeks"] = (
    X_prediction["Patient_Week"].str.extract(r".*_(.*)")[0].astype(int)
)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]].copy()



## === cell 7
C1, C2 = 70.0, 1000.0


def laplace_metric_np(y_true_fvc, y_pred_fvc, sigma):
    sigma_clip = np.maximum(sigma, C1)
    delta = np.minimum(np.abs(y_true_fvc - y_pred_fvc), C2)
    sq2 = np.sqrt(2.0)
    metric = -(sq2 * delta) / sigma_clip - np.log(sq2 * sigma_clip)
    return float(np.mean(metric))




## === cell 8


def eval_score_np(y_true_fvc, y_pred_fvc, sigma):
    return laplace_metric_np(
        np.asarray(y_true_fvc), np.asarray(y_pred_fvc), np.asarray(sigma)
    )




## === cell 9
SELECTED_COLUMNS = [
    "Weeks",
    "Percent",
    "Age",
    "Sex",
    "Min_week",
    "Base_FVC",
    "Base_week",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
    "Patient_Class",
]


def fit_patient_trend(df_patient):
    """
    Fit FVC ~ a + b*Weeks for a single patient.
    Returns (a, b, resid_mae).
    """
    w = df_patient["Weeks"].values.astype(float)
    y = df_patient["FVC"].values.astype(float)
    if len(df_patient) < 2 or np.all(w == w[0]):
        a = float(np.median(y))
        b = 0.0
        resid = np.abs(y - a)
    else:
        A = np.vstack([np.ones_like(w), w]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        resid = np.abs(y - (a + b * w))
    resid_mae = float(np.mean(resid)) if len(resid) else 200.0
    return a, b, resid_mae




## === cell 10
patient_models = {}
resid_maes = []

for pid, g in train.groupby("Patient"):
    a, b, r = fit_patient_trend(g)
    patient_models[pid] = (a, b, r)
    resid_maes.append(r)

global_resid_mae = float(np.median(resid_maes)) if len(resid_maes) else 250.0

all_w = train["Weeks"].values.astype(float)
all_y = train["FVC"].values.astype(float)
A = np.vstack([np.ones_like(all_w), all_w]).T
coef_all, *_ = np.linalg.lstsq(A, all_y, rcond=None)
a_all, b_all = float(coef_all[0]), float(coef_all[1])



## === cell 11
test_base = raw_test.set_index("Patient")[
    ["Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
].copy()
test_base = test_base.rename(columns={"Weeks": "BaseWeeks", "FVC": "BaseFVC"})



## === cell 12
pred_fvc = np.zeros(len(X_prediction), dtype=float)
pred_sigma = np.zeros(len(X_prediction), dtype=float)

for i, row in enumerate(X_prediction.itertuples(index=False)):
    pid = row.Patient
    wk = float(row.Weeks)

    if pid in patient_models:
        a, b, r = patient_models[pid]
        b_use = 0.85 * b + 0.15 * b_all
        fvc_hat = a + b_use * wk
        sigma_hat = max(70.0, 1.25 * r)
    else:
        if pid in test_base.index:
            base_wk = float(test_base.loc[pid, "BaseWeeks"])
            base_fvc = float(test_base.loc[pid, "BaseFVC"])
            fvc_hat = base_fvc + b_all * (wk - base_wk)
        else:
            fvc_hat = a_all + b_all * wk
        sigma_hat = max(70.0, 1.25 * global_resid_mae)

    pred_fvc[i] = fvc_hat
    pred_sigma[i] = sigma_hat



## === cell 13
subm = X_prediction[["Patient_Week"]].copy()
subm["FVC"] = pred_fvc
subm["Confidence"] = pred_sigma

subm["FVC"] = subm["FVC"].round().astype(int)
subm["Confidence"] = subm["Confidence"].clip(lower=70).round().astype(int)



## === cell 14
for r in raw_test.itertuples(index=False):
    key = f"{r.Patient}_{int(r.Weeks)}"
    m = subm["Patient_Week"].values == key
    if np.any(m):
        subm.loc[m, "FVC"] = int(round(r.FVC))
        subm.loc[m, "Confidence"] = 70  # minimum effective confidence per metric



## === cell 15
subm = subm[["Patient_Week", "FVC", "Confidence"]]
subm.to_csv("submission.csv", index=False)

print(subm.head())
print("Wrote submission.csv with shape:", subm.shape)
