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

-7.084316949960436

# 6. Current score

-15.27891

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76189) has done: 'I fix the environment-breaking TensorFlow import issue by forcing the pure-Python protobuf backend before importing TF (this resolves the `MessageFactory.GetPrototype` crash). Then I correct the broken Kaggle file paths and remove dependencies on missing external pickles/datasets by fitting `data_preparation()` directly from the provided `train.csv` and building the needed `Weight` column safely. I also update the Adam optimizer arguments to the current Keras API (`learning_rate` instead of `lr/decay`) while keeping the same intended hyperparameters via `weight_decay`. Finally, I ensure `X_prediction` is created correctly from `sample_submission.csv` and that a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved -8.76189) has done: 'I fix the environment-breaking TensorFlow/protobuf crash by setting the protobuf backend before *any* TensorFlow-related import and falling back to a safe import if the first attempt fails. Then I fix the Keras compile error by passing `metrics=[score]` (newer Keras requires a list/tuple/dict) while keeping the same model and loss logic. Next I correct the test/sample_submission feature build so `X_prediction` contains the required clinical columns (`Sex`, `SmokingStatus`, etc.) and doesn’t reference a missing `FVC` column, which is what caused the downstream `KeyError: 'Sex'`. Finally, I keep the same training/inference loop but make prediction arrays the right shape and ensure a valid `submission.csv` is always written with the exact required columns.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before any TF import and (if needed) downgrading protobuf in-process to a compatible version (this is the only reliable way to avoid `MessageFactory.GetPrototype` in Kaggle images where TF/protobuf are mismatched). Then I fix the `None values not supported` training crash by ensuring every feature used in `SELECTED_COLUMNS` is fully populated after merges/encoding (fill missing numerics with train medians and missing categoricals with train modes before `data_preparation`). Finally, I keep the exact same model/loss/training loop but make the fold split deterministic and non-overlapping (still 8 folds) to slightly improve stability/score toward your target without changing the core approach, and I always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -15.27891) has done: 'I fix the environment-breaking TensorFlow/protobuf crash by avoiding TensorFlow entirely (the current error happens at import time) and replacing the training/prediction part with a lightweight, deterministic clinical-feature regression that still outputs both FVC and Confidence in the required format. I also fix the immediate runtime error in training (`None values not supported`) by ensuring all model inputs are numeric and fully imputed (even though we won’t use TF afterwards). Finally, I improve score toward your target by predicting per-patient FVC trend using a robust linear fit on each patient’s history (much better than a constant baseline), and set Confidence based on training residual dispersion with clipping aligned to the metric.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error


def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_all(20)



## === cell 1
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing: {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing: {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"

train = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("train/test/sample:", train.shape, raw_test.shape, sample_sub.shape)
print("train cols:", train.columns.tolist())



## === cell 2
base = (
    train.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    .rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC", "Percent": "Base_Percent"})
)
train = train.merge(
    base[["Patient", "Min_week", "Base_FVC", "Base_Percent"]], on="Patient", how="left"
)
train["Base_week"] = train["Weeks"] - train["Min_week"]

print(train[["Patient", "Weeks", "Min_week", "Base_week", "Base_FVC"]].head())



## === cell 3
X_prediction = sample_sub.copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)

baseline_test = raw_test.copy()
baseline_test = baseline_test.rename(
    columns={"Weeks": "Min_week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)

X_prediction = X_prediction.merge(
    baseline_test[
        [
            "Patient",
            "Min_week",
            "Base_FVC",
            "Base_Percent",
            "Age",
            "Sex",
            "SmokingStatus",
        ]
    ],
    how="left",
    on="Patient",
)
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]

print(X_prediction.head())
print("X_prediction columns:", X_prediction.columns.tolist())



## === cell 4
for df in (train, X_prediction):
    for col in ["Sex", "SmokingStatus"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", np.nan)
    for col in [
        "Weeks",
        "FVC",
        "Percent",
        "Age",
        "Min_week",
        "Base_FVC",
        "Base_week",
        "Base_Percent",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

num_fill = {
    "Percent": float(train["Percent"].median()),
    "Age": float(train["Age"].median()),
    "Weeks": float(train["Weeks"].median()),
    "Min_week": float(train["Min_week"].median()),
    "Base_FVC": float(train["Base_FVC"].median()),
    "Base_week": float(train["Base_week"].median()),
}
cat_fill = {}
for col in ["Sex", "SmokingStatus"]:
    mode_val = train[col].mode(dropna=True)
    cat_fill[col] = mode_val.iloc[0] if len(mode_val) else "Unknown"

for df in (train, X_prediction):
    for k, v in num_fill.items():
        if k in df.columns:
            df[k] = df[k].fillna(v)
    for k, v in cat_fill.items():
        if k in df.columns:
            df[k] = df[k].fillna(v).astype(str)

train["FVC"] = (
    pd.to_numeric(train["FVC"], errors="coerce")
    .fillna(float(train["FVC"].median()))
    .astype(float)
)




## === cell 5
def fit_patient_trend(df_patient: pd.DataFrame):
    w = df_patient["Weeks"].values.astype(float)
    y = df_patient["FVC"].values.astype(float)

    if len(y) < 2 or np.all(w == w[0]):
        a = float(np.median(y))
        b = 0.0
        resid = y - a
        return a, b, resid

    X = np.vstack([np.ones_like(w), w]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    resid = y - (a + b * w)

    if len(y) >= 10:
        keep = np.argsort(np.abs(resid))[: int(np.ceil(0.9 * len(y)))]
        Xk, yk = X[keep], y[keep]
        beta2, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
        a, b = float(beta2[0]), float(beta2[1])
        resid = y - (a + b * w)

    return a, b, resid


patient_models = {}
all_resid = []

for pid, g in train.groupby("Patient", sort=False):
    a, b, resid = fit_patient_trend(g)
    patient_models[pid] = (a, b)
    all_resid.append(resid)

all_resid = np.concatenate(all_resid) if len(all_resid) else np.array([0.0])
global_mae = float(np.mean(np.abs(all_resid)))
global_sigma = float(np.std(all_resid))

print("global_mae:", global_mae, "global_sigma:", global_sigma)



## === cell 6
pred_fvc = np.zeros(len(X_prediction), dtype=float)
for i, (pid, wk) in enumerate(
    zip(X_prediction["Patient"].values, X_prediction["Weeks"].values)
):
    if pid in patient_models:
        a, b = patient_models[pid]
        pred_fvc[i] = a + b * float(wk)
    else:
        pred_fvc[i] = float(train["FVC"].median())

conf = max(70.0, 0.9 * global_mae + 0.1 * global_sigma)
pred_conf = np.full(len(X_prediction), conf, dtype=float)

X_prediction["FVC1"] = pred_fvc
X_prediction["Confidence1"] = pred_conf



## === cell 7
subm = sample_sub.copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

pred_map = X_prediction.set_index("Patient_Week")[["FVC1", "Confidence1"]]
subm = subm.merge(pred_map, how="left", left_on="Patient_Week", right_index=True)

mask = ~subm["FVC1"].isna()
subm.loc[mask, "FVC"] = subm.loc[mask, "FVC1"].astype(float)
subm.loc[mask, "Confidence"] = subm.loc[mask, "Confidence1"].astype(float)

otest = pd.read_csv(TEST_PATH)
for i in range(len(otest)):
    pw = f"{otest.Patient.iloc[i]}_{int(otest.Weeks.iloc[i])}"
    subm.loc[subm["Patient_Week"] == pw, "FVC"] = float(otest.FVC.iloc[i])
    subm.loc[subm["Patient_Week"] == pw, "Confidence"] = 0.1

subm["Confidence"] = (
    pd.to_numeric(subm["Confidence"], errors="coerce")
    .fillna(100.0)
    .astype(float)
    .clip(lower=0.1)
)
subm["FVC"] = pd.to_numeric(subm["FVC"], errors="coerce").fillna(3020.0).astype(float)

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Confidence (const, post-clip):", float(submission["Confidence"].median()))
