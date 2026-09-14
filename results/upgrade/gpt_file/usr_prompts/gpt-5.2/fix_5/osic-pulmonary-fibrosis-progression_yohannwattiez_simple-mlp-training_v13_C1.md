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

-7.06085209924736

# 6. Current score

-11.42053

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.53155) has done: 'I fix the environment/runtime issues by removing the TensorFlow import that triggers the protobuf `MessageFactory.GetPrototype` error and replacing it with a lightweight, deterministic baseline that uses only pandas/numpy (which are already available). I also fix broken file paths (`/kaggle/input/prep-data/...`) to use the actual competition dataset CSVs, and remove the missing pickled preprocessor dependency by computing the needed features directly from `train.csv`/`test.csv`. To move score toward the target (and also get a reasonable non-error submission), I fit a simple per-patient linear trend on FVC vs Weeks from the training history and predict test weeks accordingly, while setting `Confidence` to a safe clipped value (>=70). The script run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved -15.58135) has done: 'I keep your per-patient linear trend core logic intact, but make two minimal metric-aligned adjustments to move the score up toward the target. First, I anchor each patient’s trend prediction to the known baseline measurement from `test.csv` (same slope, adjusted intercept) to reduce systematic per-patient bias on the scored weeks. Second, I set `Confidence` using a single calibrated constant derived from training residual MAE (then clipped to >=70), which typically improves the Laplace log-likelihood compared to per-patient std that can be too small/large or unstable for patients with few points. The script still run end-to-end and write a valid `submission.csv` with the required schema and row order matching `sample_submission.csv`.'
- What this solution (achieved -8.53155) has done: 'Your current gap to the target is large and negative (worse than target), so we should improve score with minimal, metric-aligned changes while keeping your per-patient linear trend and baseline anchoring intact. The biggest remaining issue is that you’re anchoring to `test.csv`’s provided FVC (baseline week), but the competition’s scored weeks are typically far from baseline; anchoring should instead use each test patient’s *own last observed* FVC/Week from `train.csv` when available (it is, since test patients appear in train history in this dataset), which reduces bias without changing the model form. Second, we compute the global calibrated sigma using the metric-optimal Laplace MLE (`sqrt(2)*MAE`) on residuals, and use a slightly shrunk per-patient sigma (blended with global) to avoid under/over-confidence while staying stable and deterministic. These are small post-fit calibration/anchoring adjustments that generally move the Laplace log-likelihood upward toward your target without changing the core approach.'
- What this solution (achieved -11.42053) has done: 'I keep your per-patient linear trend + intercept anchoring logic intact, but adjust two metric-facing details that typically improve Laplace log-likelihood without changing the modeling approach. First, I calibrate `Confidence` using the metric-optimal constant sigma computed directly from training absolute residuals and use that single value for all rows (your current blend can over-penalize via `-log(sigma)` when sigma is too large). Second, I compute the residuals used for sigma calibration with the same “last observation anchoring” you use at inference time, so the confidence better matches the actual prediction error distribution. These are minimal, deterministic changes aimed at moving the score up from -8.53 toward your -7.06 target.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd




## === cell 1
def seed_all(seed: int = 20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_all(20)



## === cell 2
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
raw_test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

assert {"Patient", "Weeks", "FVC"}.issubset(train.columns)
assert {"Patient", "Weeks", "FVC"}.issubset(raw_test.columns)
assert {"Patient_Week", "FVC", "Confidence"}.issubset(sample_sub.columns)



## === cell 3
X_prediction = sample_sub[["Patient_Week"]].copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")[0]
X_prediction["Weeks"] = (
    X_prediction["Patient_Week"].str.extract(r".*_(.*)")[0].astype(int)
)

X_prediction = X_prediction.merge(
    raw_test[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]],
    on="Patient",
    how="left",
    suffixes=("", "_base"),
)

X_prediction = X_prediction.rename(
    columns={
        "Weeks_base": "Base_week",
        "FVC_base": "Base_FVC",
        "Percent": "Base_percent",
    }
)

if "Weeks_x" in X_prediction.columns:
    X_prediction = X_prediction.rename(columns={"Weeks_x": "Weeks"})
if "Weeks_y" in X_prediction.columns:
    X_prediction = X_prediction.rename(columns={"Weeks_y": "Base_week"})
if "FVC_y" in X_prediction.columns:
    X_prediction = X_prediction.rename(columns={"FVC_y": "Base_FVC"})
if "FVC_x" in X_prediction.columns:
    X_prediction = X_prediction.drop(columns=["FVC_x"])

for col in ["Patient", "Weeks", "Patient_Week", "Base_week", "Base_FVC"]:
    if col not in X_prediction.columns:
        X_prediction[col] = np.nan

X_prediction["Base_week"] = pd.to_numeric(X_prediction["Base_week"], errors="coerce")
X_prediction["Base_FVC"] = pd.to_numeric(X_prediction["Base_FVC"], errors="coerce")




## === cell 4
def fit_patient_trend(df_patient: pd.DataFrame):
    x = df_patient["Weeks"].to_numpy(dtype=float)
    y = df_patient["FVC"].to_numpy(dtype=float)

    if len(df_patient) < 2 or np.all(x == x[0]):
        return float(np.mean(y)), 0.0

    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 1e-12:
        return float(y_mean), 0.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = float(y_mean - slope * x_mean)
    return intercept, slope


patient_params = {}

for pid, g in train.groupby("Patient", sort=False):
    intercept, slope = fit_patient_trend(g)
    resid_std = 0.0
    if len(g) > 1:
        x = g["Weeks"].to_numpy(dtype=float)
        y = g["FVC"].to_numpy(dtype=float)
        yhat = intercept + slope * x
        resid_std = float(np.std(y - yhat))
    patient_params[pid] = (intercept, slope, resid_std)

global_fvc_mean = float(train["FVC"].mean())
global_fvc_std = float(train["FVC"].std())

train_sorted = train.sort_values(["Patient", "Weeks"], ascending=[True, True])
last_obs = (
    train_sorted.groupby("Patient", sort=False)
    .tail(1)[["Patient", "Weeks", "FVC"]]
    .copy()
)
last_obs = last_obs.rename(columns={"Weeks": "Last_week", "FVC": "Last_FVC"})
last_obs_map = dict(
    zip(
        last_obs["Patient"].values,
        zip(last_obs["Last_week"].values, last_obs["Last_FVC"].values),
    )
)

all_abs_resid_anchored = []
for pid, g in train.groupby("Patient", sort=False):
    intercept, slope, _ = patient_params.get(
        pid, (global_fvc_mean, 0.0, global_fvc_std)
    )

    if pid in last_obs_map:
        lw, lfvc = last_obs_map[pid]
        if np.isfinite(lw) and np.isfinite(lfvc):
            intercept = float(lfvc - slope * float(lw))

    x = g["Weeks"].to_numpy(dtype=float)
    y = g["FVC"].to_numpy(dtype=float)
    yhat = intercept + slope * x
    all_abs_resid_anchored.extend(np.abs(y - yhat).tolist())

global_abs_resid_mae = (
    float(np.mean(all_abs_resid_anchored))
    if len(all_abs_resid_anchored)
    else global_fvc_std
)

calibrated_sigma = float(np.clip(np.sqrt(2.0) * global_abs_resid_mae, 70.0, 400.0))



## === cell 5
pred_fvc = np.empty(len(X_prediction), dtype=float)
pred_conf = np.empty(len(X_prediction), dtype=float)

for i, (pid, w, bw, bfvc) in enumerate(
    zip(
        X_prediction["Patient"].values,
        X_prediction["Weeks"].values,
        X_prediction["Base_week"].values,
        X_prediction["Base_FVC"].values,
    )
):
    params = patient_params.get(pid, None)
    if params is None:
        intercept, slope, resid_std = global_fvc_mean, 0.0, global_fvc_std
    else:
        intercept, slope, resid_std = params

    if pid in last_obs_map:
        lw, lfvc = last_obs_map[pid]
        if np.isfinite(lw) and np.isfinite(lfvc):
            intercept = float(lfvc - slope * float(lw))
    elif np.isfinite(bw) and np.isfinite(bfvc):
        intercept = float(bfvc - slope * float(bw))

    yhat = intercept + slope * float(w)
    pred_fvc[i] = yhat

    pred_conf[i] = calibrated_sigma

pred_fvc = np.clip(pred_fvc, 500.0, 6000.0)



## === cell 6
sub = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": pred_fvc.astype(np.float32),
        "Confidence": pred_conf.astype(np.float32),
    }
)

sub["FVC"] = sub["FVC"].fillna(global_fvc_mean).astype(np.float32)
sub["Confidence"] = (
    sub["Confidence"].fillna(calibrated_sigma).clip(lower=70.0).astype(np.float32)
)

sub = sub[["Patient_Week", "FVC", "Confidence"]]
assert len(sub) == len(sample_sub)



## === cell 7
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Global calibrated sigma used:", calibrated_sigma)
print("Wrote submission.csv with shape:", sub.shape)
