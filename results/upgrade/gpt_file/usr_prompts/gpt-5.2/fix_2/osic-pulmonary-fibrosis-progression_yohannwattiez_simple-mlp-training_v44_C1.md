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

-6.964932511772696

# 6. Current score

-12.47955

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved -12.47955) has done: 'I fix the environment-breaking TensorFlow import issue by switching to a pure NumPy/scikit-learn approach while keeping the overall “predict FVC + confidence per Patient_Week” semantics and producing a valid `submission.csv`. I also fix the broken file paths by reading from the provided `/kaggle/input/osic-pulmonary-fibrosis-progression/` dataset, and remove dependencies on missing external pickles/datasets. To move score toward the target (and avoid the very poor constant-3020 baseline), I fit a simple per-patient linear trend on train data and use it to extrapolate FVC for the requested weeks; confidence be calibrated from train residuals and clipped to the competition’s minimum of 70. The script run end-to-end within time limits and write a correctly formatted submission.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error




## === cell 1
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_all(20)



## === cell 2

DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing: {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"

train = pd.read_csv(TRAIN_CSV)
raw_test = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)



## === cell 3
X_pred = sample_sub[["Patient_Week"]].copy()
X_pred["Patient"] = X_pred["Patient_Week"].str.extract(r"^(.*)_")[0]
X_pred["Weeks"] = X_pred["Patient_Week"].str.extract(r"_(.*)$")[0].astype(int)

X_pred = X_pred.merge(
    raw_test[
        ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].rename(columns={"Weeks": "Base_Week", "FVC": "Base_FVC"}),
    on="Patient",
    how="left",
)

if X_pred["Base_FVC"].isna().any():
    missing = X_pred.loc[X_pred["Base_FVC"].isna(), "Patient"].unique()[:5]
    raise ValueError(f"Missing baseline info for some patients, example: {missing}")



## === cell 4

global_fvc_median = float(train["FVC"].median())
global_slope_median = float(
    train.groupby("Patient")
    .apply(
        lambda df: (
            np.polyfit(df["Weeks"].values, df["FVC"].values, 1)[0]
            if df["Weeks"].nunique() > 1
            else np.nan
        )
    )
    .median()
)
if not np.isfinite(global_slope_median):
    global_slope_median = 0.0

patient_models = {}
patient_sigmas = {}

residuals_all = []

for pid, df in train.groupby("Patient"):
    d = df.sort_values("Weeks")
    x = d["Weeks"].values.astype(float)
    y = d["FVC"].values.astype(float)

    if len(d) >= 2 and np.unique(x).size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope = global_slope_median
        intercept = float(y.mean() - slope * x.mean())

    yhat = intercept + slope * x
    resid = y - yhat
    residuals_all.append(resid)

    mae = float(np.mean(np.abs(resid))) if resid.size else 200.0
    sigma = max(70.0, 1.5 * mae)

    patient_models[pid] = (intercept, slope)
    patient_sigmas[pid] = sigma

residuals_all = (
    np.concatenate(residuals_all) if len(residuals_all) else np.array([200.0])
)
global_mae = float(np.mean(np.abs(residuals_all)))
global_sigma = max(70.0, 1.5 * global_mae)




## === cell 5
def predict_row(pid, week):
    if pid in patient_models:
        intercept, slope = patient_models[pid]
        fvc = intercept + slope * float(week)
        sigma = patient_sigmas.get(pid, global_sigma)
    else:
        fvc = global_fvc_median
        sigma = global_sigma
    return fvc, sigma


pred_fvc = np.zeros(len(X_pred), dtype=np.float32)
pred_conf = np.zeros(len(X_pred), dtype=np.float32)

for i, (pid, wk) in enumerate(zip(X_pred["Patient"].values, X_pred["Weeks"].values)):
    f, s = predict_row(pid, wk)
    pred_fvc[i] = f
    pred_conf[i] = s

pred_fvc = np.where(np.isfinite(pred_fvc), pred_fvc, global_fvc_median).astype(
    np.float32
)
pred_conf = np.where(np.isfinite(pred_conf), pred_conf, global_sigma).astype(np.float32)
pred_conf = np.clip(pred_conf, 70.0, 1000.0)



## === cell 6
baseline_map = raw_test.set_index(["Patient", "Weeks"])["FVC"].to_dict()

for i, (pid, wk) in enumerate(zip(X_pred["Patient"].values, X_pred["Weeks"].values)):
    key = (pid, int(wk))
    if key in baseline_map:
        pred_fvc[i] = float(baseline_map[key])
        pred_conf[i] = 70.0



## === cell 7
subm = pd.DataFrame(
    {
        "Patient_Week": X_pred["Patient_Week"].values,
        "FVC": np.round(pred_fvc).astype(int),
        "Confidence": pred_conf.astype(float),
    }
)

subm = sample_sub[["Patient_Week"]].merge(subm, on="Patient_Week", how="left")

assert subm.shape[0] == sample_sub.shape[0]
assert list(subm.columns) == ["Patient_Week", "FVC", "Confidence"]
if subm[["FVC", "Confidence"]].isna().any().any():
    raise ValueError("Submission contains NaNs; aborting.")

subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print(subm.head())
