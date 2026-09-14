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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

-7.1388

# 6. Current score

-11.23608

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -11.02591) has done: 'Your code didn’t yield a Kaggle score mainly because it reads from `../input/...`, which doesn’t match the provided filesystem (`/kaggle/data/...` here), so it likely fails before writing `submission.csv`. I make the smallest change to robustly locate the CSVs from the available paths and ensure the script always reaches the submission-writing cell. To move the metric toward the target (higher is better), I also make a minimal, metric-aware adjustment: set a single global `Confidence` based on the training residual spread (and clipped at 70) instead of a hardcoded 100, while keeping your model and prediction logic unchanged.'
- What this solution (achieved -10.99001) has done: 'Your current pipeline likely underperforms because it feeds the model a “Last FVC” at inference time that is based on your own prior predictions, which creates error accumulation across weeks and hurts the final-three-week accuracy. To move the score upward toward the target while preserving your linear regression core, I change inference to use a patient’s *observed* baseline FVC as `Last FVC` for all weeks (a minimal, metric-consistent stabilization). I also set the fixed `Confidence` using a robust residual scale (MAD→sigma) computed from training residuals, then clip at 70 as required; this usually improves Laplace log-likelihood versus a plain std when there are outliers. All file paths and submission alignment with `sample_submission.csv` stay the same.'
- What this solution (achieved -10.99001) has done: 'You’re currently well below the target (gap ≈ -3.85), so we should cautiously improve the Laplace log-likelihood without changing your model or training loop. The biggest metric-relevant lever left is `Confidence`: with Laplace NLL, the best constant sigma is the mean absolute error (MAE) of residuals (then clipped at 70), not a MAD→normal sigma; switching to a Laplace-consistent estimate usually improves score directly. Second, your features include `Weeks` but not the derived “time since baseline”; keeping core logic intact, we can add `Weeks Passed` consistently to both train/test features (it was computed then dropped), which typically improves linear fit for this task without changing the model class. Finally, we keep your stable inference choice (fixed baseline `Last FVC`) and keep submission alignment exactly as sample.'
- What this solution (achieved -11.39034) has done: 'You’re currently below the target (gap ≈ -3.85), so we make the smallest metric-aligned changes that can lift the Laplace log-likelihood without changing the model class or training loop. The biggest remaining lever is better per-row `Confidence`: for this metric, a constant or near-constant sigma is often suboptimal, so we estimate a patient-specific sigma from training residuals (robust MAE) and then clip at 70 as required. Second, we apply a minimal post-processing to `FVC` predictions (clip to a plausible range derived from training) to reduce extreme errors that get capped at Δ=1000 but still hurt via the log term when sigma is small. Finally, we ensure train/test one-hot columns align deterministically (same dummy columns) so inference matches training features exactly, reducing silent feature mismatch.'
- What this solution (achieved -11.41864) has done: 'We make two metric-aligned, minimal changes to improve the Laplace log-likelihood without changing your linear regression training core. First, we compute the optimal constant `Confidence` for the Laplace metric (the mean clipped absolute error at 1000 on train residuals), and use that constant for every row; this avoids noisy per-patient sigma estimates that often hurt due to the `-log(sigma)` term. Second, we clip predictions to a slightly wider, robust range (1st–99th percentile instead of 0.5–99.5) to reduce harmful tail clipping artifacts while still preventing extreme outliers. Everything else (features, model, inference loop, submission alignment) stays the same and it still write a valid `submission.csv`.'
- What this solution (achieved -11.41864) has done: 'Your score is far below the target (current -11.4186 vs target -7.1388; higher is better), so we should make small, metric-aligned improvements without changing your linear regression core. The biggest controllable lever under this constraint is making `Confidence` better match the evaluation’s Laplace likelihood: using an out-of-sample (CV) residual MAE instead of in-sample residuals avoids overly-optimistic (too small) sigma that hurts the `-log(sigma)` term on test. Second, your current “global sigma” is computed from training predictions made on the same data the model was fit on; switching to patient-grouped CV predictions preserves your model/training loop but yields a more realistic confidence estimate. Everything else (features, model, inference logic, submission alignment and format) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved -11.41864) has done: 'Your current approach is held back by a subtle train/test feature mismatch: you train the linear model with `Sex`/`SmokingStatus` dropped (and without `Weeks Passed` explicitly added), but at inference you build rows containing those one-hot columns plus `Weeks Passed`, then reindex to the model’s feature list—so those extra columns are silently ignored and `Weeks Passed` is missing (filled as 0), degrading predictions. I make the minimal fix to ensure the *same feature engineering* is applied to both train and test: keep `Weeks Passed` in training features and keep one-hot columns consistently in both. I also keep your existing patient-grouped CV confidence estimate (since it’s metric-aligned) and keep the rest of your model/training loop and submission alignment unchanged.'
- What this solution (achieved -11.48093) has done: 'Your current score is well below the target (−11.4186 vs −7.1388; higher is better), so we should make a small, metric-aligned improvement without changing your model class or training loop. The biggest low-risk lever is the mismatch between what you train on (all historical rows) and what you’re evaluated on (final 3 weeks per patient): we can keep the same linear regression but train it on the *same “final-three” style rows* (i.e., for each patient, predict FVC at week t using the previous visit as `Last FVC`). This preserves your “Last FVC / First FVC / Weeks Passed” feature idea, but makes the learned mapping closer to the test-time situation. We also recompute the global `Confidence` from patient-grouped CV on that same “final-three” training subset (still Laplace-optimal MAE with Δ-clipping), which typically improves Laplace log-likelihood. Everything else (feature engineering, inference loop using fixed baseline `Last FVC`, submission alignment/format) stays the same.'
- What this solution (achieved -11.23608) has done: 'We keep your linear regression setup and final-three-style training subset intact, and only adjust two metric-relevant pieces that can lift the Laplace log-likelihood without changing the core approach. First, we compute `Confidence` in a way that is consistent with the Laplace metric by estimating the Laplace scale `b` from patient-grouped CV absolute residuals and converting to the metric’s sigma via `sigma = sqrt(2)*b` (your current code effectively uses `sigma=b`, which is mis-scaled and usually hurts the score). Second, we do the delta-clipping for residuals in the same direction as the metric (clip `|residual|` to 1000 before averaging) and keep the required `sigma>=70` clipping. Everything else (features, model, inference loop, submission alignment/format, paths) stays the same and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

tf = None  # kept as a placeholder name to preserve any downstream references if present

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)




## === cell 1
def _find_competition_file(filename: str) -> str:
    candidates = [
        f"../input/osic-pulmonary-fibrosis-progression/{filename}",
        f"/kaggle/input/osic-pulmonary-fibrosis-progression/{filename}",
        f"/kaggle/data/osic-pulmonary-fibrosis-progression/{filename}",
        f"/kaggle/data/{filename}",
        f"/kaggle/input/{filename}",
        f"../input/{filename}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {filename}. Tried: {candidates}")


train_path = _find_competition_file("train.csv")
test_path = _find_competition_file("test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Loaded train:", train.shape, "from", train_path)
print("Loaded test :", test.shape, "from", test_path)



## === cell 2
train.head()



## === cell 3
train.info()



## === cell 4
test.head()



## === cell 5
test.info()



## === cell 6
train_patients = train.Patient.unique()



## === cell 7
fig, ax = plt.subplots(5, 1, figsize=(10, 20))

for i in range(5):
    patient_log = train[train["Patient"] == train_patients[i]]

    ax[i].set_title(train_patients[i])
    ax[i].plot(patient_log["Weeks"], patient_log["FVC"])



## === cell 8
train = train.sort_values(["Patient", "Weeks"]).reset_index(drop=True)
train.loc[0, "Last FVC"] = train.loc[0, "FVC"]



## === cell 9
for i in range(1, len(train)):
    patient = train.loc[i, "Patient"]
    last_patient = train.loc[i - 1, "Patient"]

    if patient == last_patient:
        train.loc[i, "Last FVC"] = train.loc[i - 1, "FVC"]
    else:
        train.loc[i, "Last FVC"] = train.loc[i, "FVC"]



## === cell 10
train.head()



## === cell 11
patient_list = train.Patient.unique()



## === cell 12
patient_log = train[train["Patient"] == "ID00007637202177411956430"]
patient_log = patient_log.sort_values(by="Weeks")  # ensure sort is applied
patient_log.FVC.values[0]



## === cell 13
start_fvc_dict = {}
start_week_dict = {}

for patient in patient_list:
    patient_log = train[train["Patient"] == patient].sort_values(by="Weeks")

    start_fvc = patient_log.FVC.values[0]
    start_week = patient_log.Weeks.values[0]

    start_fvc_dict[patient] = start_fvc
    start_week_dict[patient] = start_week



## === cell 14
for i in range(len(train)):
    train.loc[i, "First FVC"] = start_fvc_dict[train.loc[i, "Patient"]]
    train.loc[i, "First Week"] = start_week_dict[train.loc[i, "Patient"]]



## === cell 15
train.head()



## === cell 16
train["Weeks Passed"] = train["Weeks"] - train["First Week"]



## === cell 17
train.head()




## === cell 18
def calculate_height(row):
    if row["Sex"] == "Male":
        return row["FVC"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FVC"] / (21.78 - 0.101 * row["Age"])


train["height"] = train.apply(calculate_height, axis=1)



## === cell 19
train.head()



## === cell 20
sex_dummies = pd.get_dummies(train["Sex"])
smoking_dummies = pd.get_dummies(train["SmokingStatus"])



## === cell 21
smoking_dummies.head()



## === cell 22
train = train.join(sex_dummies)
train = train.join(smoking_dummies)



## === cell 23
train.head()



## === cell 24
train = train.drop(columns=["Sex", "SmokingStatus"])



## === cell 25
train_full = train.copy()

train_full["_is_first_visit"] = train_full["Last FVC"].astype(float) == train_full[
    "FVC"
].astype(float)

train_model_df = (
    train_full.loc[~train_full["_is_first_visit"]]
    .sort_values(["Patient", "Weeks"])
    .groupby("Patient", group_keys=False)
    .tail(3)
    .drop(columns=["_is_first_visit"])
    .reset_index(drop=True)
)

print("Training rows (full):", train.shape[0])
print("Training rows (final-three-style subset):", train_model_df.shape[0])

labels = train_model_df.pop("FVC")
patients = train_model_df.pop("Patient")



## === cell 26
train_model_df.head()



## === cell 27
model = linear_model.LinearRegression()



## === cell 28
model.fit(train_model_df, labels)



## === cell 29
plt.bar(train_model_df.columns.values, model.coef_)
plt.xticks(rotation=45)



## === cell 30
predictions = model.predict(train_model_df)

loss = mean_squared_error(labels, predictions, squared=False)
print("Loss (train subset RMSE): {0:.2f}".format(loss))



## === cell 31
train_model_df["FVC"] = labels
train_model_df["prediction"] = predictions
train_model_df["Patient"] = patients



## === cell 32
train_model_df.head()



## === cell 33
plt.scatter(predictions, labels)

plt.xlabel("predictions")
plt.ylabel("FVC (labels)")



## === cell 34
delta = predictions - labels
plt.hist(delta, bins=20)



## === cell 35
fig, ax = plt.subplots(5, 1, figsize=(10, 20))

for i in range(5):
    pid = train_patients[i]
    patient_log = train_model_df[train_model_df["Patient"] == pid]

    ax[i].set_title(pid)
    ax[i].plot(patient_log["Weeks"], patient_log["FVC"], label="truth")
    ax[i].plot(patient_log["Weeks"], patient_log["prediction"], label="prediction")
    ax[i].legend()



## === cell 36
test_feat = test.copy()


def height_from_baseline(row):
    if row["Sex"] == "Male":
        return row["FVC"] / (27.63 - 0.112 * row["Age"])
    else:
        return row["FVC"] / (21.78 - 0.101 * row["Age"])


test_feat["height"] = test_feat.apply(height_from_baseline, axis=1)

test_sex = pd.get_dummies(test_feat["Sex"])
test_smoke = pd.get_dummies(test_feat["SmokingStatus"])
for c in sex_dummies.columns:
    if c not in test_sex.columns:
        test_sex[c] = 0
for c in smoking_dummies.columns:
    if c not in test_smoke.columns:
        test_smoke[c] = 0
test_sex = test_sex[sex_dummies.columns]
test_smoke = test_smoke[smoking_dummies.columns]

test_feat = test_feat.join(test_sex).join(test_smoke)

patient_weeks = []
patients_out = []
weeks_out = []
fvcs = []
confidences = []

train_columns = list(model.feature_names_in_)
X_full = train_model_df[train_columns].copy()
y_full = train_model_df["FVC"].values.astype(float)
patient_full = train_model_df["Patient"].values

oof_pred = np.empty(len(train_model_df), dtype=float)
oof_pred[:] = np.nan

unique_patients = pd.unique(patient_full)
K = 5
rng = np.random.RandomState(RANDOM_SEED)
shuffled = unique_patients.copy()
rng.shuffle(shuffled)
folds = np.array_split(shuffled, K)

for k in range(K):
    val_patients = set(folds[k])
    val_mask = np.array([p in val_patients for p in patient_full], dtype=bool)
    tr_mask = ~val_mask

    m = linear_model.LinearRegression()
    m.fit(X_full.loc[tr_mask], y_full[tr_mask])
    oof_pred[val_mask] = m.predict(X_full.loc[val_mask])

assert np.isfinite(oof_pred).all(), "OOF predictions contain NaNs/Infs; CV failed."

oof_resid = (oof_pred - y_full).astype(float)
oof_abs_clipped = np.minimum(np.abs(oof_resid), 1000.0)
laplace_b = float(np.mean(oof_abs_clipped))
global_laplace_sigma = float(np.sqrt(2.0) * laplace_b)
global_conf = float(max(70.0, global_laplace_sigma))
print(
    "Using global confidence from patient-grouped CV on final-three-style subset (Laplace b=MAE with delta-clip; sigma=sqrt(2)*b; clipped at 70):",
    global_conf,
)

fvc_min = float(np.percentile(train_full["FVC"].values.astype(float), 1.0))
fvc_max = float(np.percentile(train_full["FVC"].values.astype(float), 99.0))
print("Clipping predicted FVC to training-derived range:", (fvc_min, fvc_max))

for patient in test_feat.Patient.unique():
    patient_row = test_feat[test_feat["Patient"] == patient].iloc[0]

    start_week = float(patient_row["Weeks"])
    fvc0 = float(patient_row["FVC"])
    percent0 = float(patient_row["Percent"])
    age0 = float(patient_row["Age"])
    height0 = float(patient_row["height"])

    last_fvc_fixed = fvc0

    for j in range(-12, 134):
        week = float(j)

        row_dict = {
            "Weeks": week,
            "Weeks Passed": week - start_week,
            "Percent": percent0,
            "Age": age0,
            "Last FVC": last_fvc_fixed,
            "First FVC": fvc0,
            "First Week": start_week,
            "height": height0,
        }

        for col in test_sex.columns:
            row_dict[col] = float(patient_row.get(col, 0.0))
        for col in test_smoke.columns:
            row_dict[col] = float(patient_row.get(col, 0.0))

        X_row = pd.DataFrame([row_dict]).reindex(columns=train_columns, fill_value=0.0)

        prediction = float(model.predict(X_row)[0])
        prediction = float(np.clip(prediction, fvc_min, fvc_max))

        patient_weeks.append(f"{patient}_{j}")
        patients_out.append(patient)
        weeks_out.append(j)
        fvcs.append(prediction)
        confidences.append(global_conf)

submission = pd.DataFrame(
    data={
        "Patient_Week": patient_weeks,
        "Patient": patients_out,
        "Weeks": weeks_out,
        "FVC": fvcs,
        "Confidence": confidences,
    }
)



## === cell 37
fig, ax = plt.subplots(5, 1, figsize=(10, 20))

patient_list = list(test.Patient.unique())
for i in range(min(5, len(patient_list))):
    patient_log = submission[submission["Patient"] == patient_list[i]]

    ax[i].set_title(patient_list[i])
    ax[i].plot(patient_log["Weeks"], patient_log["FVC"])



## === cell 38
submission = submission.drop(columns=["Patient", "Weeks"])



## === cell 39
submission.head()



## === cell 40
submission = submission[["Patient_Week", "FVC", "Confidence"]]

sample_path = _find_competition_file("sample_submission.csv")
sample = pd.read_csv(sample_path)
submission = sample[["Patient_Week"]].merge(submission, on="Patient_Week", how="left")
assert (
    submission["FVC"].notna().all() and submission["Confidence"].notna().all()
), "Missing predictions for some Patient_Week"

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", list(submission.columns))
print(submission.head())
