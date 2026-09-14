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

-6.885968373235216

# 6. Current score

-15.27911

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.64042) has done: 'I remove/guard the problematic optional imports that trigger the protobuf `MessageFactory.GetPrototype` crash and keep the rest of the pipeline unchanged. Then I fix the pandas `DataFrame.append` deprecation by using `pd.concat`, which unblocks creation of the combined `data` frame used by later cells. The original notebook also tries to load a non-existent external `model.h5`; since it’s unavailable in this environment, I minimally replace that with a small KFold-trained tabular Keras model that outputs the same 3-quantile predictions the rest of your code expects (so the submission logic remains the same). Finally, I ensure `SmokingStatus` is referenced correctly, replace deprecated `np.float`, and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.64029) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which is a common Kaggle runtime issue. Then I fix the submission post-processing bug where the code tries to inverse-scale `FVC` even though `FVC` was never scaled; instead, I use the model’s predicted median quantile directly as `FVC` in ml. Finally, I keep the same core KFold tabular model and quantile output logic, but ensure confidence is computed robustly (non-negative, clipped to at least 70) and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -24.64046) has done: 'I fix the execution blockers while keeping your existing pipeline and KFold quantile-regression model intact. First, I harden the TensorFlow import against the protobuf `GetPrototype` crash by falling back to a safe environment setup and (if needed) a clean import path. Second, I fix the submission post-processing bug: `FVC` is not part of the scaled feature list, so inverse-scaling via `scaler.data_min_` can’t work; instead, I train/predict in the original ml scale (as your `y_train` already is) and use predicted quantiles directly for `FVC` and `Confidence`. Finally, I make categorical mapping robust to missing values and ensure a valid `submission.csv` is always written with the required columns.'
- What this solution (achieved -24.6404) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by removing the forced pure-Python protobuf setting and instead forcing a compatible protobuf implementation for TF in this environment. I keep the rest of your pipeline (data prep, scaling, KFold quantile-regression tabular model, and submission formatting) unchanged so behavior and semantics remain the same. I also add a small, score-neutral safety fallback: if TensorFlow still fails to import, the code produce a valid submission using a simple per-patient linear trend fit from the training data (so you always get a CSV). This unblocks end-to-end execution and should also improve score substantially versus the current crash/no-run state, while keeping core logic intact when TF works.'
- What this solution (achieved -24.64035) has done: 'I fix the TensorFlow/protobuf import crash that prevents the notebook from running by forcing TensorFlow to use the pure-Python protobuf implementation before importing TF, which is the most common reliable workaround in Kaggle OSIC notebooks. I keep your existing KFold quantile-regression tabular model and training loop unchanged, but I also ensure the TF failure path still produces a valid `submission.csv` as it already does. Additionally, I make the TF import block robust by retrying the import once after setting the protobuf environment variables, so you don’t silently fall into the weaker fallback due to a transient import ordering issue. No changes are made to the model architecture, loss, features, or submission formatting beyond making execution stable end-to-end.'
- What this solution (achieved -9.97744) has done: 'I fix the TensorFlow/protobuf crash that currently stops execution by avoiding TensorFlow entirely (it’s not required here) and using a deterministic, patient-level linear regression with sensible fallbacks; this keeps the pipeline end-to-end and produces a valid `submission.csv`. Then I improve the score toward your target by calibrating predictions and confidence in a metric-aware way: for each test patient, estimate an FVC trend from similar training patients and set `Confidence` to reflect residual uncertainty while respecting the 70-ml clip. These changes are minimal in scope (imports/training block only) and preserve the overall data preparation and submission formatting. The result should run within the time limit and substantially improve over the current very low score by producing non-degenerate, patient-specific predictions with reasonable uncertainty.'
- What this solution (achieved -12.24743) has done: 'Your current score (-9.977) is below the target (-6.886), so we should improve predictions while keeping your patient-level linear-trend + neighbor borrowing logic intact. The biggest gain with minimal change is to make `Confidence` metric-aligned: your code currently outputs `Confidence = |q80-q20|` which is ~1.68×sigma and thus over-penalizes via `-log(sigma)`; we instead output `Confidence ≈ (q80-q20)/1.683`. Next, we anchor each patient’s predicted median trajectory to their known baseline point by blending the borrowed-slope intercept with a gentle global-drift prior (so outlier slopes don’t explode), while preserving the same KNN-median mechanism. These two changes are small, deterministic, and directly target the Laplace log-likelihood without changing the overall approach or I/O.'
- What this solution (achieved -13.31966) has done: 'We keep your fallback patient-level linear trend + KNN slope borrowing exactly as-is, and only make two metric-aligned adjustments that should improve the score toward the target without changing the overall approach. First, we calibrate `Confidence` directly from your constructed `sigma` instead of reconstructing it via quantile spread (which can still be mismatched), because the Laplace log-likelihood strongly rewards well-calibrated (not overly large) sigmas. Second, we slightly reduce the time-based widening (`+ 2.0 * dt`) to a gentler value, because excessive confidence inflation hurts the `-log(sigma)` term across many weeks and is a common cause of scores like -12. These are minimal, localized changes that preserve your pipeline, output schema, and runtime, and they should move the score upward (less negative) toward -6.886.'
- What this solution (achieved -14.81504) has done: 'Your current score (-13.31966) is worse than the target (-6.88597), so we should improve it (make it less negative) with minimal, metric-aligned tweaks. The biggest likely issue is that your fallback creates an overly wide prediction interval (`spread = 0.8*sigma`) that makes the derived quantiles inconsistent with the confidence model; we make the quantiles consistent with a Normal approximation by using `spread = 0.8416*sigma` so (q20,q80) match sigma. Next, we slightly reduce time-based uncertainty growth (`sigma = base_sigma + 1.0*dt`) to `0.5*dt`, because the metric penalizes large sigma via `-log(sigma)` across many rows. Finally, we make KNN weighting distance-aware (weighted median) while keeping the same neighbor-borrowing core logic, which typically improves FVC accuracy without changing the overall approach.'
- What this solution (achieved -16.1604) has done: 'Your current score (-14.815) is far below the target (-6.886), so we should improve it (make it less negative) with minimal, metric-aligned adjustments while keeping your existing fallback logic (per-patient linear fit + KNN slope borrowing + baseline anchoring) intact. The biggest likely gain is reducing systematic FVC bias by borrowing not just slopes but also a “typical residual offset at baseline” from neighbors, then applying that small offset to the anchored intercept (this preserves the same linear trajectory form). Next, we calibrate `Confidence` using neighbor RMSE but shrink it slightly toward the 70ml clip (over-large sigma is strongly penalized by `-log(sigma)`), and we soften the time-uncertainty growth a bit to avoid inflating sigma across many rows. These are localized changes inside the fallback prediction block and keep I/O/submission formatting unchanged.'
- What this solution (achieved -15.27911) has done: 'Your current score (-16.1604) is far below the target (-6.8859), so we should improve (make it less negative) while keeping your fallback linear-trajectory + KNN borrowing core logic unchanged. The most direct metric-aligned fix is to stop generating predictions for *all* weeks in `sample_submission.csv` (1908 rows) and instead predict only the *three scored weeks per patient* (Weeks = 0, 26, 52) derived from `test.csv`, which removes many off-target rows that otherwise drag down the averaged score. While doing that, we keep the same KNN slope/offset borrowing and baseline anchoring, but we also tighten `Confidence` slightly (still clipped at 70) to reduce the `-log(sigma)` penalty now that we’re only scoring a few rows per patient. These are localized changes to the submission construction and the fallback prediction loop; the model form, features, and prediction semantics remain the same.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

TF_AVAILABLE = False
TF_IMPORT_ERROR = "Disabled to avoid protobuf MessageFactory.GetPrototype crash"

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TensorFlow disabled/fallback path active:", TF_IMPORT_ERROR)



## === cell 1
EPOCHS = 5
NUM_IMAGES = 140
BATCH_SIZE = 4
FOLDS = 5
IMAGE_DIM = (NUM_IMAGES, 60, 60)

COMP_DIR = "../input/osic-pulmonary-fibrosis-progression/"
TRAIN_PATH = "../input/osic-pulmonary-fibrosis-progression/train"
TEST_PATH = "../input/osic-pulmonary-fibrosis-progression/test"
SUB_PATH = "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"



## === cell 2
comp_dir = "../input/osic-pulmonary-fibrosis-progression"

train_data = pd.read_csv(os.path.join(comp_dir, "train.csv"))
sub = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))
test_data = pd.read_csv(os.path.join(comp_dir, "test.csv"))

train_data = train_data.drop_duplicates(
    keep=False, subset=["Patient", "Weeks"]
).reset_index(drop=True)



## === cell 3
train_data_u = train_data.drop_duplicates(subset=["Patient"]).copy()
train_data_u = train_data_u.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
train_data_u["Typical_FVC"] = (
    train_data_u["Base_FVC"].values / train_data_u["Base_Percent"].values
) * 100.0
train_data = train_data.merge(
    train_data_u.drop(["Age", "Sex", "SmokingStatus"], axis=1), on="Patient", how="left"
)



## === cell 4
scored_weeks = [0, 26, 52]
sub = pd.DataFrame(
    [(p, w, f"{p}_{w}") for p in test_data["Patient"].unique() for w in scored_weeks],
    columns=["Patient", "Weeks", "Patient_Week"],
)



## === cell 5
test_data = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
test_data["Typical_FVC"] = (
    test_data["Base_FVC"].values / test_data["Base_Percent"].values
) * 100.0
sub = sub.merge(test_data, how="left", on="Patient")



## === cell 6
train_data["Type"] = "train"
sub["Type"] = "test"

data = pd.concat([train_data, sub], axis=0, ignore_index=True)



## === cell 7
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

Categorical_cols = ["Sex", "SmokingStatus"]



## === cell 8
for c in Continuos_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
train_mask = data["Type"] == "train"
for c in Continuos_cols:
    fill_val = float(np.nanmedian(data.loc[train_mask, c].values))
    data[c] = data[c].fillna(fill_val)

scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols].values)



## === cell 9
data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1})
data["Sex"] = (
    data["Sex"].fillna(data.loc[train_mask, "Sex"].mode().iloc[0]).astype(np.int32)
)

smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
data["SmokingStatus"] = data["SmokingStatus"].map(smoke_map)
data["SmokingStatus"] = (
    data["SmokingStatus"]
    .fillna(data.loc[train_mask, "SmokingStatus"].mode().iloc[0])
    .astype(np.int32)
)



## === cell 10
x_cols = ["Weeks", "Base_Week", "Base_FVC", "Sex", "Age"]

x_train = data.loc[data["Type"] == "train", x_cols].values.astype(np.float32)
y_train = data.loc[data["Type"] == "train", prediction_col].values.astype(np.float32)

x_test = data.loc[data["Type"] == "test", x_cols].values.astype(np.float32)

train_patient_ids = data.loc[data["Type"] == "train", "Patient"].values
test_patient_ids = data.loc[data["Type"] == "test", "Patient"].values

print("Shapes:", x_train.shape, y_train.shape, x_test.shape)



## === cell 11
if TF_AVAILABLE:
    pass
else:
    test_pred = None



## === cell 12
if not TF_AVAILABLE:
    tr = train_data.copy()
    te = test_data.copy()

    tr_base = (
        tr.sort_values(["Patient", "Weeks"])
        .groupby("Patient", as_index=False)
        .first()[
            [
                "Patient",
                "Age",
                "Sex",
                "SmokingStatus",
                "Base_FVC",
                "Base_Percent",
                "Base_Week",
            ]
        ]
    )

    slopes = {}
    intercepts = {}
    resid_rmse = {}
    base_resid = {}

    global_median = float(np.median(tr["FVC"].values))

    for pid, grp in tr.groupby("Patient"):
        w = grp["Weeks"].values.astype(np.float64)
        y = grp["FVC"].values.astype(np.float64)
        if len(grp) >= 2 and np.std(w) > 1e-9:
            b, a = np.polyfit(w, y, 1)  # y = b*w + a
            yhat = a + b * w
            rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
            slopes[pid] = float(b)
            intercepts[pid] = float(a)
            resid_rmse[pid] = rmse

            w0 = float(w[np.argmin(w)])
            y0 = float(y[np.argmin(w)])
            base_resid[pid] = float(y0 - (a + b * w0))
        else:
            slopes[pid] = 0.0
            a0 = float(y[0]) if len(y) else global_median
            intercepts[pid] = a0
            resid_rmse[pid] = 250.0
            base_resid[pid] = 0.0

    all_slopes = np.array(list(slopes.values()), dtype=np.float64)
    global_slope = float(np.median(all_slopes)) if len(all_slopes) else 0.0

    sex_map = {"Male": 0, "Female": 1}
    smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    tr_base_enc = tr_base.copy()
    tr_base_enc["Sex_enc"] = tr_base_enc["Sex"].map(sex_map).fillna(0).astype(np.int32)
    tr_base_enc["Smoke_enc"] = (
        tr_base_enc["SmokingStatus"].map(smoke_map).fillna(0).astype(np.int32)
    )

    te_enc = te.copy()
    te_enc["Sex_enc"] = te_enc["Sex"].map(sex_map).fillna(0).astype(np.int32)
    te_enc["Smoke_enc"] = (
        te_enc["SmokingStatus"].map(smoke_map).fillna(0).astype(np.int32)
    )

    feat_cols = ["Age", "Sex_enc", "Smoke_enc", "Base_FVC", "Base_Percent"]
    Xtr = tr_base_enc[feat_cols].astype(np.float64).values
    Xte = te_enc[feat_cols].astype(np.float64).values

    med = np.nanmedian(Xtr, axis=0)
    mad = np.nanmedian(np.abs(Xtr - med), axis=0)
    mad[mad < 1e-6] = 1.0
    Xtr_s = (Xtr - med) / mad
    Xte_s = (Xte - med) / mad

    pid_list = tr_base_enc["Patient"].values
    slope_arr = np.array([slopes[p] for p in pid_list], dtype=np.float64)
    rmse_arr = np.array([resid_rmse[p] for p in pid_list], dtype=np.float64)
    base_resid_arr = np.array([base_resid[p] for p in pid_list], dtype=np.float64)

    KNN = 30  # keep same neighbor count for stability
    preds = []
    confs = []

    base_fvc_by_patient = dict(zip(te["Patient"].values, te["Base_FVC"].values))
    base_week_by_patient = dict(zip(te["Patient"].values, te["Base_Week"].values))

    def weighted_median(values, weights):
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        order = np.argsort(values)
        v = values[order]
        w = weights[order]
        cw = np.cumsum(w)
        cutoff = 0.5 * np.sum(w)
        return float(v[np.searchsorted(cw, cutoff, side="left")])

    for i in range(Xte_s.shape[0]):
        d2 = np.sum((Xtr_s - Xte_s[i]) ** 2, axis=1)
        k = min(KNN, len(d2))
        nn_idx = np.argpartition(d2, k - 1)[:k]

        nn_slopes = slope_arr[nn_idx]
        nn_rmse = rmse_arr[nn_idx]
        nn_d2 = d2[nn_idx]
        nn_base_resid = base_resid_arr[nn_idx]

        wts = 1.0 / (nn_d2 + 1e-6)

        b_knn = weighted_median(nn_slopes, wts)

        resid0 = weighted_median(nn_base_resid, wts)
        resid0 = float(np.clip(resid0, -200.0, 200.0))

        base_sigma = weighted_median(nn_rmse, wts)

        base_sigma = max(float(base_sigma) * 0.75, 70.0)

        shrink = 0.85
        b = shrink * b_knn + (1.0 - shrink) * global_slope

        pid = te.iloc[i]["Patient"]
        base_fvc = float(base_fvc_by_patient.get(pid, global_median))
        base_week = float(base_week_by_patient.get(pid, 0.0))

        a = (base_fvc + resid0) - b * base_week

        preds_i = []
        confs_i = []
        for _, r in sub[sub["Patient"] == pid].iterrows():
            wk = float(r["Weeks"])
            fvc = a + b * wk
            dt = abs(wk - base_week)

            sigma = base_sigma + 0.25 * dt
            sigma = max(sigma, 70.0)

            spread = 0.8416212335729143 * sigma
            preds_i.append([fvc - spread, fvc, fvc + spread])
            confs_i.append(sigma)

        preds.extend(preds_i)
        confs.extend(confs_i)

    test_pred = np.asarray(preds, dtype=np.float32)
    test_conf = np.asarray(confs, dtype=np.float32)



## === cell 13
pred = test_pred.astype(np.float32)

q20_ml = pred[:, 0]
q50_ml = pred[:, 1]
q80_ml = pred[:, 2]

fvc_pred_ml = q50_ml

if "test_conf" in globals() and test_conf is not None:
    conf_ml = test_conf.astype(np.float32)
else:
    Z_DIFF_80_20 = 2.0 * 0.8416212335729143  # ~1.6832424671458286
    conf_ml = np.abs(q80_ml - q20_ml) / Z_DIFF_80_20
conf_ml = np.maximum(conf_ml, 70.0)

sub_out = sub.copy()
sub_out["FVC"] = fvc_pred_ml.astype(np.float32)
sub_out["Confidence"] = conf_ml.astype(np.float32)

subm = sub_out[["Patient_Week", "FVC", "Confidence"]].copy()
subm = subm.replace([np.inf, -np.inf], np.nan)

fallback_fvc = float(np.nanmedian(train_data["FVC"].values))
subm = subm.fillna({"FVC": fallback_fvc, "Confidence": 200.0})

subm.to_csv("submission.csv", index=False)
print(subm.head())
print("Wrote submission.csv with shape:", subm.shape)
print("Columns:", list(subm.columns))
