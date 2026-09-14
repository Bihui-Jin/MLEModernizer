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

-6.8983417380335

# 6. Current score

-8.10411

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.19985) has done: 'I fix the immediate runtime/import failure by removing the unused `pydicom`/image pipeline that triggers the protobuf `MessageFactory.GetPrototype` error, since the provided environment doesn’t support it reliably and the external `model.h5` is also missing. I also fix pandas 2.x incompatibilities (`DataFrame.append`, `np.float`) and the smoking-status column typo so tabular preprocessing runs end-to-end. Because the pretrained model file isn’t available, I keep the same tabular feature set and train a small Keras MLP to predict FVC, then generate a valid `submission.csv` with the required columns. Finally, I set a stable, reasonable constant `Confidence` (>=70) to produce a valid Laplace metric submission without changing evaluation semantics.'
- What this solution (achieved -7.66229) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` incompatibility seen in some Kaggle images. I also make the training target consistent with the model output (currently you’re training a 1-unit regression model against a 1-column target but later you treat the output as if it were quantiles), and I keep the model architecture/training loop the same. Finally, I slightly improve metric-alignment without changing core modeling by using a per-row, data-driven `Confidence` derived from training residuals (clipped to >=70), which should move the score toward the target band versus a fixed constant.'
- What this solution (achieved -7.90816) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (it’s not required for this tabular solution) and replacing the Keras MLP with an equivalent small sklearn MLP regressor, keeping the same “2 hidden layers of 64 ReLU units” core model idea. I also correct a silent but major preprocessing bug: you scale `Percent` but your features use `Age` etc., and the base columns (`Base_Percent`, `Typical_FVC`) weren’t included in the required-columns check, so they could become NaN without being caught; this fix is score-positive and still within the same intended feature set. Finally, I keep the calibrated constant `Confidence` derived from training residuals (clipped to ≥70) and ensure the submission is written as `submission.csv` with the exact required columns.'
- What this solution (achieved -7.8638) has done: 'We keep your tabular MLPRegressor core exactly the same, but make two small, metric-aligned adjustments that typically improve the Laplace log-likelihood without changing the modeling approach. First, we train on a more stable target by predicting the *delta from the baseline FVC* (FVC - Base_FVC) and then add Base_FVC back at inference; this usually reduces patient-level bias and improves generalization with minimal logic change. Second, we set `Confidence` using an out-of-fold (OOF) residual estimate (GroupKFold by Patient) instead of in-sample residuals, which makes sigma less over-optimistic and tends to improve the competition metric. The submission format/path remain identical and the script still runs end-to-end within constraints.'
- What this solution (achieved -7.74976) has done: 'We make two metric-aligned tweaks that keep your core MLPRegressor and delta-from-baseline target unchanged, but should improve the Laplace log-likelihood toward the target by reducing systematic error and better-calibrating sigma. First, we add back `Typical_FVC`, `Percent`, and `Base_Percent` into `x_cols` (they’re already engineered/scaled but currently unused), which usually improves FVC accuracy without changing the modeling approach. Second, we replace the single global `Confidence` with a per-row confidence that grows with distance from baseline week using OOF residual spread, which typically improves the metric versus an overconfident constant sigma while staying legitimate and simple. Submission writing/format and all paths remain the same.'
- What this solution (achieved -7.76771) has done: 'To move the score upward toward the target with minimal change, I keep your exact tabular MLPRegressor + delta-from-baseline setup, but fix one metric-relevant issue: you currently add **scaled** `Base_FVC` back to the predicted delta (because `Base_FVC` was MinMax-scaled), which mis-scales predictions and hurts FVC accuracy. I preserve the scaled feature columns for training, but also store the **original (unscaled)** `Base_FVC`/`Weeks`/`Base_Week` before scaling and use those originals when reconstructing `pred_fvc` and computing `dt` for confidence binning. This keeps core logic identical while aligning units correctly, which should improve the Laplace log-likelihood toward your target. Submission writing and paths remain unchanged.'
- What this solution (achieved -7.76771) has done: 'We keep your exact tabular MLPRegressor + “predict delta from baseline” approach, but fix one remaining data issue that can suppress score: you’re currently taking the “base row” from an arbitrary week per patient (because `drop_duplicates` without sorting may pick a non-baseline measurement). I minimally change this to define `Base_*` using each patient’s Week=0 row when available (otherwise fallback to that patient’s earliest week), and keep everything else (features, scaling, OOF sigma logic, training) the same. This should improve FVC accuracy (and thus Laplace log-likelihood) without changing evaluation semantics. Submission generation stays identical and still writes `submission.csv`.'
- What this solution (achieved -8.10411) has done: 'We keep your exact MLPRegressor + delta-from-baseline setup, but fix two metric-relevant issues that can hold the score back. First, your current `train_data.drop_duplicates(keep=False, ...)` silently drops *all* rows for duplicated Patient/Week pairs; switching to `keep="first"` avoids losing signal and usually improves generalization with no modeling change. Second, your confidence is currently a piecewise-constant function of time distance; we keep the same OOF-residual calibration approach but make `Confidence` a smooth, monotonic function of `dt` (with the same OOF-derived scale and still clipped at 70), which typically improves the Laplace log-likelihood by reducing overconfidence at larger horizons. Everything else (features, scaling, CV grouping, model hyperparameters, submission format/path) stays the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
EPOCHS = 50
BATCH_SIZE = 32  # kept for parity; sklearn uses batch_size separately below

COMP_DIR = "../input/osic-pulmonary-fibrosis-progression/"
if not os.path.exists(COMP_DIR):
    COMP_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"

TRAIN_CSV = os.path.join(COMP_DIR, "train.csv")
TEST_CSV = os.path.join(COMP_DIR, "test.csv")
SUB_CSV = os.path.join(COMP_DIR, "sample_submission.csv")

train_data = pd.read_csv(TRAIN_CSV)
test_data = pd.read_csv(TEST_CSV)
sub = pd.read_csv(SUB_CSV)

train_data = train_data.drop_duplicates(
    subset=["Patient", "Weeks"], keep="first"
).reset_index(drop=True)



## === cell 2
train_sorted = train_data.sort_values(["Patient", "Weeks"]).reset_index(drop=True)
has_w0 = train_sorted["Weeks"].eq(0)

base_rows = []
for pid, g in train_sorted.groupby("Patient", sort=False):
    g0 = g[g["Weeks"] == 0]
    if len(g0) > 0:
        base_rows.append(g0.iloc[0])
    else:
        base_rows.append(g.iloc[0])  # earliest week fallback
train_data_u = pd.DataFrame(base_rows).reset_index(drop=True)

train_data_u = train_data_u.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
train_data_u["Typical_FVC"] = (
    train_data_u["Base_FVC"].values / train_data_u["Base_Percent"].values
) * 100.0

train_data = train_data.merge(
    train_data_u.drop(["Age", "Sex", "SmokingStatus"], axis=1),
    on="Patient",
    how="left",
)



## === cell 3
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub.drop(["Confidence"], axis=1)
sub = sub[["Patient", "Weeks", "Patient_Week"]]



## === cell 4
test_base = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
).copy()
test_base["Typical_FVC"] = (
    test_base["Base_FVC"].values / test_base["Base_Percent"].values
) * 100.0

sub = sub.merge(test_base, how="left", on="Patient")



## === cell 5
train_data["Type"] = "train"
sub["Type"] = "test"

data = pd.concat([train_data, sub], axis=0, ignore_index=True)

needed_cols = [
    "Patient",
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Percent",
    "Base_Percent",
    "Typical_FVC",
    "Age",
    "Sex",
    "SmokingStatus",
    "Type",
]
missing = [c for c in needed_cols if c not in data.columns]
if missing:
    raise ValueError(f"Missing required columns after merge: {missing}")



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

data["Weeks_orig"] = data["Weeks"].astype(np.float32)
data["Base_Week_orig"] = data["Base_Week"].astype(np.float32)
data["Base_FVC_orig"] = data["Base_FVC"].astype(np.float32)

scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols])



## === cell 7
sex_m = np.zeros((len(data), 1), dtype=np.float32)
sex_f = np.zeros((len(data), 1), dtype=np.float32)
sm_es = np.zeros((len(data), 1), dtype=np.float32)
sm_ns = np.zeros((len(data), 1), dtype=np.float32)
sm_cs = np.zeros((len(data), 1), dtype=np.float32)

sex_vals = data["Sex"].fillna("Unknown").values
smoke_vals = data["SmokingStatus"].fillna("Unknown").values

for i in range(len(data)):
    if sex_vals[i] == "Male":
        sex_m[i] = 1.0
    elif sex_vals[i] == "Female":
        sex_f[i] = 1.0

for i in range(len(data)):
    if smoke_vals[i] == "Ex-smoker":
        sm_es[i] = 1.0
    elif smoke_vals[i] == "Never smoked":
        sm_ns[i] = 1.0
    elif smoke_vals[i] == "Currently smokes":
        sm_cs[i] = 1.0
    else:
        pass

data["sex_m"] = sex_m
data["sex_f"] = sex_f
data["sm_es"] = sm_es
data["sm_ns"] = sm_ns
data["sm_cs"] = sm_cs



## === cell 8
x_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Age",
    "Percent",
    "Base_Percent",
    "sex_m",
    "sex_f",
    "sm_es",
    "sm_ns",
    "sm_cs",
]

x_train = data.loc[data["Type"] == "train", x_cols].values.astype(np.float32)
y_train_fvc = (
    data.loc[data["Type"] == "train", prediction_col]
    .values.astype(np.float32)
    .reshape(-1)
)
x_test = data.loc[data["Type"] == "test", x_cols].values.astype(np.float32)

base_fvc_train = (
    data.loc[data["Type"] == "train", "Base_FVC_orig"]
    .values.astype(np.float32)
    .reshape(-1)
)
base_fvc_test = (
    data.loc[data["Type"] == "test", "Base_FVC_orig"]
    .values.astype(np.float32)
    .reshape(-1)
)
y_train = (y_train_fvc - base_fvc_train).astype(np.float32)

train_col_means = np.nanmean(x_train, axis=0)
inds = np.where(np.isnan(x_train))
x_train[inds] = np.take(train_col_means, inds[1])
inds = np.where(np.isnan(x_test))
x_test[inds] = np.take(train_col_means, inds[1])

x_train.shape, y_train.shape, x_test.shape



## === cell 9
model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    solver="adam",
    alpha=0.0,  # closer to no explicit regularization like the TF baseline
    batch_size=BATCH_SIZE,
    learning_rate_init=1e-3,
    max_iter=EPOCHS,
    shuffle=True,
    random_state=SEED,
    early_stopping=False,  # do NOT introduce early stopping
    n_iter_no_change=EPOCHS + 1,
    verbose=False,
)



## === cell 10
groups = data.loc[data["Type"] == "train", "Patient"].values
gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

oof_pred = np.zeros_like(y_train, dtype=np.float32)
for tr_idx, va_idx in gkf.split(x_train, y_train, groups=groups):
    m = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        alpha=0.0,
        batch_size=BATCH_SIZE,
        learning_rate_init=1e-3,
        max_iter=EPOCHS,
        shuffle=True,
        random_state=SEED,
        early_stopping=False,
        n_iter_no_change=EPOCHS + 1,
        verbose=False,
    )
    m.fit(x_train[tr_idx], y_train[tr_idx])
    oof_pred[va_idx] = m.predict(x_train[va_idx]).astype(np.float32)

model.fit(x_train, y_train)



## === cell 11
pred_delta = model.predict(x_test).reshape(-1).astype(np.float32)
pred_fvc = (pred_delta + base_fvc_test).astype(np.float32)

residuals = (y_train.reshape(-1) - oof_pred.reshape(-1)).astype(np.float32)

weeks_train = data.loc[data["Type"] == "train", "Weeks_orig"].values.astype(np.float32)
base_week_train = data.loc[data["Type"] == "train", "Base_Week_orig"].values.astype(
    np.float32
)
dt_train = np.abs(weeks_train - base_week_train)

weeks_test = data.loc[data["Type"] == "test", "Weeks_orig"].values.astype(np.float32)
base_week_test = data.loc[data["Type"] == "test", "Base_Week_orig"].values.astype(
    np.float32
)
dt_test = np.abs(weeks_test - base_week_test)

mad = np.median(np.abs(residuals - np.median(residuals)))
sigma_global = (
    float(1.4826 * mad)
    if np.isfinite(mad) and mad > 0
    else float(np.std(residuals) + 1e-6)
)
sigma_global = max(70.0, sigma_global)

bin_edges = np.array([0.0, 4.0, 12.0, 24.0, 1e9], dtype=np.float32)
sigma_by_bin = np.zeros(len(bin_edges) - 1, dtype=np.float32)

for b in range(len(sigma_by_bin)):
    mask = (dt_train >= bin_edges[b]) & (dt_train < bin_edges[b + 1])
    if np.sum(mask) >= 30:
        r = residuals[mask]
        mad_b = np.median(np.abs(r - np.median(r)))
        s = (
            float(1.4826 * mad_b)
            if np.isfinite(mad_b) and mad_b > 0
            else float(np.std(r) + 1e-6)
        )
        sigma_by_bin[b] = max(70.0, s)
    else:
        sigma_by_bin[b] = sigma_global

midpoints = np.array(
    [2.0, 8.0, 18.0, 36.0], dtype=np.float32
)  # representative dt per bin
sigma_by_bin_mono = np.maximum.accumulate(sigma_by_bin).astype(np.float32)

dt_test_clip = np.clip(dt_test.astype(np.float32), midpoints.min(), midpoints.max())
pred_conf = np.interp(dt_test_clip, midpoints, sigma_by_bin_mono).astype(np.float32)
pred_conf = np.maximum(pred_conf, 70.0).astype(np.float32)

sub_out = sub.copy()
sub_out["FVC"] = pred_fvc
sub_out["Confidence"] = pred_conf

subm = sub_out[["Patient_Week", "FVC", "Confidence"]].copy()
subm.to_csv("submission.csv", index=False)

print(subm.head())
print("Sigma global (OOF) =", sigma_global)
print(
    "Sigma by dt bins (raw) =",
    list(
        zip(
            [f"[{bin_edges[i]}, {bin_edges[i+1]})" for i in range(len(sigma_by_bin))],
            sigma_by_bin.tolist(),
        )
    ),
)
print("Sigma by dt bins (monotone) =", sigma_by_bin_mono.tolist())
print("Wrote submission.csv with shape:", subm.shape)
