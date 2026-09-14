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

-9.00301

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.53155) has done: 'I fix the environment/runtime issues by removing the TensorFlow import that triggers the protobuf `MessageFactory.GetPrototype` error and replacing it with a lightweight, deterministic baseline that uses only pandas/numpy (which are already available). I also fix broken file paths (`/kaggle/input/prep-data/...`) to use the actual competition dataset CSVs, and remove the missing pickled preprocessor dependency by computing the needed features directly from `train.csv`/`test.csv`. To move score toward the target (and also get a reasonable non-error submission), I fit a simple per-patient linear trend on FVC vs Weeks from the training history and predict test weeks accordingly, while setting `Confidence` to a safe clipped value (>=70). The script run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved -15.58135) has done: 'I keep your per-patient linear trend core logic intact, but make two minimal metric-aligned adjustments to move the score up toward the target. First, I anchor each patient’s trend prediction to the known baseline measurement from `test.csv` (same slope, adjusted intercept) to reduce systematic per-patient bias on the scored weeks. Second, I set `Confidence` using a single calibrated constant derived from training residual MAE (then clipped to >=70), which typically improves the Laplace log-likelihood compared to per-patient std that can be too small/large or unstable for patients with few points. The script still run end-to-end and write a valid `submission.csv` with the required schema and row order matching `sample_submission.csv`.'
- What this solution (achieved -8.53155) has done: 'Your current gap to the target is large and negative (worse than target), so we should improve score with minimal, metric-aligned changes while keeping your per-patient linear trend and baseline anchoring intact. The biggest remaining issue is that you’re anchoring to `test.csv`’s provided FVC (baseline week), but the competition’s scored weeks are typically far from baseline; anchoring should instead use each test patient’s *own last observed* FVC/Week from `train.csv` when available (it is, since test patients appear in train history in this dataset), which reduces bias without changing the model form. Second, we compute the global calibrated sigma using the metric-optimal Laplace MLE (`sqrt(2)*MAE`) on residuals, and use a slightly shrunk per-patient sigma (blended with global) to avoid under/over-confidence while staying stable and deterministic. These are small post-fit calibration/anchoring adjustments that generally move the Laplace log-likelihood upward toward your target without changing the core approach.'
- What this solution (achieved -11.42053) has done: 'I keep your per-patient linear trend + intercept anchoring logic intact, but adjust two metric-facing details that typically improve Laplace log-likelihood without changing the modeling approach. First, I calibrate `Confidence` using the metric-optimal constant sigma computed directly from training absolute residuals and use that single value for all rows (your current blend can over-penalize via `-log(sigma)` when sigma is too large). Second, I compute the residuals used for sigma calibration with the same “last observation anchoring” you use at inference time, so the confidence better matches the actual prediction error distribution. These are minimal, deterministic changes aimed at moving the score up from -8.53 toward your -7.06 target.'
- What this solution (achieved -11.99934) has done: 'We keep your per-patient linear trend + last-observation intercept anchoring intact, but fix a key mismatch with the competition’s scoring: only the final three weeks per patient are evaluated, while your sigma calibration currently uses all historical weeks and can over/underestimate uncertainty for the extrapolation regime. I recalibrate the single global `Confidence` using anchored residuals evaluated specifically at each patient’s last three observed weeks (a closer proxy for “final three” behavior), which typically increases the Laplace log-likelihood without changing the model form. I also add a very small stabilization: ensure every test patient has a usable slope by falling back to a global slope (fit on train) instead of forcing slope=0 when missing, which reduces extreme bias for unseen patients while preserving the same linear model logic. The output submission format, paths, and deterministic behavior remain unchanged.'
- What this solution (achieved -11.99934) has done: 'Your current gap to the target is large (−11.999 vs −7.061, higher is better), so we should improve score with minimal metric-aligned tweaks while keeping the same per-patient linear trend + anchoring logic. The biggest low-risk gain is to stop using a single constant sigma and instead use a per-row confidence that increases with extrapolation distance from the anchor week, which better matches the competition’s “final three weeks” uncertainty pattern and improves the Laplace log-likelihood without changing the prediction model. We calibrate this distance-based sigma using anchored residuals measured at +12/+24/+36 weeks beyond each patient’s last observed week (a closer proxy to the test scoring horizon than “last three observed weeks”), then clip at 70 as required. Everything else (data paths, trend fit, anchoring, submission schema/order) stays the same and the script still writes `submission.csv`.'
- What this solution (achieved -8.5354) has done: 'Your current score (-11.999) is far below the target (-7.061), so we should improve (increase) it with minimal, metric-aligned changes while keeping the same per-patient linear trend + intercept anchoring. The biggest issue is your confidence calibration: it currently uses residuals on each patient’s *last 3 observed weeks* but then applies that to *future extrapolation*, so sigma is miscalibrated for the scored horizon. I recalibrate the distance-based sigma using true out-of-sample residuals computed by fitting the trend on each patient’s history *excluding the last 3 points* and evaluating error on those held-out last 3 (which better matches “future” prediction uncertainty). I also add a small, safe slope clamp (winsorize to train-wide 1st–99th percentile) to prevent extreme extrapolations that can dominate error, without changing the linear model form or training approach.'
- What this solution (achieved -9.00301) has done: 'Your current score (-8.5354) is below the target (-7.06085), so we should improve it slightly while keeping the same per-patient linear trend + last-observation anchoring logic. The minimal metric-aligned change is to recalibrate the distance-based confidence model in a way that better matches the evaluation’s “final three visits” horizon: estimate sigma using held-out future residuals at patient-specific +Δ weeks (matching each patient’s actual last-3-step gaps), rather than using generic near/far quantiles. Then use the metric-optimal Laplace sigma mapping (`sigma ≈ sqrt(2)*MAE`) with a robust (median-based) fit and keep your required clipping (>=70) and the same slope clamp/prediction pipeline. This typically increases Laplace log-likelihood by improving confidence calibration without changing the FVC predictor.'
- What this solution (achieved -9.00301) has done: 'We keep your per-patient linear trend + last-observation anchoring exactly as-is and only adjust the confidence calibration to be more metric-aligned, since your current score is below the target and the largest remaining lever is sigma. Specifically, instead of binning deltas and fitting a line on binned medians (which can miscalibrate), we fit a nonnegative linear model `MAE ≈ a + b*delta` directly on all held-out (delta, |residual|) pairs using closed-form least squares with robust trimming, then map to `sigma = sqrt(2)*MAE` and clip. We also widen the allowed sigma upper clip (still safe) to avoid overly harsh penalties when predictions are off, which usually increases Laplace log-likelihood. These are minimal changes confined to confidence estimation; prediction FVC logic, anchoring, and file paths/order remain unchanged and it still writes a valid `submission.csv`.'

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


def fit_global_trend(df: pd.DataFrame):
    x = df["Weeks"].to_numpy(dtype=float)
    y = df["FVC"].to_numpy(dtype=float)
    if len(df) < 2:
        return global_fvc_mean, 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 1e-12:
        return y_mean, 0.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = float(y_mean - slope * x_mean)
    return intercept, slope


global_intercept, global_slope = fit_global_trend(train)

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

all_slopes = np.array(
    [v[1] for v in patient_params.values() if np.isfinite(v[1])], dtype=float
)
if all_slopes.size > 10:
    slope_lo, slope_hi = np.quantile(all_slopes, [0.01, 0.99])
else:
    slope_lo, slope_hi = -np.inf, np.inf


def clamp_slope(s: float) -> float:
    if not np.isfinite(s):
        return global_slope
    if abs(s) < 1e-12:
        return global_slope
    return float(np.clip(s, slope_lo, slope_hi))


abs_resids = []
deltas = []

for pid, g in train.groupby("Patient", sort=False):
    g_sorted = g.sort_values("Weeks")
    if len(g_sorted) < 4:
        continue

    g_fit = g_sorted.iloc[:-3].copy()
    g_hold = g_sorted.iloc[-3:].copy()

    intercept_fit, slope_fit = fit_patient_trend(g_fit)
    slope_fit = clamp_slope(slope_fit)

    lw = float(g_fit["Weeks"].iloc[-1])
    lfvc = float(g_fit["FVC"].iloc[-1])
    intercept_fit = float(lfvc - slope_fit * lw)

    xh = g_hold["Weeks"].to_numpy(dtype=float)
    yh = g_hold["FVC"].to_numpy(dtype=float)
    yhat_h = intercept_fit + slope_fit * xh

    abs_resids.extend(np.abs(yh - yhat_h).tolist())
    deltas.extend(np.abs(xh - lw).tolist())

abs_resids = np.asarray(abs_resids, dtype=float)
deltas = np.asarray(deltas, dtype=float)

if abs_resids.size == 0:
    abs_resids = np.asarray(
        [float(train.groupby("Patient")["FVC"].diff().abs().median())], dtype=float
    )
    deltas = np.asarray([0.0], dtype=float)

mask = np.isfinite(deltas) & np.isfinite(abs_resids) & (deltas >= 0) & (abs_resids >= 0)
d = deltas[mask]
r = abs_resids[mask]

if d.size == 0:
    a = float(np.median(abs_resids))
    b = 0.0
else:
    r_cap = float(np.quantile(r, 0.95)) if r.size >= 20 else float(np.max(r))
    keep = r <= r_cap
    d2 = d[keep]
    r2 = r[keep]

    if d2.size < 2 or np.allclose(d2, d2[0]):
        a = float(np.median(r2)) if r2.size else float(np.median(r))
        b = 0.0
    else:
        X = np.vstack([np.ones_like(d2), d2]).T
        beta, _, _, _ = np.linalg.lstsq(X, r2, rcond=None)
        a = float(max(0.0, beta[0]))
        b = float(max(0.0, beta[1]))

SIGMA_MIN = 70.0
SIGMA_MAX = 1000.0


def sigma_from_delta(delta_weeks: float) -> float:
    mae = a + b * float(abs(delta_weeks))
    return float(np.clip(np.sqrt(2.0) * mae, SIGMA_MIN, SIGMA_MAX))


global_abs_resid_mae = float(np.median(r)) if r.size else float(np.median(abs_resids))
calibrated_sigma_const = float(
    np.clip(np.sqrt(2.0) * global_abs_resid_mae, SIGMA_MIN, SIGMA_MAX)
)



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
        intercept, slope, resid_std = global_intercept, global_slope, global_fvc_std
    else:
        intercept, slope, resid_std = params

    slope = clamp_slope(slope)

    anchor_week = None
    if pid in last_obs_map:
        lw, lfvc = last_obs_map[pid]
        if np.isfinite(lw) and np.isfinite(lfvc):
            intercept = float(lfvc - slope * float(lw))
            anchor_week = float(lw)
    elif np.isfinite(bw) and np.isfinite(bfvc):
        intercept = float(bfvc - slope * float(bw))
        anchor_week = float(bw)

    yhat = intercept + slope * float(w)
    pred_fvc[i] = yhat

    if anchor_week is None or not np.isfinite(anchor_week):
        pred_conf[i] = calibrated_sigma_const
    else:
        pred_conf[i] = sigma_from_delta(float(w) - float(anchor_week))

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
    sub["Confidence"].fillna(calibrated_sigma_const).clip(lower=70.0).astype(np.float32)
)

sub = sub[["Patient_Week", "FVC", "Confidence"]]
assert len(sub) == len(sample_sub)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Const sigma fallback:", calibrated_sigma_const)
print("Distance-sigma params (a,b):", float(a), float(b))
print("Global slope fallback used:", global_slope)
print(
    "Slope clamp [lo,hi]:",
    float(slope_lo) if np.isfinite(slope_lo) else slope_lo,
    float(slope_hi) if np.isfinite(slope_hi) else slope_hi,
)
print("Wrote submission.csv with shape:", sub.shape)
