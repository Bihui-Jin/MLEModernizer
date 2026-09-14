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

-6.889932744569319

# 6. Current score

-9.86719

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.19901) has done: 'I fix the immediate runtime blocker caused by `pydicom`/protobuf incompatibility by removing DICOM-related imports and keeping the pipeline purely tabular (the current script never successfully uses the image model anyway). I also replace deprecated `DataFrame.append` with `pd.concat`, fix column name typos (`Smoking_status` → `SmokingStatus`), and replace deprecated `np.float` with `np.float32` so feature building runs. Since the referenced external pretrained model file is missing, I train the same kind of small dense Keras regressor directly from the tabular features and then generate a valid `submission.csv` matching `sample_submission.csv` order. Finally, I output constant Confidence=70 (metric-clipped minimum) to ensure a valid confidence column and avoid illegal values.'
- What this solution (achieved -8.95815) has done: 'I fix the runtime crash happening at import time by removing the unnecessary TensorFlow dependency (it triggers a protobuf `MessageFactory.GetPrototype` incompatibility in this Kaggle image) and replacing the modeling step with a pure-pandas/sklearn tabular baseline that runs reliably. To nudge the score upward from -8.199 toward the target -6.889 (higher is better), I implement a simple per-patient linear regression of FVC vs Weeks on the full training history and use that to extrapolate test weeks; this is a common strong baseline for OSIC and should improve accuracy without touching any CT/DICOM logic. I also produce a robust Confidence estimate from training residual dispersion (clipped to >=70 as required), and ensure the submission matches `sample_submission.csv` ordering exactly and writes `submission.csv`. All other data paths and the overall tabular feature approach remain minimal and stable.'
- What this solution (achieved -8.76814) has done: 'Your current score (-8.95815) is worse than the target (-6.88993), so we should improve accuracy while keeping your per-patient linear fit core logic intact. The biggest low-risk gain is to fit the per-patient slope/intercept using a robust procedure (reduce outlier influence) and then anchor intercept to the test baseline as you already do; this preserves the same “linear per patient” model but usually improves FVC predictions. Next, improve Confidence calibration by using a global residual model that increases uncertainty as you extrapolate farther from the baseline week (this can improve the metric without changing FVC predictions drastically). Finally, keep submission ordering identical and keep all paths unchanged.'
- What this solution (achieved -8.29176) has done: 'Your current score (-8.76814) is worse than the target (-6.88993), so we should improve accuracy a bit while keeping your “per-patient linear fit anchored to test baseline + calibrated confidence” core logic intact. The biggest low-risk gain is to estimate each patient’s slope with a more robust linear fit (iteratively reweighted least squares / Huber-style), which reduces outlier influence without changing the overall approach. Next, adjust the confidence model to combine (not max) the patient residual sigma with the extrapolation-distance uncertainty in quadrature; this usually gives better-calibrated sigma for the Laplace metric. Finally, keep the same submission alignment but remove the aggressive FVC clipping to 1–99% quantiles (it can hurt if true extremes exist), replacing it with only the metric’s natural 1000-ml cap via confidence rather than clipping predictions.'
- What this solution (achieved -8.2984) has done: 'Your current score (-8.29176) is worse than the target (-6.88993), so we should improve accuracy/calibration while keeping the same “per-patient robust linear fit anchored to test baseline + calibrated confidence” approach. The biggest low-risk gain is to estimate each patient’s slope/intercept around their *baseline week* (week 0 in train for that patient) rather than around the median week; this matches how test is defined and typically improves extrapolation. Next, we keep your confidence formula but recalibrate the residual-vs-distance fit using distances to each patient’s baseline week (not median), which better matches the scoring weeks and tends to improve the Laplace log-likelihood. All paths, outputs, and the submission schema remain unchanged.'
- What this solution (achieved -8.29748) has done: 'You’re currently below the target (score -8.2984 vs target -6.8899; higher is better), so we should improve accuracy/calibration with minimal changes while keeping your per-patient robust linear model intact. The main low-risk win is to fit each patient’s slope/intercept using weeks centered at that patient’s baseline week (0 if present), which stabilizes the intercept and typically extrapolates better to the scored future weeks. Next, we keep your “anchor intercept to test baseline” behavior but make the distance-based uncertainty consistent with that same baseline definition (distance to the test baseline week), improving Confidence calibration for the Laplace metric. Finally, we remove the hard FVC clipping (500–6500) to avoid unnecessary bias; the metric already caps large errors via Δ and you already clip Confidence to legal bounds.'
- What this solution (achieved -8.29748) has done: 'We keep your per-patient robust linear model and the same “anchor intercept to test baseline” behavior, but fix a key training-test mismatch: your current fit is inadvertently using each patient’s earliest record (not the week-0 baseline) when deriving Typical_FVC and related fields, which can distort slopes/confidence calibration. We minimally change the baseline selection to use the row closest to week 0 per patient (train), matching how test is defined, and we keep everything else intact. Next, we make the distance-based confidence calibration use centered weeks from that same baseline definition (still linear in |Δweek|), which usually improves Laplace log-likelihood without changing the core approach. The script still runs end-to-end and writes a valid `submission.csv` with the correct schema and ordering.'
- What this solution (achieved -8.3161) has done: 'We keep your per-patient robust linear fit + “anchor intercept to test baseline” exactly as-is, but fix the main score drag: the Confidence model is currently double-counting uncertainty by adding a distance-based term on top of a residual sigma computed from *all* weeks (which already includes distance/extrapolation effects). We recalibrate confidence in a minimal way by (1) estimating a per-patient “near-baseline” residual sigma using only points close to that patient’s baseline-like visit, and (2) learning the distance-to-error relationship from those near-baseline residuals, then combining them in quadrature. This usually improves the Laplace log-likelihood (higher is better) without changing the FVC predictions. We also make the confidence mapping consistent (fit to expected |error| then convert to sigma) while keeping clipping and submission formatting unchanged.'
- What this solution (achieved -8.6772) has done: 'Your current score (-8.3161) is worse than the target (-6.8899), so we should improve the metric with minimal risk while keeping the same per-patient robust linear fit + test-baseline anchoring. The largest likely drag is overestimated Confidence (sigma) due to combining a per-patient residual term with a second distance-based term; this hurts the `-log(sigma)` part of the metric. I keep your FVC predictions logic intact, but recalibrate Confidence more conservatively by (1) fitting the distance→expected absolute error curve on the full residuals (not only “near” points), and (2) using that distance-based sigma as the main term while only adding a small per-patient floor (max) rather than adding in quadrature. This typically increases the score (less overconfidence penalty) without changing the core modeling approach.'
- What this solution (achieved -9.86719) has done: 'Your current score (-8.6772) is below the target (-6.8899), so we should improve the metric by increasing accuracy and (especially) better-calibrating Confidence without changing the per-patient robust linear model or the “anchor intercept to test baseline” behavior. The minimal high-impact issue is that the global distance→error fit is currently unweighted across patients, so patients with many measurements dominate calibration; we compute per-patient slopes and then take robust aggregates so calibration matches test patients more evenly. Next, we calibrate Confidence directly against the Laplace objective by fitting the distance model to the *expected clipped absolute error* (cap at 1000) and mapping it to sigma; this avoids over-penalizing with too-large sigmas while respecting the metric’s cap. All file paths, prediction generation, and submission formatting remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -9.86719) has done: 'We keep your per-patient robust linear fit and the “anchor intercept to the test baseline” exactly as-is, but fix a calibration mismatch in the Confidence model that likely worsened the score: the distance→error line (d0,d1) is learned from residuals around each patient’s *train baseline-like week*, but applied using distance to the *test baseline week*. We instead learn (d0,d1) using distances computed to the same baseline definition used at inference (week=0 when available, otherwise the closest-to-0 week), which aligns calibration with how you score predictions. Then we make the distance-based expected abs error monotonic and non-negative (avoids negative intercept/slope artifacts) while keeping the same sigma mapping/clipping and submission formatting. This is a minimal change focused purely on improving Laplace log-likelihood via better Confidence calibration (without changing the FVC prediction logic).'
- What this solution (achieved -9.86719) has done: 'Your current score (-9.86719) is well below the target (-6.88993), so we should improve (increase) it with minimal risk while keeping your per-patient robust linear model and test-baseline anchoring unchanged. The biggest likely issue is the Confidence calibration: you currently estimate the distance→error model using only training-time distances to a train “baseline-like” week, but at inference you use distances to the test baseline week; we recalibrate (d0, d1) using distances to the *same definition used at inference* (week 0 if present else closest-to-0), and do it on a per-patient basis to avoid patients with many rows dominating. We also make the distance→expected-|error| fit robust and monotone (non-negative slope/intercept) and then map to sigma exactly as you already do, preserving evaluation semantics. No changes to the FVC prediction logic, data paths, or submission formatting.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler



## === cell 1
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## === cell 2
EPOCHS = 5
BATCH_SIZE = 32
FOLDS = 5

COMP_DIR = "../input/osic-pulmonary-fibrosis-progression/"
SUB_PATH = os.path.join(COMP_DIR, "sample_submission.csv")



## === cell 3
comp_dir = "../input/osic-pulmonary-fibrosis-progression"

train_data = pd.read_csv(os.path.join(comp_dir, "train.csv"))
sub = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))
test_data = pd.read_csv(os.path.join(comp_dir, "test.csv"))

train_data.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])



## === cell 4
train_data_u = train_data.copy()
train_data_u["abs_Weeks"] = train_data_u["Weeks"].abs()
train_data_u = (
    train_data_u.sort_values(["Patient", "abs_Weeks", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()
    .drop(columns=["abs_Weeks"])
)
train_data_u = train_data_u.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
train_data_u["Typical_FVC"] = (
    train_data_u.Base_FVC.values / train_data_u.Base_Percent.values
) * 100.0

train_data = train_data.merge(
    train_data_u.drop(["Age", "Sex", "SmokingStatus"], axis=1), on="Patient", how="left"
)



## === cell 5
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))

sub = sub[["Patient", "Weeks", "Patient_Week"]].copy()



## === cell 6
test_data = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
test_data["Typical_FVC"] = (
    test_data.Base_FVC.values / test_data.Base_Percent.values
) * 100.0
sub = sub.merge(test_data, how="left", on="Patient")



## === cell 7
train_data["Type"] = "train"
sub["Type"] = "test"

data = pd.concat([train_data, sub], axis=0, ignore_index=True)



## === cell 8
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



## === cell 9
scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols])



## === cell 10
cat_df = pd.get_dummies(data[Categorical_cols], prefix=Categorical_cols, dummy_na=True)
data = pd.concat([data.drop(columns=Categorical_cols), cat_df], axis=1)



## === cell 11
base_x_cols = ["Weeks", "Base_Week", "Base_FVC", "Age"]
cat_cols = [
    c for c in data.columns if c.startswith("Sex_") or c.startswith("SmokingStatus_")
]
x_cols = base_x_cols + cat_cols

missing = [c for c in x_cols + prediction_col + ["Type"] if c not in data.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")



## === cell 12
train_raw = train_data.copy()
test_raw = pd.read_csv(os.path.join(comp_dir, "test.csv"))
sample = pd.read_csv(SUB_PATH)[["Patient_Week"]].copy()
sample["Patient"] = sample["Patient_Week"].str.split("_").str[0]
sample["Weeks"] = sample["Patient_Week"].str.split("_").str[1].astype(int)


def robust_line_fit_irls(w, y, iters=15, k=1.345, eps=1e-6):
    w = np.asarray(w, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(w) < 2 or np.std(w) == 0:
        return 0.0, float(y[0]), np.nan

    X = np.vstack([w, np.ones_like(w)]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a, b = float(beta[0]), float(beta[1])

    for _ in range(iters):
        r = y - (a * w + b)
        mad = np.median(np.abs(r - np.median(r))) + eps
        s = 1.4826 * mad + eps
        u = r / (k * s)
        wgt = np.ones_like(u)
        mask = np.abs(u) > 1.0
        wgt[mask] = 1.0 / (np.abs(u[mask]) + eps)

        W = np.sqrt(wgt)
        Xw = X * W[:, None]
        yw = y * W
        beta_new = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        a_new, b_new = float(beta_new[0]), float(beta_new[1])

        if abs(a_new - a) < 1e-9 and abs(b_new - b) < 1e-6:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    r = y - (a * w + b)
    sig = float(np.std(r)) if len(r) > 1 else np.nan
    return a, b, sig


def pick_baseline_week(w):
    w = np.asarray(w, dtype=np.float64)
    if np.any(w == 0.0):
        return 0.0
    return float(w[np.argmin(np.abs(w))])


patient_params = {}

per_patient_d0 = []
per_patient_d1 = []
per_patient_global_sig = []

patient_sigma0 = {}
NEAR_WEEKS = 8.0  # kept (used for patient floor), core approach unchanged

for pid, g in train_raw.groupby("Patient"):
    g = g.sort_values("Weeks")
    w = g["Weeks"].values.astype(np.float64)
    y = g["FVC"].values.astype(np.float64)

    w0 = pick_baseline_week(
        w
    )  # baseline-like visit (matches inference baseline notion)
    wc = w - w0  # centered weeks around baseline-like visit

    if len(g) >= 2 and np.std(wc) > 0:
        a, b_center, _ = robust_line_fit_irls(wc, y, iters=15)
        b = float(b_center - a * w0)

        yhat = a * w + b
        resid = y - yhat
        patient_params[pid] = (float(a), float(b), np.nan)

        dw = np.abs(w - w0)

        near_mask = dw <= NEAR_WEEKS
        if np.sum(near_mask) >= 2:
            sig0 = float(np.std(resid[near_mask]))
        else:
            sig0 = float(np.std(resid)) if len(resid) > 1 else np.nan
        patient_sigma0[pid] = sig0

        abs_r = np.minimum(np.abs(resid), 1000.0)

        if len(abs_r) >= 4 and np.std(dw) > 0:
            q = np.quantile(abs_r, 0.95)
            m = abs_r <= q
            if np.sum(m) >= 4 and np.std(dw[m]) > 0:
                d1_i, d0_i = np.polyfit(dw[m], abs_r[m], deg=1)
            else:
                d1_i, d0_i = np.polyfit(dw, abs_r, deg=1)

            d0_i = float(max(d0_i, 0.0))
            d1_i = float(max(d1_i, 0.0))
            per_patient_d0.append(d0_i)
            per_patient_d1.append(d1_i)
        else:
            per_patient_d0.append(float(np.median(abs_r)) if len(abs_r) else 200.0)
            per_patient_d1.append(0.0)

        per_patient_global_sig.append(
            float(np.std(resid)) if len(resid) > 1 else np.nan
        )
    else:
        b = float(y[0])
        a = 0.0
        patient_params[pid] = (a, b, np.nan)
        patient_sigma0[pid] = np.nan
        per_patient_d0.append(200.0)
        per_patient_d1.append(0.0)
        per_patient_global_sig.append(np.nan)

pps = np.asarray(
    [s for s in per_patient_global_sig if np.isfinite(s)], dtype=np.float64
)
global_sigma = float(np.median(pps)) if len(pps) else 200.0
global_sigma = max(global_sigma, 70.0)

d0 = float(np.median(per_patient_d0)) if len(per_patient_d0) else float(global_sigma)
d1 = float(np.median(per_patient_d1)) if len(per_patient_d1) else 0.0
d0 = float(max(d0, 0.0))
d1 = float(max(d1, 0.0))

d0 = float(max(d0, 30.0))

test_base = test_raw.set_index("Patient")[["Weeks", "FVC"]].rename(
    columns={"Weeks": "BaseWeek", "FVC": "BaseFVC"}
)

fvc_pred = np.zeros(len(sample), dtype=np.float32)
conf_pred = np.zeros(len(sample), dtype=np.float32)

for i, row in sample.iterrows():
    pid = row["Patient"]
    w = float(row["Weeks"])

    if pid in patient_params:
        a, b, _ = patient_params[pid]

        if pid in test_base.index:
            bw = float(test_base.loc[pid, "BaseWeek"])
            bfvc = float(test_base.loc[pid, "BaseFVC"])
            b = bfvc - a * bw  # anchor intercept to test baseline (unchanged)
        else:
            bw = float(np.nan)

        pred = a * w + b

        ps0 = patient_sigma0.get(pid, np.nan)
        patient_floor = float(ps0) if np.isfinite(ps0) else global_sigma
    else:
        if pid in test_base.index:
            pred = float(test_base.loc[pid, "BaseFVC"])
            bw = float(test_base.loc[pid, "BaseWeek"])
        else:
            pred = float(train_raw["FVC"].median())
            bw = float(np.nan)
        patient_floor = global_sigma

    if np.isfinite(bw):
        dw = abs(w - bw)
        eabs = float(d0 + d1 * dw)
        sigma_distance = float(eabs * np.sqrt(np.pi / 2.0))
        sigma = float(max(patient_floor, sigma_distance))
    else:
        sigma = float(patient_floor)

    fvc_pred[i] = np.float32(pred)
    conf_pred[i] = np.float32(max(sigma, 70.0))



## === cell 13
conf_pred = np.clip(conf_pred, 70.0, 1000.0).astype(np.float32)
fvc_pred = fvc_pred.astype(np.float32)



## === cell 14
subm = pd.DataFrame(
    {
        "Patient_Week": sample["Patient_Week"].values,
        "FVC": fvc_pred,
        "Confidence": conf_pred,
    }
)

subm = pd.read_csv(SUB_PATH)[["Patient_Week"]].merge(
    subm, on="Patient_Week", how="left"
)
subm["FVC"] = subm["FVC"].fillna(float(train_raw["FVC"].median())).astype(np.float32)
subm["Confidence"] = subm["Confidence"].fillna(70.0).astype(np.float32)

subm.to_csv("submission.csv", index=False)
subm.head()



## === cell 15
assert os.path.exists("submission.csv")
assert list(subm.columns) == ["Patient_Week", "FVC", "Confidence"]
assert len(subm) == len(pd.read_csv(SUB_PATH))
print("Wrote submission.csv with shape:", subm.shape)
print(subm.describe(include="all"))
