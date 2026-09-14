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
pymc3==3.11.4
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

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

-6.8657

# 6. Current score

-8.32705

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.60189) has done: 'I fix the immediate runtime failure by removing the incompatible `pymc3/theano` import and switching to a scikit-learn linear mixed-effects approximation that preserves your core intent: patient-specific intercepts and slopes over `Weeks`. I also fix the downstream `LabelEncoder` scoping/NameErrors by ensuring all needed imports and variables exist before use. Finally, I ensure the submission matches `sample_submission.csv` exactly (same `Patient_Week` rows/order), and produce a valid `submission.csv` with clipped positive confidence (>=70) to align with the metric and avoid invalid values.'
- What this solution (achieved -8.17868) has done: 'The crashes come from using a `LabelEncoder` fit only on train patients, but test/sample_submission contain unseen patient IDs; that also cascades into `n_patients` not being defined because cell 1 never finishes. I fix this by fitting the encoder on the union of train+test patients (score-neutral but required for inference), and recomputing `n_patients` after encoding. To keep the model behavior consistent, I keep the same ridge “patient intercept + patient slope” design matrix and training, but for patients with no training history (test-only) I fall back to a global intercept/slope estimated from train so predictions are defined. Finally, I guarantee a valid `submission.csv` with exact sample_submission row order and `Confidence >= 70`.'
- What this solution (achieved -8.22614) has done: 'I keep your ridge “patient intercept + patient slope” model unchanged, and only adjust the Confidence estimation to better match the Laplace Log Likelihood metric (which strongly rewards well-calibrated, not-too-small and not-too-large σ). Concretely, I replace the raw per-patient residual std (very noisy for patients with few points) with a James–Stein-style shrinkage toward the global sigma based on each patient’s number of observations, which usually improves this metric without altering the FVC predictions at all. I also set a single global multiplicative calibration factor on sigma using out-of-fold residuals on train (same model, no early stopping), which typically moves the score upward toward your target by improving σ calibration. The submission format, row order, and minimum Confidence≥70 are preserved exactly.'
- What this solution (achieved -8.24615) has done: 'Your current gap to the target is about 19.8% (−8.226 vs −6.865; higher is better), so we should improve score but keep the same ridge “patient intercept + patient slope” core. The most leverage with minimal change is Confidence calibration: the Laplace metric rewards sigmas that match the actual residual scale, and your coarse grid over a single multiplicative scale is likely leaving points on the table. I keep your per-patient shrinkage exactly as-is, but (1) choose the sigma scale with a simple 1D golden-section search on out-of-fold residuals (same OOF setup, just finer optimization), and (2) add a very small additive sigma floor learned from OOF (sigma = max(best_scale*sigma_patient + sigma_add, 70)), which often improves calibration when residuals have a constant noise component. This does not change FVC predictions at all and should move the score upward toward your target.'
- What this solution (achieved -8.25537) has done: 'I keep your ridge “patient intercept + patient slope” model exactly the same and only adjust the Confidence calibration, because that is the lowest-risk lever for improving the Laplace log-likelihood score. Your current calibration optimizes scale and add sequentially; I replace that with a tiny 2D coordinate-ascent (alternating golden-section searches) on out-of-fold residuals, which typically finds a better (scale, add) pair without changing any FVC predictions. I also optimize on the exact submission-relevant subset of OOF rows (the last three weeks per patient), matching what Kaggle scores, again without changing core logic. Finally, I keep the same constraints (Confidence clipped at ≥70) and the exact sample_submission row order.'
- What this solution (achieved -8.25537) has done: 'Your current score (-8.25537) is below the target (-6.8657), so we should improve (increase) the score with the smallest, lowest-risk change. The model predictions for FVC are already fixed by your ridge “patient intercept + patient slope” setup, so the best lever is Confidence calibration because the Laplace metric is very sensitive to σ. I keep your shrinkage-based per-patient sigma exactly as-is, but change the sigma calibration objective to match Kaggle scoring more closely by calibrating on per-patient *future* points (weeks after baseline) rather than “last 3 observed” (which often includes negative weeks and mismatches the scored regime). Concretely: in OOF, for each patient I hold out the top-3 *maximum Weeks* and only calibrate (scale, add) on those, which is still a minimal change and should move the score upward toward the target without changing FVC predictions.'
- What this solution (achieved -8.25537) has done: 'I keep your ridge “patient intercept + patient slope” model and the same per-patient shrinkage sigma construction, and only adjust the Confidence calibration step because that’s the lowest-risk lever for the Laplace metric. Specifically, I replace the OOF calibration subset (currently “Weeks>0 then tail(3)”) with a submission-aligned subset: for each patient, calibrate on the three *largest Weeks* (the regime Kaggle scores, since test asks for final visits), which avoids mixing in early/negative weeks that distort sigma. I also constrain the additive term search range using the OOF residual scale on that same subset so the optimizer doesn’t wander into overly-large σ that hurts score. Everything else (paths, training loop structure, submission order/format, Confidence≥70 clipping) remains unchanged.'
- What this solution (achieved -8.22878) has done: 'I keep your ridge “patient intercept + patient slope” model and the shrinkage-based per-patient sigma exactly as-is, and only adjust the Confidence calibration because your current score is below the target and this is the lowest-risk lever for the Laplace metric. Specifically, I (1) calibrate Confidence on an OOF subset that more closely matches what Kaggle scores: the last three *largest weeks after baseline* (Weeks>0) per patient when available, otherwise the last three largest weeks, and (2) replace the coordinate-ascent over (scale, add) with a direct 2D coarse-to-fine grid search bounded by the OOF residual scale; this is still minimal and deterministic, but less prone to landing in a suboptimal corner for this non-linear objective. FVC predictions, submission format, row order, and Confidence≥70 clipping remain unchanged.'
- What this solution (achieved -8.44649) has done: 'I keep your ridge “patient intercept + patient slope” model and your per-patient shrinkage sigma unchanged, and only adjust the Confidence calibration to better match the competition metric. The main minimal fix is to calibrate sigma on the exact same regime Kaggle scores: the three largest Weeks per patient (not “Weeks>0 else tail(3)”), because the test asks for the final three visits and early/negative weeks distort optimal σ. I also change the OOF sigma base used for calibration to be derived from OOF residuals (patient-level, then shrunk), so the calibration step is self-consistent and better aligned with generalization without touching FVC predictions. Finally, I keep the same deterministic coarse-to-fine grid search and submission formatting/order, still clipping Confidence to ≥70.'
- What this solution (achieved -8.32705) has done: 'I keep your ridge “patient intercept + patient slope” model and all feature construction unchanged, and only adjust the Confidence calibration because that is the safest lever to increase the Laplace log-likelihood toward the target. The main issue is a train/test mismatch: you calibrate (scale, add) using OOF residual-based sigmas, but then apply it to *in-sample* per-patient sigmas (`sigma_by_patient`), which breaks the calibration mapping and can hurt score. I instead compute a per-patient sigma on train using the same OOF-residual pipeline (still shrunk) and use that as the base sigma for test-time Confidence, while leaving FVC predictions untouched. This is a minimal change, deterministic, and directly aligned to the metric.'
- What this solution (achieved -8.32705) has done: 'We’re currently below the target (−8.327 vs −6.865; higher is better), so the safest way to move upward without touching your FVC core model is to improve Confidence calibration because the Laplace metric is highly sensitive to σ. I keep the same ridge “patient intercept + patient slope” design matrix and training, and keep your per-patient OOF-residual shrinkage sigma base. The minimal change is to calibrate (scale, add) using exactly the same aggregation level that Kaggle scores: per Patient_Week, but effectively “per patient’s final three weeks” — and to reduce overfitting/noise by optimizing the calibration parameters on a patient-averaged objective (each patient contributes equally) rather than row-weighted (patients with more rows dominate). This does not alter any FVC predictions; it only adjusts the sigma mapping in a more scoring-aligned way.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)

train = train.copy()
test = test.copy()
train.drop(train[train.Patient == "ID00197637202246865691526"].index, inplace=True)

all_patients = pd.Index(train["Patient"]).append(pd.Index(test["Patient"])).unique()
le_id = LabelEncoder()
le_id.fit(all_patients.values)

train["PatientID"] = le_id.transform(train["Patient"])
test["PatientID"] = le_id.transform(test["Patient"])

n_patients = int(len(le_id.classes_))

print(
    "train rows:",
    train.shape,
    "test rows:",
    test.shape,
    "sample_sub rows:",
    sample_sub.shape,
)
print(
    "unique patients train/test:",
    train["Patient"].nunique(),
    test["Patient"].nunique(),
    "encoder patients:",
    n_patients,
)




## === cell 2
def build_design_matrix(patient_ids: np.ndarray, weeks: np.ndarray, n_patients: int):
    patient_ids = patient_ids.astype(int)
    weeks = weeks.astype(float)
    n = len(patient_ids)

    X0 = np.zeros((n, n_patients), dtype=np.float32)
    X0[np.arange(n), patient_ids] = 1.0

    X1 = X0 * weeks.reshape(-1, 1)
    X = np.concatenate([X0, X1], axis=1)
    return X


def laplace_metric(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> float:
    sigma_c = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return float(
        np.mean(-np.sqrt(2.0) * delta / sigma_c - np.log(np.sqrt(2.0) * sigma_c))
    )


FVC_obs = train["FVC"].values.astype(float)
Weeks = train["Weeks"].values.astype(float)
PatientID = train["PatientID"].values.astype(int)

X = build_design_matrix(PatientID, Weeks, n_patients)

ridge = Ridge(alpha=50.0, fit_intercept=False, random_state=RANDOM_SEED)
ridge.fit(X, FVC_obs)

coef = ridge.coef_.astype(np.float32)
a_hat = coef[:n_patients].copy()
b_hat = coef[n_patients:].copy()

train_pred = X @ coef
resid = FVC_obs - train_pred

sigma_global = float(np.std(resid, ddof=1)) if len(resid) > 1 else 150.0

resid_df = pd.DataFrame({"PatientID": PatientID, "resid": resid.astype(np.float32)})
sigma_raw = resid_df.groupby("PatientID")["resid"].std(ddof=1)
count_by_patient = resid_df.groupby("PatientID")["resid"].size()

sigma_raw = sigma_raw.reindex(np.arange(n_patients))
count_by_patient = count_by_patient.reindex(np.arange(n_patients)).fillna(0).astype(int)

k_shrink = 6.0  # keep unchanged (core approach)
w = count_by_patient.values.astype(float) / (
    count_by_patient.values.astype(float) + k_shrink
)
sigma_shrunk = (w * sigma_raw.fillna(sigma_global).values) + ((1.0 - w) * sigma_global)
sigma_by_patient = pd.Series(sigma_shrunk, index=np.arange(n_patients), dtype=float)

global_slope, global_intercept = np.polyfit(Weeks, FVC_obs, deg=1)
train_patient_ids = set(train["PatientID"].unique().tolist())
missing_patient_ids = np.array(
    [pid for pid in range(n_patients) if pid not in train_patient_ids], dtype=int
)
if len(missing_patient_ids) > 0:
    a_hat[missing_patient_ids] = float(global_intercept)
    b_hat[missing_patient_ids] = float(global_slope)
    sigma_by_patient.iloc[missing_patient_ids] = sigma_global

gkf = GroupKFold(n_splits=5)
groups = PatientID
oof_pred = np.zeros_like(FVC_obs, dtype=float)
for tr_idx, va_idx in gkf.split(X, FVC_obs, groups=groups):
    ridge_cv = Ridge(alpha=50.0, fit_intercept=False, random_state=RANDOM_SEED)
    ridge_cv.fit(X[tr_idx], FVC_obs[tr_idx])
    oof_pred[va_idx] = (X[va_idx] @ ridge_cv.coef_).astype(float)

oof_resid = (FVC_obs - oof_pred).astype(float)
sigma_oof_global = (
    float(np.std(oof_resid, ddof=1)) if len(oof_resid) > 1 else sigma_global
)

oof_resid_df = pd.DataFrame(
    {"PatientID": PatientID.astype(int), "oof_resid": oof_resid.astype(np.float32)}
)
sigma_oof_raw = oof_resid_df.groupby("PatientID")["oof_resid"].std(ddof=1)
count_oof_by_patient = oof_resid_df.groupby("PatientID")["oof_resid"].size()

sigma_oof_raw = sigma_oof_raw.reindex(np.arange(n_patients))
count_oof_by_patient = (
    count_oof_by_patient.reindex(np.arange(n_patients)).fillna(0).astype(int)
)

w_oof = count_oof_by_patient.values.astype(float) / (
    count_oof_by_patient.values.astype(float) + k_shrink
)
sigma_oof_shrunk = (w_oof * sigma_oof_raw.fillna(sigma_oof_global).values) + (
    (1.0 - w_oof) * sigma_oof_global
)
sigma_oof_by_patient_series = pd.Series(
    sigma_oof_shrunk, index=np.arange(n_patients), dtype=float
)

pid_arr = PatientID.astype(int)
sigma_base_all = sigma_oof_by_patient_series.iloc[pid_arr].values.astype(float)

train_tmp = train[["PatientID", "Weeks"]].copy()
train_tmp["row_idx"] = np.arange(len(train_tmp))
train_tmp = train_tmp.sort_values(["PatientID", "Weeks"])

rows_for_cal = []
for pid, g in train_tmp.groupby("PatientID", sort=False):
    sel = g.tail(3)  # keep: three largest weeks per patient (submission-aligned)
    rows_for_cal.append(sel["row_idx"].values)

cal_idx = np.concatenate(rows_for_cal).astype(int)
mask_cal = np.zeros(len(train_tmp), dtype=bool)
mask_cal[cal_idx] = True

y_true_cal = FVC_obs[mask_cal]
y_pred_cal = oof_pred[mask_cal]
sigma_base_cal = sigma_base_all[mask_cal]
pid_cal = PatientID[mask_cal].astype(int)

sigma_oof_cal = (
    float(np.std(y_true_cal - y_pred_cal, ddof=1))
    if len(y_true_cal) > 1
    else sigma_oof_global
)

cal_df = pd.DataFrame(
    {
        "PatientID": pid_cal,
        "y_true": y_true_cal.astype(float),
        "y_pred": y_pred_cal.astype(float),
        "sigma_base": sigma_base_cal.astype(float),
    }
)


def score_for_params_patient_avg(scale, add) -> float:
    scale = float(scale)
    add = float(add)

    def _patient_score(g):
        sig = g["sigma_base"].values * scale + add
        return laplace_metric(g["y_true"].values, g["y_pred"].values, sig)

    return float(cal_df.groupby("PatientID", sort=False).apply(_patient_score).mean())


scale_lo, scale_hi = 0.4, 2.4
add_lo = 0.0
add_hi = max(120.0, 1.25 * float(sigma_oof_cal))

best_scale, best_add = 1.0, 0.0
best_score = -1e18

scale_grid = np.linspace(scale_lo, scale_hi, 41)  # step ~0.05
add_grid = np.linspace(add_lo, add_hi, 41)
for s in scale_grid:
    for a in add_grid:
        sc = score_for_params_patient_avg(s, a)
        if sc > best_score:
            best_score = sc
            best_scale, best_add = float(s), float(a)

for _ in range(2):
    s_lo = max(scale_lo, best_scale - 0.12)
    s_hi = min(scale_hi, best_scale + 0.12)
    a_lo = max(add_lo, best_add - 0.20 * add_hi)
    a_hi = min(add_hi, best_add + 0.20 * add_hi)

    scale_grid = np.linspace(s_lo, s_hi, 31)
    add_grid = np.linspace(a_lo, a_hi, 31)
    for s in scale_grid:
        for a in add_grid:
            sc = score_for_params_patient_avg(s, a)
            if sc > best_score:
                best_score = sc
                best_scale, best_add = float(s), float(a)

best_score_final = best_score

sigma_submit_base_by_patient = sigma_oof_by_patient_series.copy()
if len(missing_patient_ids) > 0:
    sigma_submit_base_by_patient.iloc[missing_patient_ids] = sigma_oof_global

print("Estimated global sigma (in-sample):", sigma_global)
print("Estimated global sigma (OOF):", sigma_oof_global)
print("Estimated cal-subset sigma (OOF):", sigma_oof_cal)
print(
    "Chosen sigma scale (OOF subset, patient-avg):",
    best_scale,
    "chosen sigma add (OOF subset, patient-avg):",
    best_add,
)
print("OOF metric on calibration subset (patient-avg, final):", best_score_final)
print(
    "Global fallback (intercept, slope):", float(global_intercept), float(global_slope)
)
print("Test-only patients filled:", int(len(missing_patient_ids)))
print("Per-patient sigma (train, shrunk) summary:", sigma_by_patient.describe())
print(
    "Per-patient sigma (OOF, shrunk) summary:", sigma_oof_by_patient_series.describe()
)



## === cell 3
sub = sample_sub.copy()
sub[["Patient", "Weeks"]] = sub["Patient_Week"].str.split("_", n=1, expand=True)
sub["Weeks"] = sub["Weeks"].astype(int)

sub["PatientID"] = le_id.transform(sub["Patient"]).astype(int)

sub_fvc = (
    a_hat[sub["PatientID"].values]
    + b_hat[sub["PatientID"].values] * sub["Weeks"].values
)
sub["FVC"] = sub_fvc.astype(float)

sub_sigma = sigma_submit_base_by_patient.iloc[sub["PatientID"].values].values.astype(
    float
)
sub_sigma = np.maximum(sub_sigma * float(best_scale) + float(best_add), 70.0)
sub["Confidence"] = sub_sigma

final = sub[["Patient_Week", "FVC", "Confidence"]].copy()

final["FVC"] = final["FVC"].astype(float)
final["Confidence"] = final["Confidence"].astype(float)
final = final.fillna(
    {"FVC": float(np.nanmedian(final["FVC"].values)), "Confidence": 70.0}
)
final["Confidence"] = np.maximum(final["Confidence"].values, 70.0)

final = (
    final.set_index("Patient_Week").reindex(sample_sub["Patient_Week"]).reset_index()
)

final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final.shape)
print(final.head())
print(final.tail())
assert list(final.columns) == ["Patient_Week", "FVC", "Confidence"]
assert final.shape[0] == sample_sub.shape[0]
assert np.all(np.isfinite(final["FVC"].values))
assert np.all(final["Confidence"].values >= 70.0)
