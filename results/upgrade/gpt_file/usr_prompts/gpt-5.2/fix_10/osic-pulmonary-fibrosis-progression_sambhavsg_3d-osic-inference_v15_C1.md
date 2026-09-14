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

-6.894224487668252

# 6. Current score

-7.98981

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.17447) has done: 'I fix the two main blockers preventing an end-to-end run: (1) the import-time crash coming from `pydicom` (protobuf incompatibility) by removing the unused DICOM/image pipeline and related imports, and (2) pandas API breakage (`DataFrame.append` removed) by switching to `pd.concat`. I also remove the dependency on a missing external pretrained model file (`../input/tab-data-osic/dense_model.h5`) by training the same tabular model structure in-notebook on the provided `train.csv` and then predicting for `sample_submission.csv` rows. Finally, I ensure the submission is correctly aligned to `Patient_Week` and written as `submission.csv` with the exact required columns (`Patient_Week,FVC,Confidence`).'
- What this solution (achieved -8.17163) has done: 'I fix the import-time crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` incompatibility). Then I make the metric-alignment bug fix: your `score()` currently computes the *negative* of the competition metric, so the model is trained in the wrong direction; I correct it to match the Kaggle formula (higher is better), which should improve your score toward the target without changing the overall modeling approach. Finally, I add a small post-processing step to ensure the predicted quantiles are ordered (q20 ≤ q50 ≤ q80) so Confidence is non-negative and stable, preventing pathological outputs.'
- What this solution (achieved -7.84634) has done: 'I fix the import-time protobuf crash by avoiding TensorFlow entirely (it’s the only thing failing) while keeping the same tabular feature pipeline and training-by-folds approach. To preserve the core “predict FVC + uncertainty” semantics without changing the overall data logic, I switch the model backend to a lightweight scikit-learn regressor for FVC and compute a per-row Confidence from out-of-fold residuals (clipped at 70), which aligns with the competition metric. I also ensure predictions are on the original FVC scale (your current code trains on raw FVC but predicts from minmax-scaled features, so that’s fine) and that the submission is perfectly aligned to `sample_submission.csv`’s `Patient_Week` ordering. These changes are minimal and targeted: remove the TensorFlow dependency that crashes and keep the rest of the pipeline stable while nudging score upward via better-calibrated confidence.'
- What this solution (achieved -7.84136) has done: 'Your current score (-7.84634) is worse than the target (-6.8942), so we should cautiously improve (increase) it with minimal, metric-aligned changes. The biggest low-risk gain here is to make `Confidence` patient-specific (and slightly week-dependent) rather than a single global constant, using out-of-fold residual dispersion learned from training—this directly targets the competition metric without changing the model itself. We also fix a subtle leakage/feature mismatch by defining `Base_*` from the true per-patient baseline week (closest to Week 0) instead of an arbitrary first row after `drop_duplicates`, which improves the tabular signal while keeping the same feature set and regressor. Finally, we keep the exact submission alignment to `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved -8.08874) has done: 'We keep your regressor and feature set unchanged, and only adjust the Confidence calibration because that directly impacts the Laplace log-likelihood metric and is the lowest-risk lever to move score upward toward the target. Specifically, we estimate per-patient residual dispersion more robustly by pooling each patient’s std toward the global std (shrinkage) to avoid noisy over/under-confidence from few observations. Then we apply a single global scaling factor to Confidence (computed from out-of-fold residuals) to better match the competition’s clipping behavior (70/1000) without changing the FVC predictions. These changes are minimal, deterministic, and should improve (increase) your score from -7.84 toward -6.89.'
- What this solution (achieved -7.92572) has done: 'We keep your model and features identical, and only adjust the Confidence calibration because that’s the safest lever to increase the Laplace log-likelihood toward the target. Right now Confidence varies by patient/week, but it’s derived from raw residual std which is noisy and not necessarily optimal for the Laplace metric; we (1) compute a Laplace-consistent scale per patient using MAD (more robust than std), (2) apply shrinkage in the *scale* domain, and (3) choose a single global multiplicative scaling for Confidence by directly maximizing the out-of-fold competition metric (no change to FVC predictions). This is deterministic, fast, and directly metric-aligned while preserving your training loop and prediction semantics. The submission writing and alignment remain unchanged.'
- What this solution (achieved -7.91591) has done: 'We keep your regressor, features, folds, and the entire training/prediction flow unchanged, and only tune the Confidence calibration (the safest lever for this metric). Your current Confidence is capped at 500 and has a week inflation; both can hurt the Laplace log-likelihood because overly-large σ reduces the score via the `-log(σ)` term. We (1) remove the hard upper cap of 500 (keep only the competition’s lower clip at 70), (2) re-optimize the global Confidence scale over a wider, denser range, and (3) slightly reduce the week-dependent inflation so σ doesn’t grow too aggressively far from baseline. These are minimal, metric-aligned changes that should increase the score from -7.92572 toward the target -6.8942.'
- What this solution (achieved -8.01526) has done: 'Your current score (-7.9159) is worse than the target (-6.8942), so we should increase it with the smallest, metric-aligned change. The safest lever (without altering your regressor/features/folds) is improving Confidence calibration: your current sigma uses a MAD→Normal conversion (÷0.6745), but the competition metric is Laplace-based, so calibrating sigma from a Laplace-consistent scale (MAD / ln(2)) typically improves the log-likelihood without touching FVC predictions. I keep your exact patient-level shrinkage + week-factor structure, and only swap the robust scale estimator/conversion plus re-optimize the single global scale on OOF using the same grid search. This preserves core logic and evaluation semantics while directly targeting the metric term that’s currently miscalibrated.'
- What this solution (achieved -7.98981) has done: 'We should increase the score (make it less negative) toward the target, and the safest lever without changing your regressor/features/folds is Confidence calibration (it directly affects the Laplace log-likelihood). Your current week-factor uses `abs(Weeks)` and is based on the scaled `Weeks` (because you overwrite `data["Weeks"]` with MinMaxScaler), which unintentionally ties Confidence inflation to the wrong scale; I preserve the same structure but compute week inflation from the original (unscaled) week numbers. I also tune the week-inflation strength `alpha` jointly with the global scale on OOF to better match the metric, keeping the exact same patient-sigma computation and leaving FVC predictions untouched. Finally, I ensure the submission remains perfectly aligned to `sample_submission.csv` and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor

SEED = 42
np.random.seed(SEED)



## === cell 1
EPOCHS = 5
BATCH_SIZE = 32
FOLDS = 5

COMP_DIR = "../input/osic-pulmonary-fibrosis-progression/"

TRAIN_CSV = os.path.join(COMP_DIR, "train.csv")
TEST_CSV = os.path.join(COMP_DIR, "test.csv")
SUB_PATH = os.path.join(COMP_DIR, "sample_submission.csv")



## === cell 2
train_data = pd.read_csv(TRAIN_CSV)
test_data = pd.read_csv(TEST_CSV)
sub = pd.read_csv(SUB_PATH)

train_data = train_data.drop_duplicates(
    keep=False, subset=["Patient", "Weeks"]
).reset_index(drop=True)

train_data.head(), test_data.head(), sub.head()



## === cell 3
td = train_data.copy()
td["_abs_week"] = td["Weeks"].abs()
train_base_idx = (
    td.sort_values(["Patient", "_abs_week", "Weeks"])
    .groupby("Patient", sort=False)
    .head(1)
    .index
)
train_data_u = td.loc[train_base_idx].drop(columns=["_abs_week"]).copy()

train_data_u = train_data_u.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
train_data_u["Typical_FVC"] = (
    train_data_u["Base_FVC"].values / train_data_u["Base_Percent"].values
) * 100.0

train_data = train_data.merge(
    train_data_u.drop(["Age", "Sex", "SmokingStatus"], axis=1), on="Patient", how="left"
)

train_data.head()



## === cell 4
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Patient_Week"]].copy()

test_data_b = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
).copy()
test_data_b["Typical_FVC"] = (
    test_data_b["Base_FVC"].values / test_data_b["Base_Percent"].values
) * 100.0

sub = sub.merge(test_data_b, how="left", on="Patient")
sub.head()



## === cell 5
train_data = train_data.copy()
train_data["Type"] = "train"
sub["Type"] = "test"

data = pd.concat([train_data, sub], axis=0, ignore_index=True)
data.shape, data["Type"].value_counts()



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

missing_cols = [c for c in Continuos_cols if c not in data.columns]
missing_cols



## === cell 7
data["Weeks_raw"] = data["Weeks"].astype(np.int32)
data["Base_Week_raw"] = data["Base_Week"].astype(np.int32)

scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols])

data[Continuos_cols].describe().T



## === cell 8
data["sex_m"] = (data["Sex"] == "Male").astype(np.float32)
data["sex_f"] = (data["Sex"] == "Female").astype(np.float32)

data["sm_es"] = (data["SmokingStatus"] == "Ex-smoker").astype(np.float32)
data["sm_ns"] = (data["SmokingStatus"] == "Never smoked").astype(np.float32)
data["sm_cs"] = (data["SmokingStatus"] == "Currently smokes").astype(np.float32)

unknown_smoke = data[["sm_es", "sm_ns", "sm_cs"]].sum(axis=1) == 0
data.loc[unknown_smoke, "sm_cs"] = 1.0

x_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Age",
    "sex_m",
    "sex_f",
    "sm_es",
    "sm_ns",
    "sm_cs",
]



## === cell 9
train_mask = data["Type"] == "train"
test_mask = data["Type"] == "test"

x_train = data.loc[train_mask, x_cols].values.astype(np.float32)
y_train = data.loc[train_mask, prediction_col].values.astype(np.float32).reshape(-1)

x_test = data.loc[test_mask, x_cols].values.astype(np.float32)

test_patient_week = data.loc[test_mask, "Patient_Week"].values
test_patients = data.loc[test_mask, "Patient"].values.astype(str)

test_weeks_raw = data.loc[test_mask, "Weeks_raw"].values.astype(np.int32)
train_patients = data.loc[train_mask, "Patient"].values.astype(str)
train_weeks_raw = data.loc[train_mask, "Weeks_raw"].values.astype(np.int32)

x_train.shape, y_train.shape, x_test.shape, len(test_patient_week)



## === cell 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_pred = np.zeros(x_train.shape[0], dtype=np.float32)
pred_test = np.zeros(x_test.shape[0], dtype=np.float32)

fold_sigmas = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train), 1):
    model = GradientBoostingRegressor(
        random_state=SEED + fold,
        loss="squared_error",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
    )
    model.fit(x_train[tr_idx], y_train[tr_idx])

    va_pred = model.predict(x_train[va_idx]).astype(np.float32)
    oof_pred[va_idx] = va_pred

    resid = (y_train[va_idx] - va_pred).astype(np.float32)
    resid = np.clip(resid, -1000.0, 1000.0)
    sigma = float(np.std(resid))
    fold_sigmas.append(max(sigma, 70.0))

    pred_test += model.predict(x_test).astype(np.float32) / FOLDS

float(np.mean(fold_sigmas)), np.min(fold_sigmas), np.max(fold_sigmas)




## === cell 11
def competition_metric(y_true, y_pred, sigma):
    sigma_c = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return (-np.sqrt(2.0) * delta / sigma_c) - np.log(np.sqrt(2.0) * sigma_c)


oof_resid = (y_train.astype(np.float32) - oof_pred.astype(np.float32)).astype(
    np.float32
)
oof_resid = np.clip(oof_resid, -1000.0, 1000.0)

oof_df = pd.DataFrame(
    {"Patient": train_patients, "Weeks_raw": train_weeks_raw, "resid": oof_resid}
)

_LN2 = float(np.log(2.0))
_SQRT2 = float(np.sqrt(2.0))


def mad_to_sigma_laplace(x):
    x = np.asarray(x, dtype=np.float32)
    mad = float(np.median(np.abs(x)))
    b = mad / _LN2 if mad > 0 else 0.0
    sigma = _SQRT2 * b
    return float(sigma)


global_mad = float(np.median(np.abs(oof_resid.astype(np.float32))))
global_sigma_robust = max(
    (_SQRT2 * (global_mad / _LN2 if global_mad > 0 else 0.0)), 70.0
)
global_sigma_std = max(float(np.nanstd(oof_resid.astype(np.float32), ddof=0)), 70.0)
global_sigma = float(0.65 * global_sigma_robust + 0.35 * global_sigma_std)
global_sigma = max(global_sigma, 70.0)

pat_grp = oof_df.groupby("Patient")["resid"]
pat_cnt = pat_grp.size().astype(np.float32)
pat_mad_sigma = pat_grp.apply(mad_to_sigma_laplace).astype(np.float32)

k = 8.0
w = (pat_cnt / (pat_cnt + k)).astype(np.float32)
patient_sigma = (
    w * pat_mad_sigma.fillna(global_sigma) + (1.0 - w) * global_sigma
).astype(np.float32)
patient_sigma = np.clip(patient_sigma, 70.0, np.inf).astype(np.float32)

base_conf = np.array(
    [patient_sigma.get(p, global_sigma) for p in test_patients], dtype=np.float32
)
base_conf = np.clip(base_conf, 70.0, np.inf).astype(np.float32)

train_base_sigma = np.array(
    [patient_sigma.get(p, global_sigma) for p in train_patients], dtype=np.float32
)
train_base_sigma = np.clip(train_base_sigma, 70.0, np.inf).astype(np.float32)

y_true = y_train.astype(np.float32)
y_pred = oof_pred.astype(np.float32)

abs_w_test = np.abs(test_weeks_raw.astype(np.float32))
abs_w_train = np.abs(train_weeks_raw.astype(np.float32))

alpha_grid = np.array([0.00, 0.04, 0.07, 0.10, 0.13], dtype=np.float32)
scales = np.linspace(0.25, 1.50, 251).astype(np.float32)

best_score = -1e18
best_scale = 1.0
best_alpha = 0.10

for alpha in alpha_grid:
    week_factor_train = 1.0 + alpha * (abs_w_train / 100.0)
    week_factor_train = np.clip(week_factor_train, 1.0, 1.25).astype(np.float32)
    sigma_train_nominal = np.clip(
        train_base_sigma * week_factor_train, 70.0, np.inf
    ).astype(np.float32)

    for s in scales:
        sig = np.clip(sigma_train_nominal * s, 70.0, np.inf).astype(np.float32)
        score = float(np.mean(competition_metric(y_true, y_pred, sig)))
        if score > best_score:
            best_score = score
            best_scale = float(s)
            best_alpha = float(alpha)

week_factor_test = 1.0 + best_alpha * (abs_w_test / 100.0)
week_factor_test = np.clip(week_factor_test, 1.0, 1.25).astype(np.float32)

conf = np.clip(base_conf * week_factor_test * best_scale, 70.0, np.inf).astype(
    np.float32
)
fvc_pred = pred_test.astype(np.float32)

subm = pd.DataFrame(
    {"Patient_Week": test_patient_week, "FVC": fvc_pred, "Confidence": conf}
)
subm = sub[["Patient_Week"]].merge(subm, on="Patient_Week", how="left")

subm["FVC"] = subm["FVC"].fillna(np.nanmedian(subm["FVC"].values)).astype(np.float32)
subm["Confidence"] = subm["Confidence"].fillna(global_sigma).astype(np.float32)
subm["Confidence"] = np.maximum(subm["Confidence"].values.astype(np.float32), 70.0)

print("OOF metric (using calibrated sigma):", best_score)
print("Chosen global Confidence scale:", best_scale)
print("Chosen week alpha:", best_alpha)
subm.head(), subm.shape



## === cell 12
subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print("Confidence summary:", pd.Series(subm["Confidence"]).describe())
print(subm.head(10))
