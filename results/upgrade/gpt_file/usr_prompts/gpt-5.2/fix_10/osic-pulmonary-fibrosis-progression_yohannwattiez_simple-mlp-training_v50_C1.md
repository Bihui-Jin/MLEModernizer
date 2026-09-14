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

-8.52911

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76189) has done: 'I fix the environment-breaking TensorFlow import issue by forcing the pure-Python protobuf backend before importing TF (this resolves the `MessageFactory.GetPrototype` crash). Then I correct the broken Kaggle file paths and remove dependencies on missing external pickles/datasets by fitting `data_preparation()` directly from the provided `train.csv` and building the needed `Weight` column safely. I also update the Adam optimizer arguments to the current Keras API (`learning_rate` instead of `lr/decay`) while keeping the same intended hyperparameters via `weight_decay`. Finally, I ensure `X_prediction` is created correctly from `sample_submission.csv` and that a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved -8.76189) has done: 'I fix the environment-breaking TensorFlow/protobuf crash by setting the protobuf backend before *any* TensorFlow-related import and falling back to a safe import if the first attempt fails. Then I fix the Keras compile error by passing `metrics=[score]` (newer Keras requires a list/tuple/dict) while keeping the same model and loss logic. Next I correct the test/sample_submission feature build so `X_prediction` contains the required clinical columns (`Sex`, `SmokingStatus`, etc.) and doesn’t reference a missing `FVC` column, which is what caused the downstream `KeyError: 'Sex'`. Finally, I keep the same training/inference loop but make prediction arrays the right shape and ensure a valid `submission.csv` is always written with the exact required columns.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before any TF import and (if needed) downgrading protobuf in-process to a compatible version (this is the only reliable way to avoid `MessageFactory.GetPrototype` in Kaggle images where TF/protobuf are mismatched). Then I fix the `None values not supported` training crash by ensuring every feature used in `SELECTED_COLUMNS` is fully populated after merges/encoding (fill missing numerics with train medians and missing categoricals with train modes before `data_preparation`). Finally, I keep the exact same model/loss/training loop but make the fold split deterministic and non-overlapping (still 8 folds) to slightly improve stability/score toward your target without changing the core approach, and I always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -15.27891) has done: 'I fix the environment-breaking TensorFlow/protobuf crash by avoiding TensorFlow entirely (the current error happens at import time) and replacing the training/prediction part with a lightweight, deterministic clinical-feature regression that still outputs both FVC and Confidence in the required format. I also fix the immediate runtime error in training (`None values not supported`) by ensuring all model inputs are numeric and fully imputed (even though we won’t use TF afterwards). Finally, I improve score toward your target by predicting per-patient FVC trend using a robust linear fit on each patient’s history (much better than a constant baseline), and set Confidence based on training residual dispersion with clipping aligned to the metric.'
- What this solution (achieved -14.8671) has done: 'Your current score is far below the target, so we should improve the predictions without changing the overall “per-patient trend” core approach. The biggest easy win is to fit each patient’s FVC trend relative to their own baseline visit week (Week=0 aligned) rather than using absolute calendar weeks, which reduces intercept/slope bias across patients and typically improves this competition metric. Next, we set per-row Confidence using the training residual spread as a function of “weeks-from-baseline magnitude” (still simple and deterministic), and we remove the harmful override that sets the known test baseline row Confidence to 0.1 (the metric clips sigma to 70 anyway, and ultra-low sigma is not beneficial). These are minimal changes that keep the same modeling logic (linear fit per patient + residual-based sigma) while moving the score toward your target.'
- What this solution (achieved -13.26799) has done: 'I fix the root cause of the `Age`/`Sex`/`SmokingStatus` KeyErrors by preventing column-name collisions created by merging `base` back into `train` (which currently produces `Age_x/Age_y` etc.). Then I make the imputation robust to missing columns and ensure `Sex`/`SmokingStatus` always exist for both train and prediction frames, so `group_stats` is defined and inference runs. Finally, I keep the existing per-patient linear trend + residual-based confidence logic unchanged, and ensure `FVC1/Confidence1` are created before writing a correctly formatted `submission.csv`.'
- What this solution (achieved -8.49254) has done: 'Your current score (-13.27) is far worse than the target (-7.08), so we should legitimately improve the metric with the smallest possible change to your existing per-patient linear trend + residual-based confidence approach. The biggest low-risk gain is to stop predicting for *all* weeks in `sample_submission` and instead predict only the **final three visits per test patient** (the only ones scored); for earlier weeks we copy the known baseline FVC and use a safe confidence. Next, we compute those final-three weeks directly from `sample_submission.csv` (no hidden info) and also calibrate confidence for scored rows only, because overconfident predictions on scored rows are heavily penalized by the Laplace term. These changes keep your core model/trend fitting identical, but align inference/post-processing with the competition’s scoring behavior to move the score upward toward the target.'
- What this solution (achieved -8.52911) has done: 'We keep your per-patient linear trend model unchanged and focus only on confidence calibration and the scored-row postprocessing, because your current gap to target (-8.49 vs -7.08; higher is better) is most sensitive to sigma choices under the Laplace log-likelihood. Concretely: (1) replace the fixed `* 1.25` confidence inflation with a small grid search on a single global multiplier (trained on out-of-fold residuals) to pick the multiplier that best matches the competition metric; (2) additionally learn a single global additive confidence offset (again via OOF metric) to avoid systematic under/over-confidence. This preserves the same prediction core (a+b*week), only tunes how Confidence is mapped, and should move the score upward toward your target without altering evaluation semantics. The submission writing, paths, and row alignment remain the same.'

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
train_sorted = train.sort_values(["Patient", "Weeks"]).copy()

w0 = (
    train_sorted.loc[train_sorted["Weeks"] == 0]
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    .rename(
        columns={"Weeks": "Base_Week0", "FVC": "Base_FVC0", "Percent": "Base_Percent0"}
    )
)

wmin = (
    train_sorted.groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    .rename(
        columns={
            "Weeks": "Min_week",
            "FVC": "Base_FVC_min",
            "Percent": "Base_Percent_min",
        }
    )
)

base = wmin.merge(
    w0[["Patient", "Base_Week0", "Base_FVC0", "Base_Percent0"]],
    on="Patient",
    how="left",
)

base["Base_week_anchor"] = np.where(
    base["Base_Week0"].notna(), 0.0, base["Min_week"].astype(float)
)
base["Base_FVC"] = np.where(
    base["Base_FVC0"].notna(),
    base["Base_FVC0"].astype(float),
    base["Base_FVC_min"].astype(float),
)
base["Base_Percent"] = np.where(
    base["Base_Percent0"].notna(),
    base["Base_Percent0"].astype(float),
    base["Base_Percent_min"].astype(float),
)

train = train.merge(
    base[["Patient", "Base_week_anchor", "Base_FVC", "Base_Percent"]],
    on="Patient",
    how="left",
)

train["Base_week"] = train["Weeks"].astype(float) - train["Base_week_anchor"].astype(
    float
)

print(train[["Patient", "Weeks", "Base_week_anchor", "Base_week", "Base_FVC"]].head())
print("train cols after base merge:", train.columns.tolist())



## === cell 3
X_prediction = sample_sub.copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)

baseline_test = raw_test.copy().rename(
    columns={"Weeks": "Base_week_anchor", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)

X_prediction = X_prediction.merge(
    baseline_test[
        [
            "Patient",
            "Base_week_anchor",
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
X_prediction["Base_week"] = X_prediction["Weeks"].astype(float) - X_prediction[
    "Base_week_anchor"
].astype(float)

print(X_prediction.head())
print("X_prediction columns:", X_prediction.columns.tolist())



## === cell 4
for df in (train, X_prediction):
    for required in ["Age", "Sex", "SmokingStatus", "Percent"]:
        if required not in df.columns:
            df[required] = np.nan

for df in (train, X_prediction):
    for col in ["Sex", "SmokingStatus"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", np.nan).replace("None", np.nan)
    for col in [
        "Weeks",
        "FVC",
        "Percent",
        "Age",
        "Base_week_anchor",
        "Base_FVC",
        "Base_week",
        "Base_Percent",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def safe_median(s, fallback=0.0):
    s = pd.to_numeric(s, errors="coerce")
    v = s.median()
    return float(v) if pd.notna(v) else float(fallback)


num_fill = {
    "Percent": safe_median(train.get("Percent", pd.Series(dtype=float)), fallback=0.0),
    "Age": safe_median(train.get("Age", pd.Series(dtype=float)), fallback=60.0),
    "Weeks": safe_median(train.get("Weeks", pd.Series(dtype=float)), fallback=0.0),
    "Base_week_anchor": safe_median(
        train.get("Base_week_anchor", pd.Series(dtype=float)), fallback=0.0
    ),
    "Base_FVC": safe_median(
        train.get("Base_FVC", pd.Series(dtype=float)),
        fallback=safe_median(train.get("FVC", pd.Series(dtype=float)), fallback=3000.0),
    ),
    "Base_week": safe_median(
        train.get("Base_week", pd.Series(dtype=float)), fallback=0.0
    ),
    "Base_Percent": safe_median(
        train.get("Base_Percent", pd.Series(dtype=float)),
        fallback=safe_median(
            train.get("Percent", pd.Series(dtype=float)), fallback=0.0
        ),
    ),
}

cat_fill = {}
for col in ["Sex", "SmokingStatus"]:
    if col in train.columns:
        mode_val = train[col].mode(dropna=True)
        cat_fill[col] = mode_val.iloc[0] if len(mode_val) else "Unknown"
    else:
        cat_fill[col] = "Unknown"

for df in (train, X_prediction):
    for k, v in num_fill.items():
        if k in df.columns:
            df[k] = df[k].fillna(v)
    for k, v in cat_fill.items():
        if k in df.columns:
            df[k] = df[k].fillna(v).astype(str)

train["FVC"] = (
    pd.to_numeric(train["FVC"], errors="coerce")
    .fillna(float(pd.to_numeric(train["FVC"], errors="coerce").median()))
    .astype(float)
)

print("After fill, train cols:", train.columns.tolist())
print(
    "After fill, missing Sex/SmokingStatus counts:",
    train["Sex"].isna().sum(),
    train["SmokingStatus"].isna().sum(),
)




## === cell 5
def fit_patient_trend(df_patient: pd.DataFrame):
    w = df_patient["Base_week"].values.astype(float)
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
patient_sigma = {}
all_resid = []
all_abs_baseweek = []

for pid, g in train.groupby("Patient", sort=False):
    a, b, resid = fit_patient_trend(g)
    patient_models[pid] = (a, b)
    all_resid.append(resid)
    all_abs_baseweek.append(np.abs(g["Base_week"].values.astype(float)))

    s = float(np.std(resid)) if len(resid) > 1 else float(np.abs(resid).mean())
    if not np.isfinite(s) or s <= 0:
        s = float(np.abs(resid).mean()) if len(resid) else 0.0
    patient_sigma[pid] = float(max(70.0, s))

all_resid = np.concatenate(all_resid) if len(all_resid) else np.array([0.0])
all_abs_baseweek = (
    np.concatenate(all_abs_baseweek) if len(all_abs_baseweek) else np.array([0.0])
)

global_mae = float(np.mean(np.abs(all_resid)))
global_sigma = (
    float(np.std(all_resid)) if np.isfinite(np.std(all_resid)) else float(global_mae)
)

bw = all_abs_baseweek
abs_r = np.abs(all_resid)
if len(bw) >= 10 and np.std(bw) > 1e-6:
    A = np.vstack([np.ones_like(bw), bw]).T
    coef, *_ = np.linalg.lstsq(A, abs_r, rcond=None)
    conf_a = float(max(0.0, coef[0]))
    conf_b = float(max(0.0, coef[1]))
else:
    conf_a = float(global_mae)
    conf_b = 0.0

pm_rows = []
for pid, (a, b) in patient_models.items():
    row_sel = train.loc[train["Patient"] == pid, ["Sex", "SmokingStatus"]]
    if len(row_sel) == 0:
        sex0, smoke0 = "Unknown", "Unknown"
    else:
        row0 = row_sel.iloc[0]
        sex0, smoke0 = str(row0["Sex"]), str(row0["SmokingStatus"])
    pm_rows.append((pid, sex0, smoke0, a, b, patient_sigma.get(pid, 70.0)))

pm_df = pd.DataFrame(
    pm_rows, columns=["Patient", "Sex", "SmokingStatus", "a", "b", "sigma"]
)

group_stats = (
    pm_df.groupby(["Sex", "SmokingStatus"], dropna=False)
    .agg(a_med=("a", "median"), b_med=("b", "median"), sig_med=("sigma", "median"))
    .reset_index()
)

global_a_med = (
    float(pm_df["a"].median()) if len(pm_df) else float(train["FVC"].median())
)
global_b_med = float(pm_df["b"].median()) if len(pm_df) else 0.0
global_sig_med = float(pm_df["sigma"].median()) if len(pm_df) else 70.0

print(
    "global_mae:",
    global_mae,
    "global_sigma:",
    global_sigma,
    "conf_a:",
    conf_a,
    "conf_b:",
    conf_b,
    "global_a_med:",
    global_a_med,
    "global_b_med:",
    global_b_med,
)



## === cell 6
pred_fvc = np.zeros(len(X_prediction), dtype=float)
pred_conf = np.zeros(len(X_prediction), dtype=float)

X_pred2 = X_prediction.merge(group_stats, on=["Sex", "SmokingStatus"], how="left")

for i, (pid, bw0, a_med, b_med, sig_med) in enumerate(
    zip(
        X_pred2["Patient"].values,
        X_pred2["Base_week"].values,
        X_pred2["a_med"].values,
        X_pred2["b_med"].values,
        X_pred2["sig_med"].values,
    )
):
    bwf = float(bw0)

    if pid in patient_models:
        a, b = patient_models[pid]
        pred_fvc[i] = a + b * bwf
        base_sig = patient_sigma.get(pid, global_sig_med)
    else:
        aa = float(a_med) if np.isfinite(a_med) else global_a_med
        bb = float(b_med) if np.isfinite(b_med) else global_b_med
        pred_fvc[i] = aa + bb * bwf
        base_sig = float(sig_med) if np.isfinite(sig_med) else global_sig_med

    c = 0.65 * base_sig + 0.35 * (conf_a + conf_b * abs(bwf))
    c = 0.85 * c + 0.15 * global_sigma
    pred_conf[i] = max(70.0, float(c))

X_prediction["FVC1"] = pred_fvc
X_prediction["Confidence1"] = pred_conf

print("Pred summaries:", np.nanmean(pred_fvc), np.nanmean(pred_conf))



## === cell 7


def laplace_metric_np(y_true, y_pred, sigma):
    sigma_c = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return -np.sqrt(2.0) * delta / sigma_c - np.log(np.sqrt(2.0) * sigma_c)


def build_point_and_baseconf(df: pd.DataFrame):
    df2 = df.merge(group_stats, on=["Sex", "SmokingStatus"], how="left")
    y_true = df2["FVC"].values.astype(float)
    y_pred = np.zeros(len(df2), dtype=float)
    base_conf = np.zeros(len(df2), dtype=float)

    for i, (pid, bwf, a_med, b_med, sig_med) in enumerate(
        zip(
            df2["Patient"].values,
            df2["Base_week"].values.astype(float),
            df2["a_med"].values,
            df2["b_med"].values,
            df2["sig_med"].values,
        )
    ):
        if pid in patient_models:
            a, b = patient_models[pid]
            y_pred[i] = a + b * float(bwf)
            base_sig = patient_sigma.get(pid, global_sig_med)
        else:
            aa = float(a_med) if np.isfinite(a_med) else global_a_med
            bb = float(b_med) if np.isfinite(b_med) else global_b_med
            y_pred[i] = aa + bb * float(bwf)
            base_sig = float(sig_med) if np.isfinite(sig_med) else global_sig_med

        c = 0.65 * base_sig + 0.35 * (conf_a + conf_b * abs(float(bwf)))
        c = 0.85 * c + 0.15 * global_sigma
        base_conf[i] = max(70.0, float(c))

    return y_true, y_pred, base_conf


y_true_cal, y_pred_cal, base_conf_cal = build_point_and_baseconf(train)

mult_grid = np.array([1.00, 1.10, 1.20, 1.25, 1.30, 1.40], dtype=float)
off_grid = np.array([0.0, 20.0, 40.0, 60.0, 80.0], dtype=float)

best = (-1e18, 1.25, 0.0)  # (metric, mult, offset)
for m in mult_grid:
    for off in off_grid:
        sigma_try = np.maximum(70.0, base_conf_cal * m + off)
        score = float(np.mean(laplace_metric_np(y_true_cal, y_pred_cal, sigma_try)))
        if score > best[0]:
            best = (score, float(m), float(off))

best_score, best_mult, best_off = best
print(
    "Calibrated confidence params on train:",
    {"mult": best_mult, "offset": best_off, "metric": best_score},
)



## === cell 8
subm = sample_sub.copy()
subm["Patient"] = subm["Patient_Week"].str.extract(r"(.*)_.*")
subm["Weeks"] = subm["Patient_Week"].str.extract(r".*_(.*)").astype(int)

last3_weeks = (
    subm.groupby("Patient")["Weeks"]
    .apply(lambda s: set(s.sort_values().tail(3).tolist()))
    .to_dict()
)
is_scored = subm.apply(
    lambda r: r["Weeks"] in last3_weeks.get(r["Patient"], set()), axis=1
)

baseline_fvc_map = raw_test.set_index("Patient")["FVC"].to_dict()
subm["FVC"] = subm["Patient"].map(baseline_fvc_map).astype(float)
subm["Confidence"] = 150.0  # safe default (clipped at 70 anyway)

pred_map = X_prediction.set_index("Patient_Week")[["FVC1", "Confidence1"]]
subm = subm.merge(pred_map, how="left", left_on="Patient_Week", right_index=True)

mask_scored_pred = is_scored.values & (~subm["FVC1"].isna())
subm.loc[mask_scored_pred, "FVC"] = subm.loc[mask_scored_pred, "FVC1"].astype(float)

subm.loc[mask_scored_pred, "Confidence"] = (
    subm.loc[mask_scored_pred, "Confidence1"].astype(float) * best_mult + best_off
)

for i in range(len(raw_test)):
    pw = f"{raw_test.Patient.iloc[i]}_{int(raw_test.Weeks.iloc[i])}"
    subm.loc[subm["Patient_Week"] == pw, "FVC"] = float(raw_test.FVC.iloc[i])
    subm.loc[subm["Patient_Week"] == pw, "Confidence"] = max(
        70.0, float(subm.loc[subm["Patient_Week"] == pw, "Confidence"].iloc[0])
    )

subm["Confidence"] = (
    pd.to_numeric(subm["Confidence"], errors="coerce")
    .fillna(150.0)
    .astype(float)
    .clip(lower=70.0, upper=1000.0)
)
subm["FVC"] = (
    pd.to_numeric(subm["FVC"], errors="coerce")
    .fillna(float(raw_test["FVC"].median()))
    .astype(float)
)

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Scored rows count (final-3 per patient):", int(is_scored.sum()))
print("Confidence summary:", submission["Confidence"].describe())
