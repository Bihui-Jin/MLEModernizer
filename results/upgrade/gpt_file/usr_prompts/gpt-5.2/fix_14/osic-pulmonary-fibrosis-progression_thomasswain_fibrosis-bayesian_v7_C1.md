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

-6.8665

# 6. Current score

-8.58579

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.98469) has done: 'I remove the PyMC3/Theano dependency that is crashing under the provided NumPy version by replacing it with a mathematically equivalent hierarchical linear model fit using scikit-learn (same per-patient intercept/slope structure). I keep the core semantics the same: learn population priors for intercept/slope, infer per-patient linear trajectories over Weeks, then generate FVC predictions plus an uncertainty estimate used as `Confidence`. I also fix pathing to the provided dataset location, ensure the label encoder is available, and make sure the produced `submission.csv` matches `sample_submission.csv` (same `Patient_Week` rows/order) so it is accepted by Kaggle. Finally, I keep runtime safely under the limit by avoiding MCMC and using closed-form / ridge regression while preserving the original modeling idea.'
- What this solution (achieved -8.59435) has done: 'I keep your per-patient intercept/slope Ridge setup unchanged, but tune the two places that most directly affect the competition metric: (1) the ridge regularization strength (controls bias/variance of FVC predictions), and (2) the uncertainty calibration (Confidence), which heavily impacts the Laplace log-likelihood. To do this minimally and safely, I add a tiny patient-level CV on the training set’s last-3 visits to pick `ridge_alpha` from a small grid and a single multiplicative `sigma_scale` to better match the metric’s optimal sigma behavior. This does not change the model form or training approach—only hyperparameters and confidence calibration—so it should move the score upward toward your target. The submission format/ordering remains identical and still writes `submission.csv`.'
- What this solution (achieved -8.59435) has done: 'I make two minimal, metric-aligned tweaks to move your score up toward the target without changing the ridge per-patient intercept/slope core logic: (1) tune the uncertainty calibration with a slightly richer `sigma_scale_grid` and allow `sigma_add` (a small additive confidence floor) since the Laplace metric strongly rewards well-calibrated sigma, and (2) choose hyperparameters using a CV proxy that matches Kaggle more closely by validating on each patient’s *last three weeks relative to that patient’s max week* (instead of simply tail(3) after sorting), which better mimics “final visits” semantics. Everything else (design matrix, Ridge with fit_intercept=False, prediction formula) remains the same, and the script still writes `submission.csv` in the exact sample submission order.'
- What this solution (achieved -8.59436) has done: 'I keep your per-patient intercept/slope Ridge model exactly as-is and only adjust the two places that most directly move the Laplace Log Likelihood: the CV proxy selection (to better match Kaggle’s “final three visits” per patient) and the confidence calibration. Concretely, I (1) change the CV validation to score only on each validation patient’s last-3 observed weeks (instead of choosing 3 points closest to the max week, which can accidentally include non-final points), and (2) slightly expand the `sigma_scale`/`sigma_add` grids so the calibration can move closer to the target score without changing model form. Everything else (design matrix, Ridge fit, prediction formula, submission alignment/order and output `submission.csv`) stays the same.'
- What this solution (achieved -8.59415) has done: 'I keep your per-patient intercept/slope Ridge model and training loop intact, and only make small, metric-aligned adjustments that should raise the score toward the target. Specifically, I (1) tune the inflation formula’s two coefficients (week-based and low-data-based) via the same patient-fold CV proxy you already use, because Confidence calibration is a primary driver of the Laplace log-likelihood, and (2) slightly widen the sigma calibration grids so CV can pick a better Confidence without changing semantics. Everything else—design matrix, Ridge fit, prediction formula, submission row order/format—stays the same, and the script still writes `submission.csv`.'
- What this solution (achieved -8.59415) has done: 'I keep your per-patient intercept/slope Ridge setup and the existing CV proxy, but make two minimal, metric-aligned adjustments that should lift the score toward the target: (1) tune a constant multiplicative “final confidence inflation” that is applied only at submission time (not affecting FVC), since the Laplace LLL is highly sensitive to sigma calibration; and (2) slightly extend the Ridge alpha grid in the direction that often improves generalization for this design matrix without changing the model form. Both are selected via the same patient-fold, last-3-weeks CV proxy you already use, so semantics stay identical while improving calibration. Output format, row order, and paths remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved -8.59417) has done: 'I keep your ridge per-patient intercept/slope model and the same CV proxy, but make a minimal metric-aligned change to how CV selects the “last three” points: instead of taking the last three rows by week, it take the three largest observed weeks (ensuring uniqueness) per patient, which better matches “final visits” semantics when duplicate weeks exist. Then I slightly expand the confidence-calibration search space (only the sigma-related grids) because your current score suggests Confidence is still under-calibrated for the Laplace LLL, and this is the least invasive lever that can improve score without changing FVC predictions. Finally, I keep submission formatting/ordering identical, still writing `submission.csv`.'
- What this solution (achieved -8.59413) has done: 'I keep your per-patient intercept/slope Ridge model and training/prediction flow unchanged, and only adjust the CV objective so it matches Kaggle’s metric semantics more closely: per-patient averaging (each patient contributes equally via their last-3 points) rather than pooling all points together. This is a minimal change but can materially shift the chosen sigma calibration and ridge alpha in the direction that improves the public score toward your target. I also add a tiny “sigma_global” grid (computed from CV residuals) so Confidence calibration can move without changing FVC predictions, and keep the submission row order identical to `sample_submission.csv`. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -8.59412) has done: 'Your current score is below the target (gap ≈ -1.73), so we should cautiously improve without changing the ridge per-patient intercept/slope core logic. The smallest lever with the biggest impact on this metric is Confidence calibration, so I (1) adjust the CV proxy to better match Kaggle’s *test-time setting* by computing sigma_global from training folds but evaluating on validation folds (instead of mixing), and (2) slightly expand the confidence search space (only sigma-related grids) to allow CV to pick a better-calibrated uncertainty. I also make one minimal metric-consistent guard: clip final Confidence to be at least 70 at submission time (since Kaggle clips anyway, predicting <70 only hurts via the log term). Submission format, ordering, model form, and prediction equations remain unchanged.'
- What this solution (achieved -8.58573) has done: 'The timeout is dominated by the enormous nested grid search in CV: it repeatedly copies DataFrames, rebuilds dense design matrices, and recomputes the same groupby/apply work for every hyperparameter combination. I keep the exact Ridge setup and the exact scoring semantics, but make the CV loop provably equivalent and much faster by (1) precomputing per-fold predictions and per-row features once per alpha, (2) replacing slow `groupby().apply()` and per-candidate DataFrame copies with vectorized “last-3-per-patient” selection using sorting + `cumcount`, and (3) computing the Laplace log-likelihood fully in NumPy and aggregating per-patient means via `np.bincount`. These changes preserve the algorithm’s logic and results (up to negligible floating-point ordering differences) while reducing redundant work by orders of magnitude.'
- What this solution (achieved -8.58579) has done: 'The timeout is dominated by the huge nested hyperparameter grid search in cell 3 (millions of candidate evaluations) and repeated per-fold/per-candidate Python loops and recomputation. I preserve the exact model (per-patient Ridge with intercept disabled) and the exact evaluation semantics, but replace the brute-force grid enumeration with an equivalent coordinate-descent over the same discrete grids (no approximations; it still chooses values only from your grids). I also vectorize the per-fold score computation and cache fold-specific arrays so each candidate only does fast NumPy ops (no repeated bincount/groupby or Python inner loops). Finally, I remove expensive directory walking/plotting that is irrelevant to producing `submission.csv` within the 600s limit.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(DATA_DIR):
    alt = "/kaggle/data/osic-pulmonary-fibrosis-progression"
    if os.path.exists(alt):
        DATA_DIR = alt

print("Using DATA_DIR:", DATA_DIR)

for fn in ("train.csv", "test.csv", "sample_submission.csv"):
    p = os.path.join(DATA_DIR, fn)
    print("Found:", p, "exists:", os.path.exists(p))




## === cell 1
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_raw = train.copy()
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

train.drop(train[train.Patient == "ID00197637202246865691526"].index, inplace=True)

exclude_test_patient_data_from_trainset = False
if exclude_test_patient_data_from_trainset:
    train = train[~train["Patient"].isin(test["Patient"].unique())]

all_pat = pd.concat(
    [train[["Patient"]], test[["Patient"]]], axis=0, ignore_index=True
).drop_duplicates()

le_id = LabelEncoder()
le_id.fit(all_pat["Patient"].values)

train["PatientID"] = le_id.transform(train["Patient"])
test["PatientID"] = le_id.transform(test["Patient"])

n_patients = len(le_id.classes_)
print("n_patients:", n_patients, "train rows:", len(train), "test rows:", len(test))
train.head()




## === cell 2
def _build_design(df, n_patients):
    patient_ids = df["PatientID"].values.astype(int)
    weeks = df["Weeks"].values.astype(float)
    N = len(df)
    X = np.zeros((N, 2 * n_patients), dtype=np.float32)
    rows = np.arange(N)
    X[rows, patient_ids] = 1.0
    X[rows, n_patients + patient_ids] = weeks
    return X


def _laplace_lll(y_true, y_pred, sigma):
    sigma_c = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return -np.sqrt(2.0) * delta / sigma_c - np.log(np.sqrt(2.0) * sigma_c)


def _select_patient_last3_by_weeks(df_one_patient):
    d = df_one_patient.sort_values("Weeks")
    d = d.drop_duplicates(subset=["Weeks"], keep="last")
    return d.tail(3)


def _sigma_inflation(weeks, count_pid, week_coef, count_coef):
    weeks = weeks.astype(float)
    count_pid = count_pid.astype(float)
    return (
        1.0
        + week_coef * (np.abs(weeks) / 100.0)
        + count_coef / np.sqrt(np.maximum(count_pid, 1.0))
    )


def _select_last3_idx_patient_weeks(patient_ids, weeks):
    order = np.lexsort(
        (np.arange(len(patient_ids), dtype=np.int64), weeks, patient_ids)
    )
    pid_s = patient_ids[order]
    w_s = weeks[order]

    is_last = np.ones(len(order), dtype=bool)
    same_pid = pid_s[1:] == pid_s[:-1]
    same_week = w_s[1:] == w_s[:-1]
    is_last[:-1] = ~(same_pid & same_week)

    order2 = order[is_last]

    rev = order2[::-1]
    pid_rev = patient_ids[rev]
    new_group = np.empty(len(pid_rev), dtype=bool)
    new_group[0] = True
    new_group[1:] = pid_rev[1:] != pid_rev[:-1]
    grp = np.cumsum(new_group) - 1
    pos = np.arange(len(pid_rev), dtype=np.int64)
    start = np.empty_like(pos)
    start[0] = 0
    start[1:] = np.where(new_group[1:], pos[1:], 0)
    start = np.maximum.accumulate(start)
    cc = pos - start
    keep_rev = cc < 3

    keep_idx = rev[keep_rev]
    keep_idx.sort()
    return keep_idx


def _patient_weighted_lll_numpy_compact(patient_ids, y_true, y_pred, sigma):
    lll = _laplace_lll(y_true, y_pred, sigma)
    uniq, inv = np.unique(patient_ids, return_inverse=True)
    sums = np.bincount(inv, weights=lll).astype(np.float64)
    cnts = np.bincount(inv).astype(np.float64)
    means = sums / cnts
    return float(means.mean())


unique_pids = np.array(sorted(train["PatientID"].unique()))
rng = np.random.RandomState(RANDOM_STATE)
rng.shuffle(unique_pids)
n_folds = 5
folds = np.array_split(unique_pids, n_folds)

alpha_grid = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

sigma_scale_grid = [
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.8,
    2.0,
    2.2,
    2.5,
]
sigma_add_grid = [
    0.0,
    10.0,
    20.0,
    40.0,
    60.0,
    80.0,
    100.0,
    130.0,
    160.0,
    200.0,
    250.0,
    300.0,
]

week_coef_grid = [0.10, 0.15, 0.20]
count_coef_grid = [0.25, 0.35, 0.45]

final_conf_mult_grid = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
sigma_global_mult_grid = [0.80, 0.85, 0.925, 1.0, 1.075, 1.15, 1.25]
fvc_bias_grid = [-150.0, -100.0, -60.0, -30.0, 0.0, 30.0, 60.0, 100.0, 150.0]
sigma_bias_grid = [0.0, 15.0, 30.0, 45.0, 60.0, 80.0, 100.0]

train_pid_all = train["PatientID"].values.astype(np.int32)
train_w_all = train["Weeks"].values.astype(np.float32)
train_y_all = train["FVC"].values.astype(np.float32)


def _make_fold_cache_for_alpha(alpha):
    fold_cache = []
    fold_sigma_global = []

    for k in range(n_folds):
        val_pids = folds[k]
        is_val = np.isin(train_pid_all, val_pids)
        tr_idx = np.where(~is_val)[0]
        va_idx = np.where(is_val)[0]

        tr_pid = train_pid_all[tr_idx].astype(int)
        tr_w = train_w_all[tr_idx].astype(float)

        X_tr = np.zeros((len(tr_idx), 2 * n_patients), dtype=np.float32)
        r = np.arange(len(tr_idx))
        X_tr[r, tr_pid] = 1.0
        X_tr[r, n_patients + tr_pid] = tr_w
        y_tr = train_y_all[tr_idx].astype(float)

        m = Ridge(alpha=alpha, fit_intercept=False, random_state=RANDOM_STATE)
        m.fit(X_tr, y_tr)

        y_tr_pred = m.predict(X_tr)
        resid_tr = y_tr - y_tr_pred
        sigma_global_tr = float(np.sqrt(np.mean(resid_tr**2)))
        fold_sigma_global.append(sigma_global_tr)

        va_pid = train_pid_all[va_idx].astype(np.int32)
        va_w = train_w_all[va_idx].astype(np.float32)

        X_va = np.zeros((len(va_idx), 2 * n_patients), dtype=np.float32)
        r2 = np.arange(len(va_idx))
        X_va[r2, va_pid] = 1.0
        X_va[r2, n_patients + va_pid] = va_w
        y_va_pred = m.predict(X_va).astype(np.float32)

        counts_tr = np.bincount(tr_pid, minlength=n_patients).astype(np.float32)
        count_pid_va = counts_tr[va_pid]

        keep_local = _select_last3_idx_patient_weeks(va_pid, va_w)

        pid_l3 = va_pid[keep_local].astype(np.int32)
        w_l3 = va_w[keep_local].astype(np.float32)
        y_l3 = train_y_all[va_idx][keep_local].astype(np.float32)
        pred_l3 = y_va_pred[keep_local].astype(np.float32)
        cnt_l3 = count_pid_va[keep_local].astype(np.float32)

        uniq_pid, inv = np.unique(pid_l3, return_inverse=True)
        fold_cache.append(
            {
                "pid": pid_l3,
                "w": w_l3,
                "y": y_l3,
                "pred": pred_l3,
                "cnt": cnt_l3,
                "inv": inv.astype(np.int32),
                "n_uniq": int(len(uniq_pid)),
            }
        )

    sigma_global_all = float(np.mean(fold_sigma_global))
    return fold_cache, sigma_global_all


def _score_params_on_cache(
    fold_cache,
    sigma_global_all,
    week_coef,
    count_coef,
    sg_mult,
    sscale,
    sadd,
    sbias,
    fmult,
    fvc_bias,
):
    scores = []
    sg = sigma_global_all * float(sg_mult)
    wk = float(week_coef)
    cc = float(count_coef)
    ssc = float(sscale)
    sad = float(sadd)
    sbi = float(sbias)
    fm = float(fmult)
    fb = float(fvc_bias)

    for fc in fold_cache:
        w = fc["w"].astype(np.float64)
        cnt = fc["cnt"].astype(np.float64)
        infl = 1.0 + wk * (np.abs(w) / 100.0) + cc / np.sqrt(np.maximum(cnt, 1.0))
        sigma_vec = (sg * infl * ssc + sad + sbi) * fm

        lll = _laplace_lll(
            fc["y"].astype(np.float64),
            (fc["pred"].astype(np.float64) + fb),
            sigma_vec,
        )
        sums = np.bincount(fc["inv"], weights=lll, minlength=fc["n_uniq"]).astype(
            np.float64
        )
        cnts = np.bincount(fc["inv"], minlength=fc["n_uniq"]).astype(np.float64)
        scores.append(float((sums / cnts).mean()))
    return float(np.mean(scores))


def _coordinate_descent_discrete(fold_cache, sigma_global_all):
    cur = {
        "week_coef": week_coef_grid[0],
        "count_coef": count_coef_grid[0],
        "sg_mult": sigma_global_mult_grid[0],
        "sscale": sigma_scale_grid[0],
        "sadd": sigma_add_grid[0],
        "sbias": sigma_bias_grid[0],
        "fmult": final_conf_mult_grid[0],
        "fvc_bias": fvc_bias_grid[0],
    }
    cur_score = _score_params_on_cache(fold_cache, sigma_global_all, **cur)

    improved = True
    while improved:
        improved = False
        for name, grid in [
            ("week_coef", week_coef_grid),
            ("count_coef", count_coef_grid),
            ("sg_mult", sigma_global_mult_grid),
            ("sscale", sigma_scale_grid),
            ("sadd", sigma_add_grid),
            ("sbias", sigma_bias_grid),
            ("fmult", final_conf_mult_grid),
            ("fvc_bias", fvc_bias_grid),
        ]:
            best_val = cur[name]
            best_sc = cur_score
            for v in grid:
                if v == cur[name]:
                    continue
                trial = dict(cur)
                trial[name] = v
                sc = _score_params_on_cache(fold_cache, sigma_global_all, **trial)
                if sc > best_sc:
                    best_sc = sc
                    best_val = v
            if best_sc > cur_score:
                cur[name] = best_val
                cur_score = best_sc
                improved = True

    return cur_score, cur


best = None  # (mean_score, alpha, params_dict)

for alpha in alpha_grid:
    fold_cache, sigma_global_all = _make_fold_cache_for_alpha(alpha)
    score, params = _coordinate_descent_discrete(fold_cache, sigma_global_all)
    cand = (float(score), float(alpha), params)
    if (best is None) or (cand[0] > best[0]):
        best = cand

best_lll, ridge_alpha, params = best
week_coef = params["week_coef"]
count_coef = params["count_coef"]
sigma_scale = params["sscale"]
sigma_add = params["sadd"]
sigma_bias = params["sbias"]
final_conf_mult = params["fmult"]
sigma_global_mult = params["sg_mult"]
fvc_bias = params["fvc_bias"]

print(
    "Chosen "
    f"ridge_alpha={ridge_alpha}, week_coef={week_coef}, count_coef={count_coef}, "
    f"sigma_scale={sigma_scale}, sigma_add={sigma_add}, sigma_bias={sigma_bias}, "
    f"final_conf_mult={final_conf_mult}, sigma_global_mult={sigma_global_mult}, fvc_bias={fvc_bias} "
    f"via CV proxy LLL={best_lll:.5f}"
)

patient_ids = train["PatientID"].values.astype(int)
weeks = train["Weeks"].values.astype(float)
y_fvc = train["FVC"].values.astype(float)

N = len(train)
X = np.zeros((N, 2 * n_patients), dtype=np.float32)
rows = np.arange(N)
X[rows, patient_ids] = 1.0
X[rows, n_patients + patient_ids] = weeks

model_a = Ridge(alpha=ridge_alpha, fit_intercept=False, random_state=RANDOM_STATE)
model_a.fit(X, y_fvc)

coef = model_a.coef_.astype(np.float64)
a_hat = coef[:n_patients]
b_hat = coef[n_patients:]

y_pred_train = model_a.predict(X)
resid = y_fvc - y_pred_train
sigma_global = float(np.sqrt(np.mean(resid**2))) * float(sigma_global_mult)
print("sigma_global (RMSE on train rows, after mult):", sigma_global)




## === cell 3
weeks_grid = np.arange(-12, 134, dtype=np.int16)
pred_template = pd.DataFrame(
    {
        "PatientID": np.repeat(np.arange(n_patients, dtype=np.int32), len(weeks_grid)),
        "Weeks": np.tile(weeks_grid, n_patients),
    }
)

pid = pred_template["PatientID"].values.astype(int)
w = pred_template["Weeks"].values.astype(float)

pred_template["FVC_pred"] = a_hat[pid] + b_hat[pid] * w + float(fvc_bias)

counts = (
    train.groupby("PatientID")
    .size()
    .reindex(range(n_patients))
    .fillna(0)
    .values.astype(float)
)
count_pid = counts[pid]

inflation = _sigma_inflation(w, count_pid, float(week_coef), float(count_coef))
pred_template["sigma"] = (
    sigma_global * inflation * float(sigma_scale) + float(sigma_add) + float(sigma_bias)
) * float(final_conf_mult)

pred_template.head()




## === cell 4
df = pd.DataFrame(
    {
        "Patient": le_id.inverse_transform(
            pred_template["PatientID"].values.astype(int)
        ),
        "Weeks": pred_template["Weeks"].values.astype(int),
        "FVC_pred": pred_template["FVC_pred"].values.astype(float),
        "sigma": pred_template["sigma"].values.astype(float),
    }
)
df["FVC_inf"] = df["FVC_pred"] - df["sigma"]
df["FVC_sup"] = df["FVC_pred"] + df["sigma"]

df = pd.merge(
    df, train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
)
df = df.rename(columns={"FVC": "FVC_true"})
df.head()




## === cell 5
pass




## === cell 6
use_only_last_3_measures = True

eval_df = df.dropna(subset=["FVC_true"]).copy()
if use_only_last_3_measures:
    y_eval = (
        eval_df.groupby("Patient", group_keys=False)
        .apply(lambda d: _select_patient_last3_by_weeks(d.sort_values("Weeks")))
        .copy()
    )
else:
    y_eval = eval_df.copy()

rmse = float(
    np.sqrt(np.mean((y_eval["FVC_pred"].values - y_eval["FVC_true"].values) ** 2))
)
print(f"RMSE (mean): {rmse:.1f} ml")

sigma_c = y_eval["sigma"].values.copy()
sigma_c[sigma_c < 70] = 70
delta = np.abs(y_eval["FVC_pred"].values - y_eval["FVC_true"].values)
delta[delta > 1000] = 1000
lll = -np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)
print(f"Laplace Log Likelihood (proxy on train final-3): {float(np.mean(lll)):.4f}")




## === cell 7
sub = sample_sub.copy()
tmp = sub["Patient_Week"].str.rsplit("_", n=1, expand=True)
sub["Patient"] = tmp[0]
sub["Weeks"] = tmp[1].astype(int)

sub["PatientID"] = le_id.transform(sub["Patient"].values)
pid = sub["PatientID"].values.astype(int)
w = sub["Weeks"].values.astype(float)

sub_fvc = a_hat[pid] + b_hat[pid] * w + float(fvc_bias)

count_pid = counts[pid]
inflation = _sigma_inflation(w, count_pid, float(week_coef), float(count_coef))
sub_sigma = (
    sigma_global * inflation * float(sigma_scale) + float(sigma_add) + float(sigma_bias)
) * float(final_conf_mult)

final = pd.DataFrame(
    {
        "Patient_Week": sub["Patient_Week"].values,
        "FVC": sub_fvc.astype(float),
        "Confidence": sub_sigma.astype(float),
    }
)

final["FVC"] = final["FVC"].fillna(final["FVC"].median())
final["Confidence"] = final["Confidence"].fillna(max(70.0, sigma_global))
final["Confidence"] = final["Confidence"].clip(lower=70.0)

final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final.shape)
final.head()
