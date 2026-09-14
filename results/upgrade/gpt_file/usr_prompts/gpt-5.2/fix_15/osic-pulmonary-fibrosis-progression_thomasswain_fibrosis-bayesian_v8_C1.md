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

-6.9351

# 6. Current score

-24.65932

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.55606) has done: 'The crash comes from building the design matrix with `n_patients = nunique()` while `PatientID` values are global label-encoded indices; when some IDs are missing in a subset, `PatientID.max()` can exceed `nunique()-1`, causing out-of-bounds indexing. I fix this by sizing the design matrix using the global number of encoded patients (`len(le_id.classes_)`) consistently in both training and prediction. I also make `_build_design_matrix` explicitly validate indices to fail fast if anything unexpected happens. These changes are score-neutral (same model/logic) but unblock end-to-end execution and ensure a valid `submission.csv` is written.'
- What this solution (achieved -9.72024) has done: 'You’re below the target (current -9.556 vs target -6.935; higher is better), so we make the smallest change that legitimately improves the metric without changing the model form: calibrate the submitted `Confidence` (sigma) to better match the Laplace log-likelihood clipping behavior. Instead of using the training RMSE directly, we compute per-patient residual spread on the last-3 visits (same visits used in scoring) and use a robust global confidence derived from those residuals (still a single constant confidence for all rows, preserving your core ridge model and prediction logic). This typically improves the score because the metric heavily penalizes under/over-confident sigma, and RMSE is not the right scale for Laplace-like residuals. We keep everything else identical and still write a valid `submission.csv`.'
- What this solution (achieved -9.86338) has done: 'You’re below the target (current -9.72024 vs target -6.9351; higher is better), so we should improve the metric with the smallest change that doesn’t alter your ridge-per-patient intercept/slope core logic. The biggest lever left is confidence calibration: the Laplace metric is very sensitive to sigma, and using a single robust sigma from absolute errors can still be miscalibrated. I keep your model and predictions identical, but compute the submission `Confidence` by directly maximizing the training Laplace log-likelihood on the *last-3 visits per patient* (matching the scoring setup), which typically moves the score upward without changing the model. I also add a tiny safeguard to ensure the submission rows align exactly with `sample_submission.csv`.'
- What this solution (achieved -9.86366) has done: 'You’re currently below the target (−9.863 vs −6.935; higher is better), so we should improve score with the smallest change that can legitimately help without touching the ridge-per-patient intercept/slope logic. The biggest remaining lever is the **Confidence** calibration: instead of a coarse multiplicative grid around a heuristic sigma, we can **directly maximize the training Laplace log-likelihood** (on last-3 visits per patient, matching scoring) using a fast 1D search over sigma. This keeps predictions identical and only changes the submitted confidence value in a metric-aligned way. I also compute the optimal sigma from the closed-form candidate region around the clip point (70) using a bounded search to keep runtime < 600s and preserve stability.'
- What this solution (achieved -9.86366) has done: 'Your score is far below the target (−9.86 vs −6.94; higher is better), so we should improve metric alignment with minimal risk while preserving your per-patient intercept+slope ridge core. The most effective small lever left is to calibrate **Confidence** correctly: your current optimization mistakenly uses `abs_err.mean()` (a constant) inside the likelihood, which is not equivalent to averaging the Laplace log-likelihood per row and can pick a suboptimal sigma. I fix the sigma objective to compute the true mean Laplace log-likelihood over all last-3 residuals (with the same clipping rules as Kaggle), then do the same bounded 1D grid search. Predictions (FVC) stay identical; only Confidence changes, and we still write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved -9.86399) has done: 'We keep your ridge per-patient intercept+slope model and prediction pipeline unchanged, and only adjust the **Confidence** calibration because that’s the only lever that can materially move the Laplace Log Likelihood without changing FVC predictions. Your current sigma search is based on a mean-absolute-error surrogate; instead we directly maximize the exact Kaggle metric (with both delta and sigma clipping) on the **same last-3-per-patient rows** used for calibration. To avoid overfitting to the clipped region and to stay stable, we do a simple 1D grid search over a reasonable sigma range and pick the maximizer; this is fast (<1s) and preserves evaluation semantics. We also set `trace["sigma"] = sigma_hat` so any downstream use of `df["sigma"]` is consistent, while the submission still uses the same `sigma_hat`.'
- What this solution (achieved -9.86321) has done: 'We keep your ridge per-patient intercept+slope model exactly as-is and only adjust the confidence calibration, since that’s the only lever left that can materially improve the Laplace log-likelihood without changing FVC predictions. Your current sigma optimization uses all last-3 residuals but can be dominated by patients with noisier/denser residual patterns; to better match the competition (3 scored points per patient), we optimize sigma on a per-patient-averaged objective (each patient contributes equally). We also switch the sigma search to a small, deterministic two-stage grid (coarse then fine around the best) to avoid edge-picking and improve stability. Submission row order and formatting remain strictly aligned to `sample_submission.csv`, and we still write `submission.csv`.'
- What this solution (achieved -9.86321) has done: 'You’re well below the target (current −9.863 vs target −6.935; higher is better), so we should *increase* score with the smallest change that doesn’t touch your ridge per-patient intercept+slope logic. The main remaining lever is **Confidence**: your current global sigma is optimized patient-weighted, but Kaggle’s metric is averaged over *rows*, and your per-patient weighting can miscalibrate sigma and hurt the score. I switch sigma calibration to **directly maximize the exact Kaggle Laplace log-likelihood averaged over all last-3 rows** (still using the same predictions; only Confidence changes). I keep the same deterministic two-stage grid search and submission alignment to `sample_submission.csv`.'
- What this solution (achieved -9.86327) has done: 'We keep your ridge per-patient intercept+slope model and week template exactly unchanged, and only adjust **Confidence** calibration since that’s the only lever allowed that can legitimately move the Laplace log-likelihood without altering FVC predictions. Your current sigma search is correct in form, but the search range can be too wide/high and the optimum for this metric is usually near `sqrt(2)*median(|err|)` (and never below 70 due to clipping), so we switch to a much tighter, data-driven bracket and a deterministic refinement around the analytic center. This reduces the chance of picking an overly-large sigma (which hurts via `-log(sigma)`) and should improve score toward the target while preserving evaluation semantics. We also compute and print the in-sample calibrated LLL to sanity-check the sigma choice (no change to submission format/order).'
- What this solution (achieved -8.74247) has done: 'Your current ridge per-patient intercept+slope core is fine; the main reason you’re far from the target is that this linear fit is systematically biased at the “final 3 visits” horizon, and confidence-only calibration can’t recover that. With minimal change to the same core logic, I add a single global “week anchor” centering (fit/predict on Weeks centered at each patient’s last observed train week), which preserves the same model form but reduces extrapolation error on last-3 visits. I keep your exact Laplace-metric sigma calibration on last-3 rows, but recompute it after the centering so Confidence remains metric-aligned. Submission row alignment and format remain identical and it still write `submission.csv`.'
- What this solution (achieved -24.65932) has done: 'Your current score (−8.742) is worse than the target (−6.935), so we should *increase* it with the smallest change that doesn’t alter your ridge per-patient intercept/slope core. The main remaining issue is that you’re predicting for **all weeks** but only **three weeks per patient** are scored; calibrating `Confidence` using residuals from many non-scored weeks miscalibrates sigma and hurts the Laplace LLL. I keep your model and FVC predictions identical, but (1) build the exact scored rows from `sample_submission.csv`, (2) calibrate a single global `sigma_hat` by maximizing the exact Kaggle metric **on those scored weeks** (train patients only), and (3) submit in exact sample order to avoid any alignment issues.'
- What this solution (achieved -24.65932) has done: 'I fix the baseline-week mapping construction, which currently returns a 2D object and breaks `Series.map(...).astype(float)` in both the centering step and the submission-template step. Then I ensure `WeeksCentered` is always created for both `train` and `sub` before `model_fit()` and before building `template_scored`, resolving the downstream `KeyError: 'WeeksCentered'`. These changes preserve your ridge per-patient intercept/slope core logic and keep your confidence calibration approach intact, but unblock end-to-end execution. Finally, I keep the submission strictly aligned to `sample_submission.csv` and write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge

DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"

print("Available input files (first ~30):")
n_print = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if n_print < 30:
            print(os.path.join(dirname, filename))
            n_print += 1
        else:
            break



## === cell 1
train_raw = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

train_raw = train_raw.copy()
train_raw.drop(
    train_raw[train_raw.Patient == "ID00197637202246865691526"].index, inplace=True
)

le_id = LabelEncoder()
le_id.fit(pd.concat([train_raw["Patient"], test["Patient"]], axis=0).values)

train = train_raw.copy()
train["PatientID"] = le_id.transform(train["Patient"])
test = test.copy()
test["PatientID"] = le_id.transform(test["Patient"])

N_PATIENTS_GLOBAL = int(len(le_id.classes_))

train.head()



## === cell 2
_all_pt_weeks = pd.concat(
    [train[["Patient", "Weeks"]], test[["Patient", "Weeks"]]], axis=0, ignore_index=True
).copy()
_all_pt_weeks["Weeks"] = _all_pt_weeks["Weeks"].astype(np.float64)


def _closest_to_zero_week(g: pd.DataFrame) -> float:
    w = g["Weeks"].values
    return float(w[np.argmin(np.abs(w - 0.0))])


_baseline_week_by_patient = (
    _all_pt_weeks.groupby("Patient", sort=False, as_index=True)
    .apply(_closest_to_zero_week)
    .to_dict()
)


def _add_week_anchor_and_center(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["WeekAnchor"] = out["Patient"].map(_baseline_week_by_patient)
    out["WeekAnchor"] = out["WeekAnchor"].astype(np.float64).fillna(0.0)
    out["WeeksCentered"] = out["Weeks"].astype(np.float64) - out["WeekAnchor"]
    return out


train = _add_week_anchor_and_center(train)
test = _add_week_anchor_and_center(test)

train[["Patient", "Weeks", "WeekAnchor", "WeeksCentered"]].head()




## === cell 3
def _build_design_matrix(weeks: np.ndarray, patient_ids: np.ndarray, n_patients: int):
    """Create X = [I(patient), I(patient)*weeks] so each patient has its own intercept and slope.

    Bug fix: n_patients must be >= patient_ids.max()+1. We also validate indices to avoid silent misalignment.
    """
    weeks = np.asarray(weeks, dtype=np.float64)
    patient_ids = np.asarray(patient_ids, dtype=int)

    if patient_ids.size == 0:
        return np.zeros((0, 2 * n_patients), dtype=np.float64)

    pid_min = int(patient_ids.min())
    pid_max = int(patient_ids.max())
    if pid_min < 0 or pid_max >= n_patients:
        raise IndexError(
            f"PatientID out of bounds for design matrix: min={pid_min}, max={pid_max}, "
            f"but n_patients={n_patients}. Ensure n_patients is global (len(le_id.classes_))."
        )

    n = len(weeks)
    X = np.zeros((n, 2 * n_patients), dtype=np.float64)
    rows = np.arange(n)
    X[rows, patient_ids] = 1.0
    X[rows, n_patients + patient_ids] = weeks
    return X


def model_fit(data, examine=True):
    n_patients = int(len(le_id.classes_))
    y = data["FVC"].values.astype(np.float64)

    weeks = data["WeeksCentered"].values.astype(np.float64)
    pids = data["PatientID"].values.astype(int)

    X = _build_design_matrix(weeks, pids, n_patients)

    model = Ridge(alpha=1.0, fit_intercept=False, random_state=0)
    model.fit(X, y)

    y_hat = model.predict(X)
    resid = y - y_hat
    sigma = float(np.sqrt(np.mean(resid**2)))

    trace = {
        "sigma": sigma,
        "n_patients": n_patients,
    }  # lightweight stand-in for downstream usage

    if examine:
        print(f"Fitted ridge model with n_patients={n_patients}, sigma≈{sigma:.2f}")

    return model, trace




## === cell 4
def generate_template(data):
    pred_template = []
    for patient in data["Patient"].unique():
        df = pd.DataFrame(columns=["PatientID", "Weeks", "Patient"])
        df["Weeks"] = np.arange(-12, 134)
        df["Patient"] = patient
        pred_template.append(df)
    pred_template = pd.concat(pred_template, ignore_index=True)
    pred_template["PatientID"] = le_id.transform(pred_template["Patient"])

    pred_template["WeekAnchor"] = pred_template["Patient"].map(
        _baseline_week_by_patient
    )
    pred_template["WeekAnchor"] = (
        pred_template["WeekAnchor"].astype(np.float64).fillna(0.0)
    )
    pred_template["WeeksCentered"] = (
        pred_template["Weeks"].astype(np.float64) - pred_template["WeekAnchor"]
    )
    return pred_template




## === cell 5
def model_predict(model, trace, template):
    n_patients = int(trace["n_patients"])

    if "WeeksCentered" in template.columns:
        weeks = template["WeeksCentered"].values.astype(np.float64)
    else:
        weeks = template["Weeks"].values.astype(np.float64)

    pids = template["PatientID"].values.astype(int)

    X = _build_design_matrix(weeks, pids, n_patients)
    pred = model.predict(X)

    df = pd.DataFrame(
        {
            "Patient": le_id.inverse_transform(pids),
            "Weeks": template["Weeks"].values,
            "FVC_pred": pred,
        }
    )

    df["sigma"] = float(trace["sigma"])

    df["FVC_inf"] = df["FVC_pred"] - df["sigma"]
    df["FVC_sup"] = df["FVC_pred"] + df["sigma"]

    if "FVC" in train.columns:
        df = pd.merge(
            df, train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
        )
        df = df.rename(columns={"FVC": "FVC_true"})
    return df




## === cell 6
def examine_predictions(data):
    f, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, patient in enumerate(
        np.random.choice(data["Patient"].unique(), size=3, replace=False)
    ):
        ax = axes[i]
        df = data[data["Patient"] == patient].sort_values("Weeks")
        x = df["Weeks"]
        ax.set_title(patient)
        if "FVC_true" in df.columns:
            ax.plot(x, df["FVC_true"], "o")
        ax.plot(x, df["FVC_pred"])
        ax.fill_between(x, df["FVC_inf"], df["FVC_sup"], alpha=0.3, color="#ffcd3c")
        ax.set_ylabel("FVC")




## === cell 7
def evaluate_predictions(df, use_only_last_3_measures=True, examine=False):
    if "FVC_true" not in df.columns:
        raise ValueError(
            "evaluate_predictions requires FVC_true column (merge predictions with train first)."
        )

    if use_only_last_3_measures:
        y = df.dropna(subset=["FVC_true"]).groupby("Patient").tail(3).copy()
    else:
        y = df.dropna(subset=["FVC_true"]).copy()

    sigma_c = y["sigma"].values.astype(np.float64)
    sigma_c[sigma_c < 70] = 70.0
    delta = (y["FVC_pred"] - y["FVC_true"]).abs().values.astype(np.float64)
    delta[delta > 1000] = 1000.0
    lll = -np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

    y["sigma_c"] = sigma_c
    y["delta_c"] = delta
    y["main_loss"] = y["delta_c"] / y["sigma_c"]
    if examine:
        sns.histplot(y["main_loss"], bins=100)
        plt.show()

    return float(np.mean(lll))




## === cell 8
def evaluation_cycle(train_df, valid_df=None, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model, trace = model_fit(train_df, examine=examine_trace)
    print("")

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train_df)
    pred_train = model_predict(model, trace, template_train)
    if examine_preds:
        examine_predictions(pred_train)
    lll_train = evaluate_predictions(pred_train)

    pred_valid, lll_valid = None, None
    if valid_df is not None:
        print("Examine true vs predictions for validation data ...")
        template_valid = generate_template(valid_df)
        pred_valid = model_predict(model, trace, template_valid)
        if examine_preds:
            examine_predictions(pred_valid)
        lll_valid = evaluate_predictions(pred_valid)

    return pred_train, pred_valid, lll_train, lll_valid




## === cell 9
sub = sample_sub.copy()
sub[["Patient", "Weeks"]] = sub["Patient_Week"].str.split("_", expand=True)
sub["Weeks"] = sub["Weeks"].astype(int)
sub["PatientID"] = le_id.transform(sub["Patient"])

sub["WeekAnchor"] = sub["Patient"].map(_baseline_week_by_patient)
sub["WeekAnchor"] = sub["WeekAnchor"].astype(np.float64).fillna(0.0)
sub["WeeksCentered"] = sub["Weeks"].astype(np.float64) - sub["WeekAnchor"]

sub.head()



## === cell 10
print("Fit model on full training data ...")
model, trace = model_fit(train, examine=True)

template_scored = sub[["PatientID", "Weeks", "Patient", "WeeksCentered"]].copy()
pred_scored_trainmerge = model_predict(model, trace, template_scored)

y_scored = pred_scored_trainmerge.dropna(subset=["FVC_true"]).copy()


def _mean_lll_row_avg_for_sigma(sigmas: np.ndarray, y_df: pd.DataFrame) -> np.ndarray:
    sigmas = np.asarray(sigmas, dtype=np.float64)

    abs_err_vec = (y_df["FVC_pred"] - y_df["FVC_true"]).abs().values.astype(np.float64)
    abs_err_vec = np.clip(abs_err_vec, 0.0, 1000.0)

    sigma_c = np.maximum(sigmas, 70.0)
    lll = -np.sqrt(2.0) * (abs_err_vec[None, :] / sigma_c[:, None]) - np.log(
        np.sqrt(2.0) * sigma_c[:, None]
    )
    return lll.mean(axis=1)


def _optimize_sigma_lll_row_avg(y_df: pd.DataFrame) -> float:
    if y_df.shape[0] == 0:
        return 70.0

    abs_err_vec = (y_df["FVC_pred"] - y_df["FVC_true"]).abs().values.astype(np.float64)
    abs_err_vec = np.clip(abs_err_vec, 0.0, 1000.0)
    if not np.all(np.isfinite(abs_err_vec)):
        return 70.0

    med = float(np.median(abs_err_vec))
    s_center = float(max(70.0, np.sqrt(2.0) * med))

    lo = 70.0
    hi = float(min(1500.0, max(200.0, 3.0 * s_center)))

    grid1 = np.linspace(lo, hi, 801, dtype=np.float64)
    lll1 = _mean_lll_row_avg_for_sigma(grid1, y_df)
    s1 = float(grid1[int(np.argmax(lll1))])

    span = max(30.0, 0.10 * s1)
    lo2 = float(max(70.0, s1 - span))
    hi2 = float(min(1500.0, s1 + span))
    grid2 = np.linspace(lo2, hi2, 1201, dtype=np.float64)
    lll2 = _mean_lll_row_avg_for_sigma(grid2, y_df)
    s2 = float(grid2[int(np.argmax(lll2))])

    return s2


sigma_hat = float(_optimize_sigma_lll_row_avg(y_scored))
trace["sigma"] = sigma_hat  # keep downstream consistent

train_lll_scored_cal = float(
    _mean_lll_row_avg_for_sigma(np.array([sigma_hat]), y_scored)[0]
)
print(
    f"Calibrated global confidence sigma_hat={sigma_hat:.2f} "
    f"(optimized exact row-avg Laplace LLL on scored-week rows where train labels exist). "
    f"In-sample scored-week LLL: {train_lll_scored_cal:.5f}"
)

pred_sub = model_predict(model, trace, template_scored)

sub_out = pd.DataFrame(
    {
        "Patient_Week": sub["Patient_Week"].values,
        "FVC": pred_sub["FVC_pred"].values.astype(np.float64),
        "Confidence": np.full(len(sub), sigma_hat, dtype=np.float64),
    }
)

sub_out = (
    sub_out.set_index("Patient_Week")
    .reindex(sample_sub["Patient_Week"].values)
    .reset_index()
)

if sub_out["FVC"].isna().any() or sub_out["Confidence"].isna().any():
    sub_out["FVC"] = sub_out["FVC"].fillna(sample_sub["FVC"].astype(np.float64))
    sub_out["Confidence"] = sub_out["Confidence"].fillna(
        sample_sub["Confidence"].astype(np.float64)
    )

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
sub_out.head()
