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

-6.8603

# 6. Current score

-8.6651

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.57933) has done: 'The crash comes from a mismatch between how many unique `PatientID`s exist in the training dataframe and the actual numeric range of `PatientID` values (IDs were encoded on train+test, so they’re not guaranteed to be 0..n_train-1). I fix this by building the design matrix using a compact per-training-patient index mapping, while keeping the same ridge-per-patient intercept/slope core logic. Then I update prediction to use the same mapping for patients seen in training, and safely fall back to baseline FVC plus a global mean slope for any unseen patients (shouldn’t happen here, but prevents future runtime errors). Finally, I keep the submission formatting identical and ensure `submission.csv` is always written with required columns.'
- What this solution (achieved -8.7699) has done: 'Your current approach is already structurally aligned with the competition (patient-specific intercept+slope with Ridge), but the biggest score gap typically comes from mis-calibrated uncertainty and from learning patient slopes on absolute weeks instead of *relative-to-baseline* weeks. I keep the exact same Ridge-per-patient-intercept/slope core logic, but (1) train/predict using `Weeks - Weeks_base` so the intercept corresponds to each patient’s baseline FVC level more cleanly, and (2) calibrate `Confidence` using a robust per-class residual scale (MAD-based) with a single global multiplier tuned to be closer to the Laplace optimum (instead of the current fixed `*1.25`). These are minimal, metric-aligned changes that usually improve Laplace log-likelihood without changing the model family or training loop. The submission format and file name stay identical (`submission.csv`).'
- What this solution (achieved -8.68403) has done: 'I make two minimal, metric-aligned adjustments that keep your exact “one global Ridge with per-patient intercept+slope (fit_intercept=False) + class-based sigma” core logic intact. First, I fix baseline-week alignment for test patients by deriving `Weeks_base`/`FVC_base` from `test` (not `train`) when computing `weeks_rel` at prediction time; the current code incorrectly uses train baselines for test IDs, which hurts FVC and uncertainty calibration. Second, I compute `sigma_multiplier` once from training residuals via a tiny grid search that maximizes the competition Laplace score on the training residuals (no early stopping, no sampling), replacing the current hard-coded `0.95` so confidence is better calibrated. These changes should improve the score (move it upward toward -6.8603) without altering the modeling approach or submission format, and still writes `submission.csv`.'
- What this solution (achieved -8.82826) has done: 'I fix the OOF calibration crash by ensuring the validation predictions retain the `Class` column (it was being dropped inside `model_predict`), and by making `model_predict` robust to templates that don’t contain baseline columns by pulling baselines from the appropriate global train/test tables. Then I make sigma-multiplier calibration deterministic and usable in the final fit so `sigma_multiplier_oof` is always defined. Finally, I ensure the submission join is complete and always writes `submission.csv` with the exact required columns and no missing values.'
- What this solution (achieved -8.6031) has done: 'Your current ridge-per-patient intercept+slope logic is fine; the biggest remaining gap to the target score is usually from (a) training on *all weeks*, even though Kaggle scores only the last 3 visits, and (b) confidence calibration being optimized on a mismatched distribution. I keep the same model family and prediction pipeline, but train the ridge and calibrate the sigma multiplier using only each patient’s last 3 measurements (the exact scoring subset), which typically increases the Laplace score and should move you toward -6.8603 with minimal risk. I also make one small, metric-aligned post-processing change: clip the output Confidence to be at least 70 (matching the metric), to avoid accidental underconfidence/overconfidence artifacts. Submission formatting and paths remain identical, and the code still writes `submission.csv` end-to-end.'
- What this solution (achieved -8.59713) has done: 'Your current ridge-per-patient intercept+slope pipeline is already stable, so the smallest score-moving change is to align training and sigma calibration with what Kaggle actually scores: the *final 3 visits per patient*. Right now you train/calibrate only on the last-3 subset, but you still use a fixed alpha=10 and a relatively narrow sigma-multiplier search range; both can be slightly mis-calibrated for that subset and hurt the Laplace log-likelihood. I keep the exact same model family and prediction semantics, but (1) tune Ridge `alpha` using OOF Laplace score on the last-3 subset (tiny grid), and (2) widen the sigma-multiplier grid a bit and calibrate it after selecting `alpha`, then fit once on all last-3 and write `submission.csv` exactly as before. These are minimal, metric-aligned calibration changes and should move your score upward toward the target without changing the core approach.'
- What this solution (achieved -8.59652) has done: 'I keep your exact ridge “per-patient intercept+slope” core model and the last-3-only training choice, but improve score toward the target by making two minimal, metric-aligned calibration changes. First, I tune `Ridge(alpha)` on a slightly wider (but still tiny) grid using the same OOF Laplace computation you already use, because the current coarse grid can leave you under-regularized/over-regularized on the last-3 subset. Second, I calibrate `sigma_multiplier` on a slightly wider range and also add a single global additive sigma term (in quadrature) fitted by OOF to better match the Laplace likelihood’s preference for a nonzero irreducible noise floor; this preserves your class-based sigma structure and only adjusts confidence calibration. The submission schema, filename, and all feature logic remain unchanged.'
- What this solution (achieved -8.59544) has done: 'I keep your exact “global Ridge with per-patient intercept+slope (fit_intercept=False) trained on last-3 visits” approach, and only make two score-aligned adjustments that tend to raise the Laplace log-likelihood without changing the model family. First, I select the Ridge `alpha` by OOF Laplace score using the *final evaluated pipeline* (including the OOF-calibrated sigma multiplier/additive term), because tuning alpha using only `sigma_by_class` can pick an alpha that’s suboptimal once confidence calibration is applied. Second, I calibrate confidence on a slightly richer but still tiny candidate set (a few more `sigma_add` values) while keeping the same functional form `sqrt((sigma_base*m)^2 + add^2)`; this often improves the likelihood more than further model changes. Submission writing stays identical (`submission.csv` with correct columns/order), and all paths remain unchanged.'
- What this solution (achieved -8.59544) has done: 'You’re currently well below the target (−8.59544 vs −6.8603), so we should cautiously improve score without changing the core “global Ridge with per-patient intercept+slope on last-3 visits + class-based sigma” logic. The smallest high-impact fix is to stop generating predictions for all weeks (−12..133) and instead predict exactly the `Patient_Week` rows that Kaggle evaluates (the ones in `sample_submission.csv`), which avoids any join/misalignment edge cases and ensures each required row is predicted directly. Second, we keep your existing OOF calibration approach but make it consistent with submission-time prediction by always using the same template structure (patient/week/class only) and avoiding any dependence on template weeks being int ranges. This preserves evaluation semantics while typically improving the Laplace score by preventing subtle week/base mismatches and missing-row fallbacks.'
- What this solution (achieved -8.59544) has done: 'I keep your exact “global Ridge with per-patient intercept+slope on last-3 visits + class-based sigma with (mult, add) calibration” core pipeline, but fix a subtle mismatch that can hurt score: your training baselines (`Weeks_base`, `FVC_base`) are currently computed from each patient’s *minimum week*, whereas Kaggle’s test baseline is the provided measurement at `Weeks` in `test.csv` (typically week 0). I change baseline computation to use each patient’s `Weeks==0` row when available (fallback to min-week only if week 0 is missing), and use the same rule consistently for both train and test. This preserves the same model family and training loop, but aligns the “weeks_rel” feature and intercept meaning between train and test, which typically improves FVC accuracy and confidence calibration toward your target score. Submission generation remains identical and still writes `submission.csv` with the required columns.'
- What this solution (achieved -8.6651) has done: 'We keep your exact Ridge per-patient intercept+slope core model and last-3 training setup, but fix a metric-relevant mismatch: you currently calibrate sigma only on the last-3 rows, while Kaggle evaluates weeks that are later than the provided baseline (often far beyond those last-3). I minimally change OOF calibration to use a “submission-like” set of weeks for each validation patient (the same Patient_Week rows as in `sample_submission.csv`) while still using only training labels where available (no leakage), so confidence is calibrated on the correct week distribution. Then we keep your alpha tuning loop intact but make it use this submission-like calibration/evaluation, which typically raises Laplace log-likelihood toward your target without changing the model family. Submission generation stays identical and still writes `submission.csv` with the required columns.'
- What this solution (achieved -8.6651) has done: 'I keep your exact Ridge per-patient intercept+slope model and the “train on last-3 visits” setup, but make two minimal, metric-aligned fixes to move the score upward toward the target. First, I stop using `reference_df=df_va` when building the “submission-like” validation template: it can drop many validation patients because `df_va` often doesn’t contain `PatientID/Class` consistently for all patients, which weakens OOF calibration; instead I use stable metadata from the full `train` table. Second, I cache baseline maps once (instead of rebuilding them inside every `model_predict` call) and ensure they always come from the correct split (train vs test), which reduces subtle inconsistencies in weeks_rel computation during OOF calibration and final prediction. These changes do not alter the model family, features, or loss; they only make the existing calibration/evaluation pipeline consistent with the intended submission rows and should improve Laplace log-likelihood.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")



## === cell 1
train = pd.read_csv(TRAIN_CSV)
train_raw = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

train.drop(train[train.Patient == "ID00197637202246865691526"].index, inplace=True)

combined = pd.concat([train, test], axis=0, ignore_index=True).drop_duplicates()

le_id = LabelEncoder()
combined["PatientID"] = le_id.fit_transform(combined["Patient"])

pid_map = combined[["Patient", "PatientID"]].drop_duplicates()
train = train.merge(pid_map, on="Patient", how="left")
test = test.merge(pid_map, on="Patient", how="left")

train.head()




## === cell 2
def add_baselines(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    base0 = data.loc[data["Weeks"] == 0, ["Patient", "Weeks", "FVC"]].copy()
    base0 = base0.groupby("Patient", as_index=False).mean(numeric_only=True)

    minw = data[["Patient", "Weeks"]].groupby("Patient", as_index=False).min()
    minw = minw.merge(
        data[["Patient", "Weeks", "FVC"]],
        how="left",
        on=["Patient", "Weeks"],
    )
    minw = minw.groupby("Patient", as_index=False).mean(numeric_only=True)

    aux = minw.merge(
        base0.rename(columns={"Weeks": "Weeks0", "FVC": "FVC0"}),
        on="Patient",
        how="left",
    )
    aux["Weeks_base"] = np.where(aux["Weeks0"].notna(), aux["Weeks0"], aux["Weeks"])
    aux["FVC_base"] = np.where(aux["FVC0"].notna(), aux["FVC0"], aux["FVC"])
    aux = aux[["Patient", "Weeks_base", "FVC_base"]].copy()

    aux["Weeks_base"] = aux["Weeks_base"].astype(int)
    aux["FVC_base"] = aux["FVC_base"].astype(int)

    data = data.merge(aux, how="left", on="Patient")
    return data


train = add_baselines(train)
test = add_baselines(test)
train.head()




## === cell 3
def patient_class(row):
    if row["Sex"] == "Male":
        if row["SmokingStatus"] == "Currently smokes":
            return 0
        elif row["SmokingStatus"] == "Ex-smoker":
            return 1
        elif row["SmokingStatus"] == "Never smoked":
            return 2
    else:
        if row["SmokingStatus"] == "Currently smokes":
            return 3
        elif row["SmokingStatus"] == "Ex-smoker":
            return 4
        elif row["SmokingStatus"] == "Never smoked":
            return 5
    return 0


train["Class"] = train.apply(patient_class, axis=1)
test["Class"] = test.apply(patient_class, axis=1)

test.head()



## === cell 4
PatientID = train["Patient"].values
fvc_b = train.groupby("Patient").first(numeric_only=False)["FVC_base"]
fvc_b.values




## === cell 5
def _robust_sigma_mad(resid: np.ndarray) -> float:
    """Robust std estimate via MAD; returns a sigma-like scale in ml."""
    resid = np.asarray(resid, dtype=np.float64)
    if resid.size == 0:
        return np.nan
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    return 1.4826 * mad


def _laplace_lll(delta: np.ndarray, sigma: np.ndarray) -> float:
    """Competition metric averaged; sigma is clipped at 70, delta at 1000."""
    delta = np.asarray(delta, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_c = np.maximum(sigma, 70.0)
    d = np.minimum(np.abs(delta), 1000.0)
    lll = -np.sqrt(2.0) * d / sigma_c - np.log(np.sqrt(2.0) * sigma_c)
    return float(np.mean(lll))


def model_fit(data, examine=True, alpha: float = 10.0):
    required = ["FVC", "Weeks", "PatientID", "Class", "Weeks_base"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    unique_pids = np.sort(data["PatientID"].astype(int).unique())
    pid_to_ix = {int(pid): i for i, pid in enumerate(unique_pids)}
    n_patients = int(len(unique_pids))

    pid_raw = data["PatientID"].astype(int).values
    pid_ix = np.array([pid_to_ix[int(p)] for p in pid_raw], dtype=np.int32)

    weeks_rel = (
        data["Weeks"].astype(float).values - data["Weeks_base"].astype(float).values
    )
    y = data["FVC"].astype(float).values

    X = np.zeros((len(data), 2 * n_patients), dtype=np.float32)
    X[np.arange(len(data)), pid_ix] = 1.0
    X[np.arange(len(data)), n_patients + pid_ix] = weeks_rel

    ridge = Ridge(alpha=float(alpha), fit_intercept=False, random_state=RANDOM_STATE)
    ridge.fit(X, y)

    coef = ridge.coef_.astype(np.float64)
    a = coef[:n_patients]  # patient intercepts at baseline after weeks_rel transform
    b = coef[n_patients:]  # patient slopes (per week)

    y_hat = a[pid_ix] + b[pid_ix] * weeks_rel
    resid = y - y_hat

    sigma_by_class = np.zeros(6, dtype=np.float64)
    global_sigma = float(_robust_sigma_mad(resid))
    if not np.isfinite(global_sigma) or global_sigma <= 1e-6:
        s2 = float(np.std(resid))
        global_sigma = s2 if s2 > 1e-6 else 150.0

    cls_arr = data["Class"].astype(int).values
    for c in range(6):
        rc = resid[cls_arr == c]
        s = float(_robust_sigma_mad(rc))
        if (not np.isfinite(s)) or (len(rc) < 5) or (s <= 1e-6):
            s = global_sigma
        sigma_by_class[c] = max(s, 70.0)  # metric clips at 70 anyway

    global_slope = float(np.mean(b)) if len(b) else 0.0

    sigma_multiplier = 1.0

    model = {
        "n_patients": n_patients,
        "a": a,
        "b": b,
        "sigma_by_class": sigma_by_class,
        "pid_to_ix": pid_to_ix,
        "global_slope": global_slope,
        "sigma_multiplier": sigma_multiplier,
        "alpha": float(alpha),
        "sigma_add": 0.0,
    }
    trace = None  # placeholder to preserve call signature

    if examine:
        plt.figure(figsize=(6, 3))
        sns.histplot(resid, bins=50, kde=False)
        plt.title("Training residual distribution (FVC_true - FVC_pred)")
        plt.xlabel("Residual (ml)")
        plt.tight_layout()

    return model, trace




## === cell 6
def generate_template(data):
    pred_template = []
    for patient in data["Patient"].unique():
        df = pd.DataFrame(columns=["PatientID", "Weeks", "Patient", "Class"])
        df["Weeks"] = np.arange(-12, 134)
        df["Patient"] = patient
        df["Class"] = int(data.loc[data["Patient"] == patient, "Class"].max())
        pred_template.append(df)
    pred_template = pd.concat(pred_template, ignore_index=True)
    pred_template["PatientID"] = le_id.transform(pred_template["Patient"])
    return pred_template




## === cell 7
template_train_test = generate_template(test)
template_train_test.head()



## === cell 8
BASE_WEEKS_MAP_TRAIN = (
    train.groupby("Patient").first(numeric_only=False)["Weeks_base"].to_dict()
)
BASE_FVC_MAP_TRAIN = (
    train.groupby("Patient").first(numeric_only=False)["FVC_base"].to_dict()
)
BASE_WEEKS_MAP_TEST = (
    test.groupby("Patient").first(numeric_only=False)["Weeks_base"].to_dict()
)
BASE_FVC_MAP_TEST = (
    test.groupby("Patient").first(numeric_only=False)["FVC_base"].to_dict()
)

GLOBAL_MEAN_WEEKS_BASE = float(np.nanmean(train["Weeks_base"].astype(float).values))
GLOBAL_MEAN_FVC_BASE = float(np.nanmean(train["FVC_base"].astype(float).values))




## === cell 9
def model_predict(model, trace, template):
    pid_raw = template["PatientID"].astype(int).values
    weeks = template["Weeks"].astype(float).values
    cls = template["Class"].astype(int).values

    a = model["a"]
    b = model["b"]
    sigma_by_class = model["sigma_by_class"]
    pid_to_ix = model["pid_to_ix"]
    global_slope = model["global_slope"]
    sigma_multiplier = float(model.get("sigma_multiplier", 1.0))
    sigma_add = float(model.get("sigma_add", 0.0))

    pid_ix = np.full(len(pid_raw), -1, dtype=np.int32)
    for i, pr in enumerate(pid_raw):
        pid_ix[i] = pid_to_ix.get(int(pr), -1)

    seen = pid_ix >= 0

    patients = le_id.inverse_transform(pid_raw)

    base_weeks = np.array(
        [
            BASE_WEEKS_MAP_TEST.get(p, BASE_WEEKS_MAP_TRAIN.get(p, np.nan))
            for p in patients
        ],
        dtype=np.float64,
    )
    nan_bw = np.isnan(base_weeks)
    if np.any(nan_bw):
        base_weeks[nan_bw] = GLOBAL_MEAN_WEEKS_BASE

    weeks_rel = weeks - base_weeks

    fvc_pred = np.empty(len(pid_raw), dtype=np.float64)
    if np.any(seen):
        fvc_pred[seen] = a[pid_ix[seen]] + b[pid_ix[seen]] * weeks_rel[seen]

    if np.any(~seen):
        base_vals = np.array(
            [
                BASE_FVC_MAP_TEST.get(p, BASE_FVC_MAP_TRAIN.get(p, np.nan))
                for p in patients
            ],
            dtype=np.float64,
        )
        nan_base = np.isnan(base_vals)
        if np.any(nan_base):
            base_vals[nan_base] = GLOBAL_MEAN_FVC_BASE
        fvc_pred[~seen] = base_vals[~seen] + global_slope * weeks_rel[~seen]

    sigma_base = sigma_by_class[np.clip(cls, 0, 5)]
    sigma = np.sqrt((sigma_base * sigma_multiplier) ** 2 + (sigma_add**2))
    sigma = np.maximum(sigma, 70.0)

    df = pd.DataFrame(
        {
            "Patient": patients,
            "Weeks": template["Weeks"].astype(int).values,
            "Class": cls.astype(int),  # keep for calibration/evaluation
            "FVC_pred": fvc_pred,
            "sigma": sigma,
        }
    )
    df["FVC_inf"] = df["FVC_pred"] - df["sigma"]
    df["FVC_sup"] = df["FVC_pred"] + df["sigma"]

    df = pd.merge(
        df, train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
    )
    df = df.rename(columns={"FVC": "FVC_true"})
    return df




## === cell 10
def examine_predictions(data):
    n = (data["Patient"].nunique()) + 1
    f, axes = plt.subplots((n // 3) + 1, 3, figsize=(15, 5 * ((n // 3) + 1)))
    for i, patient in enumerate(data["Patient"].unique()):
        ax = axes[i // 3, i % 3]
        df = data[data["Patient"] == patient].sort_values("Weeks")
        x = df["Weeks"]
        ax.set_title(patient)
        ax.plot(x, df["FVC_true"], "o")
        ax.plot(x, df["FVC_pred"])
        sns.regplot(
            x=df["FVC_true"],
            y=df["FVC_pred"],
            ax=ax,
            ci=None,
            line_kws={"color": "red"},
        )
        ax.fill_between(x, df["FVC_inf"], df["FVC_sup"], alpha=0.5, color="#ffcd3c")
        ax.set_ylabel("FVC")
    axes[(n // 3), (n % 3)].axis("off")
    plt.tight_layout()




## === cell 11
def evaluate_predictions(df, use_only_last_3_measures=False, examine=False):
    if use_only_last_3_measures:
        y = df.dropna(subset=["FVC_true"]).groupby("Patient").tail(3)
    else:
        y = df.dropna(subset=["FVC_true"])

    sigma_c = y["sigma"].values.copy()
    sigma_c[sigma_c < 70] = 70
    delta = (y["FVC_pred"] - y["FVC_true"]).abs().values
    delta[delta > 1000] = 1000
    lll = -np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

    if examine:
        main_loss = delta / sigma_c
        plt.figure(figsize=(6, 3))
        plt.hist(main_loss, bins=100)
        plt.title("delta/sigma clipped")
        plt.tight_layout()

    return float(np.mean(lll))




## === cell 12
def evaluation_cycle(train_df, valid_df, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model, trace = model_fit(train_df, examine=examine_trace)

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train_df)
    pred_train = model_predict(model, trace, template_train)
    if examine_preds:
        examine_predictions(pred_train)
    lll_train = evaluate_predictions(pred_train)

    pred_valid, lll_valid = None, None
    if valid_df is not None and len(valid_df) > 0:
        print("Examine true vs predictions for validation data ...")
        template_valid = generate_template(valid_df)
        pred_valid = model_predict(model, trace, template_valid)
        if examine_preds:
            examine_predictions(pred_valid)
        lll_valid = evaluate_predictions(pred_valid)

    return pred_train, pred_valid, lll_train, lll_valid




## === cell 13
for fold in range(0):
    examine_lll = True

    all_patients = train["Patient"].unique()
    validation_patients = np.random.choice(all_patients, size=20, replace=False)
    df_valid = train[train["Patient"].isin(validation_patients)]
    df_train = train[~train["Patient"].isin(validation_patients)]

    df_valid_first_readings = df_valid.groupby("Patient").head(1)
    df_train = pd.concat([df_train, df_valid_first_readings], axis=0, ignore_index=True)

    print(f"Fold: {fold}")
    pred_train, pred_valid, lll_train, lll_valid = evaluation_cycle(
        df_train, df_valid, examine_trace=True, examine_preds=True
    )

    print(f"Laplace Log Likelihoods for fold: {fold}")
    print(f"Training:     {lll_train:.4f}")
    print(f"Validation:   {lll_valid:.4f}")
    print("")

    evaluate_predictions(pred_train, use_only_last_3_measures=True, examine=examine_lll)
    evaluate_predictions(pred_valid, use_only_last_3_measures=True, examine=examine_lll)



## === cell 14
train_last3 = (
    train.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .tail(3)
    .copy()
)

sample_sub = pd.read_csv(SAMPLE_SUB)
_sample_pw = sample_sub["Patient_Week"].str.rsplit("_", n=1, expand=True)
SAMPLE_WEEKS_BY_PATIENT = (
    pd.DataFrame(
        {"Patient": _sample_pw[0].values, "Weeks": _sample_pw[1].astype(int).values}
    )
    .groupby("Patient")["Weeks"]
    .apply(lambda x: np.sort(x.unique()))
    .to_dict()
)

TRAIN_META = (
    train[["Patient", "PatientID", "Class"]]
    .drop_duplicates("Patient")
    .set_index("Patient")
)


def _build_submission_like_template_for_patients(
    patients: np.ndarray, reference_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    meta = (
        reference_df[["Patient", "PatientID", "Class"]]
        .drop_duplicates("Patient")
        .set_index("Patient")
    )
    for p in patients:
        if p not in SAMPLE_WEEKS_BY_PATIENT:
            continue
        weeks = SAMPLE_WEEKS_BY_PATIENT[p]
        if p not in meta.index:
            continue
        pid = int(meta.loc[p, "PatientID"])
        cls = int(meta.loc[p, "Class"])
        rows.append(
            pd.DataFrame(
                {
                    "Patient": p,
                    "PatientID": pid,
                    "Class": cls,
                    "Weeks": weeks.astype(int),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["Patient", "PatientID", "Weeks", "Class"])
    return pd.concat(rows, ignore_index=True)


def _build_submission_like_template_for_patients_stable_meta(
    patients: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for p in patients:
        if p not in SAMPLE_WEEKS_BY_PATIENT:
            continue
        if p not in TRAIN_META.index:
            continue
        weeks = SAMPLE_WEEKS_BY_PATIENT[p]
        pid = int(TRAIN_META.loc[p, "PatientID"])
        cls = int(TRAIN_META.loc[p, "Class"])
        rows.append(
            pd.DataFrame(
                {
                    "Patient": p,
                    "PatientID": pid,
                    "Class": cls,
                    "Weeks": weeks.astype(int),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["Patient", "PatientID", "Weeks", "Class"])
    return pd.concat(rows, ignore_index=True)


def calibrate_sigma_oof(
    data: pd.DataFrame, n_splits: int = 5, alpha: float = 10.0
) -> tuple[float, float]:
    patients = np.array(sorted(data["Patient"].unique()))
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(patients)
    folds = np.array_split(patients, n_splits)

    all_delta = []
    all_sigma_base = []

    for k in range(n_splits):
        valid_patients = np.array(sorted(folds[k].tolist()))
        df_tr = data[~data["Patient"].isin(valid_patients)].copy()

        m, trc = model_fit(df_tr, examine=False, alpha=float(alpha))

        tmp = _build_submission_like_template_for_patients_stable_meta(valid_patients)
        if len(tmp) == 0:
            continue

        pred_va = model_predict(m, trc, tmp)
        y = pred_va.dropna(subset=["FVC_true"]).copy()
        if len(y) == 0:
            continue

        delta = (y["FVC_pred"].values - y["FVC_true"].values).astype(np.float64)
        sigma_base = m["sigma_by_class"][
            np.clip(y["Class"].astype(int).values, 0, 5)
        ].astype(np.float64)

        all_delta.append(delta)
        all_sigma_base.append(sigma_base)

    if len(all_delta) == 0:
        return 1.0, 0.0

    delta = np.concatenate(all_delta, axis=0)
    sigma_base = np.concatenate(all_sigma_base, axis=0)

    mult_candidates = np.linspace(0.50, 2.00, 76)  # 0.50..2.00 step 0.02
    add_candidates = np.array([0.0, 15.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0])

    best_mult = 1.0
    best_add = 0.0
    best_lll = -np.inf

    for mlt in mult_candidates:
        for add in add_candidates:
            sigma = np.sqrt((sigma_base * mlt) ** 2 + (add**2))
            lll = _laplace_lll(delta=delta, sigma=sigma)
            if lll > best_lll:
                best_lll = float(lll)
                best_mult = float(mlt)
                best_add = float(add)

    return float(best_mult), float(best_add)


def tune_ridge_alpha_oof_with_calibrated_sigma(
    data: pd.DataFrame, n_splits: int = 5
) -> tuple[float, float, float]:
    patients = np.array(sorted(data["Patient"].unique()))
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(patients)
    folds = np.array_split(patients, n_splits)

    alphas = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

    best_alpha = 10.0
    best_mult = 1.0
    best_add = 0.0
    best_score = -np.inf

    for alpha in alphas:
        mult, add = calibrate_sigma_oof(data, n_splits=n_splits, alpha=float(alpha))

        all_delta = []
        all_sigma_base = []

        for k in range(n_splits):
            valid_patients = np.array(sorted(folds[k].tolist()))
            df_tr = data[~data["Patient"].isin(valid_patients)].copy()

            m, trc = model_fit(df_tr, examine=False, alpha=float(alpha))

            tmp = _build_submission_like_template_for_patients_stable_meta(
                valid_patients
            )
            if len(tmp) == 0:
                continue

            pred_va = model_predict(m, trc, tmp)
            y = pred_va.dropna(subset=["FVC_true"]).copy()
            if len(y) == 0:
                continue

            delta = (y["FVC_pred"].values - y["FVC_true"].values).astype(np.float64)
            sigma_base = m["sigma_by_class"][
                np.clip(y["Class"].astype(int).values, 0, 5)
            ].astype(np.float64)

            all_delta.append(delta)
            all_sigma_base.append(sigma_base)

        if len(all_delta) == 0:
            continue

        delta = np.concatenate(all_delta, axis=0)
        sigma_base = np.concatenate(all_sigma_base, axis=0)
        sigma = np.sqrt((sigma_base * mult) ** 2 + (add**2))

        score = _laplace_lll(delta=delta, sigma=sigma)
        if score > best_score:
            best_score = float(score)
            best_alpha = float(alpha)
            best_mult = float(mult)
            best_add = float(add)

    return float(best_alpha), float(best_mult), float(best_add)


best_alpha, sigma_multiplier_oof, sigma_add_oof = (
    tune_ridge_alpha_oof_with_calibrated_sigma(train_last3, n_splits=5)
)

print("OOF best alpha (last3, submission-like calib):", best_alpha)
print("OOF sigma_multiplier (last3):", sigma_multiplier_oof)
print("OOF sigma_add (last3):", sigma_add_oof)



## === cell 15
print("Fit model ...")
model, trace = model_fit(train_last3, examine=False, alpha=float(best_alpha))

model["sigma_multiplier"] = float(sigma_multiplier_oof)
model["sigma_add"] = float(sigma_add_oof)
print("")

print("Make predictions for test sample_submission rows ...")
sample = pd.read_csv(SAMPLE_SUB)
final = sample[["Patient_Week"]].copy()

pw_split = final["Patient_Week"].str.rsplit("_", n=1, expand=True)
final["Patient"] = pw_split[0].values
final["Weeks"] = pw_split[1].astype(int).values

test_meta = test[["Patient", "PatientID", "Class"]].drop_duplicates("Patient")
final = final.merge(test_meta, on="Patient", how="left")

if final["PatientID"].isna().any():
    final.loc[final["PatientID"].isna(), "PatientID"] = le_id.transform(
        final.loc[final["PatientID"].isna(), "Patient"]
    )
if final["Class"].isna().any():
    final.loc[final["Class"].isna(), "Class"] = int(test["Class"].mode().iloc[0])

pred_test = model_predict(
    model, trace, final[["Patient", "PatientID", "Weeks", "Class"]]
)

pred_test = pred_test.copy()
pred_test["Patient_Week"] = pred_test["Patient"] + "_" + pred_test["Weeks"].astype(str)
pred_map = pred_test.set_index("Patient_Week")[["FVC_pred", "sigma"]]
final = final.join(pred_map, on="Patient_Week")

if final["FVC_pred"].isna().any():
    base_fvc = test.groupby("Patient").first(numeric_only=False)["FVC"].to_dict()
    missing_mask = final["FVC_pred"].isna()
    final.loc[missing_mask, "FVC_pred"] = (
        final.loc[missing_mask, "Patient"].map(base_fvc).astype(float)
    )

if final["sigma"].isna().any():
    final.loc[final["sigma"].isna(), "sigma"] = 150.0

final = final.rename(columns={"FVC_pred": "FVC", "sigma": "Confidence"})
final["FVC"] = final["FVC"].astype(float)
final["Confidence"] = final["Confidence"].astype(float)

final["Confidence"] = final["Confidence"].clip(lower=70.0)

final = final[["Patient_Week", "FVC", "Confidence"]]
final.to_csv("submission.csv", index=False)
print(final.shape)
final.head()
