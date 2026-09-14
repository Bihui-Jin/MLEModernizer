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

-6.8731

# 6. Current score

-13.40821

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -12.88684) has done: 'I remove the `pymc3` dependency (it fails with NumPy 1.26) and replace it with a lightweight linear per-patient regression that preserves the original core assumption `FVC = a + b * Weeks` and produces a per-row uncertainty `sigma` for the Laplace metric. I fix the missing imports and ensure `PatientID` is created consistently for both train/test, then make `generate_template()` independent of any global encoder. Finally, I write `submission.csv` with the exact required columns and align it to `sample_submission.csv` so the file is always valid and complete.'
- What this solution (achieved -11.57641) has done: 'Your current score is far below the target (gap ≈ -6.01, higher is better), so we should improve but with minimal changes that keep the same per-patient linear model. The biggest lever consistent with your core logic is the confidence (`sigma`) calibration: the Laplace metric strongly rewards not underestimating uncertainty, and your current residual-based sigma is often too small/unstable, which hurts score. I keep the same `FVC = a + b*Weeks` fit, but compute a more robust per-patient sigma (add a small floor, use an unbiased estimate, and blend with a global sigma) and then lightly inflate sigma at prediction time to better match the metric’s clipping behavior. This typically improves the public LB substantially for this baseline without changing architecture/training loops, and still writes a valid `submission.csv`.'
- What this solution (achieved -14.95959) has done: 'Your current score (-11.576) is well below the target (-6.873), so we should improve (increase) it while keeping the same per-patient linear regression core. The biggest safe lever for this metric is confidence calibration: we compute per-patient sigma from regression residuals (not raw FVC std), blend it with a robust global sigma, and then choose a single mild inflation factor that improves the Laplace metric without changing the prediction model. I also fix the small baseline bug (baseline should be week 0, not “minimum week”) to better anchor each patient’s trajectory, while leaving the rest of the pipeline intact. Finally, we keep submission alignment to `sample_submission.csv` exactly as you do now, ensuring a valid `submission.csv`.'
- What this solution (achieved -14.33392) has done: 'You’re far below the target (current -14.96 vs target -6.87, higher is better), so we should improve score but keep the same per-patient linear model `FVC = a + b*Weeks`. The safest high-impact change for this metric is calibrating `Confidence`: if sigma is too small, the Laplace score is heavily penalized, so we make sigma more robust by (1) computing it from residuals but using a median-absolute-deviation fallback for small sample sizes, (2) blending per-patient sigma with a global sigma computed robustly, and (3) applying a slightly stronger (but still mild) inflation factor at prediction time. This preserves your core logic and evaluation semantics while typically improving OSIC LLL materially. The submission writing logic remains the same and still aligns exactly to `sample_submission.csv`.'
- What this solution (achieved -13.19835) has done: 'I keep your per-patient linear fit `FVC = a + b*Weeks` unchanged and focus only on confidence calibration, since the Laplace metric heavily penalizes underconfident (too-small) sigmas and your current score is far below the target (needs improvement). The smallest high-impact tweak is to (1) compute a more stable *global* sigma from per-patient residual sigmas using a trimmed median, and (2) slightly increase the blend toward the global sigma so patients with few/noisy points don’t get overly optimistic confidence. I also replace the fixed `1.25` inflation with a slightly stronger but still mild factor to move your score upward toward the target band without changing predictions’ core structure. Submission writing and alignment to `sample_submission.csv` stays identical to ensure a valid `submission.csv`.'
- What this solution (achieved -15.06) has done: 'Your current score (-13.198) is far below the target (-6.873, higher is better), so we should improve it while preserving the same per-patient linear model. The smallest high-impact change for this metric is to stop over-inflating `sigma`: your `1.45 * s` likely makes predictions too uncertain, which reduces the Laplace log-likelihood via the `-log(sigma)` term once errors are not huge. I keep the exact same `FVC = a + b*Weeks` fit and the same residual-based sigma construction, but reduce the prediction-time inflation factor to a milder value and apply the same confidence fallback consistently. This should move the score upward (less negative) toward the target without changing the core modeling approach or submission schema.'
- What this solution (achieved -13.72133) has done: 'Your score is far below the target (current -15.06 vs target -6.8731; higher is better), so we should improve while keeping your per-patient linear fit unchanged. The biggest safe lever without changing the model is calibrating `Confidence` because the Laplace metric heavily penalizes under/over-confident sigmas via both the `delta/sigma` and `-log(sigma)` terms. I keep the same `FVC = a + b*Weeks` fitting and prediction logic, but (1) compute a more stable global sigma from training residuals, (2) use a slightly stronger global blending for patients with few points, and (3) tune `SIGMA_INFLATION` upward modestly (your current 1.15 appears too “tight”, leading to large `delta/sigma` penalties). Submission writing remains identical and still aligns exactly to `sample_submission.csv`.'
- What this solution (achieved -14.53853) has done: 'Your current score (-13.721) is far below the target (-6.873, higher is better), so we should improve while keeping the same per-patient linear model `FVC = a + b * Weeks`. The most direct lever that doesn’t change the prediction core is the `Confidence` calibration: right now the global blending is quite strong and then you inflate again, which often makes `sigma` too large and hurts the `-log(sigma)` term. I make a minimal change by slightly reducing the *global* blend weight (so per-patient residual information matters more) and modestly lowering `SIGMA_INFLATION`, which should increase the Laplace log-likelihood toward the target without changing the model form. The submission creation and alignment to `sample_submission.csv` stays identical so you still always write a valid `submission.csv`.'
- What this solution (achieved -13.21299) has done: 'Your gap to target is large (current -14.54 vs target -6.87; higher is better), so we should improve while keeping the same per-patient linear model `FVC = a + b*Weeks`. The most effective minimal lever here is the `Confidence` calibration: right now sigma is often too large (hurting the `-log(sigma)` term) and also not tied to how far from baseline week you predict, which can under/over-penalize the `delta/sigma` term. I keep the same fit and same prediction formula for FVC, but (1) slightly reduce the base sigma inflation and (2) add a tiny, week-distance-based uncertainty growth (common in this competition) so sigma is tighter near baseline and only modestly larger far away. This preserves evaluation semantics and produces the same `submission.csv` schema aligned to `sample_submission.csv`.'
- What this solution (achieved -12.16874) has done: 'You’re far below the target (current -13.213 vs target -6.873; higher is better), so we should improve score while keeping your per-patient linear model `FVC = a + b*Weeks` unchanged. The biggest minimal lever here is confidence calibration: your current `sigma` is often too tight near baseline and not large enough for the far future, which makes the `delta/sigma` penalty dominate. I keep the same fit and prediction formula, but (1) make sigma grow slightly more with week distance (conservative uncertainty growth), and (2) reduce the base inflation a touch so sigma isn’t uniformly oversized (helping the `-log(sigma)` term). This stays within the same core logic and still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved -12.36743) has done: 'Your current score (-12.16874) is well below the target (-6.8731), so we should improve (increase) it while keeping the exact same per-patient linear model `FVC = a + b*Weeks`. The biggest low-risk lever is confidence calibration: your `SIGMA_WEEK_SLOPE=1.60` likely over-inflates sigma for distant weeks, which hurts the `-log(sigma)` term more than it helps `delta/sigma` on this baseline, so we reduce it to a milder value. To keep sigma reasonable when predicting far from baseline without changing the model, we add a very small quadratic growth term (still only affects `Confidence`, not `FVC_pred`) which tends to improve Laplace LLL vs a large linear slope. Finally, we keep the submission alignment to `sample_submission.csv` exactly as you do now to guarantee a valid `submission.csv`.'
- What this solution (achieved -13.24241) has done: 'We’re far below the target (current -12.367 vs target -6.873; higher is better), so we should improve while keeping the exact same per-patient linear model `FVC = a + b*Weeks`. The most effective minimal lever left is the `Confidence` calibration: your current distance-based growth is likely making sigma too large far from baseline (hurting the `-log(sigma)` term) while not helping enough near the weeks that are actually scored. I make a small, safe adjustment by (1) reducing the week-distance linear and quadratic growth a bit, and (2) applying a tiny lower bound tied to the model’s per-patient sigma so confidence doesn’t collapse near baseline—this changes only `Confidence`, not `FVC_pred`. Submission writing and alignment to `sample_submission.csv` stays identical to guarantee a valid `submission.csv`.'
- What this solution (achieved -12.2268) has done: 'I keep your per-patient linear regression (`FVC = a + b*Weeks`) exactly as-is and only adjust the confidence calibration, because your current score is far below the target and the Laplace metric is very sensitive to `sigma`. The minimal change is to make `sigma` grow more for weeks far from each patient’s baseline week (where your linear extrapolation error increases), while keeping it unchanged near baseline so we don’t pay unnecessary `-log(sigma)` penalty. Concretely, I slightly increase `SIGMA_WEEK_SLOPE` and `SIGMA_WEEK_QUAD` and keep everything else (fit, prediction, submission alignment) identical. This should increase the score (less negative) toward the target without changing your model’s core logic or submission format.'
- What this solution (achieved -12.81668) has done: 'Your current score (-12.2268) is well below the target (-6.8731), so we should improve it (increase it) with the smallest change that plausibly helps the Laplace metric without changing the linear per-patient FVC model. The biggest lever here is confidence calibration: right now sigma grows with distance from baseline using fairly aggressive linear/quadratic terms, which can hurt the `-log(sigma)` part of the metric on the scored weeks. I keep the exact same `FVC = a + b*Weeks` predictions and only slightly reduce the distance-based sigma growth while keeping the same clipping/flooring behavior, so confidence is less over-inflated. Submission writing/alignment stays identical to ensure a valid `submission.csv`.'
- What this solution (achieved -13.40821) has done: 'We keep your per-patient linear fit (`FVC = a + b*Weeks`) exactly unchanged and only adjust the confidence calibration, because your score is far below target and the Laplace metric is very sensitive to sigma. The minimal likely win is to reduce the distance-based sigma growth further, since you’re predicting a wide week range but Kaggle only scores the final 3 visits, and over-inflated sigma hurts the `-log(sigma)` term a lot. Concretely, we slightly lower `SIGMA_WEEK_SLOPE` and `SIGMA_WEEK_QUAD` while leaving the residual-based per-patient/global sigma logic intact. The pipeline remains end-to-end and still writes a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings("ignore")

import os

print("Listing a few input files under /kaggle/input ...")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")




## === cell 1
train = pd.read_csv(TRAIN_CSV)
train_raw = train.copy()
test = pd.read_csv(TEST_CSV)

train.drop(train[train.Patient == "ID00197637202246865691526"].index, inplace=True)

all_patients = pd.concat(
    [train[["Patient"]], test[["Patient"]]], axis=0, ignore_index=True
)
le_id = LabelEncoder()
le_id.fit(all_patients["Patient"])

train["PatientID"] = le_id.transform(train["Patient"])
test["PatientID"] = le_id.transform(test["Patient"])

train.head()




## === cell 2
def add_baselines(data: pd.DataFrame) -> pd.DataFrame:
    base = data.loc[data["Weeks"] == 0, ["Patient", "Weeks", "FVC"]].copy()

    missing = set(data["Patient"].unique()) - set(base["Patient"].unique())
    if missing:
        aux = (
            data.loc[data["Patient"].isin(list(missing)), ["Patient", "Weeks"]]
            .groupby("Patient")["Weeks"]
            .min()
            .reset_index()
        )
        aux = pd.merge(
            aux,
            data[["Patient", "Weeks", "FVC"]],
            how="left",
            on=["Patient", "Weeks"],
        )
        aux = aux.groupby("Patient", as_index=False).mean()
        base = pd.concat([base, aux], ignore_index=True)

    base = base.groupby("Patient", as_index=False).mean()
    base["Weeks"] = base["Weeks"].round().astype(int)
    base["FVC"] = base["FVC"].round().astype(int)

    data = pd.merge(data, base, how="left", on="Patient", suffixes=("", "_base"))
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


train["Class"] = train.apply(patient_class, axis=1).astype(int)
test["Class"] = test.apply(patient_class, axis=1).astype(int)

train[["Patient", "Weeks", "FVC", "PatientID", "Class"]].head()




## === cell 4
PatientID_vals = train["Patient"].values
fvc_b = train.groupby("Patient").first()["FVC_base"]
age = train.groupby("PatientID").first()["Age"]
age.head()




## === cell 5
def _robust_sigma_from_residuals(resid: np.ndarray) -> float:
    """
    Robust sigma estimator used only for confidence calibration.
    This keeps the same linear fit core logic, but stabilizes sigma to improve Laplace metric.
    """
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return np.nan
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    return float(1.4826 * mad)


def _trimmed_median(x: np.ndarray, trim_q: float = 0.1) -> float:
    """
    Small confidence-calibration helper: a trimmed median is more stable than a raw median
    when residual sigmas have heavy tails. This affects only sigma, not FVC predictions.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    lo, hi = np.quantile(x, [trim_q, 1.0 - trim_q])
    xs = x[(x >= lo) & (x <= hi)]
    if xs.size == 0:
        xs = x
    return float(np.median(xs))


def model_fit(data: pd.DataFrame, examine: bool = True):
    params = {}

    resid_sigmas = []
    resid_sigmas_rob = []

    for pid, g in data.groupby("Patient"):
        g = g.sort_values("Weeks")
        x = g["Weeks"].values.astype(float)
        y = g["FVC"].values.astype(float)
        if len(g) >= 3 and np.std(x) > 0:
            b, a = np.polyfit(x, y, 1)
            yhat = a + b * x
            resid = y - yhat

            s_std = float(np.std(resid, ddof=1))
            s_mad = _robust_sigma_from_residuals(resid)

            if np.isfinite(s_std) and s_std > 1.0:
                resid_sigmas.append(s_std)
            if np.isfinite(s_mad) and s_mad > 1.0:
                resid_sigmas_rob.append(s_mad)

    global_sigma = (
        _trimmed_median(np.array(resid_sigmas_rob), trim_q=0.05)
        if len(resid_sigmas_rob)
        else np.nan
    )
    if not np.isfinite(global_sigma) or global_sigma <= 1.0:
        global_sigma = (
            _trimmed_median(np.array(resid_sigmas), trim_q=0.05)
            if len(resid_sigmas)
            else np.nan
        )
    if not np.isfinite(global_sigma) or global_sigma <= 1.0:
        global_sigma = 200.0

    for pid, g in data.groupby("Patient"):
        g = g.sort_values("Weeks")
        x = g["Weeks"].values.astype(float)
        y = g["FVC"].values.astype(float)

        if len(g) >= 2 and np.std(x) > 0:
            b, a = np.polyfit(x, y, 1)
            yhat = a + b * x
            resid = y - yhat

            if len(resid) >= 4:
                sigma_res = float(np.std(resid, ddof=1))
                sigma_rob = _robust_sigma_from_residuals(resid)
                if np.isfinite(sigma_rob) and sigma_rob > 1.0:
                    sigma_res = 0.7 * sigma_res + 0.3 * sigma_rob
            else:
                sigma_res = _robust_sigma_from_residuals(resid)

            if not np.isfinite(sigma_res) or sigma_res <= 1.0:
                sigma_res = global_sigma

            sigma = 0.55 * sigma_res + 0.45 * global_sigma

            sigma = max(float(sigma), 70.0)
        else:
            a = float(y[0]) if len(y) else 2000.0
            b = 0.0
            sigma = max(float(global_sigma), 70.0)

        params[pid] = {"a": float(a), "b": float(b), "sigma": float(sigma)}

    model = {"params": params, "global_sigma": float(global_sigma)}
    trace = None  # kept for API compatibility

    return model, trace




## === cell 6
def generate_template(data: pd.DataFrame):
    pred_template = []
    for patient in data["Patient"].unique():
        df = pd.DataFrame(columns=["PatientID", "Weeks", "Patient", "Class"])
        df["Weeks"] = np.arange(-12, 134)
        df["Patient"] = patient
        df["Class"] = int(data.loc[data["Patient"] == patient, "Class"].max())
        df["PatientID"] = int(le_id.transform([patient])[0])
        pred_template.append(df)
    pred_template = pd.concat(pred_template, ignore_index=True)
    pred_template["PatientID"] = pred_template["PatientID"].astype(int)
    pred_template["Weeks"] = pred_template["Weeks"].astype(int)
    pred_template["Class"] = pred_template["Class"].astype(int)
    return pred_template


template_train_test = generate_template(test)
template_train_test.head()




## === cell 7
SIGMA_INFLATION = 1.03

SIGMA_WEEK_SLOPE = 0.55  # was 0.70
SIGMA_WEEK_QUAD = 0.008  # was 0.012

SIGMA_REL_FLOOR = 0.80


def model_predict(model, trace, template: pd.DataFrame):
    params = model["params"]
    global_sigma = model["global_sigma"]

    df = pd.DataFrame(index=np.arange(len(template)))
    df["Patient"] = template["Patient"].values
    df["Weeks"] = template["Weeks"].values.astype(int)

    a = np.zeros(len(template), dtype=float)
    b = np.zeros(len(template), dtype=float)
    s = np.zeros(len(template), dtype=float)

    for i, pid in enumerate(df["Patient"].values):
        p = params.get(pid, None)
        if p is None:
            a[i] = 2000.0
            b[i] = 0.0
            s[i] = global_sigma
        else:
            a[i] = p["a"]
            b[i] = p["b"]
            s[i] = p["sigma"]

    base_week_map = {}
    if "Weeks_base" in train.columns:
        base_week_map.update(train.groupby("Patient")["Weeks_base"].first().to_dict())
    if "Weeks_base" in test.columns:
        base_week_map.update(test.groupby("Patient")["Weeks_base"].first().to_dict())

    w0 = np.array(
        [float(base_week_map.get(pid, 0.0)) for pid in df["Patient"].values],
        dtype=float,
    )

    df["FVC_pred"] = a + b * df["Weeks"].values.astype(float)

    dist = np.abs(df["Weeks"].values.astype(float) - w0)

    sigma_dist = SIGMA_WEEK_SLOPE * dist + SIGMA_WEEK_QUAD * (dist**2)
    sigma_base = SIGMA_INFLATION * s
    df["sigma"] = np.maximum(sigma_base + sigma_dist, SIGMA_REL_FLOOR * s)

    df["FVC_inf"] = df["FVC_pred"] - df["sigma"]
    df["FVC_sup"] = df["FVC_pred"] + df["sigma"]

    df = pd.merge(
        df, train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
    )
    df = df.rename(columns={"FVC": "FVC_true"})
    return df




## === cell 8
def examine_predictions(data):
    n = (data["Patient"].nunique()) + 1
    f, axes = plt.subplots((n // 3) + 1, 3, figsize=(15, 5 * ((n // 3) + 1)))
    axes = np.array(axes).reshape(((n // 3) + 1, 3))
    for i, patient in enumerate(data["Patient"].unique()):
        ax = axes[i // 3, i % 3]
        dfp = data[data["Patient"] == patient].sort_values("Weeks")
        x = dfp["Weeks"]
        ax.set_title(patient)
        ax.plot(x, dfp["FVC_true"], "o")
        ax.plot(x, dfp["FVC_pred"])
        ax.fill_between(x, dfp["FVC_inf"], dfp["FVC_sup"], alpha=0.3, color="#ffcd3c")
        ax.set_ylabel("FVC")
    plt.tight_layout()




## === cell 9
def evaluate_predictions(df, use_only_last_3_measures=False, examine=False):
    if use_only_last_3_measures:
        y = df.dropna(subset=["FVC_true"]).groupby("Patient").tail(3)
    else:
        y = df.dropna(subset=["FVC_true"])

    sigma_c = y["sigma"].values.astype(float)
    sigma_c[sigma_c < 70] = 70

    delta = (y["FVC_pred"] - y["FVC_true"]).abs().values.astype(float)
    delta[delta > 1000] = 1000

    lll = -np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

    y = y.copy()
    y["lll"] = lll
    patient_llls = y.groupby("Patient")["lll"].mean()

    if examine:
        plt.hist(patient_llls, bins=50)

    return float(np.mean(lll)), patient_llls




## === cell 10
def evaluation_cycle(train_df, valid_df, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model, trace = model_fit(train_df, examine=False)

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train_df)
    pred_train = model_predict(model, trace, template_train)
    if examine_preds:
        examine_predictions(pred_train)
    lll_train, train_patient_llls = evaluate_predictions(pred_train)

    pred_valid, lll_valid, valid_patient_llls = None, None, None
    if valid_df is not None:
        print("Examine true vs predictions for validation data ...")
        template_valid = generate_template(valid_df)
        pred_valid = model_predict(model, trace, template_valid)
        if examine_preds:
            examine_predictions(pred_valid)
        lll_valid, valid_patient_llls = evaluate_predictions(pred_valid)

    return (
        pred_train,
        pred_valid,
        lll_train,
        lll_valid,
        train_patient_llls,
        valid_patient_llls,
    )




## === cell 11
lll_trains = []
lll_valids = []

fold = 0
kfold = KFold(3, shuffle=True, random_state=1)
all_patients = train["Patient"].unique()

for fold in range(0):
    validation_patients = np.random.choice(all_patients, size=30, replace=False)
    df_valid = train[train["Patient"].isin(validation_patients)]
    df_train = train[~train["Patient"].isin(validation_patients)]

    df_valid_first_readings = df_valid.groupby("Patient").head(1)
    df_train = pd.concat([df_train, df_valid_first_readings], axis=0, ignore_index=True)

    print(f"Fold: {fold}")
    (
        pred_train,
        pred_valid,
        lll_train,
        lll_valid,
        train_patient_llls,
        valid_patient_llls,
    ) = evaluation_cycle(df_train, df_valid, examine_trace=False, examine_preds=False)

    print(f"Laplace Log Likelihoods for fold: {fold}")
    print(f"Training:     {lll_train:.4f}")
    print(f"Validation:   {lll_valid:.4f}")

    lll_trains.append(lll_train)
    lll_valids.append(lll_valid)

print(f"Training LLLs:   {*lll_trains,}")
print(f"Validation LLLs: {*lll_valids,}")




## === cell 12
print("Fit model ...")
model, trace = model_fit(train, examine=False)
print("")

print("Make predictions for test data ...")
template_test = generate_template(test)
pred_test = model_predict(model, trace, template_test)

sample = pd.read_csv(SAMPLE_SUB_CSV)

tmp = sample["Patient_Week"].str.split("_", n=1, expand=True)
sample_pat = tmp[0].values
sample_week = tmp[1].astype(int).values

pred_key = pred_test.copy()
pred_key["Patient_Week"] = (
    pred_key["Patient"].astype(str) + "_" + pred_key["Weeks"].astype(str)
)
pred_key = pred_key.set_index("Patient_Week")[["FVC_pred", "sigma"]]

final = sample.copy()
final = final.set_index("Patient_Week")

fallback_fvc = test.set_index("Patient")["FVC"].to_dict()
fvc_out = []
conf_out = []

for pw in final.index:
    if pw in pred_key.index:
        fvc_out.append(float(pred_key.loc[pw, "FVC_pred"]))
        conf_out.append(float(pred_key.loc[pw, "sigma"]))
    else:
        pat, wk = pw.split("_", 1)
        fvc_out.append(float(fallback_fvc.get(pat, 2000.0)))
        conf_out.append(float(SIGMA_INFLATION * model["global_sigma"]))

final["FVC"] = np.round(np.array(fvc_out)).astype(int)
final["Confidence"] = np.array(conf_out).astype(float)
final["Confidence"] = np.maximum(final["Confidence"], 70.0)

final = final.reset_index()

final.to_csv("submission.csv", index=False)
print(final.shape)
print(final.head())
print("Wrote submission.csv")
