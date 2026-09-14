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

-6.8568

# 6. Current score

-8.22633

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.73879) has done: 'I remove the `pymc3` dependency (it crashes with NumPy 1.26 via Theano’s deprecated `np.bool`) and replace it with a minimal, deterministic clinical-only baseline that still matches the competition’s required output format. I also fix the missing `LabelEncoder` import usage by eliminating the now-unneeded encoding path and ensuring the template/submission align exactly to `sample_submission.csv`’s `Patient_Week` IDs (so no missing weeks or extra rows). Finally, I compute a reasonable confidence per prediction using training residuals from a per-patient linear fit, clipped to the metric’s effective floor (70), which should yield a sensible score and always produce a valid `submission.csv`.'
- What this solution (achieved -12.94987) has done: 'I keep your per-patient linear fit core logic intact and only adjust the parts that most directly affect the Laplace log-likelihood: (1) make the “baseline” use Week==0 when available (instead of minimum week), which better matches the test-time setup and improves extrapolation, and (2) tune the global multiplier applied to residual-based uncertainty (Confidence) to better calibrate σ toward the metric’s optimum rather than over-penalizing via the log(σ) term. These are minimal, deterministic changes that don’t alter the model family or training loop, but should move your score upward toward the target. The script still writes a valid `submission.csv` aligned exactly to `sample_submission.csv`.'
- What this solution (achieved -12.94987) has done: 'We keep your per-patient linear fit and submission alignment exactly as-is, and only make two minimal changes that directly affect the Laplace log-likelihood score. First, we compute each patient’s “anchor” (Weeks0/FVC0) using Week==0 when available (falling back to earliest week otherwise), which makes the fallback intercepts consistent with the test setup and improves extrapolation for sparse patients. Second, we set the Confidence multiplier using a quick deterministic calibration on the training set (grid over a few multipliers) to move the resulting score upward without changing the model family or adding any training loop. These changes are small, fast, and should improve the score toward your target while preserving core logic.'
- What this solution (achieved -10.47626) has done: 'I keep your per-patient linear fit exactly as-is and only adjust the uncertainty calibration, because the Laplace metric is very sensitive to σ and your current grid is likely overfitting to in-sample residuals (hurting leaderboard score). Specifically, I (1) pick `CONF_MULT` using out-of-fold (GroupKFold by Patient) predictions so the σ scaling better reflects test-time error, and (2) slightly widen the multiplier search range while staying deterministic and fast. This preserves the same model family and prediction logic, but should move the score upward toward your target by improving confidence calibration rather than changing FVC predictions. The script still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved -9.45671) has done: 'Your current FVC point predictions are already reasonable for a clinical-only per-patient linear fit; the biggest remaining gap to the target is likely coming from miscalibrated per-patient uncertainty (Confidence), since the Laplace metric strongly rewards well-chosen σ. I keep your exact fitting/prediction logic intact and only make the σ calibration more robust by (1) selecting `CONF_MULT` using the median (more stable than mean under heavy tails) of the OOF Laplace scores and (2) slightly widening and densifying the multiplier grid around your current best region to better match the target without changing the model family. This is deterministic, fast, and only touches the uncertainty post-processing used by the metric. The script still writes a valid `submission.csv` aligned 1:1 with `sample_submission.csv`.'
- What this solution (achieved -9.13108) has done: 'I keep your per-patient linear fitting and point-prediction logic unchanged and focus only on improving the Laplace metric via better uncertainty (Confidence) calibration, since that is the most sensitive lever left. Specifically, I (1) select `CONF_MULT` by maximizing the *mean* OOF Laplace score (closer to Kaggle’s averaging than the median you currently use), and (2) add a tiny deterministic local refinement around the best grid value to avoid missing the optimum due to coarse steps. These changes are minimal, fast (still just OOF predictions once), and should move the score upward toward your target without altering the model family or training semantics. The script still write a valid `submission.csv` aligned 1:1 to `sample_submission.csv`.'
- What this solution (achieved -8.48307) has done: 'I keep your per-patient linear fit and point predictions exactly the same, and only adjust the Confidence post-processing because that is the most metric-sensitive lever and your current score is still below target. Specifically, I calibrate the confidence multiplier using a patient-level OOF score computed on per-patient *last 3 visits* (matching the test scoring focus) instead of averaging across every historical row, which tends to miscalibrate σ. I also allow a per-patient “effective sigma” that grows mildly with distance from that patient’s baseline week (weeks_base), which is a minimal, deterministic adjustment that usually improves Laplace log-likelihood without changing the FVC model. The script still runs end-to-end and writes a valid `submission.csv` aligned 1:1 to `sample_submission.csv`.'
- What this solution (achieved -8.42424) has done: 'We keep your per-patient linear FVC model exactly the same and only adjust the uncertainty calibration in a way that better matches the competition’s scoring setup. Specifically, we compute OOF calibration using the *final-three-weeks per patient* but (crucially) score it against predictions for those exact target weeks rather than whatever rows happen to be the last three in the fold, which reduces a subtle misalignment between calibration and test-time targets. We also add a very small, deterministic refinement around the best (mult, alpha) pair in 2D (not just mult) to land closer to the metric optimum without changing modeling. These changes are minimal, fast, and aimed at improving your current score (-8.483) upward toward the target (-6.8568).'
- What this solution (achieved -8.35102) has done: 'I keep your per-patient linear FVC model and OOF calibration structure unchanged, and only make minimal changes that directly improve the Laplace metric by better aligning the “weeks_base” used in sigma growth with the test-time reality (baseline week=0). Concretely, I ensure `Weeks_base` is forced to 0 for the test set (and for the training last3 calibration rows when Week 0 exists), so the confidence inflation term `|week - weeks_base|` isn’t inadvertently anchored to an early negative week. I also make the OOF folds deterministic by using a fixed GroupKFold split ordering (stable patient ordering), reducing variance in the chosen (mult, alpha) without changing semantics. These changes should improve calibration and move your score upward toward the target while preserving the core logic and producing the same valid submission format.'
- What this solution (achieved -8.43186) has done: 'I keep your per-patient linear FVC model and the same OOF calibration structure, and only make a minimal change to the Confidence calibration to move the score up toward the target. The main adjustment is to calibrate `CONF_MULT` and `CONF_ALPHA` using the same Laplace metric but computed on an OOF “simulation” of the test setup: for each patient we use their known baseline row (Week==0 if available else earliest) and predict their last 3 visits, then score those predictions. This better matches how the competition evaluates (baseline known, last-3 predicted) without changing your model family or adding any training loop. The submission generation remains identical and still writes a valid `submission.csv` aligned 1:1 to `sample_submission.csv`.'
- What this solution (achieved -8.35102) has done: 'I keep your per-patient linear FVC model exactly as-is and only adjust the uncertainty calibration, since the Laplace metric is most sensitive to σ and your current score is still below target. The minimal change is to calibrate `CONF_MULT`/`CONF_ALPHA` using a stricter OOF setup that mirrors the test condition: fit slopes/intercepts (and residual σ) on each patient’s *full history* in the training fold, but only score on that patient’s last-3 visits; this avoids underestimating σ from fitting to a single baseline row. I also apply the same fold-specific residual σ (from full-history fit) rather than from the baseline-only table, keeping everything deterministic and fast. This should move the score upward toward your target while preserving identical prediction semantics and producing the same valid `submission.csv`.'
- What this solution (achieved -8.35126) has done: 'I keep your per-patient linear FVC model and OOF calibration structure intact, and only make the smallest changes that are likely to improve the Laplace log-likelihood toward your target by improving uncertainty calibration. The main adjustment is to compute the residual standard deviation per patient more robustly (use RMSE instead of sample std with ddof=1) and to add a small, deterministic “sigma floor” component mixed into every prediction (so Confidence is not overly optimistic for low-residual patients, which is heavily penalized by the metric when errors occur). I then re-run your existing OOF grid/refinement to select `CONF_MULT/CONF_ALPHA` under this improved sigma definition, without changing your prediction semantics or submission format. This should move the score upward from -8.351 toward -6.8568 while remaining deterministic and fast.'
- What this solution (achieved -8.22633) has done: 'I keep your per-patient linear FVC model exactly the same and only adjust the uncertainty calibration in ways that better match the competition’s evaluation setup (baseline known, last-3 predicted) while staying deterministic and fast. Specifically, I (1) compute OOF calibration targets as each patient’s “last 3 visits” but scored against predictions made from a baseline-anchored fit (Week==0 when available, otherwise earliest), and (2) add a tiny per-patient heteroscedastic component proportional to the patient’s fitted |slope| to avoid overconfident σ for fast decliners (this only affects Confidence). These are minimal changes focused on moving your score upward from -8.351 toward the -6.8568 target without changing point prediction semantics or submission format. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -8.22633) has done: 'I keep your per-patient linear FVC model and the same OOF calibration setup, and make only a minimal metric-aligned adjustment to Confidence calibration. The main issue is that your OOF score currently uses each patient’s per-patient residual RMSE directly, but test-time uncertainty tends to be dominated by cross-patient variability (and the metric heavily penalizes being overconfident), so we add a tiny deterministic “global residual” component into the per-row base sigma before applying your existing (mult, alpha, mix, slope_beta) calibration. This preserves the exact prediction semantics for FVC and only changes Confidence in a controlled way that typically improves Laplace log-likelihood (moves score upward toward -6.8568). Finally, we keep output alignment exactly to `sample_submission.csv` and still write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)

DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)

train = train[train.Patient != "ID00197637202246865691526"].copy()

for df in (train, test):
    df["Weeks"] = df["Weeks"].astype(int)
    df["FVC"] = df["FVC"].astype(float)
    df["Percent"] = df["Percent"].astype(float)
    df["Age"] = df["Age"].astype(float)

print(train.shape, test.shape, sample_sub.shape)
print(sample_sub.head())




## === cell 1
def add_baselines(data: pd.DataFrame) -> pd.DataFrame:
    week0 = data[data["Weeks"] == 0][
        ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ].copy()

    minw = data[["Patient", "Weeks"]].groupby("Patient").min().reset_index()
    minw = pd.merge(
        minw,
        data[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]],
        how="left",
        on=["Patient", "Weeks"],
    )

    aux = pd.concat([week0, minw], axis=0, ignore_index=True)
    aux = aux.dropna(subset=["Patient", "Weeks", "FVC"])
    aux = aux.sort_values(["Patient", "Weeks"]).drop_duplicates("Patient", keep="first")

    aux = aux.groupby("Patient", as_index=False).mean(numeric_only=True)
    aux["Weeks"] = aux["Weeks"].astype(int)
    aux["FVC"] = aux["FVC"].round().astype(int)
    aux["Percent"] = aux["Percent"].astype(float)

    aux = aux.rename(
        columns={"Weeks": "Weeks_base", "FVC": "FVC_base", "Percent": "Percent_base"}
    )
    out = pd.merge(
        data,
        aux[["Patient", "Weeks_base", "FVC_base", "Percent_base"]],
        how="left",
        on="Patient",
    )
    return out


train = add_baselines(train)
test = add_baselines(test)

test["Weeks_base"] = 0

train.head()




## === cell 2
def patient_class(row):
    sex = row.get("Sex", None)
    smoke = row.get("SmokingStatus", None)
    mapping = {
        ("Male", "Currently smokes"): 0,
        ("Male", "Ex-smoker"): 1,
        ("Male", "Never smoked"): 2,
        ("Female", "Currently smokes"): 3,
        ("Female", "Ex-smoker"): 4,
        ("Female", "Never smoked"): 5,
    }
    return mapping.get((sex, smoke), 0)


train["Class"] = train.apply(patient_class, axis=1).astype(int)
test["Class"] = test.apply(patient_class, axis=1).astype(int)

train[["Patient", "Weeks", "FVC", "FVC_base", "Weeks_base", "Class"]].head()




## === cell 3
def fit_patient_lines(train_df: pd.DataFrame):
    slopes = {}
    intercepts = {}
    resid_stds = {}

    patient_groups = train_df.groupby("Patient")
    global_slopes = []
    global_resids = []

    for pid, g in patient_groups:
        g = g.sort_values("Weeks")
        x = g["Weeks"].values.astype(float)
        y = g["FVC"].values.astype(float)

        if len(g) >= 2 and np.std(x) > 0:
            m, c = np.polyfit(x, y, 1)
            yhat = m * x + c
            resid = y - yhat

            rs = float(np.sqrt(np.mean(resid**2)))

            global_slopes.append(m)
            global_resids.append(resid)
        else:
            m, c, rs = np.nan, np.nan, np.nan

        slopes[pid] = m
        intercepts[pid] = c
        resid_stds[pid] = rs

    if len(global_slopes) == 0:
        m_global = -3.0
    else:
        m_global = float(np.median(global_slopes))

    if len(global_resids) == 0:
        sigma_global = 200.0
    else:
        all_res = np.concatenate(global_resids)
        sigma_global = float(np.sqrt(np.mean(all_res**2)))
        if not np.isfinite(sigma_global) or sigma_global <= 0:
            sigma_global = 200.0

    g0 = (
        train_df[train_df["Weeks"] == 0].sort_values("Weeks").groupby("Patient").first()
    )
    gmin = train_df.sort_values("Weeks").groupby("Patient").first()
    base_map = gmin[["Weeks", "FVC"]].rename(columns={"Weeks": "Weeks0", "FVC": "FVC0"})
    base_map.loc[g0.index, "Weeks0"] = 0.0
    base_map.loc[g0.index, "FVC0"] = g0["FVC"].astype(float).values

    for pid in slopes.keys():
        if not np.isfinite(slopes[pid]) or not np.isfinite(intercepts[pid]):
            if pid in base_map.index:
                w0 = float(base_map.loc[pid, "Weeks0"])
                f0 = float(base_map.loc[pid, "FVC0"])
                slopes[pid] = m_global
                intercepts[pid] = f0 - slopes[pid] * w0
            else:
                slopes[pid] = m_global
                intercepts[pid] = float(train_df["FVC"].median())

        if not np.isfinite(resid_stds[pid]) or resid_stds[pid] <= 0:
            resid_stds[pid] = sigma_global

    return slopes, intercepts, resid_stds, m_global, sigma_global


slopes, intercepts, resid_stds, m_global, sigma_global = fit_patient_lines(train)

print("Global slope:", m_global)
print("Global residual sigma:", sigma_global)



## === cell 4
from sklearn.model_selection import GroupKFold


def laplace_metric(y_true, y_pred, sigma):
    sigma_clip = np.maximum(sigma, 70.0)
    delta = np.minimum(np.abs(y_true - y_pred), 1000.0)
    return -(np.sqrt(2.0) * delta) / sigma_clip - np.log(np.sqrt(2.0) * sigma_clip)


def predict_fvc(patient, week):
    m = slopes.get(patient, m_global)
    c = intercepts.get(patient, float(train["FVC"].median()))
    return m * float(week) + c


def make_sigma(
    base_sigma,
    weeks,
    weeks_base,
    alpha,
    sigma_floor_mix,
    slope_abs=None,
    slope_beta=0.0,
):
    base_sigma = np.asarray(base_sigma, dtype=float)
    weeks = np.asarray(weeks, dtype=float)
    weeks_base = np.asarray(weeks_base, dtype=float)

    eff = base_sigma * (1.0 + float(alpha) * np.abs(weeks - weeks_base))

    if slope_abs is not None and float(slope_beta) > 0.0:
        slope_abs = np.asarray(slope_abs, dtype=float)
        eff = eff + float(slope_beta) * slope_abs

    floor = float(sigma_global)
    return (1.0 - float(sigma_floor_mix)) * eff + float(sigma_floor_mix) * floor


train_sorted = train.sort_values(["Patient", "Weeks"]).copy()

last3 = (
    train_sorted.groupby("Patient")
    .tail(3)[["Patient", "Weeks", "FVC", "Weeks_base"]]
    .copy()
).reset_index(drop=True)

has_week0 = set(train.loc[train["Weeks"] == 0, "Patient"].unique().tolist())
mask0 = last3["Patient"].isin(has_week0)
last3.loc[mask0, "Weeks_base"] = 0

train_for_split = train_sorted.reset_index(drop=True)
n_splits = min(5, pd.Series(train["Patient"]).nunique())
gkf = GroupKFold(n_splits=n_splits)

patient_to_fold = {}
fold_id = 0
for tr_idx, va_idx in gkf.split(
    train_for_split, groups=train_for_split["Patient"].values
):
    va_pats = pd.unique(train_for_split.iloc[va_idx]["Patient"])
    va_pats = np.sort(va_pats.astype(str))
    for p in va_pats:
        patient_to_fold[p] = fold_id
    fold_id += 1

last3_fold = last3["Patient"].map(patient_to_fold).values.astype(int)

oof_pred = np.zeros(len(last3), dtype=float)
oof_sigma_base = np.zeros(len(last3), dtype=float)
oof_slope_abs = np.zeros(len(last3), dtype=float)

oof_weeks = last3["Weeks"].values.astype(float)
oof_weeks_base = last3["Weeks_base"].fillna(0).values.astype(float)
y_true = last3["FVC"].values.astype(float)

OOF_BASE_SIGMA_GLOBAL_MIX = (
    0.20  # small, deterministic; tuned to be conservative and minimal
)

for fid in range(fold_id):
    tr_full = train_sorted[train_sorted["Patient"].map(patient_to_fold) != fid].copy()
    va_mask = last3_fold == fid
    va_df = last3.loc[va_mask].copy()

    s_tr, i_tr, r_tr, m_g_tr, sig_g_tr = fit_patient_lines(tr_full)

    def _pred(pid, wk):
        m = s_tr.get(pid, m_g_tr)
        c = i_tr.get(pid, float(tr_full["FVC"].median()))
        return m * float(wk) + c

    oof_pred[va_mask] = va_df.apply(
        lambda r: _pred(r["Patient"], r["Weeks"]), axis=1
    ).values.astype(float)

    base_sig = va_df["Patient"].map(r_tr).fillna(sig_g_tr).values.astype(float)
    base_sig = (
        1.0 - OOF_BASE_SIGMA_GLOBAL_MIX
    ) * base_sig + OOF_BASE_SIGMA_GLOBAL_MIX * float(sig_g_tr)
    oof_sigma_base[va_mask] = base_sig

    oof_slope_abs[va_mask] = (
        va_df["Patient"].map(s_tr).fillna(m_g_tr).astype(float).abs().values
    )


def oof_score_for_params(
    mult: float, alpha: float, sigma_floor_mix: float, slope_beta: float
) -> float:
    sigma_eff = make_sigma(
        oof_sigma_base,
        oof_weeks,
        oof_weeks_base,
        alpha,
        sigma_floor_mix,
        slope_abs=oof_slope_abs,
        slope_beta=slope_beta,
    ) * float(mult)
    scores = laplace_metric(y_true, oof_pred, sigma_eff)
    return float(np.mean(scores))


mult_grid = np.array(
    [0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 0.95, 1.10, 1.25, 1.45, 1.65, 1.90, 2.20],
    dtype=float,
)
alpha_grid = np.array([0.0, 0.003, 0.006, 0.010, 0.015], dtype=float)
mix_grid = np.array([0.0, 0.05, 0.10, 0.15], dtype=float)
slope_beta_grid = np.array([0.0, 2.0, 5.0, 10.0], dtype=float)

best_mult = 1.0
best_alpha = 0.0
best_mix = 0.0
best_sbeta = 0.0
best_score = -np.inf

for sbeta in slope_beta_grid:
    for mix in mix_grid:
        for alpha in alpha_grid:
            for mult in mult_grid:
                s = oof_score_for_params(mult, alpha, mix, sbeta)
                if s > best_score:
                    best_score = s
                    best_mult = float(mult)
                    best_alpha = float(alpha)
                    best_mix = float(mix)
                    best_sbeta = float(sbeta)

mult_refine = np.linspace(max(0.1, best_mult * 0.85), best_mult * 1.15, 11)
alpha_refine = np.clip(
    np.array(
        [
            best_alpha - 0.003,
            best_alpha - 0.0015,
            best_alpha,
            best_alpha + 0.0015,
            best_alpha + 0.003,
        ],
        dtype=float,
    ),
    0.0,
    0.03,
)
mix_refine = np.clip(
    np.array(
        [
            best_mix - 0.05,
            best_mix - 0.025,
            best_mix,
            best_mix + 0.025,
            best_mix + 0.05,
        ],
        dtype=float,
    ),
    0.0,
    0.30,
)
sbeta_refine = np.clip(
    np.array(
        [
            best_sbeta - 5.0,
            best_sbeta - 2.5,
            best_sbeta,
            best_sbeta + 2.5,
            best_sbeta + 5.0,
        ],
        dtype=float,
    ),
    0.0,
    25.0,
)

for sbeta in sbeta_refine:
    for mix in mix_refine:
        for alpha in alpha_refine:
            for mult in mult_refine:
                s = oof_score_for_params(mult, alpha, mix, sbeta)
                if s > best_score:
                    best_score = s
                    best_mult = float(mult)
                    best_alpha = float(alpha)
                    best_mix = float(mix)
                    best_sbeta = float(sbeta)

CONF_MULT = best_mult
CONF_ALPHA = best_alpha
CONF_MIX = best_mix
CONF_SLOPE_BETA = best_sbeta

print(
    "Chosen CONF_MULT/CONF_ALPHA/CONF_MIX/CONF_SLOPE_BETA:",
    CONF_MULT,
    CONF_ALPHA,
    CONF_MIX,
    CONF_SLOPE_BETA,
    " (OOF mean approx:",
    best_score,
    ")",
)

sub = sample_sub.copy()
sub["Patient"] = sub["Patient_Week"].str.split("_").str[0]
sub["Weeks"] = sub["Patient_Week"].str.split("_").str[1].astype(int)

sub["FVC"] = sub.apply(lambda r: predict_fvc(r["Patient"], r["Weeks"]), axis=1)

SUB_BASE_SIGMA_GLOBAL_MIX = OOF_BASE_SIGMA_GLOBAL_MIX

base_sigma_sub = (
    sub["Patient"].map(resid_stds).fillna(sigma_global).astype(float).values
)
base_sigma_sub = (
    1.0 - SUB_BASE_SIGMA_GLOBAL_MIX
) * base_sigma_sub + SUB_BASE_SIGMA_GLOBAL_MIX * float(sigma_global)

weeks_sub = sub["Weeks"].values.astype(float)

weeks_base_sub = test.set_index("Patient")["Weeks_base"].to_dict()
weeks_base_sub = sub["Patient"].map(weeks_base_sub).fillna(0).astype(float).values

slope_abs_sub = sub["Patient"].map(slopes).fillna(m_global).astype(float).abs().values

sub["Confidence"] = (
    make_sigma(
        base_sigma_sub,
        weeks_sub,
        weeks_base_sub,
        CONF_ALPHA,
        CONF_MIX,
        slope_abs=slope_abs_sub,
        slope_beta=CONF_SLOPE_BETA,
    )
    * CONF_MULT
)
sub["Confidence"] = sub["Confidence"].clip(lower=70)

sub["FVC"] = np.round(sub["FVC"]).astype(int)

submission = sub[["Patient_Week", "FVC", "Confidence"]].copy()

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["Patient_Week", "FVC", "Confidence"]
assert submission["Patient_Week"].is_unique

submission.head()



## === cell 5
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Shape:", submission.shape)
print(submission.head())
print(submission.tail())
