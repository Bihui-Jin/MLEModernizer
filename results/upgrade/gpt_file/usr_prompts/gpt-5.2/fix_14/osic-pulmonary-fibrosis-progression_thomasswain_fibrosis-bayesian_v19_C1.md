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

-6.896

# 6. Current score

-10.81761

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.04539) has done: 'I fix the submission-building bug caused by filling NaNs with a method object instead of per-row baseline FVC values, which currently makes the `FVC_pred` column contain non-numeric objects and crashes on `.astype(float)`. I replace that `.fillna(test.set_index(...).to_dict().get)` with a proper merge against test baseline FVC and then fill remaining missing predictions with that numeric baseline. I also ensure `PatientID` is added to the prediction template (it was missing), which can otherwise break prediction indexing in some runs. These changes are execution/stability fixes and keep the modeling logic intact while producing a valid `submission.csv`.'
- What this solution (achieved -8.91967) has done: 'Your current score is below the target (gap = -8.04539 − (-6.896) = -1.14939), so we should cautiously improve it without changing the model structure. The biggest low-risk gain for this metric is calibrating the submitted `Confidence` (sigma): your code currently multiplies class sigma by 1.5, which usually over-penalizes via `-log(sigma)` when residuals aren’t that large. I keep the same per-patient linear fit and class-based sigma estimation, but remove that extra 1.5 inflation and instead apply a small, global calibration factor learned from training data to better match the Laplace metric. This is minimal, fast, preserves core logic, and directly targets the evaluation formula.'
- What this solution (achieved -8.91967) has done: 'We keep your per-patient closed-form Bayesian linear regression exactly as-is and focus only on confidence (sigma) calibration, since your current score is below the target and the Laplace metric is very sensitive to sigma. The current calibration searches a tiny grid and optimizes on the same in-sample residuals, which can choose an overly optimistic sigma and hurt the public LB; we instead pick a more stable calibration by doing patient-wise CV on the training set and selecting the sigma scale that maximizes the CV Laplace score. This preserves the modeling logic (same a_hat/b_hat, same class sigma estimate), only changing the single global `SIGMA_CALIBRATION` value in a more reliable way. Finally, we keep submission formatting identical and ensure Confidence is clipped to >=70 as required.'
- What this solution (achieved -8.99286) has done: 'Your current score (-8.91967) is below the target (-6.896), so we should make a small, metric-aligned improvement without changing the per-patient Bayesian linear regression itself. The biggest low-risk lever is the global confidence scaling: instead of picking `SIGMA_CALIBRATION` from a very narrow grid, we run the same patient-wise CV selection but on a slightly wider (still small) scale range and with one refinement step around the best value, which tends to better match the Laplace metric’s tradeoff. We keep the same `a_hat/b_hat`, same class-wise sigma estimation, and only adjust the single global multiplicative sigma calibration more robustly. Submission writing, column names, and clipping to `>=70` remain unchanged.'
- What this solution (achieved -8.99286) has done: 'Your score is below the target (gap = -8.99286 − (-6.896) = -2.09686), so we should make a small, metric-aligned improvement without altering the per-patient Bayesian linear regression core. The most impactful minimal change here is to calibrate `Confidence` more robustly: your current CV uses all weeks, but Kaggle scoring uses only the last 3 visits per patient, so the sigma scale selected is misaligned and tends to underperform on LB. I keep the exact same model and sigma-per-class computation, but change the CV selection to optimize the Laplace metric on the last-3 measurements only (patient-wise), and slightly widen the refine neighborhood for stability. Submission building and clipping semantics stay identical, and it still write a valid `submission.csv`.'
- What this solution (achieved -10.81302) has done: 'The crash comes from fitting `a_hat/b_hat` arrays sized only to the fold’s *seen* PatientIDs, while the validation template still contains global-encoded PatientIDs that can exceed that size. I fix this by sizing the coefficient arrays using the global `le_id` cardinality (so indexing is always valid) and by safely skipping patients not present in the fold’s `base` table. This is a correctness/stability fix that keeps your closed-form per-patient Bayesian linear regression and sigma-per-class logic unchanged, and it let the CV sigma calibration and submission generation run end-to-end. I also make the template generator take an explicit encoder to avoid any accidental mismatch, while keeping the same template weeks and downstream merge logic.'
- What this solution (achieved -10.81304) has done: 'Your current score (-10.81302) is well below the target (-6.896), so we should improve it with the smallest metric-aligned change while keeping your per-patient Bayesian linear regression intact. The biggest issue is that you are evaluating/predicting FVC using raw `Weeks`, even though your priors (`FVC_base`, `Weeks_base`) are defined at each patient’s baseline week; this mismatch hurts both slope/intercept estimation and predictions. I minimally change the model to use `t = Weeks - Weeks_base` (time since baseline) everywhere in fitting and prediction, while leaving the same closed-form solve, priors, sigma-per-class logic, and CV calibration flow. This keeps semantics consistent with the competition setup (baseline at Week 0 per patient in test) and typically yields a sizable uplift without altering the core approach.'
- What this solution (achieved -10.81304) has done: 'We keep your per-patient Bayesian linear regression and the “last-3 only” CV calibration framework intact, but fix a key issue: you calibrate `SIGMA_CALIBRATION` via CV and then never refit the final model using the full training data with that chosen calibration (you currently predict test with a model fit *before* calibration). We therefore (1) run the CV calibration first, then (2) refit the model on all of `train`, then (3) create test predictions using the refit model and calibrated sigma. This is a minimal change that directly targets the Laplace metric tradeoff (via better confidence calibration actually applied to the final model) and should move your score upward toward the target.'
- What this solution (achieved -10.81761) has done: 'We keep your per-patient closed-form Bayesian linear regression exactly as-is and focus on one minimal, metric-aligned lever: the confidence (sigma) used in the Laplace log-likelihood. Right now `sigma_class` is estimated from raw residuals, but those residuals include large outliers and also include patient-week predictions for weeks outside the observed range (because `model_predict` uses a full template), both of which distort sigma and hurt the metric. We compute `sigma_class` only on *observed training rows* (not template rows) and use a *robust* estimate (MAD -> sigma) with clipping, which usually improves the LLL by avoiding overly large sigma driven by outliers while keeping core logic identical. Everything else (CV calibration, refit-on-full-train, submission merge/format) remains the same and still produces `submission.csv`.'
- What this solution (achieved -10.81761) has done: 'We keep your per-patient closed-form Bayesian linear regression and the whole CV-calibration flow intact, but make one metric-aligned adjustment that tends to improve LB with minimal risk: calibrate the confidence scale using a patient-wise CV that scores only the last-3 measurements (as you already do) but using a more stable fold assignment (deterministic KFold on unique patients) and slightly expanded calibration range up to 2.0 (your current upper bound 1.5 can force sigmas too small if residuals are larger on test, hurting the log term tradeoff). This change only affects the single scalar `SIGMA_CALIBRATION` and preserves all modeling/prediction semantics. We also ensure `sigma` in the final submission is clipped to the competition’s effective range (>=70) before writing, without changing the predicted FVCs. The script still runs end-to-end and writes `submission.csv` with the required columns.'
- What this solution (achieved -10.81761) has done: 'We keep your per-patient Bayesian linear regression exactly the same, and focus on the one lever that directly affects the Laplace metric: the submitted `Confidence` (sigma). Right now `sigma_class` is derived from residuals of a model fit with a fixed `sigma_global=150`, which can miscalibrate uncertainty; instead we compute `sigma_class` from per-class residual dispersion using a log-likelihood–aligned objective by choosing a single `SIGMA_GLOBAL` via the same patient-wise CV (last-3 only) you already use for `SIGMA_CALIBRATION`. This is minimal (no architecture/loop changes), fast (closed-form, small grid), and typically improves both the CV-selected sigma scale and the final LB score. Finally, we refit on full train with the chosen `SIGMA_GLOBAL` and `SIGMA_CALIBRATION`, then write a valid `submission.csv` with `Confidence` clipped to `[70, 1000]` as required.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
train_raw = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

train.drop(train[train.Patient == "ID00197637202246865691526"].index, inplace=True)

all_patients = pd.concat([train["Patient"], test["Patient"]], axis=0, ignore_index=True)
le_id = LabelEncoder()
le_id.fit(all_patients)

train["PatientID"] = le_id.transform(train["Patient"])
test["PatientID"] = le_id.transform(test["Patient"])

train.head()




## === cell 1
def add_baselines(data: pd.DataFrame) -> pd.DataFrame:
    aux = data[["Patient", "Weeks"]].groupby("Patient").min().reset_index()
    aux = pd.merge(
        aux, data[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
    )
    aux = aux.groupby("Patient").mean(numeric_only=True).reset_index()
    aux["Weeks"] = aux["Weeks"].astype(int)
    aux["FVC"] = aux["FVC"].astype(int)
    data = pd.merge(data, aux, how="left", on="Patient", suffixes=("", "_base"))
    return data


train = add_baselines(train)
test = add_baselines(test)
train.head()




## === cell 2
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


train["Class"] = train.apply(patient_class, axis=1)
test["Class"] = test.apply(patient_class, axis=1)

test.head()



## === cell 3
PatientID = train["Patient"].values
fvc_b = train.groupby("Patient").first()["FVC_base"]
fvc_b.values[:5]




## === cell 4
def model_fit(data, examine=True, n_patients_global=None, sigma_global=150.0):
    tau_a = 1000.0
    tau_b = 5.0
    lam_a = 1.0 / (tau_a**2)
    lam_b = 1.0 / (tau_b**2)

    patients = data["PatientID"].unique()

    if n_patients_global is None:
        n_patients = int(data["PatientID"].max()) + 1
    else:
        n_patients = int(n_patients_global)

    a_hat = np.full(n_patients, np.nan, dtype=np.float64)
    b_hat = np.full(n_patients, np.nan, dtype=np.float64)

    base = data.groupby("PatientID").first()[["FVC_base", "Weeks_base", "Class"]].copy()

    sigma_global = float(sigma_global)

    for pid in patients:
        if pid not in base.index:
            continue

        dfp = data[data["PatientID"] == pid]
        x = dfp["Weeks"].values.astype(np.float64) - float(base.loc[pid, "Weeks_base"])
        y = dfp["FVC"].values.astype(np.float64)

        if len(dfp) < 2:
            a_hat[pid] = float(base.loc[pid, "FVC_base"])
            b_hat[pid] = -4.0  # align to original mu_b mean
            continue

        X = np.vstack([np.ones_like(x), x]).T  # (n,2)
        XtX = X.T @ X
        P = np.array([[lam_a, 0.0], [0.0, lam_b]])
        mu0 = np.array([float(base.loc[pid, "FVC_base"]), -4.0])  # prior mean at t=0

        s2 = sigma_global**2
        A = XtX / s2 + P
        rhs = (X.T @ y) / s2 + (P @ mu0)
        beta = np.linalg.solve(A, rhs)
        a_hat[pid], b_hat[pid] = beta[0], beta[1]

    df_obs = data[["PatientID", "Weeks", "FVC", "Class"]].copy()
    weeks_base_map = base["Weeks_base"].to_dict()
    t_base = df_obs["PatientID"].map(weeks_base_map).astype(np.float64).values
    x_all = df_obs["Weeks"].values.astype(np.float64) - t_base
    pid_all = df_obs["PatientID"].values.astype(int)

    pred = a_hat[pid_all] + b_hat[pid_all] * x_all
    resid = (df_obs["FVC"].astype(np.float64).values - pred).astype(np.float64)

    sigma_class = np.full(6, 150.0, dtype=np.float64)
    for c in range(6):
        r = resid[df_obs["Class"].values.astype(int) == c]
        r = r[np.isfinite(r)]
        if len(r) >= 5:
            med = np.median(r)
            mad = np.median(np.abs(r - med))
            robust_sigma = 1.4826 * mad  # consistent with normal std
            if np.isfinite(robust_sigma) and robust_sigma > 0:
                sigma_class[c] = float(robust_sigma)
            else:
                sigma_class[c] = float(np.std(r, ddof=1))
    sigma_class = np.clip(sigma_class, 70.0, 1000.0)

    model = {
        "a_hat": a_hat,
        "b_hat": b_hat,
        "sigma_class": sigma_class,
        "weeks_base_by_pid": base["Weeks_base"].astype(np.float64).to_dict(),
        "sigma_global": sigma_global,
    }
    trace = None

    if examine:
        print(
            "Closed-form per-patient Bayesian linear regression fitted (no MCMC trace)."
        )
        print("sigma_global used:", sigma_global)
        print("Estimated class sigmas:", sigma_class)

    return model, trace




## === cell 5
def generate_template(data, encoder):
    pred_template = []
    for patient in data["Patient"].unique():
        df = pd.DataFrame(columns=["PatientID", "Weeks", "Patient", "Class"])
        df["Weeks"] = np.arange(-12, 134)
        df["Patient"] = patient
        df["Class"] = data.loc[data["Patient"] == patient, "Class"].max()
        pred_template.append(df)
    pred_template = pd.concat(pred_template, ignore_index=True)
    pred_template["PatientID"] = encoder.transform(pred_template["Patient"])
    return pred_template




## === cell 6
template_train_test = generate_template(test, le_id)
template_train_test.head()



## === cell 7
SIGMA_CALIBRATION = 1.0  # set after fitting using training residuals (see later cell)
SIGMA_GLOBAL_FINAL = 150.0  # set via CV below


def model_predict(model, trace, template, sigma_calibration=1.0):
    a_hat = model["a_hat"]
    b_hat = model["b_hat"]
    sigma_class = model["sigma_class"]

    df = pd.DataFrame(columns=["Patient", "Weeks", "FVC_pred", "sigma"])
    df["Patient"] = template["Patient"].values
    df["Weeks"] = template["Weeks"].values.astype(int)

    pid = template["PatientID"].values.astype(int)
    weeks = template["Weeks"].values.astype(float)

    wb_map = model.get("weeks_base_by_pid", {})
    weeks_base = np.array([wb_map.get(int(p), 0.0) for p in pid], dtype=np.float64)
    t = weeks - weeks_base

    df["FVC_pred"] = a_hat[pid] + b_hat[pid] * t

    cls = template["Class"].values.astype(int)
    df["sigma"] = sigma_class[cls].astype(np.float64) * float(sigma_calibration)

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
    for i, patient in enumerate(data["Patient"].unique()):
        ax = axes[i // 3, i % 3]
        df = data[data["Patient"] == patient].sort_values("Weeks")
        x = df["Weeks"]
        ax.set_title(patient)
        ax.plot(x, df["FVC_true"], "o")
        ax.plot(x, df["FVC_pred"])
        sns.regplot(x=x, y=df["FVC_true"], ax=ax, ci=None, line_kws={"color": "red"})
        ax.fill_between(x, df["FVC_inf"], df["FVC_sup"], alpha=0.5, color="#ffcd3c")
        ax.set_ylabel("FVC")
    axes[n // 3, n % 3].plot()




## === cell 9
def evaluate_predictions(df, use_only_last_3_measures=False, examine=False):
    if use_only_last_3_measures:
        y = df.dropna().groupby("Patient").tail(3).copy()
    else:
        y = df.dropna().copy()

    sigma_c = y["sigma"].values.astype(float)
    sigma_c[sigma_c < 70] = 70
    delta = (y["FVC_pred"] - y["FVC_true"]).abs().astype(float)
    delta[delta > 1000] = 1000
    lll = -np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

    y["sigma_c"] = y["sigma"].astype(float)
    y.loc[y["sigma_c"] < 70, "sigma_c"] = 70
    y["delta_c"] = (y["FVC_pred"] - y["FVC_true"]).abs().astype(float)
    y.loc[y["delta_c"] > 1000, "delta_c"] = 1000
    y["main_loss"] = y["delta_c"] / y["sigma_c"]
    if examine:
        plt.hist(y["main_loss"], bins=100)

    return float(lll.mean())




## === cell 10
def evaluation_cycle(train, valid, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model, trace = model_fit(
        train,
        examine=examine_trace,
        n_patients_global=len(le_id.classes_),
        sigma_global=SIGMA_GLOBAL_FINAL,
    )

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train, le_id)
    pred_train = model_predict(
        model, trace, template_train, sigma_calibration=SIGMA_CALIBRATION
    )
    if examine_preds:
        examine_predictions(pred_train)
    lll_train = evaluate_predictions(pred_train)

    pred_valid = None
    lll_valid = None
    if valid is not None:
        print("Examine true vs predictions for validation data ...")
        template_valid = generate_template(valid, le_id)
        pred_valid = model_predict(
            model, trace, template_valid, sigma_calibration=SIGMA_CALIBRATION
        )
        if examine_preds:
            examine_predictions(pred_valid)
        lll_valid = evaluate_predictions(pred_valid)
    return pred_train, pred_valid, lll_train, lll_valid




## === cell 11
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




## === cell 12
def _laplace_lll_from_arrays(delta, sigma):
    sigma_c = np.clip(sigma, 70.0, 1000.0)
    delta_c = np.minimum(delta, 1000.0)
    return float(
        np.mean(-np.sqrt(2.0) * delta_c / sigma_c - np.log(np.sqrt(2.0) * sigma_c))
    )


def _cv_sigma_calibration_last3_refit(
    train_df, scales, sigma_global, n_folds=5, seed=RANDOM_SEED
):
    patients = np.array(sorted(train_df["Patient"].unique().tolist()))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    scores = []
    for s in scales:
        fold_scores = []
        for tr_idx, va_idx in kf.split(patients):
            tr_patients = patients[tr_idx]
            va_patients = patients[va_idx]

            df_tr = train_df[train_df["Patient"].isin(tr_patients)].copy()
            df_va = train_df[train_df["Patient"].isin(va_patients)].copy()

            m, t = model_fit(
                df_tr,
                examine=False,
                n_patients_global=len(le_id.classes_),
                sigma_global=float(sigma_global),
            )

            template_va = generate_template(df_va, le_id)
            pred_va = model_predict(m, t, template_va, sigma_calibration=float(s))

            obs = (
                pred_va.dropna()[["Patient", "Weeks", "FVC_pred", "FVC_true", "sigma"]]
                .sort_values(["Patient", "Weeks"])
                .groupby("Patient")
                .tail(3)
                .copy()
            )
            if len(obs) == 0:
                continue

            delta = np.abs(
                obs["FVC_pred"].astype(np.float64).values
                - obs["FVC_true"].astype(np.float64).values
            )
            sigma = obs["sigma"].astype(np.float64).values
            fold_scores.append(_laplace_lll_from_arrays(delta, sigma))

        scores.append(float(np.mean(fold_scores)) if len(fold_scores) else -1e9)

    return np.array(scores, dtype=np.float64)


sigma_global_grid = np.array([120.0, 150.0, 180.0, 220.0], dtype=np.float64)
sigma_cal_grid_for_global = np.array([0.85, 1.00, 1.15, 1.30, 1.50], dtype=np.float64)

best_global = None
best_global_score = -1e18
best_global_sigma_cal = None

for sg in sigma_global_grid:
    cv_scores_tmp = _cv_sigma_calibration_last3_refit(
        train,
        scales=sigma_cal_grid_for_global,
        sigma_global=float(sg),
        n_folds=5,
        seed=RANDOM_SEED,
    )
    idx = int(np.argmax(cv_scores_tmp))
    score = float(cv_scores_tmp[idx])
    if score > best_global_score:
        best_global_score = score
        best_global = float(sg)
        best_global_sigma_cal = float(sigma_cal_grid_for_global[idx])

SIGMA_GLOBAL_FINAL = float(best_global)

print("Sigma_global grid:", sigma_global_grid)
print(
    f"Chosen SIGMA_GLOBAL_FINAL={SIGMA_GLOBAL_FINAL:.1f} (CV metric={best_global_score:.6f})"
)

coarse_scales = np.array(
    [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.65, 1.80, 2.00],
    dtype=np.float64,
)
cv_scores_coarse = _cv_sigma_calibration_last3_refit(
    train,
    scales=coarse_scales,
    sigma_global=SIGMA_GLOBAL_FINAL,
    n_folds=5,
    seed=RANDOM_SEED,
)
best_idx = int(np.argmax(cv_scores_coarse))
best_coarse = float(coarse_scales[best_idx])

refine_scales = np.round(
    np.arange(best_coarse - 0.20, best_coarse + 0.2001, 0.025), 3
).astype(np.float64)
refine_scales = refine_scales[(refine_scales >= 0.50) & (refine_scales <= 2.00)]
cv_scores_refine = _cv_sigma_calibration_last3_refit(
    train,
    scales=refine_scales,
    sigma_global=SIGMA_GLOBAL_FINAL,
    n_folds=5,
    seed=RANDOM_SEED,
)

best_ref_idx = int(np.argmax(cv_scores_refine))
SIGMA_CALIBRATION = float(refine_scales[best_ref_idx])

print("Sigma calibration coarse grid:", coarse_scales)
print("CV mean metric per coarse scale:", np.round(cv_scores_coarse, 6))
print("Sigma calibration refined grid:", refine_scales)
print("CV mean metric per refined scale:", np.round(cv_scores_refine, 6))
print(
    f"Chosen SIGMA_CALIBRATION={SIGMA_CALIBRATION:.3f} (best CV metric={cv_scores_refine.max():.6f})"
)

print("\nRefit final model on full training data ...")
model, trace = model_fit(
    train,
    examine=False,
    n_patients_global=len(le_id.classes_),
    sigma_global=SIGMA_GLOBAL_FINAL,
)

print("Make predictions for test data ...")
template_test = generate_template(test, le_id)
pred_test = model_predict(
    model, trace, template_test, sigma_calibration=SIGMA_CALIBRATION
)

sub = sample_sub[["Patient_Week"]].copy()
sub[["Patient", "Weeks"]] = sub["Patient_Week"].str.split("_", expand=True)
sub["Weeks"] = sub["Weeks"].astype(int)

pred_map = pred_test[["Patient", "Weeks", "FVC_pred", "sigma"]].copy()
pred_map["Weeks"] = pred_map["Weeks"].astype(int)
sub = sub.merge(pred_map, how="left", on=["Patient", "Weeks"])

sub = sub.merge(
    test[["Patient", "FVC"]].rename(columns={"FVC": "FVC_baseline"}),
    how="left",
    on="Patient",
)
sub["FVC_pred"] = sub["FVC_pred"].fillna(sub["FVC_baseline"])
sub["sigma"] = sub["sigma"].fillna(150.0)

final = pd.DataFrame(
    {
        "Patient_Week": sub["Patient_Week"],
        "FVC": pd.to_numeric(sub["FVC_pred"], errors="coerce").astype(float),
        "Confidence": pd.to_numeric(sub["sigma"], errors="coerce").astype(float),
    }
)

final["Confidence"] = final["Confidence"].clip(lower=70.0, upper=1000.0)

final.to_csv("submission.csv", index=False)
print(final.shape)
final.head()
