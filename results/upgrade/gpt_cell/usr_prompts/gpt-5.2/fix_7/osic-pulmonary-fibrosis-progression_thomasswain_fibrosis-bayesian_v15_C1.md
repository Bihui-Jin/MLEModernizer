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

-14.25837

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.81761) has done: 'Diagnosis: Cell 15 crashes because `pm` is `None` (PyMC3 failed to import earlier due to NumPy 1.26 incompatibility with `pymc3==3.11.x`), and the cell explicitly raises an `ImportError` which halts execution. Since the environment cannot run the Bayesian model, the minimal way to unblock the notebook without redesigning earlier cells is to avoid raising and instead produce a deterministic, schema-correct `submission.csv` using already-available `train/test` tabular data. The submission must contain exactly the `Patient_Week`, `FVC`, and `Confidence` columns with the same `Patient_Week` keys as `sample_submission.csv`. This patch keeps I/O paths unchanged and only modifies cell 15.

Patch summary: Replace the hard failure (`raise ImportError`) with a fallback submission generator when `pm is None`. The fallback merges `sample_submission.csv` with per-patient baseline FVC from `train` (already computed via `add_baselines`), fills missing with global median baseline, and uses a constant confidence (70) to satisfy the competition constraint and submission format. When `pm` is available, the original model fitting and prediction path remains unchanged.

Updated cells: (cell 15 only)

Compatibility notes for cell k+1: Cell 15 is the last provided cell; the patch preserves the creation of `final` and writing `submission.csv` with the expected columns and row count, so any later cells expecting `final` or the output file still work.

Assumptions: `../input/osic-pulmonary-fibrosis-progression/sample_submission.csv` exists (it does per paths provided). `train` contains `Patient`, `FVC_base` from earlier cells (it does via `add_baselines`). The required `Patient_Week` keys are exactly those in `sample_submission.csv`.'
- What this solution (achieved -13.64614) has done: 'Your current score is far below the target, so we should improve it with minimal, legitimate changes while keeping the same overall “baseline-only” fallback logic (since PyMC3 can’t run here). The biggest issue is that the fallback predicts a constant FVC for all weeks per patient and uses a too-small fixed confidence (70), which gets heavily penalized for large week offsets. I keep the fallback structure but (1) learn a simple per-patient linear trend (slope) from training data using ordinary least squares on each patient’s (Weeks, FVC) history, and (2) set Confidence to a more conservative value derived from that patient’s residual error (clipped to ≥70). This preserves your core approach (tabular-only fallback) but aligns predictions better with the metric and should move the score substantially closer to the target.'
- What this solution (achieved -14.25837) has done: 'I keep your current “tabular-only fallback when `pm is None`” logic, but make two minimal, metric-aligned fixes to move the score upward toward the target: (1) shrink extreme per-patient slopes toward a class-level mean slope to reduce large week-offset errors (which are heavily penalized when Confidence is clipped), and (2) calibrate Confidence per patient-week by combining residual noise with an uncertainty term that grows with distance from that patient’s observed weeks. This preserves the same linear-per-patient prediction core, but reduces catastrophic deltas on far weeks and avoids being overconfident. I also ensure we only fit slopes on `train_raw` (true training history) and not on the concatenated `train` that includes test rows, avoiding any unintended leakage patterns. The submission format and paths remain unchanged, and it still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

try:
    import pymc3 as pm
except Exception:
    pm = None

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        break



## === cell 1
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train_raw = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

train.drop(train[train.Patient == "ID00197637202246865691526"].index, inplace=True)

train = pd.concat([train, test], axis=0, ignore_index=True).drop_duplicates()
le_id = LabelEncoder()
train["PatientID"] = le_id.fit_transform(train["Patient"])

train.head()




## === cell 2
def add_baselines(data):
    aux = data[["Patient", "Weeks"]].groupby("Patient").min().reset_index()
    aux = pd.merge(
        aux, data[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
    )
    aux = aux.groupby("Patient").mean().reset_index()
    aux["Weeks"] = aux["Weeks"].astype(int)
    aux["FVC"] = aux["FVC"].astype(int)
    data = pd.merge(data, aux, how="left", on="Patient", suffixes=("", "_base"))
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


train["Class"] = train.apply(patient_class, axis=1)
test["Class"] = test.apply(patient_class, axis=1)
test.head()
train.loc[train["Patient"] == "ID00007637202177411956430"]["Class"].max()



## === cell 4
PatientID = train["Patient"].values
fvc_b = train.groupby("Patient").first()["FVC_base"]
fvc_b.values




## === cell 5
def model_fit(data, examine=True):
    n_patients = data["Patient"].nunique()
    FVC_obs = data["FVC"].values
    Weeks = data["Weeks"].values
    PatientID = data["PatientID"].values
    patient_class = data["Class"].values
    FVC_b = data.groupby("PatientID").first()["FVC_base"]
    w_b = data.groupby("PatientID").first()["Weeks_base"]

    with pm.Model() as model:
        FVC_obs_shared = pm.Data("FVC_obs_shared", FVC_obs)
        Weeks_shared = pm.Data("Weeks_shared", Weeks)
        PatientID_shared = pm.Data("PatientID_shared", PatientID)
        patient_class_shared = pm.Data("patient_class_shared", patient_class)
        FVC_b_shared = pm.Data("FVC_b_shared", FVC_b)
        w_b_shared = pm.Data("w_b_shared", w_b)

        mu_a_base = pm.Normal("mu_a_base", mu=0.0, sigma=100)
        mu_a = FVC_b_shared + mu_a_base * w_b_shared

        sigma_a = pm.HalfNormal("sigma_a", 1000.0)
        mu_b = pm.Normal("mu_b", mu=-4.0, sigma=1)
        sigma_b = pm.HalfNormal("sigma_b", 5.0)

        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=n_patients)
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=n_patients)

        sigma = pm.HalfNormal("sigma", 150.0, shape=6)

        FVC_est = a[PatientID_shared] + b[PatientID_shared] * Weeks_shared

        FVC_like = pm.Normal(
            "FVC_like",
            mu=FVC_est,
            sigma=sigma[patient_class_shared],
            observed=FVC_obs_shared,
        )

        trace = pm.sample(4000, tune=4000, target_accept=0.9, init="adapt_diag")

    if examine:
        with model:
            pm.traceplot(trace)

    return model, trace




## === cell 6
def generate_template(data):
    pred_template = []
    for i, patient in enumerate(data["Patient"].unique()):
        df = pd.DataFrame(columns=["PatientID", "Weeks", "Patient", "Class"])
        df["Weeks"] = np.arange(-12, 134)
        df["Patient"] = patient
        df["Class"] = data.loc[data["Patient"] == patient]["Class"].max()
        pred_template.append(df)
    pred_template = pd.concat(pred_template, ignore_index=True)
    pred_template["PatientID"] = le_id.transform(pred_template["Patient"])
    return pred_template




## === cell 7
template_train_test = generate_template(test)
template_train_test.head()




## === cell 8
def model_predict(model, trace, template):
    with model:
        pm.set_data(
            {
                "PatientID_shared": template["PatientID"].values.astype(int),
                "Weeks_shared": template["Weeks"].values.astype(int),
                "FVC_obs_shared": np.zeros(len(template)).astype(int),
                "patient_class_shared": template["Class"].values.astype(int),
            }
        )
        post_pred = pm.sample_posterior_predictive(trace)
    df = pd.DataFrame(columns=["Patient", "Weeks", "FVC_pred", "sigma"])
    df["Patient"] = le_id.inverse_transform(template["PatientID"])
    df["Weeks"] = template["Weeks"]
    df["FVC_pred"] = post_pred["FVC_like"].T.mean(axis=1)
    df["sigma"] = post_pred["FVC_like"].T.std(axis=1)
    df["FVC_inf"] = df["FVC_pred"] - df["sigma"]
    df["FVC_sup"] = df["FVC_pred"] + df["sigma"]
    df = pd.merge(
        df, train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
    )
    df = df.rename(columns={"FVC": "FVC_true"})
    return df




## === cell 9
def examine_predictions(data):
    n = (data["Patient"].nunique()) + 1
    f, axes = plt.subplots((n // 3) + 1, 3, figsize=(15, 5 * ((n // 3) + 1)))
    for i, patient in enumerate(data["Patient"].unique()):
        ax = axes[i // 3, i % 3]
        df = data[data["Patient"] == patient]
        x = df["Weeks"]
        ax.set_title(patient)
        ax.plot(x, df["FVC_true"], "o")
        ax.plot(x, df["FVC_pred"])
        ax = sns.regplot(x, df["FVC_true"], ax=ax, ci=None, line_kws={"color": "red"})
        ax.fill_between(x, df["FVC_inf"], df["FVC_sup"], alpha=0.5, color="#ffcd3c")
        ax.set_ylabel("FVC")
    axes[n // 3, n % 3].plot()




## === cell 10
def evaluate_predictions(df, use_only_last_3_measures=False, examine=False):
    if use_only_last_3_measures:
        y = df.dropna().groupby("Patient").tail(3)
    else:
        y = df.dropna()

    rmse = ((y["FVC_pred"] - y["FVC_true"]) ** 2).mean() ** (1 / 2)
    mae = (y["FVC_pred"] - y["FVC_true"]).abs()
    mae_mean = (np.sqrt((y["FVC_pred"] - y["FVC_true"]) ** 2)).mean()
    mae_sd = (np.sqrt((y["FVC_pred"] - y["FVC_true"]) ** 2)).mean()
    mae_max = (np.sqrt((y["FVC_pred"] - y["FVC_true"]) ** 2)).mean()
    sigma_c = y["sigma"].values * 1.5
    sigma_c[sigma_c < 70] = 70
    delta = (y["FVC_pred"] - y["FVC_true"]).abs()
    delta[delta > 1000] = 1000
    lll = -np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

    y["sigma_c"] = y["sigma"]
    y["sigma_c"].values[y["sigma_c"].values < 70] = 70
    y["delta_c"] = (y["FVC_pred"] - y["FVC_true"]).abs()
    y["delta_c"].values[y["delta_c"].values > 1000] = 1000
    y["main_loss"] = y["delta_c"] / y["sigma_c"]
    if examine:
        plt.hist(y["main_loss"], bins=100)

    return lll.mean()




## === cell 11
def evaluation_cycle(train, valid, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model, trace = model_fit(train, examine=examine_trace)
    print("")

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train)
    template_train.head()
    pred_train = model_predict(model, trace, template_train)
    if examine_preds:
        examine_predictions(pred_train)
    lll_train = evaluate_predictions(pred_train)

    if valid is not None:
        print("Examine true vs predictions for validation data ...")
        template_valid = generate_template(valid)
        pred_valid = model_predict(model, trace, template_valid)
        if examine_preds:
            examine_predictions(pred_valid)
        lll_valid = evaluate_predictions(pred_valid)
    return pred_train, pred_valid, lll_train, lll_valid




## === cell 12
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



## === cell 13
if pm is None:
    sample = pd.read_csv(
        "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
    )

    pw = sample["Patient_Week"].str.split("_", n=1, expand=True)
    sample["Patient"] = pw[0]
    sample["Weeks"] = pw[1].astype(int)

    hist = train_raw[["Patient", "Weeks", "FVC"]].dropna().copy()
    hist["Weeks"] = hist["Weeks"].astype(float)
    hist["FVC"] = hist["FVC"].astype(float)

    patient_to_class = train[["Patient", "Class"]].drop_duplicates("Patient")

    def _fit_patient(g):
        x = g["Weeks"].values
        y = g["FVC"].values
        if len(g) < 2 or np.allclose(x.var(), 0.0):
            intercept = float(np.nanmean(y))
            slope = 0.0
            resid = y - intercept
            sigma_hat = (
                float(np.sqrt(np.mean(resid**2)))
                if np.isfinite(intercept)
                else float(np.nanstd(y, ddof=0))
            )
            wmin = float(np.nanmin(x))
            wmax = float(np.nanmax(x))
            x_mean = float(np.nanmean(x))
            return pd.Series(
                {
                    "intercept": intercept,
                    "slope": float(slope),
                    "sigma_hat": float(sigma_hat),
                    "wmin": wmin,
                    "wmax": wmax,
                    "x_mean": x_mean,
                }
            )

        x_mean = float(x.mean())
        y_mean = float(y.mean())
        denom = float(((x - x_mean) ** 2).sum())
        slope = (
            0.0 if denom <= 0 else float((((x - x_mean) * (y - y_mean)).sum()) / denom)
        )
        intercept = float(y_mean - slope * x_mean)
        resid = y - (intercept + slope * x)
        sigma_hat = float(np.sqrt(np.mean(resid**2)))  # RMSE as uncertainty proxy
        return pd.Series(
            {
                "intercept": float(intercept),
                "slope": float(slope),
                "sigma_hat": float(sigma_hat),
                "wmin": float(np.min(x)),
                "wmax": float(np.max(x)),
                "x_mean": float(x_mean),
            }
        )

    patient_params = (
        hist.groupby("Patient", sort=False).apply(_fit_patient).reset_index()
    )
    patient_params = patient_params.merge(patient_to_class, on="Patient", how="left")

    global_fvc = float(np.nanmedian(hist["FVC"].values)) if len(hist) else 2000.0
    global_sigma = (
        float(np.nanmedian(patient_params["sigma_hat"].values))
        if len(patient_params)
        else 250.0
    )
    if not np.isfinite(global_sigma) or global_sigma <= 0:
        global_sigma = 250.0

    class_slope_mean = patient_params.groupby("Class")["slope"].mean()
    global_slope_mean = (
        float(np.nanmean(patient_params["slope"].values))
        if len(patient_params)
        else 0.0
    )
    sample = sample.merge(patient_params, on="Patient", how="left")

    slope_raw = sample["slope"].astype(float)
    cls = sample["Class"]
    slope_prior = cls.map(class_slope_mean).astype(float)
    slope_prior = slope_prior.where(np.isfinite(slope_prior), global_slope_mean)

    alpha = 0.35
    slope_shrunk = (1.0 - alpha) * slope_raw + alpha * slope_prior
    slope_shrunk = slope_shrunk.where(np.isfinite(slope_shrunk), slope_prior)

    intercept = (
        sample["intercept"]
        .astype(float)
        .where(np.isfinite(sample["intercept"].astype(float)), global_fvc)
    )

    weeks_f = sample["Weeks"].astype(float)
    fvc_pred = intercept + slope_shrunk * weeks_f
    fvc_pred = fvc_pred.astype(float).where(np.isfinite(fvc_pred), global_fvc)

    sigma_hat = sample["sigma_hat"].astype(float)
    sigma_hat = sigma_hat.where(np.isfinite(sigma_hat) & (sigma_hat > 0), global_sigma)

    wmin = sample["wmin"].astype(float)
    wmax = sample["wmax"].astype(float)
    wmin = wmin.where(np.isfinite(wmin), weeks_f)
    wmax = wmax.where(np.isfinite(wmax), weeks_f)

    dist_outside = np.maximum(0.0, np.maximum(wmin - weeks_f, weeks_f - wmax))
    slope_scale = (
        slope_shrunk.abs()
        .astype(float)
        .where(np.isfinite(slope_shrunk.abs().astype(float)), 0.0)
    )
    extra_sigma = dist_outside * (0.35 * slope_scale + 12.0)

    conf = np.sqrt((1.15 * sigma_hat) ** 2 + extra_sigma**2)
    conf = conf.clip(lower=70.0)

    final = pd.DataFrame(
        {
            "Patient_Week": sample["Patient_Week"],
            "FVC": fvc_pred.values,
            "Confidence": conf.values,
        }
    )

    final = final[["Patient_Week", "FVC", "Confidence"]]
    final.to_csv("submission.csv", index=False)
    print(final.shape)
    final.head()
else:
    print("Fit model ...")
    model, trace = model_fit(train, examine=False)
    print("")

    print("Make predictions for test data ...")
    template_test = generate_template(test)
    template_test.head()
    pred_test = model_predict(model, trace, template_test)

    final = pd.DataFrame(columns=["Patient_Week", "FVC", "Confidence"])
    final["Patient_Week"] = pred_test["Patient"] + "_" + pred_test["Weeks"].astype(str)
    final["FVC"] = pred_test["FVC_pred"]
    final["Confidence"] = pred_test["sigma"]
    final.head()
    final.to_csv("submission.csv", index=False)
    print(final.shape)
    final.head()
