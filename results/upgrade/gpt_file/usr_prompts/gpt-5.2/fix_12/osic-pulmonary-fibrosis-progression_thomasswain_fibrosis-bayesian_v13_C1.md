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

-6.8592

# 6. Current score

-7.95879

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -13.3206) has done: 'I remove the failing PyMC3/Theano dependency (it’s incompatible with NumPy 1.26 here) and replace it with an equivalent hierarchical linear model fit using scikit-learn, keeping the same core intent: patient-specific intercept/slope for FVC vs Weeks and class-dependent noise. I also fix the missing `LabelEncoder` scope issues by ensuring it’s defined before use and that `PatientID` exists for every dataframe passed into fitting/prediction. Finally, I make the submission generation match `sample_submission.csv` exactly (same `Patient_Week` rows/order) and output a valid `submission.csv` with `FVC` and `Confidence` columns. This should run end-to-end within the time limit and produce a valid file for scoring.'
- What this solution (achieved -13.3206) has done: 'You’re currently under the target (gap = -13.3206 − (-6.8592) = -6.4614), so we need a legitimate score increase with minimal change. The main issue is that `generate_template(test)` assigns Weeks from -12..133 for every test patient, but the submission only scores the specific `Patient_Week` rows in `sample_submission.csv`; a mismatch here causes many `NaN` merges and forces you to fall back to baseline FVC with an arbitrary confidence, which hurts the metric. I minimally change the template generation for test-time to generate predictions exactly for the weeks present in `sample_submission.csv` per patient, ensuring every submission row gets a modeled prediction and a coherent sigma. I also compute confidence as a patient-level residual std (fallback to class/global), which typically improves calibration for this metric without changing the core linear-per-patient model.'
- What this solution (achieved -12.01677) has done: 'Your score gap is large (current -13.3206 vs target -6.8592; higher is better), and the biggest remaining low-risk improvement is to better match the competition’s evaluation setup: in test you only know a baseline FVC at Week=0, so using full per-patient histories from train to estimate per-patient slopes/intercepts for unseen test patients is effectively falling back to a global average and can be poorly calibrated. With minimal change, I keep your same per-patient linear regression core, but add a class-based prior (global intercept/slope per Sex+Smoking “Class”) and use that prior specifically for unseen patients (test) while still using patient-specific fits for seen patients (train). I also compute confidence from class-level residuals (still clipped at 70) and avoid the baseline-FVC fill fallback by ensuring every submission row gets a prediction deterministically. These changes usually improve both FVC predictions and sigma calibration for the Laplace log-likelihood without changing the overall modeling approach.'
- What this solution (achieved -8.0288) has done: 'Your current score is far below the target (gap ≈ -5.16), so we want a legitimate improvement with minimal disruption to your linear-per-patient/core prior logic. The biggest low-risk gain for this competition metric is better calibration of `Confidence` (sigma): instead of using only residual-based class sigma, we estimate sigma using the observed per-patient FVC variability around a straight-line fit across the whole training set (gives a more realistic uncertainty for unseen test patients), then apply a single multiplicative calibration factor derived from training to reduce systematic under/over-confidence. Additionally, we clip predicted FVC to a sane physiological range to avoid rare extreme extrapolations that get delta-capped at 1000 and still hurt the log term. These are small post-processing/calibration changes that preserve your model’s structure and keep the submission rows aligned exactly to `sample_submission.csv`.'
- What this solution (achieved -7.74131) has done: 'We keep your per-patient/class linear-regression core unchanged and focus on two small, metric-aligned calibration fixes to move the score up toward the target: (1) compute the sigma multiplier using an out-of-fold style calibration on patient-held-out splits (instead of in-sample train), which typically improves Confidence calibration for this Laplace metric; (2) ensure test-time predictions are anchored to each test patient’s known baseline by shifting the class-prior line so it exactly matches the provided baseline FVC at the baseline week (Weeks in `test.csv`), which reduces systematic bias without changing slopes/architecture. These are minimal post-processing/calibration changes and keep submission rows aligned exactly to `sample_submission.csv`. The script still runs end-to-end and writes `submission.csv` with the required columns.'
- What this solution (achieved -7.74131) has done: 'We keep your per-patient/class linear model exactly as-is and only fix one test-time post-processing bug that is currently injecting a huge constant bias: the baseline anchoring line `pred_test["FVC_pred"] = pred_test["FVC_pred"] + (FVC_base - FVC_pred)` collapses to `FVC_base` for all weeks, wiping out slope information and hurting the last-3-weeks predictions. We remove that redundant line and keep the later, correct anchoring step that shifts the class/prior line to match the known baseline at the baseline week using the appropriate slope (`b_used`). This is a minimal semantic change (no new model, no new training) and is expected to move your score upward toward the target by reducing systematic error while preserving the same calibrated sigma logic. The submission format and row alignment remain exactly matched to `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved -7.77481) has done: 'I fix the runtime `KeyError: 'Class'` by ensuring `model_predict()` preserves the `Class` column (and `PatientID`) in its returned dataframe, since cell 12 relies on it for baseline anchoring. I also make `generate_template()` explicitly assign `PatientID` for each patient before concatenation to avoid silent NaNs, and I keep all modeling logic (per-patient linear regression + class priors + OOF sigma calibration) unchanged. Finally, I add a small assertion right before writing the submission to guarantee the merge filled every `Patient_Week` and that the output CSV has the required columns and no missing predictions.'
- What this solution (achieved -8.22523) has done: 'We’re currently below the target (current -7.77481 vs target -6.8592, higher is better), so we want a small, legitimate improvement without changing the core per-patient/class linear model. The most leverage with minimal disruption for this metric is better `Confidence` calibration: your current sigma is constant across weeks per patient/class, but uncertainty should grow as we extrapolate away from the known baseline week. I keep your existing OOF sigma multiplier, and add a tiny, metric-aligned week-distance inflation term `sigma * (1 + k * |week - week_base|)` with `k` tuned via the same OOF procedure (patients held out) to maximize LLL on the last 3 measurements. This preserves the model structure and only adjusts confidence post-processing, then uses the chosen `k` at test-time and still writes a valid `submission.csv` matching `sample_submission.csv` rows exactly.'
- What this solution (achieved -8.22523) has done: 'We’re currently below the target (current -8.22523 vs target -6.8592; higher is better), so we want a small, legitimate uplift without changing the per-patient/class linear-regression core. The biggest low-risk issue I see is that your week-distance sigma inflation is anchored to each patient’s **minimum train week**, not the patient’s **baseline CT week (Week=0)** that the competition setup is centered around; switching that anchor typically improves confidence calibration on the last-3-weeks scoring. I change only the OOF calibration anchor map to use “week closest to 0” per patient (and keep test anchored to its provided baseline week), then re-run the same OOF search and apply it exactly as before. This keeps the model and training approach identical, but should move the score upward by better matching the metric’s uncertainty structure.'
- What this solution (achieved -7.95879) has done: 'We’re below the target (current -8.22523 vs target -6.8592; higher is better), so we want a small, legitimate uplift without changing your per-patient/class linear model. The biggest low-risk issue is inconsistent calibration: you inflate `sigma` by week-distance and then apply `sigma_mult_oof`, but the OOF multiplier was calibrated on *pre-inflation* sigma, so test-time Confidence becomes systematically too large and hurts the `-log(sigma)` term. I minimally change the OOF calibration to apply the same week-distance inflation inside the OOF loop, and then jointly choose the best `k` and `sigma_mult` on OOF predictions, matching your exact inference pipeline. Everything else (model fit/predict, baseline anchoring, submission row alignment) is preserved.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")

train = pd.read_csv(train_path)
train_raw = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

train.drop(train[train.Patient == "ID00197637202246865691526"].index, inplace=True)

all_patients = pd.concat(
    [train[["Patient"]], test[["Patient"]]], axis=0, ignore_index=True
).drop_duplicates()
le_id = LabelEncoder()
le_id.fit(all_patients["Patient"].values)

train["PatientID"] = le_id.transform(train["Patient"])
test["PatientID"] = le_id.transform(test["Patient"])

train.head()




## === cell 1
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

test.head()




## === cell 2
def model_fit(data, examine=True):
    required = {"Patient", "PatientID", "Weeks", "FVC", "Class"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"model_fit: missing columns {missing}")

    df = data.dropna(subset=["FVC", "Weeks", "PatientID", "Class"]).copy()
    df["PatientID"] = df["PatientID"].astype(int)
    df["Class"] = df["Class"].astype(int)

    a = {}
    b = {}
    resid_rows = []
    coef_rows = []
    sigma_by_patient = {}

    for pid, g in df.groupby("PatientID"):
        X = g[["Weeks"]].values
        y = g["FVC"].values

        if len(g) >= 2 and np.nanstd(g["Weeks"].values) > 0:
            lr = LinearRegression()
            lr.fit(X, y)
            a_i = float(lr.intercept_)
            b_i = float(lr.coef_[0])
        else:
            a_i = float(np.nanmean(y))
            b_i = 0.0

        a[pid] = a_i
        b[pid] = b_i

        cls_i = int(g["Class"].max())
        coef_rows.append({"Class": cls_i, "a": a_i, "b": b_i})

        yhat = a_i + b_i * g["Weeks"].values
        resid = y - yhat
        resid_rows.append(pd.DataFrame({"Class": g["Class"].values, "resid": resid}))

        sigma_by_patient[pid] = (
            float(np.nanstd(resid, ddof=1)) if len(resid) >= 2 else np.nan
        )

    resid_df = pd.concat(resid_rows, ignore_index=True)
    sigma_by_class = (
        resid_df.groupby("Class")["resid"].std(ddof=1).reindex(range(6)).values
    )
    global_sigma = float(np.nanstd(resid_df["resid"].values, ddof=1))

    sigma_by_class = np.where(np.isnan(sigma_by_class), global_sigma, sigma_by_class)
    sigma_by_class = np.maximum(sigma_by_class, 70.0)  # metric clips at 70 anyway

    pid_to_class = df.groupby("PatientID")["Class"].max().to_dict()
    sigma_by_patient_filled = {}
    for pid, s in sigma_by_patient.items():
        if np.isnan(s) or s <= 0:
            cls = int(pid_to_class.get(pid, 0))
            s = float(sigma_by_class[np.clip(cls, 0, 5)])
        sigma_by_patient_filled[int(pid)] = float(max(s, 70.0))

    coef_df = pd.DataFrame(coef_rows)
    class_a = coef_df.groupby("Class")["a"].mean().reindex(range(6)).values
    class_b = coef_df.groupby("Class")["b"].mean().reindex(range(6)).values
    global_a = float(np.nanmean(list(a.values()))) if len(a) else 1700.0
    global_b = float(np.nanmean(list(b.values()))) if len(b) else -4.0

    class_a = np.where(np.isnan(class_a), global_a, class_a).astype(float)
    class_b = np.where(np.isnan(class_b), global_b, class_b).astype(float)

    spread_by_class = (
        df.groupby("Class")["FVC"].std(ddof=1).reindex(range(6)).values.astype(float)
    )
    global_spread = float(np.nanstd(df["FVC"].values.astype(float), ddof=1))
    spread_by_class = np.where(
        np.isfinite(spread_by_class), spread_by_class, global_spread
    )
    spread_by_class = np.maximum(spread_by_class, 70.0)

    model = {
        "a": a,
        "b": b,
        "class_a": class_a,
        "class_b": class_b,
        "sigma_by_class": sigma_by_class.astype(float),
        "sigma_by_patient": sigma_by_patient_filled,
        "global_sigma": float(max(global_sigma, 70.0)),
        "global_a": float(global_a),
        "global_b": float(global_b),
        "seen_patient_ids": set(map(int, df["PatientID"].unique().tolist())),
        "spread_by_class": spread_by_class.astype(float),
        "global_spread": float(max(global_spread, 70.0)),
    }
    trace = None  # placeholder to preserve call signature

    if examine:
        some_pids = list(df["PatientID"].unique())[:6]
        f, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        for ax, pid in zip(axes, some_pids):
            g = df[df["PatientID"] == pid]
            ax.scatter(g["Weeks"], g["FVC"], s=10)
            w = np.linspace(g["Weeks"].min(), g["Weeks"].max(), 50)
            ax.plot(w, model["a"][pid] + model["b"][pid] * w)
            ax.set_title(le_id.inverse_transform([pid])[0])
            ax.set_ylabel("FVC")
            ax.set_xlabel("Weeks")
        plt.tight_layout()

    return model, trace




## === cell 3
pass




## === cell 4
def generate_template(data):
    pred_template = []
    for patient in data["Patient"].unique():
        cls_val = int(data.loc[data["Patient"] == patient, "Class"].max())
        pid_val = int(le_id.transform([patient])[0])

        df = pd.DataFrame(
            {
                "Weeks": np.arange(-12, 134, dtype=int),
                "Patient": patient,
                "Class": cls_val,
                "PatientID": pid_val,
            }
        )
        pred_template.append(df)

    pred_template = pd.concat(pred_template, ignore_index=True)
    pred_template["PatientID"] = pred_template["PatientID"].astype(int)
    pred_template["Class"] = pred_template["Class"].astype(int)
    pred_template["Weeks"] = pred_template["Weeks"].astype(int)
    return pred_template


def generate_template_from_submission(sample_submission_df, test_df):
    tmp = sample_submission_df[["Patient_Week"]].copy()
    tmp["Patient"] = tmp["Patient_Week"].str.split("_", n=1, expand=True)[0]
    tmp["Weeks"] = tmp["Patient_Week"].str.split("_", n=1, expand=True)[1].astype(int)

    tmp = tmp.merge(
        test_df[["Patient", "Class"]].drop_duplicates(), how="left", on="Patient"
    )
    tmp["Class"] = tmp["Class"].fillna(0).astype(int)

    tmp["PatientID"] = le_id.transform(tmp["Patient"])
    tmp["PatientID"] = tmp["PatientID"].astype(int)

    return tmp[["PatientID", "Weeks", "Patient", "Class"]].copy()




## === cell 5
template_train_test = generate_template(test)
template_train_test.head()



## === cell 6
pass




## === cell 7
def model_predict(model, trace, template):
    required = {"PatientID", "Weeks", "Class"}
    missing = required - set(template.columns)
    if missing:
        raise ValueError(f"model_predict: missing columns {missing}")

    pid = template["PatientID"].values.astype(int)
    weeks = template["Weeks"].values.astype(float)
    cls = template["Class"].values.astype(int)

    a_map = model["a"]
    b_map = model["b"]

    seen = model.get("seen_patient_ids", set())
    class_a = model.get("class_a", None)
    class_b = model.get("class_b", None)

    a = np.empty(len(pid), dtype=float)
    b = np.empty(len(pid), dtype=float)
    for i, (p, c) in enumerate(zip(pid, cls)):
        if int(p) in seen:
            a[i] = float(a_map.get(int(p), model["global_a"]))
            b[i] = float(b_map.get(int(p), model["global_b"]))
        else:
            c0 = int(np.clip(c, 0, 5))
            if class_a is not None and class_b is not None:
                a[i] = float(class_a[c0])
                b[i] = float(class_b[c0])
            else:
                a[i] = float(model["global_a"])
                b[i] = float(model["global_b"])

    fvc_pred = a + b * weeks
    fvc_pred = np.clip(fvc_pred, 500.0, 6000.0)

    sigma_patient_map = model.get("sigma_by_patient", {})
    sigma_pat = np.array(
        [sigma_patient_map.get(int(p), np.nan) for p in pid], dtype=float
    )

    spread_by_class = model.get("spread_by_class", None)
    if spread_by_class is not None:
        sigma_unseen_cls = spread_by_class[np.clip(cls, 0, 5)].astype(float)
    else:
        sigma_unseen_cls = model["sigma_by_class"][np.clip(cls, 0, 5)].astype(float)

    sigma_seen_cls = model["sigma_by_class"][np.clip(cls, 0, 5)].astype(float)
    is_seen = np.array([int(p) in seen for p in pid], dtype=bool)

    sigma_base = np.where(is_seen, sigma_seen_cls, sigma_unseen_cls)
    sigma = np.where(np.isfinite(sigma_pat), sigma_pat, sigma_base)

    sigma = np.where(np.isfinite(sigma), sigma, float(model.get("global_sigma", 200.0)))
    sigma = np.maximum(sigma, 70.0)

    df = pd.DataFrame(
        {
            "Patient": le_id.inverse_transform(pid),
            "PatientID": pid.astype(int),
            "Weeks": template["Weeks"].values.astype(int),
            "Class": cls.astype(int),
            "FVC_pred": fvc_pred.astype(float),
            "sigma": sigma.astype(float),
        }
    )

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
    axes = np.array(axes).reshape(-1, 3)
    for i, patient in enumerate(data["Patient"].unique()):
        ax = axes[i // 3, i % 3]
        dfp = data[data["Patient"] == patient].sort_values("Weeks")
        x = dfp["Weeks"].values
        ax.set_title(patient)
        ax.plot(x, dfp["FVC_true"], "o", markersize=3)
        ax.plot(x, dfp["FVC_pred"], linewidth=1)
        ax.fill_between(
            x, dfp["FVC_inf"].values, dfp["FVC_sup"].values, alpha=0.3, color="#ffcd3c"
        )
        ax.set_ylabel("FVC")
    plt.tight_layout()




## === cell 9
def evaluate_predictions(df, use_only_last_3_measures=True, examine=False):
    if use_only_last_3_measures:
        y = df.dropna(subset=["FVC_true"]).groupby("Patient").tail(3).copy()
    else:
        y = df.dropna(subset=["FVC_true"]).copy()

    sigma_c = y["sigma"].values.astype(float)
    sigma_c[sigma_c < 70] = 70
    delta = (y["FVC_pred"] - y["FVC_true"]).abs().values.astype(float)
    delta[delta > 1000] = 1000
    lll = -np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

    if examine:
        main_loss = delta / sigma_c
        plt.hist(main_loss, bins=100)

    return float(np.mean(lll))




## === cell 10
def evaluation_cycle(train_df, valid_df, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model, trace = model_fit(train_df, examine=examine_trace)
    print("")

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train_df)
    pred_train = model_predict(model, trace, template_train)
    if examine_preds:
        examine_predictions(pred_train)
    lll_train = evaluate_predictions(pred_train)

    pred_valid = None
    lll_valid = None
    if valid_df is not None and len(valid_df) > 0:
        print("Examine true vs predictions for validation data ...")
        template_valid = generate_template(valid_df)
        pred_valid = model_predict(model, trace, template_valid)
        if examine_preds:
            examine_predictions(pred_valid)
        lll_valid = evaluate_predictions(pred_valid)
    return pred_train, pred_valid, lll_train, lll_valid




## === cell 11
for fold in range(0):
    examine_lll = True

    all_patients = train["Patient"].unique()
    validation_patients = np.random.choice(
        all_patients, size=min(20, len(all_patients)), replace=False
    )
    df_valid = train[train["Patient"].isin(validation_patients)].copy()
    df_train = train[~train["Patient"].isin(validation_patients)].copy()

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
def calibrate_sigma_multiplier(pred_df, candidates):
    y = pred_df.dropna(subset=["FVC_true"]).groupby("Patient").tail(3).copy()

    base_sigma = y["sigma"].values.astype(float)
    base_sigma = np.maximum(base_sigma, 70.0)
    delta = (y["FVC_pred"] - y["FVC_true"]).abs().values.astype(float)
    delta = np.minimum(delta, 1000.0)

    best_m = 1.0
    best_score = -1e18
    for m in candidates:
        s = np.maximum(base_sigma * float(m), 70.0)
        lll = -np.sqrt(2) * delta / s - np.log(np.sqrt(2) * s)
        sc = float(np.mean(lll))
        if sc > best_score:
            best_score = sc
            best_m = float(m)
    return best_m, best_score


def apply_week_distance_sigma_inflation(pred_df, week0_by_patient, k):
    out = pred_df.copy()
    w0 = out["Patient"].map(week0_by_patient).astype(float).values
    w = out["Weeks"].astype(float).values
    scale = 1.0 + float(k) * np.abs(w - w0)
    out["sigma"] = np.maximum(out["sigma"].astype(float).values * scale, 70.0)
    out["FVC_inf"] = out["FVC_pred"] - out["sigma"]
    out["FVC_sup"] = out["FVC_pred"] + out["sigma"]
    return out


def _week_anchor_closest_to_zero(train_df):
    g = train_df[["Patient", "Weeks"]].dropna().copy()
    g["absw"] = g["Weeks"].abs()
    wk0 = g.sort_values(["Patient", "absw", "Weeks"]).groupby("Patient").head(1)
    return wk0.set_index("Patient")["Weeks"].astype(float).to_dict()


def calibrate_k_and_sigma_multiplier_oof(train_df, n_folds=5, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    patients = np.array(sorted(train_df["Patient"].unique()))
    rng.shuffle(patients)
    folds = np.array_split(patients, n_folds)

    week0_by_patient_full = _week_anchor_closest_to_zero(train_df)

    k_candidates = np.array(
        [0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015], dtype=float
    )
    m_candidates = np.array(
        [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.55, 1.70, 1.90, 2.10],
        dtype=float,
    )

    best_k = 0.0
    best_m = 1.0
    best_lll = -1e18

    for k in k_candidates:
        oof_preds = []
        for fold_idx in range(n_folds):
            valid_pat = set(folds[fold_idx].tolist())
            df_valid = train_df[train_df["Patient"].isin(valid_pat)].copy()
            df_train = train_df[~train_df["Patient"].isin(valid_pat)].copy()

            df_valid_first = df_valid.groupby("Patient").head(1)
            df_train_aug = pd.concat(
                [df_train, df_valid_first], axis=0, ignore_index=True
            )

            m, tr = model_fit(df_train_aug, examine=False)
            template_valid = generate_template(df_valid)
            pred_valid = model_predict(m, tr, template_valid)

            pred_valid = apply_week_distance_sigma_inflation(
                pred_valid, week0_by_patient_full, k
            )
            oof_preds.append(pred_valid)

        oof_df = pd.concat(oof_preds, ignore_index=True)

        m_star, _ = calibrate_sigma_multiplier(oof_df, m_candidates)

        tmp = oof_df.copy()
        tmp["sigma"] = np.maximum(
            tmp["sigma"].astype(float).values * float(m_star), 70.0
        )
        sc = evaluate_predictions(tmp, use_only_last_3_measures=True, examine=False)

        if sc > best_lll:
            best_lll = float(sc)
            best_k = float(k)
            best_m = float(m_star)

    k_refined = np.linspace(max(0.0, best_k - 0.005), best_k + 0.005, 9, dtype=float)
    m_refined = np.linspace(max(0.5, best_m - 0.25), best_m + 0.25, 17, dtype=float)

    for k in k_refined:
        oof_preds = []
        for fold_idx in range(n_folds):
            valid_pat = set(folds[fold_idx].tolist())
            df_valid = train_df[train_df["Patient"].isin(valid_pat)].copy()
            df_train = train_df[~train_df["Patient"].isin(valid_pat)].copy()

            df_valid_first = df_valid.groupby("Patient").head(1)
            df_train_aug = pd.concat(
                [df_train, df_valid_first], axis=0, ignore_index=True
            )

            m, tr = model_fit(df_train_aug, examine=False)
            template_valid = generate_template(df_valid)
            pred_valid = model_predict(m, tr, template_valid)

            pred_valid = apply_week_distance_sigma_inflation(
                pred_valid, week0_by_patient_full, k
            )
            oof_preds.append(pred_valid)

        oof_df = pd.concat(oof_preds, ignore_index=True)

        m_star, _ = calibrate_sigma_multiplier(oof_df, m_refined)

        tmp = oof_df.copy()
        tmp["sigma"] = np.maximum(
            tmp["sigma"].astype(float).values * float(m_star), 70.0
        )
        sc = evaluate_predictions(tmp, use_only_last_3_measures=True, examine=False)

        if sc > best_lll:
            best_lll = float(sc)
            best_k = float(k)
            best_m = float(m_star)

    return best_k, best_m, best_lll


print("Fit model ...")
model, trace = model_fit(train, examine=False)
print("")

template_train_full = generate_template(train)
pred_train_full = model_predict(model, trace, template_train_full)

k_oof, sigma_mult_oof, oof_lll_est = calibrate_k_and_sigma_multiplier_oof(
    train, n_folds=5, seed=RANDOM_SEED
)
print(
    f"Chosen week-distance sigma k (OOF, joint): {k_oof:.5f} | "
    f"Chosen sigma multiplier (OOF, joint): {sigma_mult_oof:.4f} | "
    f"OOF-est LLL: {oof_lll_est:.5f}"
)

print("Make predictions for test data (exactly matching sample_submission weeks) ...")
template_test = generate_template_from_submission(sample_sub, test)
pred_test = model_predict(model, trace, template_test)

pred_test["Patient_Week"] = pred_test["Patient"] + "_" + pred_test["Weeks"].astype(str)

test_baseline = test[["Patient", "Weeks", "FVC"]].copy()
test_baseline = test_baseline.rename(columns={"Weeks": "Weeks_base", "FVC": "FVC_base"})
pred_test = pred_test.merge(test_baseline, how="left", on="Patient")
pred_test["Weeks_base"] = pred_test["Weeks_base"].astype(float)
pred_test["FVC_base"] = pred_test["FVC_base"].astype(float)

pid_arr = pred_test["PatientID"].values.astype(int)
cls_arr = pred_test["Class"].astype(int).values

seen = model.get("seen_patient_ids", set())
b_used = np.empty(len(pred_test), dtype=float)
for i, (p, c) in enumerate(zip(pid_arr, cls_arr)):
    if int(p) in seen:
        b_used[i] = float(model["b"].get(int(p), model["global_b"]))
    else:
        c0 = int(np.clip(int(c), 0, 5))
        b_used[i] = (
            float(model["class_b"][c0])
            if model.get("class_b", None) is not None
            else float(model["global_b"])
        )

w = pred_test["Weeks"].values.astype(float)
w0 = pred_test["Weeks_base"].values.astype(float)
f0 = pred_test["FVC_base"].values.astype(float)
current_at_base = pred_test["FVC_pred"].values.astype(float) + b_used * (w0 - w)
shift = f0 - current_at_base
pred_test["FVC_pred"] = np.clip(
    pred_test["FVC_pred"].values.astype(float) + shift, 500.0, 6000.0
)

week0_by_patient_test = test.set_index("Patient")["Weeks"].astype(float).to_dict()
pred_test = apply_week_distance_sigma_inflation(pred_test, week0_by_patient_test, k_oof)

sub = sample_sub[["Patient_Week"]].copy()
sub = sub.merge(
    pred_test[["Patient_Week", "FVC_pred", "sigma"]],
    how="left",
    on="Patient_Week",
)

baseline_map = test.set_index(test["Patient"] + "_" + test["Weeks"].astype(str))[
    "FVC"
].to_dict()
sub["FVC"] = (
    sub["FVC_pred"]
    .fillna(sub["Patient_Week"].map(baseline_map))
    .fillna(model["global_a"])
    .astype(float)
)

sub["Confidence"] = (
    (sub["sigma"].astype(float) * sigma_mult_oof)
    .fillna(np.nanmean(model["sigma_by_class"]))
    .fillna(200.0)
    .astype(float)
)
sub["Confidence"] = np.maximum(sub["Confidence"].values.astype(float), 70.0)

sub = sub[["Patient_Week", "FVC", "Confidence"]]

assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["Patient_Week", "FVC", "Confidence"]
assert sub["FVC"].isna().sum() == 0
assert sub["Confidence"].isna().sum() == 0

sub.to_csv("submission.csv", index=False)

print(sub.shape)
sub.head()
