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

-6.985107322599156

# 6. Current score

-8.06003

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved -8.06003) has done: 'I fix the TensorFlow/Keras import crash by avoiding TensorFlow entirely (it fails in this environment) and replace it with a lightweight, deterministic clinical-only baseline that still produces valid `FVC` and `Confidence` for every `Patient_Week`. I also fix all broken file paths by pointing to the provided `/kaggle/input/osic-pulmonary-fibrosis-progression/` dataset and remove dependencies on missing pickles (`data_prep`, `list_patient_score`). Finally, I ensure the output submission strictly matches `sample_submission.csv` ordering and format, writing a real `submission.csv` to the working directory.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error




## === cell 1
def seed_all(seed: int = 20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_all(20)



## === cell 2
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(TRAIN_CSV)
raw_test = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

assert set(["Patient_Week", "FVC", "Confidence"]).issubset(sample_sub.columns)



## === cell 3
X_prediction = sample_sub[["Patient_Week"]].copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")[0]
X_prediction["Weeks"] = (
    X_prediction["Patient_Week"].str.extract(r".*_(.*)")[0].astype(int)
)

X_prediction = X_prediction.merge(
    raw_test, on="Patient", how="left", suffixes=("", "_base")
)

X_prediction.rename(
    columns={"Weeks_base": "Base_week", "FVC": "Base_FVC"}, inplace=True
)

needed = [
    "Patient",
    "Patient_Week",
    "Weeks",
    "Base_week",
    "Base_FVC",
    "Percent",
    "Age",
    "Sex",
    "SmokingStatus",
]
missing = [c for c in needed if c not in X_prediction.columns]
if missing:
    raise RuntimeError(f"Missing required columns after merge: {missing}")



## === cell 4
tr = train.copy()
tr = tr.sort_values(["Patient", "Weeks"]).reset_index(drop=True)

base = tr.groupby("Patient").first().reset_index()
base = base[["Patient", "Weeks", "FVC"]].rename(
    columns={"Weeks": "Base_week", "FVC": "Base_FVC"}
)

tr = tr.merge(base, on="Patient", how="left")
tr["delta_week"] = tr["Weeks"] - tr["Base_week"]


def prep_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex"] = df["Sex"].fillna("Unknown")
    df["SmokingStatus"] = df["SmokingStatus"].fillna("Unknown")
    return df


tr = prep_cats(tr)
X_prediction = prep_cats(X_prediction)




## === cell 5
def fit_group_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-patient slope and intercept fitted on (delta_week, FVC),
    plus per-patient residual MAE as an uncertainty proxy.
    """
    rows = []
    for pid, g in df.groupby("Patient"):
        x = g["delta_week"].values.astype(float)
        y = g["FVC"].values.astype(float)
        if len(g) >= 2 and np.std(x) > 0:
            b, a = np.polyfit(x, y, 1)
            yhat = a + b * x
            mae = np.mean(np.abs(y - yhat))
        else:
            a = float(g["Base_FVC"].iloc[0])
            b = 0.0
            mae = np.mean(np.abs(y - a)) if len(g) else 200.0
        rows.append((pid, a, b, mae))
    return pd.DataFrame(rows, columns=["Patient", "a", "b", "mae"])


patient_fit = fit_group_slopes(tr)

tmp = tr.merge(patient_fit[["Patient", "b", "mae"]], on="Patient", how="left")
fallback = (
    tmp.groupby(["Sex", "SmokingStatus"], dropna=False)[["b", "mae"]]
    .median()
    .reset_index()
    .rename(columns={"b": "b_fb", "mae": "mae_fb"})
)



## === cell 6
X_prediction = X_prediction.merge(patient_fit, on="Patient", how="left")
X_prediction = X_prediction.merge(fallback, on=["Sex", "SmokingStatus"], how="left")

b_used = X_prediction["b"].where(
    X_prediction["b"].notna(), X_prediction["b_fb"].fillna(0.0)
)
a_used = X_prediction["a"].where(
    X_prediction["a"].notna(), X_prediction["Base_FVC"].astype(float)
)

delta_week_pred = (X_prediction["Weeks"] - X_prediction["Base_week"]).astype(float)
fvc_pred = a_used + b_used * delta_week_pred

mae_used = X_prediction["mae"].where(
    X_prediction["mae"].notna(), X_prediction["mae_fb"].fillna(200.0)
)
conf_pred = np.maximum(70.0, 1.5 * mae_used.values.astype(float))

X_prediction["FVC_pred"] = fvc_pred.values
X_prediction["Conf_pred"] = conf_pred



## === cell 7
base_map = raw_test.set_index(["Patient", "Weeks"])["FVC"].to_dict()


def override_baseline(row):
    key = (row["Patient"], int(row["Weeks"]))
    if key in base_map:
        return (
            float(base_map[key]),
            0.1,
        )  # tiny confidence for known measurement (sigma clipped to 70 in metric anyway)
    return float(row["FVC_pred"]), float(row["Conf_pred"])


overridden = X_prediction.apply(override_baseline, axis=1, result_type="expand")
X_prediction["FVC_final"] = overridden[0]
X_prediction["Confidence_final"] = overridden[1]



## === cell 8
subm = sample_sub[["Patient_Week"]].merge(
    X_prediction[["Patient_Week", "FVC_final", "Confidence_final"]],
    on="Patient_Week",
    how="left",
)

subm["FVC"] = subm["FVC_final"].fillna(3020.0).astype(float)
subm["Confidence"] = subm["Confidence_final"].fillna(100.0).astype(float)

subm = subm[["Patient_Week", "FVC", "Confidence"]]
subm.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", subm.shape)
print(subm.head())



## === cell 9
gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(tr), dtype=float)

for fold, (trn_idx, val_idx) in enumerate(gkf.split(tr, groups=tr["Patient"]), 1):
    dtrn = tr.iloc[trn_idx].copy()
    dval = tr.iloc[val_idx].copy()

    pf = fit_group_slopes(dtrn)
    tmp2 = dtrn.merge(pf[["Patient", "b", "mae"]], on="Patient", how="left")
    fb = (
        tmp2.groupby(["Sex", "SmokingStatus"], dropna=False)[["b", "mae"]]
        .median()
        .reset_index()
        .rename(columns={"b": "b_fb", "mae": "mae_fb"})
    )

    dval = dval.merge(pf, on="Patient", how="left")
    dval = dval.merge(fb, on=["Sex", "SmokingStatus"], how="left")

    b_u = dval["b"].where(dval["b"].notna(), dval["b_fb"].fillna(0.0))
    a_u = dval["a"].where(dval["a"].notna(), dval["Base_FVC"].astype(float))
    fvc_hat = a_u + b_u * dval["delta_week"].astype(float)

    oof[val_idx] = fvc_hat.values

mae = mean_absolute_error(tr["FVC"].values, oof)
print("OOF MAE (rough sanity check):", mae)
