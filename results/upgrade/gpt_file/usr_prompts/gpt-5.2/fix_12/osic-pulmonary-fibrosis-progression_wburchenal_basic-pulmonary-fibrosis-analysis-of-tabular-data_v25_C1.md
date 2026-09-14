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
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

-6.9495

# 6. Current score

-13.00408

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I fix the two blockers preventing execution: (1) the protobuf/TensorFlow import crash by forcing the Python protobuf implementation before importing TensorFlow, and (2) the pandas `DataFrame.append` removal by replacing it with `pd.concat`. I also fix a few small TensorFlow API issues that would error in TF 2.18 (`Adam(lr=...)` and deprecated `decay`) while keeping the same model and training loop semantics. Finally, I ensure feature engineering always defines `FE`, all arrays (`z`, `pred`, etc.) are created, and that a correctly formatted `submission.csv` is written with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the two execution blockers so the notebook runs end-to-end: (1) the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation early and clearing any already-imported protobuf modules before importing TensorFlow, and (2) the shape mismatch in the custom loss/metric by making sure `y_true` is always 2D (shape `(batch, 1)`) during training/evaluation. These changes keep the same model, loss definition, and training loop semantics, but prevent the `Index out of range` error in `score()`/`qloss()`. I also keep the submission-writing logic intact and ensure a valid `submission.csv` is produced.'
- What this solution (achieved -7.95533) has done: 'I fix the two runtime blockers you’re hitting: the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing a protobuf version that is compatible with TF 2.18, and the `OperatorNotAllowedInGraphError` by removing Python boolean checks on symbolic tensors inside the custom loss/metric (using static rank checks instead). These are execution-only fixes that keep the same model, loss definition, and training loop semantics. I also keep the submission-writing logic intact and ensure a valid `submission.csv` with the exact required columns is always produced.'
- What this solution (achieved -7.95533) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` issue without changing your modeling logic). I also make the confidence post-processing consistent with the competition’s σ-clipping rule by clipping predictions to at least 70 (and avoid the current branch that can emit unrealistically small confidences), which should nudge the score upward toward the target. Finally, I keep all training/model code intact, ensure paths work in this Kaggle layout, and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -7.95533) has done: 'You’re crashing before any training because TensorFlow 2.18 is importing protobuf 6.x where `MessageFactory.GetPrototype` no longer exists; we fix this by forcing the pure-Python protobuf runtime early and clearing any already-imported `google.protobuf` modules before importing TensorFlow. I keep your model/loss/training loop unchanged, but make the sigma computation numerically safe (ensure Confidence is non-negative before clipping to 70) so the metric doesn’t get hurt by negative/invalid σ from the quantile head. Finally, I keep the same submission-writing logic and guarantee the output CSV has the exact required columns and a `.csv` suffix.'
- What this solution (achieved -12.01556) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by avoiding TensorFlow entirely (it is incompatible with protobuf 6.x in this environment) while keeping the same feature engineering and submission-writing semantics. To move the score upward toward your target with minimal logic change, I replace the broken TF model with a simple per-patient linear regression fit on the training history (predict FVC vs Weeks) and use the fitted residual spread as the Confidence (clipped to the competition’s 70 rule). I also keep the required overwrite of baseline test-week rows to the provided test FVC with Confidence=70 exactly as your code does. The result runs end-to-end under the provided packages and writes a valid `submission.csv` with the correct columns.'
- What this solution (achieved -13.39648) has done: 'You’re currently well below the target (−12.02 vs −6.95, higher is better), so we should improve score with minimal, low-risk changes that keep your per-patient linear model intact. The biggest win under this metric is better calibrated `Confidence`: right now you compute `sigma_opt` but never use it, and you also apply an arbitrary `0.996` shrink to FVC that tends to add error. I (1) remove the `0.996` scaling so predictions aren’t systematically biased, and (2) replace the per-patient residual sigma with a blended, more stable sigma using your already-computed MAE proxy (`sigma_opt`) plus the patient sigma, then clip at 70 as required. This preserves the exact modeling approach (same polyfit per patient) and only changes post-processing/calibration to better match the Laplace log-likelihood.'
- What this solution (achieved -12.32699) has done: 'Your current per-patient linear fit is fine; the biggest gap vs the target is likely coming from overconfident (too-small) or poorly calibrated confidences under the Laplace log-likelihood. I keep the exact same modeling (polyfit per patient) and only make minimal, metric-aligned calibration changes: (1) compute a more appropriate global sigma from pooled residuals (not just MAE), and (2) blend patient sigma toward that global residual sigma with a slightly stronger weight to reduce extreme under/over-confidence. I also make the confidence computation consistent (use robust residual scale + small floor) while preserving the existing override that forces baseline test-week rows to the provided FVC with Confidence=70. This should move the score upward toward your target without changing the core prediction logic.'
- What this solution (achieved -11.93745) has done: 'Your score is far below the target (−12.33 vs −6.95, higher is better), so we should improve it with minimal, low-risk changes that keep your per-patient linear fit intact. Under the Laplace log-likelihood, the biggest lever is **confidence calibration**: being too confident (σ too small) or inconsistently calibrated hurts a lot. I keep the exact same per-patient `np.polyfit` prediction for FVC, but (1) compute a more metric-aligned **global Laplace scale** from pooled residuals and (2) blend each patient’s σ toward that global scale with a slightly stronger pull, plus a small additive safety margin before the required clip at 70. This should nudge the public score upward toward your target while preserving your core modeling logic.'
- What this solution (achieved -12.5439) has done: 'You’re currently below the target (−11.94 vs −6.95; higher is better), so we should improve score with the smallest, safest change that preserves your per-patient linear `np.polyfit` core logic. The biggest lever under the Laplace log-likelihood is confidence calibration: your current blend heavily favors a single global sigma and then adds a fixed +15, which can make σ too large (overly pessimistic) and hurt the log term. I keep all FVC predictions identical, but make σ blending more metric-aligned by (1) using a robust global Laplace-optimal sigma from pooled residuals, (2) shrinking patient sigmas toward it based on how many training points the patient had (more points → trust patient sigma more), and (3) removing the fixed +15 offset in favor of a small multiplicative safety factor. This should move the score upward toward your target while keeping the modeling approach unchanged and still writing a valid `submission.csv`.'
- What this solution (achieved -13.00408) has done: 'We keep your per-patient linear `np.polyfit` FVC predictions exactly the same and only adjust the confidence calibration, since that’s the biggest lever under the Laplace log-likelihood when FVC errors are already bounded. Your current calibration likely over-inflates σ (via `*1.05` plus a fairly strong pull to a large global σ), which hurts the `-ln(sigma)` term; we reduce σ slightly and make the global σ estimate more metric-aligned (Laplace-optimal from pooled residuals) while still blending by per-patient sample size. We also clip the final confidence exactly as the metric does (floor at 70) and keep the baseline test-week override unchanged. These are minimal, metric-aligned post-processing changes intended to move the score upward toward your target without altering the modeling approach.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pydicom

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)




## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(8675309)



## === cell 2
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

print(train.head())
print(test.head())
print(sub.head())



## === cell 3
train.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(test.drop("Weeks", axis=1), on="Patient")

print(train.shape, test.shape, sub.shape)



## === cell 4
print(train.info())



## === cell 5
image_path = "../input/osic-pulmonary-fibrosis-progression/"
image_files_list = []
for dirName, subdirList, fileList in os.walk(image_path):
    for filename in fileList:
        if filename.lower().endswith(".dcm"):
            image_files_list.append(os.path.join(dirName, filename))

print("Found DICOM files:", len(image_files_list))
if len(image_files_list) > 0:
    image = pydicom.dcmread(image_files_list[0])
    plt.figure()
    plt.imshow(image.pixel_array, cmap=plt.cm.bone)
    plt.axis("off")
    plt.show()



## === cell 6
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"
data = pd.concat([train, test, sub], ignore_index=True)

data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")

base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)

data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base

COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in data[col].dropna().unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

data["age"] = (data["Age"] - data["Age"].min()) / (
    data["Age"].max() - data["Age"].min()
)
data["BASE"] = (data["min_FVC"] - data["min_FVC"].min()) / (
    data["min_FVC"].max() - data["min_FVC"].min()
)
data["week"] = (data["base_week"] - data["base_week"].min()) / (
    data["base_week"].max() - data["base_week"].min()
)
data["percent"] = (data["Percent"] - data["Percent"].min()) / (
    data["Percent"].max() - data["Percent"].min()
)
FE += ["age", "percent", "week", "BASE"]

print("FE:", FE)

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

print(train.shape, test.shape, sub.shape)



## === cell 7
train_full = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test_base = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

patient_models = {}
global_fvc_med = float(train_full["FVC"].median())
global_sigma = float(
    np.maximum(train_full.groupby("Patient")["FVC"].std().median(), 70.0)
)

patient_nobs = {}

for pid, g in train_full.groupby("Patient"):
    g = g.sort_values("Weeks")
    x = g["Weeks"].values.astype(np.float64)
    y = g["FVC"].values.astype(np.float64)

    if len(g) >= 2 and np.std(x) > 1e-9:
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = intercept + slope * x
        resid = y - y_hat
        sigma = float(np.std(resid)) if len(resid) > 1 else global_sigma
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = global_sigma
    else:
        slope, intercept = 0.0, float(np.mean(y)) if len(y) else global_fvc_med
        sigma = float(np.std(y)) if len(y) > 1 else global_sigma
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = global_sigma

    patient_models[pid] = (float(slope), float(intercept), float(sigma))
    patient_nobs[pid] = int(len(g))

print("Trained patient models:", len(patient_models))



## === cell 8
subm = sub[["Patient_Week", "Patient", "Weeks", "FVC", "Confidence"]].copy()

pred_fvc = np.zeros(len(subm), dtype=np.float32)
pred_conf = np.zeros(len(subm), dtype=np.float32)
pred_nobs = np.zeros(len(subm), dtype=np.float32)

for i, (pid, wk) in enumerate(zip(subm["Patient"].values, subm["Weeks"].values)):
    if pid in patient_models:
        slope, intercept, sigma = patient_models[pid]
        fvc_hat = intercept + slope * float(wk)
        conf_hat = sigma
        nobs = patient_nobs.get(pid, 0)
    else:
        fvc_hat = global_fvc_med
        conf_hat = global_sigma
        nobs = 0
    pred_fvc[i] = np.float32(fvc_hat)
    pred_conf[i] = np.float32(conf_hat)
    pred_nobs[i] = np.float32(nobs)

subm["FVC1"] = pred_fvc.astype(np.float32)

train_pred = []
train_true = []
train_unc = []
all_resid = []

for pid, g in train_full.groupby("Patient"):
    slope, intercept, sigma = patient_models[pid]
    yhat = intercept + slope * g["Weeks"].values.astype(np.float64)
    ytrue = g["FVC"].values.astype(np.float64)
    resid = ytrue - yhat

    train_pred.extend(list(yhat))
    train_true.extend(list(ytrue))
    train_unc.extend([sigma] * len(g))
    all_resid.extend(list(resid))

train_pred = np.asarray(train_pred, dtype=np.float32)
train_true = np.asarray(train_true, dtype=np.float32)
train_unc = np.asarray(train_unc, dtype=np.float32)
all_resid = np.asarray(all_resid, dtype=np.float32)

abs_resid = np.abs(all_resid)

b_laplace = (
    float(np.mean(abs_resid))
    if len(abs_resid)
    else float(mean_absolute_error(train_true, train_pred))
)
sigma_opt_laplace = float(np.sqrt(2.0) * b_laplace)
sigma_opt = float(np.maximum(sigma_opt_laplace, 70.0))

sigma_opt_mae = float(mean_absolute_error(train_true, train_pred))
sigma_opt_resid_std = float(np.std(all_resid)) if len(all_resid) > 1 else sigma_opt_mae
sigma_mean = float(np.mean(train_unc))

print(
    "sigma_opt_mae:",
    sigma_opt_mae,
    "sigma_opt_resid_std:",
    sigma_opt_resid_std,
    "sigma_opt_laplace:",
    sigma_opt_laplace,
    "sigma_mean:",
    sigma_mean,
)

k = 1.0  # was 3.0; smaller => more weight to patient sigma for typical patients
w = (pred_nobs / (pred_nobs + k)).astype(np.float32)  # in [0,1)

calib_sigma = (w * pred_conf + (1.0 - w) * np.float32(sigma_opt)).astype(np.float32)

calib_sigma = (calib_sigma * 0.98).astype(np.float32)

subm["Confidence1"] = np.maximum(calib_sigma, 1e-3).astype(np.float32)

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]
subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
    ~subm.FVC1.isnull(), "Confidence1"
]

subm["FVC"] = subm["FVC"].astype(np.float32)
subm["Confidence"] = subm["Confidence"].astype(np.float32).clip(lower=70.0)



## === cell 9
otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print(
    "Confidence min/max:",
    float(submission["Confidence"].min()),
    float(submission["Confidence"].max()),
)
