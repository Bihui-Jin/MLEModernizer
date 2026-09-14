# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tqdm==4.67.1

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

-6.916

# 6. Current score

-8.76216

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I fix the runtime blockers without changing the model core: (1) resolve the TensorFlow import crash caused by an incompatible protobuf version by forcing the pure‑Python protobuf implementation before importing TF, (2) update deprecated seaborn/pandas API usages so exploratory cells don’t error, and (3) replace the removed `DataFrame.append` with `pd.concat` and ensure the merged `data` keeps the `Weeks` column. Then I make minimal compatibility fixes for Keras (use `learning_rate` instead of `lr`) and ensure the training labels have the expected shape `(n, 1)` for the custom loss/metric functions. Finally, I ensure a valid `submission.csv` with exactly the required columns is always written.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow import crash by forcing a protobuf implementation compatible with TF 2.18 (setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is no longer sufficient with protobuf 6.x) so the notebook can start training. Then I fix the `None values not supported` training error by ensuring all feature/label arrays contain no NaNs/Infs after the train/test/sample concat and feature engineering (this happens because `FVC` is missing for the synthetic sample rows). Finally, I keep the model and training loop unchanged but make the submission generation robust (guarantee correct dtypes, no NaNs, and required columns), which should also improve score vs. silently broken/NaN predictions.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow import crash by switching to a protobuf version compatible with TF 2.18 (the current env uses protobuf 6.x, which triggers the `MessageFactory.GetPrototype` error). Then I fix the `None values not supported` training failure by ensuring `y` contains no missing values (the concatenated `data` includes sample rows with `FVC` missing, and those can leak into `min_FVC` unless we compute baseline using only real measured rows). Finally, I keep the model/training loop intact but make the feature scaling non-leaky (fit scalers on train only and apply to val/test) which should legitimately improve score toward the target without changing the core approach, and I keep the submission writing robust with required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring the training label `FVC` has no missing values after the train/test/sample concat: `min_FVC` be computed using only real training rows, and then any remaining missing `FVC` in the training subset be dropped (not imputed) to keep labels valid and preserve semantics. I also store and reuse per-feature scalers (fit on train only) as before, but add finite-value guards after feature engineering to prevent hidden NaNs/Infs from reaching TensorFlow. These are execution/stability fixes that should also improve score by preventing corrupted rows from harming training. The model, loss, and training loop remain unchanged, and a valid `submission.csv` is always written.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring the computed baseline `min_FVC` exists for every patient in train/val/test: when a patient’s earliest week isn’t present in the training rows, we compute the baseline from that patient’s own earliest available row in `data` (train+val+test), and then fill any remaining gaps using per-patient medians. I also add strict finite-value guards immediately before `model.fit()` to catch/repair any NaN/Inf that may still slip in, without changing the model, loss, or training loop. Finally, I keep the submission generation the same but make sure the written `submission.csv` is always complete and aligned, which is score-neutral besides preventing broken predictions.'
- What this solution (achieved -8.76216) has done: 'I fix the training crash (`ValueError: None values not supported`) by ensuring that `X`, `y`, and the fold-sliced arrays passed into `model.fit()` contain no `None`/object values (these can survive the current NaN/Inf guards) and are strictly `float32`. I do this with a minimal “sanitize to numeric + fill” step right after feature assembly and again inside the KFold loop, without changing the model, loss, or training procedure. This should both unblock end-to-end execution and improve score versus training on corrupted/invalid rows. I also make submission generation robust to any remaining NaNs while keeping the same semantics and required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the pipeline-breaking dtype issue in the feature/label sanitization step by forcing all feature columns and the FVC label to numeric before any `np.isfinite` checks, and by guarding against object dtypes coming from merges. Then I ensure `X`, `y`, and `test_` are always created (so downstream cells don’t hit `NameError`) and remain strictly finite `float32`. Finally, I keep the model/training loop unchanged but make the submission creation robust so it always writes a valid `submission.csv` with the required columns and aligned to `sample_submission.csv`. These changes are execution/stability fixes and should also improve score versus the current broken run (no submission produced).'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import subprocess


def _ensure_protobuf_compat():
    """
    Bugfix: TF 2.18 is often incompatible with protobuf 6.x in Kaggle images.
    Pin protobuf to a TF-compatible version BEFORE importing TensorFlow.
    """
    try:
        import google.protobuf  # noqa
        from google.protobuf import __version__ as pb_ver  # noqa

        major = int(pb_ver.split(".")[0])
    except Exception:
        major = None

    if major is None or major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib

        importlib.invalidate_caches()


_ensure_protobuf_compat()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
from PIL import Image
import random
import pydicom
import gc
import warnings

warnings.filterwarnings("ignore")




## === cell 1
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold




## === cell 2
df = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
sample = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)




## === cell 3
df.head()




## === cell 4
df.info()




## === cell 5
print(f"Out of 1549 entried there were only {df['Patient'].nunique()} unique patients")




## === cell 6
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
sns.countplot(data=df, x="Sex", ax=ax1).set_title("GENDER COUNT OF GIVEN DATA")
sns.countplot(data=df, x="SmokingStatus", ax=ax2).set_title(
    "SMOKING STATUS COUNT OF GIVEN DATA"
)
plt.tight_layout()
plt.show()




## === cell 7
print("Since the data has dupilcated values let's recheck by dropping duplicates")
print()
print("######### GENDER ##########")
print()
print(df[["Patient", "Sex", "SmokingStatus"]].drop_duplicates()["Sex"].value_counts())
print()
print("####### SMOKING STATUS ########")
print()
print(
    df[["Patient", "Sex", "SmokingStatus"]]
    .drop_duplicates()["SmokingStatus"]
    .value_counts()
)




## === cell 8
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

print(
    f"FVC minimum value - {df['FVC'].min()}, FVC maximum value - {df['FVC'].max()}                 \
          WEEKS minimum value - {df['Weeks'].min()}, WEEKS maximum value - {df['Weeks'].max()}"
)

sns.histplot(df["FVC"], ax=ax1, bins=40, color="salmon", kde=False).set_title(
    "FVC DISTRIBUTION"
)
sns.histplot(df["Weeks"], ax=ax2, bins=40, color="salmon", kde=False).set_title(
    "WEEKS DISTRIBUTION"
)
plt.tight_layout()
plt.show()




## === cell 9
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

print(
    f"Age minimum value - {df['Age'].min()}, Age maximum value - {df['Age'].max()}                 \
          Percent minimum value - {df['Percent'].min()}, Percent maximum value - {df['Percent'].max()}"
)

sns.histplot(df["Age"], ax=ax1, bins=40, color="plum", kde=False).set_title(
    "AGE DISTRIBUTION"
)
sns.histplot(df["Percent"], ax=ax2, bins=40, color="plum", kde=False).set_title(
    "PERCENT DISTRIBUTION"
)
plt.tight_layout()
plt.show()




## === cell 10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
sns.boxplot(x=df["Percent"], ax=ax1, palette="winter", orient="h").set_title(
    "PERCENTAGE DETAILS"
)
sns.boxplot(x=df["Age"], ax=ax2, palette="winter", orient="h").set_title("AGE DETAILS")
plt.tight_layout()
plt.show()




## === cell 11
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
sns.boxplot(x=df["FVC"], ax=ax1, palette="winter", orient="h").set_title("FVC DETAILS")
sns.boxplot(x=df["Weeks"], ax=ax2, palette="winter", orient="h").set_title(
    "WEEKS DETAILS"
)
plt.tight_layout()
plt.show()




## === cell 12
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
sns.boxplot(data=df, x="Sex", y="FVC", palette="winter", orient="v", ax=ax1).set_title(
    "GENDER VS FVC"
)
sns.boxplot(
    data=df, x="SmokingStatus", y="FVC", palette="winter", orient="v", ax=ax2
).set_title("SMOKING STAT VS FVC")
plt.tight_layout()
plt.show()




## === cell 13
print("######## FVC #########\n")
print(f"Mean FVC of Male {df[df['Sex']=='Male']['FVC'].mean()}")
print(f"Mean FVC of Female {df[df['Sex']=='Female']['FVC'].mean()}\n")

print("######## Smoking Stat #########\n")
print(f"Mean FVC of Smoker {df[df['SmokingStatus']=='Currently smokes']['FVC'].mean()}")
print(f"Mean FVC of Ex-Smoker {df[df['SmokingStatus']=='Ex-smoker']['FVC'].mean()}")
print(
    f"Mean FVC of Never Smoked {df[df['SmokingStatus']=='Never smoked']['FVC'].mean()}"
)




## === cell 14
sns.pairplot(
    hue="Sex",
    data=df,
    x_vars=["Weeks", "FVC", "Percent", "Age"],
    y_vars=["Weeks", "FVC", "Percent", "Age"],
    height=3,
)
plt.show()




## === cell 15
print("FVC decreases over time for most of the cases")
plt.figure(figsize=(15, 10))
_ = sns.lineplot(x=df["Weeks"], y=df["FVC"], hue=df["Patient"], size=1, legend=False)
plt.show()




## === cell 16
df["Photo count"] = 0
names = df["Patient"].unique()
for name in names:
    file = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/" + name
    if os.path.isdir(file):
        df.loc[df["Patient"] == name, "Photo count"] = len(os.listdir(file))
    else:
        df.loc[df["Patient"] == name, "Photo count"] = 0

data_photos = df.groupby(by="Patient")["Photo count"].first().reset_index(drop=False)
data_photos = data_photos.sort_values(["Photo count"]).reset_index(drop=True)
data_photos["Photo count"].describe()




## === cell 17
plt.figure(figsize=(20, 5))
sns.histplot(data_photos["Photo count"], bins=200, kde=False)
plt.show()




## === cell 18
patient_dir = (
    "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430"
)

if os.path.isdir(patient_dir):
    files = []
    for dcm in list(os.listdir(patient_dir)):
        files.append(dcm)

    files.sort(key=lambda f: int(re.findall(r"\d+", f)[0]))

    datasets = []
    for dcm in files:
        path = patient_dir + "/" + dcm
        datasets.append(pydicom.dcmread(path))

    fig = plt.figure(figsize=(16, 6))
    columns = 10
    rows = 3

    for i in range(min(columns * rows, len(datasets))):
        img = datasets[i].pixel_array
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img, cmap="plasma")
        plt.axis("off")
    plt.show()




## === cell 19
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)




## === cell 20
df.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])




## === cell 21
sample["Patient"] = sample["Patient_Week"].apply(lambda x: x.split("_")[0])
sample["Weeks"] = sample["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sample = sample[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sample = sample.merge(test.drop("Weeks", axis=1), on="Patient")




## === cell 22
df["WHERE"] = "train"
test["WHERE"] = "val"
sample["WHERE"] = "test"

data = pd.concat([df, test, sample], axis=0, ignore_index=True, sort=False)




## === cell 23
data["min_week"] = data.groupby("Patient")["Weeks"].transform("min")




## === cell 24
base_train = data.loc[
    (data["WHERE"] == "train") & (data["Weeks"] == data["min_week"]),
    ["Patient", "FVC"],
].copy()
base_train.columns = ["Patient", "min_FVC"]

base_any = (
    data.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)[["Patient", "FVC"]]
    .first()
)
base_any.columns = ["Patient", "min_FVC_any"]

data = data.merge(base_train, on="Patient", how="left")
data = data.merge(base_any, on="Patient", how="left")

data["min_FVC"] = data["min_FVC"].fillna(data["min_FVC_any"])
data.drop(columns=["min_FVC_any"], inplace=True)

data["min_FVC"] = data["min_FVC"].fillna(
    data.groupby("Patient")["FVC"].transform("median")
)
data["min_FVC"] = data["min_FVC"].fillna(data["FVC"].median())

data["base_week"] = data["Weeks"] - data["min_week"]
gc.collect()




## === cell 25
dummies = pd.get_dummies(data[["Sex", "SmokingStatus"]], drop_first=True)
for col in ["Sex_Male", "SmokingStatus_Ex-smoker", "SmokingStatus_Never smoked"]:
    if col not in dummies.columns:
        dummies[col] = 0
data[["Sex_Male", "SmokingStatus_Ex-smoker", "SmokingStatus_Never smoked"]] = dummies[
    ["Sex_Male", "SmokingStatus_Ex-smoker", "SmokingStatus_Never smoked"]
]

features = ["Percent", "Age", "min_FVC", "base_week"]




## === cell 26
from sklearn.preprocessing import MinMaxScaler

for col in features:
    if data[col].isna().any():
        data[col] = data[col].fillna(data.groupby("Patient")[col].transform("median"))
        data[col] = data[col].fillna(data[col].median())

train_mask = data["WHERE"] == "train"
_scalers = {}
for col in features:
    scaler = MinMaxScaler()
    data.loc[train_mask, col] = scaler.fit_transform(data.loc[train_mask, [col]])
    data.loc[~train_mask, col] = scaler.transform(data.loc[~train_mask, [col]])
    _scalers[col] = scaler




## === cell 27
df = data.loc[data.WHERE == "train"].drop("WHERE", axis=1)
test = data.loc[data.WHERE == "val"].drop("WHERE", axis=1)
sample = data.loc[data.WHERE == "test"].drop("WHERE", axis=1)




## === cell 28
features += ["Sex_Male", "SmokingStatus_Ex-smoker", "SmokingStatus_Never smoked"]




## === cell 29
df[features].head()




## === cell 30
before = len(df)
df = df.loc[df["FVC"].notna()].copy()
after = len(df)
if after != before:
    print(
        f"Dropped {before-after} training rows with missing FVC to avoid None/NaN labels."
    )

for c in features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    sample[c] = pd.to_numeric(sample[c], errors="coerce")
df["FVC"] = pd.to_numeric(df["FVC"], errors="coerce")

feat_meds = df[features].median(numeric_only=True)
df[features] = df[features].fillna(feat_meds).fillna(0.0)
sample[features] = sample[features].fillna(feat_meds).fillna(0.0)

X_num = df[features].to_numpy(dtype=np.float32, copy=True)
y_num = df["FVC"].to_numpy(dtype=np.float32, copy=True)
mask_good = np.isfinite(X_num).all(axis=1) & np.isfinite(y_num)
dropped_bad = int((~mask_good).sum())
if dropped_bad > 0:
    print(f"Dropped {dropped_bad} training rows with non-finite features/labels.")
df = df.loc[mask_good].copy()

X = np.ascontiguousarray(df[features].to_numpy(dtype=np.float32), dtype=np.float32)
y = np.ascontiguousarray(
    df["FVC"].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=np.float32
)
test_ = np.ascontiguousarray(
    sample[features].to_numpy(dtype=np.float32), dtype=np.float32
)

if not np.isfinite(y).all():
    y_med = np.nanmedian(y)
    y = np.nan_to_num(y, nan=y_med, posinf=y_med, neginf=y_med).astype(np.float32)
if not np.isfinite(X).all():
    X_med = np.nanmedian(X)
    X = np.nan_to_num(X, nan=X_med, posinf=X_med, neginf=X_med).astype(np.float32)
if not np.isfinite(test_).all():
    t_med = np.nanmedian(test_)
    test_ = np.nan_to_num(test_, nan=t_med, posinf=t_med, neginf=t_med).astype(
        np.float32
    )

print("Shapes:", "X", X.shape, "y", y.shape, "test_", test_.shape)




## === cell 31
nh = X.shape[1]
pe = np.zeros((test_.shape[0], 3), dtype=np.float32)
pred = np.zeros((X.shape[0], 3), dtype=np.float32)




## === cell 32
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # (batch, 1)
    y_pred = tf.cast(y_pred, tf.float32)  # (batch, 3)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)  # (1,3)
    y_true = tf.cast(y_true, tf.float32)  # (batch,1)
    y_pred = tf.cast(y_pred, tf.float32)  # (batch,3)
    e = y_true - y_pred  # broadcast -> (batch,3)
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model(nh):
    z = L.Input((nh,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    model = M.Model(z, preds, name="CNN")

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
    )
    model.compile(loss=mloss(0.8), optimizer=opt, metrics=[score])
    return model




## === cell 33
kf = KFold(n_splits=5, shuffle=False)
cnt = 0
EPOCHS = 800

for tr_idx, val_idx in kf.split(X):
    cnt += 1
    print(f"FOLD {cnt}")
    model = make_model(nh)

    X_tr = np.ascontiguousarray(X[tr_idx], dtype=np.float32)
    y_tr = np.ascontiguousarray(y[tr_idx], dtype=np.float32)
    X_va = np.ascontiguousarray(X[val_idx], dtype=np.float32)
    y_va = np.ascontiguousarray(y[val_idx], dtype=np.float32)

    tr_good = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr[:, 0])
    va_good = np.isfinite(X_va).all(axis=1) & np.isfinite(y_va[:, 0])
    if not tr_good.all():
        X_tr, y_tr = X_tr[tr_good], y_tr[tr_good]
    if not va_good.all():
        X_va, y_va = X_va[va_good], y_va[va_good]

    model.fit(
        X_tr,
        y_tr,
        batch_size=128,
        epochs=EPOCHS,
        validation_data=(X_va, y_va),
        verbose=0,
    )
    print("train", model.evaluate(X_tr, y_tr, verbose=0, batch_size=128))
    print("val", model.evaluate(X_va, y_va, verbose=0, batch_size=128))
    print("predict val...")
    pred[val_idx] = model.predict(X_va, batch_size=128, verbose=0)
    print("predict test...")
    pe += model.predict(test_, batch_size=128, verbose=0) / 5.0




## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3683728802.py in <cell line: 0>()
     20         X_va, y_va = X_va[va_good], y_va[va_good]
     21 
---> 22     model.fit(
     23         X_tr,
     24         y_tr,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in convert_to_tensor(x, dtype, sparse)
    135             x = tf.convert_to_tensor(x)
    136             return tf.cast(x, dtype)
--> 137         return tf.convert_to_tensor(x, dtype=dtype)
    138     elif dtype is not None and not x.dtype == dtype:
    139         if isinstance(x, tf.SparseTensor):

ValueError: None values not supported.

## === cell 34
sigma_opt = mean_absolute_error(y[:, 0], pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)




## === cell 35
idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs, 0], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()




## === cell 36
plt.hist(unc, bins=50)
plt.title("uncertainty in prediction")
plt.show()




## === cell 37
sample["FVC1"] = 0.996 * pe[:, 1]
sample["Confidence1"] = pe[:, 2] - pe[:, 0]

subm = sample[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()




## === cell 38
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]
if sigma_mean < 70:
    subm["Confidence"] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

subm["Confidence"] = (
    pd.to_numeric(subm["Confidence"], errors="coerce")
    .astype("float32")
    .clip(lower=70.0)
)




## === cell 39
otest = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0




## === cell 40
subm_out = subm[["Patient_Week", "FVC", "Confidence"]].copy()
subm_out["FVC"] = pd.to_numeric(subm_out["FVC"], errors="coerce")
subm_out["Confidence"] = pd.to_numeric(subm_out["Confidence"], errors="coerce")

subm_out["FVC"] = subm_out["FVC"].fillna(subm_out["FVC"].median())
subm_out["Confidence"] = subm_out["Confidence"].fillna(70.0).clip(lower=70.0)

subm_out["FVC"] = subm_out["FVC"].astype("float32")
subm_out["Confidence"] = subm_out["Confidence"].astype("float32")

sample_sub = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
subm_out = sample_sub[["Patient_Week"]].merge(
    subm_out, on="Patient_Week", how="left", validate="one_to_one"
)
subm_out["FVC"] = subm_out["FVC"].fillna(subm_out["FVC"].median()).astype("float32")
subm_out["Confidence"] = (
    subm_out["Confidence"].fillna(70.0).clip(lower=70.0).astype("float32")
)

subm_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm_out.shape)
print(subm_out.head())
