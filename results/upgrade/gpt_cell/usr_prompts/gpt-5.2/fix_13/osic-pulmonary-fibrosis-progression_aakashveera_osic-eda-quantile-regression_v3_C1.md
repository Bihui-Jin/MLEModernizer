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

-7.92453

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I make the pipeline reliably produce a valid submission by fixing shape/dtype issues in the custom loss/metric (your model currently trains with `y` shaped `(n,)` but the loss expects `(n,1)`), and by removing the problematic forced protobuf downgrade/exec that can prevent any submission from being generated. To move the score upward toward the target, I also correct the submission confidence for baseline test weeks: setting `Confidence=0.1` makes the metric extremely bad due to sigma clipping/log term; using 70 (or the model’s inferred uncertainty) is consistent with the evaluation and improves the score without changing the core model. Finally, I keep your model architecture/training loop intact and only adjust optimizer arguments to TensorFlow 2.18-compatible names so training runs end-to-end and writes `submission.csv`.'
- What this solution (achieved -8.76216) has done: 'Diagnosis: The crash happens while importing TensorFlow in cell 1, before any model code runs. With `protobuf==6.33.0` and `tensorflow==2.18.0`, TensorFlow’s import path can hit an incompatibility where protobuf’s `MessageFactory` no longer provides `GetPrototype`, raising `AttributeError`. This is an environment-level dependency mismatch, not an issue with your model code. The safest minimal workaround in-notebook is to force TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow.

Patch summary: In cell 1 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version) via `os.environ` before importing `tensorflow`. This avoids the failing C++/upb code path that triggers `GetPrototype` and keeps the rest of the code unchanged.

Updated cells: cell 1 only.

Compatibility notes for cell k+1: All TensorFlow/Keras symbols (`tf`, `K`, `L`, `M`) are still imported with the same names and be available to subsequent cells exactly as expected.

Assumptions: Setting the protobuf implementation env var before the first TensorFlow import is sufficient in this runtime; no other cells imported TensorFlow earlier.'
- What this solution (achieved -8.76216) has done: 'Diagnosis: The crash happens while importing TensorFlow because `protobuf==6.33.0` is incompatible with the runtime expectations of some TensorFlow components (often via TF-DF / TF metadata), leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The two environment variables set in the cell (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and version `2`) are not sufficient to prevent this in newer protobuf releases, and in some cases make it worse. The minimal deterministic fix is to pin protobuf to a TensorFlow-compatible version (commonly `<5`, e.g. `4.25.3`) before importing TensorFlow, then proceed with the same imports unchanged.

Patch summary: In cell 1 only, install a protobuf version compatible with TF 2.18.0 at runtime, then import TensorFlow/Keras exactly as before. This avoids changing any model/training logic and preserves the interfaces used by later cells.

Updated cells:'
- What this solution (achieved -8.01455) has done: 'Diagnosis: The crash happens in cell 34 during `model.fit()` because `X` (and/or `y`) contains `dtype=object`, which Keras cannot convert to tensors (“Invalid dtype: object”). This typically occurs when some feature columns were created with mixed/nullable types or booleans from `get_dummies`, leaving the underlying NumPy array as object. The model expects numeric float tensors, so we must ensure `X`, `y`, and `test_` are proper floating dtypes before fitting/predicting.

Patch summary: In cell 34, cast `X`, `y`, and `test_` to `np.float32` once right before the KFold loop (without changing model/training logic), and make sure `pred`/`pe` accumulation works unchanged. This is the smallest localized fix that keeps the same architecture, loss, metrics, epochs, folds, and evaluation semantics.

Updated cells: Only cell 34 is modified.

Compatibility notes for cell k+1: Cell 35 expects `y`, `pred` and their indexing (`y[:,0]`, `pred[:,1]`, etc.) to work; the cast preserves shapes and keeps `pred` as `(n,3)` float32, so it remains compatible.

Assumptions: The only issue is improper `object` dtype in the NumPy arrays passed to Keras; casting to float32 is valid because all feature columns and targets are numeric/one-hot encoded and should be safely representable as floats.'
- What this solution (achieved -7.92453) has done: 'Your current gap is \(-8.01455\) vs target \(-6.916\) (higher is better), so we should improve the score a bit without changing the model/training core. The biggest low-risk gain here is fixing a scaling bug: you fit a separate MinMaxScaler for each feature on the concatenated train+test+sample data, which leaks test distribution and also makes test-time preprocessing inconsistent; switching to fitting scalers on train-only and applying to others typically improves generalization and public LB. Second, your predicted uncertainty can become unrealistically large (or occasionally too small), and since the metric rewards well-calibrated sigma, we clamp predicted Confidence into a reasonable range while respecting the competition’s sigma clipping at 70. These are minimal, localized changes that keep architecture, loss, folds, and epochs intact and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import re
from PIL import Image
import random
import pydicom
import gc

import warnings

warnings.filterwarnings("ignore")



## === cell 1
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

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
sns.countplot(x=df["Sex"], ax=ax1).set_title("GENDER COUNT OF GIVEN DATA")
sns.countplot(x=df["SmokingStatus"], ax=ax2).set_title(
    "SMOKING STATUS COUNT OF GIVEN DATA"
)



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

sns.histplot(df["FVC"], ax=ax1, bins=40, color="salmon").set_title("FVC DISTRIBUTION")
sns.histplot(df["Weeks"], ax=ax2, bins=40, color="salmon").set_title(
    "WEEKS DISTRIBUTION"
)



## === cell 9
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

print(
    f"Age minimum value - {df['Age'].min()}, Age maximum value - {df['Age'].max()}                 \
          Percent minimum value - {df['Percent'].min()}, Percent maximum value - {df['Percent'].max()}"
)

sns.histplot(df["Age"], ax=ax1, bins=40, color="plum").set_title("AGE DISTRIBUTION")
sns.histplot(df["Percent"], ax=ax2, bins=40, color="plum").set_title(
    "PERCENT DISTRIBUTION"
)



## === cell 10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
sns.boxplot(x=df["Percent"], ax=ax1, palette="winter", orient="h").set_title(
    "PERCENTAGE DETAILS"
)
sns.boxplot(x=df["Age"], ax=ax2, palette="winter", orient="h").set_title("AGE DETAILS")



## === cell 11
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
sns.boxplot(x=df["FVC"], ax=ax1, palette="winter", orient="h").set_title("FVC DETAILS")
sns.boxplot(x=df["Weeks"], ax=ax2, palette="winter", orient="h").set_title(
    "WEEKS DETAILS"
)



## === cell 12
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
sns.boxplot(x=df["Sex"], y=df["FVC"], palette="winter", orient="v", ax=ax1).set_title(
    "GENDER VS FVC"
)
sns.boxplot(
    x=df["SmokingStatus"], y=df["FVC"], palette="winter", orient="v", ax=ax2
).set_title("SMOKING STAT VS FVC")



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
print("FVC decreases over time for most of the cases")
plt.figure(figsize=(15, 10))
a = sns.lineplot(x=df["Weeks"], y=df["FVC"], hue=df["Patient"], size=1, legend=False)



## === cell 15
df["Photo count"] = 0
names = df["Patient"].unique()
for name in names:
    file = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/" + name
    df.loc[df["Patient"] == name, "Photo count"] = len(os.listdir(file))

data = df.groupby(by="Patient")["Photo count"].first().reset_index(drop=False)
data = data.sort_values(["Photo count"]).reset_index(drop=True)
data["Photo count"].describe()



## === cell 16
plt.figure(figsize=(20, 5))
sns.histplot(data["Photo count"], bins=200, kde=False)



## === cell 17
patient_dir = (
    "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430"
)

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

for i in range(columns * rows):
    img = datasets[i].pixel_array
    fig.add_subplot(rows, columns, i + 1)
    plt.imshow(img, cmap="plasma")
    plt.axis("off")




## === cell 18
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)



## === cell 19
df.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])



## === cell 20
sample["Patient"] = sample["Patient_Week"].apply(lambda x: x.split("_")[0])
sample["Weeks"] = sample["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sample = sample[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sample = sample.merge(test.drop("Weeks", axis=1), on="Patient")



## === cell 21
df["WHERE"] = "train"
test["WHERE"] = "val"
sample["WHERE"] = "test"
data = pd.concat([df, test, sample], axis=0, ignore_index=True)



## === cell 22
data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")



## === cell 23
base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)



## === cell 24
data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base
gc.collect()



## === cell 25
data[["Sex_Male", "SmokingStatus_Ex-smoker", "SmokingStatus_Never smoked"]] = (
    pd.get_dummies(data[["Sex", "SmokingStatus"]], drop_first=True)
)

features = ["Percent", "Age", "min_FVC", "base_week"]



## === cell 26
from sklearn.preprocessing import MinMaxScaler

train_mask = data["WHERE"].eq("train")

for i in features:
    scaler = MinMaxScaler()
    scaler.fit(data.loc[train_mask, [i]])
    data.loc[:, i] = scaler.transform(data[[i]])



## === cell 27
df = data.loc[data.WHERE == "train"].drop("WHERE", axis=1)
test = data.loc[data.WHERE == "val"].drop("WHERE", axis=1)
sample = data.loc[data.WHERE == "test"].drop("WHERE", axis=1)



## === cell 28
features += ["Sex_Male", "SmokingStatus_Ex-smoker", "SmokingStatus_Never smoked"]



## === cell 29
df[features].head()



## === cell 30
X = df[features].values
y = df["FVC"].values
test_ = sample[features].values

y = y.astype("float32").reshape(-1, 1)



## === cell 31
nh = X.shape[1]
pe = np.zeros((test_.shape[0], 3), dtype=np.float32)
pred = np.zeros((X.shape[0], 3), dtype=np.float32)



## === cell 32
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.constant(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
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
    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, decay=0.01, amsgrad=False
        ),
        metrics=[score],
    )
    return model




## === cell 33
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)
test_ = np.asarray(test_, dtype=np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cnt = 0
EPOCHS = 800

for tr_idx, val_idx in kf.split(X):
    cnt += 1
    print(f"FOLD {cnt}")
    model = make_model(nh)
    model.fit(
        X[tr_idx],
        y[tr_idx],
        batch_size=128,
        epochs=EPOCHS,
        validation_data=(X[val_idx], y[val_idx]),
        verbose=0,
    )
    print("train", model.evaluate(X[tr_idx], y[tr_idx], verbose=0, batch_size=128))
    print("val", model.evaluate(X[val_idx], y[val_idx], verbose=0, batch_size=128))
    print("predict val...")
    pred[val_idx] = model.predict(X[val_idx], batch_size=128, verbose=0)
    print("predict test...")
    pe += model.predict(test_, batch_size=128, verbose=0) / 5.0



## === cell 34
sigma_opt = mean_absolute_error(y[:, 0], pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
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



## === cell 39
otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    pw = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == pw, "FVC"] = otest.FVC[i]
    subm.loc[subm["Patient_Week"] == pw, "Confidence"] = 70.0

subm["Confidence"] = subm["Confidence"].astype("float32")
subm["Confidence"] = np.maximum(subm["Confidence"].values, 70.0)
subm["Confidence"] = np.minimum(subm["Confidence"].values, 300.0)



## === cell 40
subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    subm[["Patient_Week", "FVC", "Confidence"]].shape,
)
print(subm[["Patient_Week", "FVC", "Confidence"]].head())
