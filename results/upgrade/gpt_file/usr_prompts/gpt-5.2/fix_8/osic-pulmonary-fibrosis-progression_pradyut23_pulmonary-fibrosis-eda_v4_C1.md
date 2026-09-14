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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
tf_keras==2.18.0
ydata-profiling==4.17.0

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

-6.9708

# 6. Current score

-8.24736

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76207) has done: 'I fix the notebook so it runs end-to-end by updating deprecated/changed APIs (seaborn call signatures, pandas `.append`, `pydicom.read_file`, and removing the incompatible `pandas_profiling` import that breaks under the current environment). I also correct the dataset merge/splitting logic so `data` is created deterministically, weeks are parsed as integers, and dummy columns are guaranteed to exist (even if a category is missing). Finally, I fix the TensorFlow/Keras issues by importing `K` before it’s used, removing the unavailable `tensorflow_addons` import, updating the optimizer arguments, and ensuring the model is trained with correctly shaped targets (replicating FVC to 3 quantiles) so predictions and submission creation work and write `submission.csv`.'
- What this solution (achieved -7.92788) has done: 'The runtime errors come from mixing Keras 3 backend APIs (`keras.backend` no longer exposes `mean`) and a protobuf incompatibility that can be triggered by some TF/Keras import paths. I fix this by removing reliance on `K.mean` and using pure `tf.reduce_mean`/`tf.math` in the custom loss/metric (same math, score-neutral aside from negligible FP differences), and by using `tf.keras` consistently for layers/models to avoid the protobuf `MessageFactory.GetPrototype` crash. I also make the training/plotting robust so later cells don’t crash when training fails, ensuring `submission.csv` is always written. Core model architecture, training loop, features, and loss formulation remain unchanged.'
- What this solution (achieved -8.66395) has done: 'I fix the runtime crash in the TensorFlow/Keras import path that’s triggering the protobuf `MessageFactory.GetPrototype` error by forcing the pure-python protobuf implementation before importing TensorFlow, which is the smallest change that unblocks training/inference. I keep your model, loss, training loop, and feature pipeline identical, only adding determinism (seeds) and a defensive NaN/inf guard for `sigma` so the metric/loss can’t blow up and ruin predictions. This should run end-to-end, write a valid `submission.csv`, and is expected to improve score modestly by preventing invalid confidence values and ensuring training actually completes. No changes are made to architecture, layers, or overall optimization setup beyond stability fixes.'
- What this solution (achieved -8.83969) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure‑python protobuf implementation *and* ensuring it’s applied early enough by restarting the Python module state before importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error). I also add a small, score-positive calibration fix for `Confidence`: keep your model outputs/architecture unchanged, but compute a robust `sigma_opt` and use it consistently (clipped to ≥70) to avoid overconfident predictions, which typically improves the Laplace log-likelihood toward your target. Finally, I make submission generation more robust (type coercion, finite checks) while keeping the same column names and writing `submission.csv` end-to-end.'
- What this solution (achieved -9.17442) has done: 'I fix the protobuf/TensorFlow crash causing `MessageFactory.GetPrototype` by ensuring we use the C++ protobuf runtime (not the pure-python one) and by importing TensorFlow first in a clean, consistent way. This is a minimal, score-neutral change that unblocks training/inference so the pipeline runs end-to-end. I also add small defensive guards so the custom loss/metric and `sigma_opt` computation can’t produce NaN/inf and break submission generation. The model, loss math, features, folds, epochs, and submission formatting remain the same so your score should improve mainly because the model actually trains and produces calibrated confidence.'
- What this solution (achieved -8.90267) has done: 'I fix the runtime crash (`MessageFactory` has no `GetPrototype`) by forcing the pure‑python protobuf implementation *before* importing TensorFlow, which is the smallest reliable workaround in this Kaggle environment. I keep your model/loss/training loop identical, but add lightweight guards to ensure the custom loss functions always receive the expected shapes and remain finite (score-neutral aside from preventing NaNs). To nudge the score toward the target (higher is better) without changing core logic, I calibrate `Confidence` per row using the model’s predicted interval width blended with `sigma_opt` and clipped at 70, which usually improves Laplace log-likelihood vs using a constant sigma for all rows. The script still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.24736) has done: 'We need to fix the protobuf/TensorFlow crash (`MessageFactory` missing `GetPrototype`) that happens at the first `import tensorflow as tf`, because it prevents training and therefore prevents producing a stable submission and score improvements. The minimal reliable fix in this Kaggle setup is to force the C++ protobuf runtime (not the python one) *before any TensorFlow import* and to avoid setting the python protobuf implementation env vars that trigger the crash. I keep your model, losses, folds, epochs, and feature pipeline the same; the only score-impacting change is that training actually run again, which should move your score back toward the target band. I also add a tiny safeguard to ensure `submission.csv` is always written even if some training fold fails, without changing predictions when training succeeds.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



## === cell 1
train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
print("Train Data:")
print(train.head())

test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
print("\n\nTest Data:")
print(test.head())

sub = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
print("\n\nSubmission File:")
print(sub.head())



## === cell 2
print(
    "Skipped ProfileReport(train) to avoid pandas_profiling/protobuf runtime incompatibilities."
)



## === cell 3
print(
    "Skipped ProfileReport(test) to avoid pandas_profiling/protobuf runtime incompatibilities."
)



## === cell 4
print(
    "Skipped ProfileReport(sub) to avoid pandas_profiling/protobuf runtime incompatibilities."
)



## === cell 5
print("Null values present in any column?")
print(train.isnull().any())



## === cell 6
print("No of unique patients:", len(train.Patient.unique()))

readings = train.groupby("Patient").Weeks.count()
print("Min no. of readings for a patient:", int(readings.min()))
print("Max no. of readings for a patient:", int(readings.max()))

fig = plt.figure(figsize=(15, 5))
sns.barplot(x=readings.index, y=readings.values, color="#7AC8BE")
plt.title("Number of Readings per Patient", size=15)
plt.xlabel("Patient", size=12)
plt.ylabel("# Readings", size=12)
plt.xticks([])
plt.show()



## === cell 7
print("Minimum aged patient:", int(train["Age"].min()))
print("Maximum aged patient:", int(train["Age"].max()))

fig = plt.figure(figsize=(10, 5))
sns.histplot(train["Age"], kde=True)
plt.title("Age Distribution", size=15)
plt.xlabel("Age", size=12)
plt.show()



## === cell 8
sex = train.groupby("Patient").Sex.first()
print("Male Patients:", int((sex == "Male").sum()))
print("Female Patients:", int((sex == "Female").sum()))

fig = plt.figure(figsize=(5, 5))
sns.countplot(x=sex.values)
plt.title("Sex Distribution", size=15)
plt.ylabel("# Patients", size=12)
plt.xlabel("Sex", size=12)
plt.show()



## === cell 9
smoke = train.groupby("Patient").SmokingStatus.first()
print("Ex-smokers:", int((smoke == "Ex-smoker").sum()))
print("Patients who never smoked:", int((smoke == "Never smoked").sum()))
print("Patients who currently smoke:", int((smoke == "Currently smokes").sum()))

fig = plt.figure(figsize=(6, 5))
sns.countplot(x=smoke.values)
plt.title("Smoking Status", size=15)
plt.ylabel("# Patients", size=12)
plt.xlabel("Status", size=12)
plt.xticks(rotation=15)
plt.show()



## === cell 10
print("Maximum FVC value:", int(train["FVC"].max()))
print("Minimum FVC value:", int(train["FVC"].min()))

fig = plt.figure(figsize=(10, 5))
sns.histplot(train["FVC"], kde=True)
plt.title("FVC Value Distribution", size=15)
plt.xlabel("FVC Value", size=12)
plt.show()



## === cell 11
print("Maximum Percentage:", float(train["Percent"].max()))
print("Minimum Percentage:", float(train["Percent"].min()))

fig = plt.figure(figsize=(10, 5))
sns.histplot(train["Percent"], kde=True)
plt.title("Percentage Distribution", size=15)
plt.xlabel("Percent", size=12)
plt.show()



## === cell 12
a = train[["Age", "SmokingStatus", "Percent"]]
fig = plt.figure(figsize=(15, 5))
for i in range(len(a.columns)):
    fig.add_subplot(1, 3, i + 1)
    sns.scatterplot(
        x=a.iloc[:, i], y=train["FVC"], hue=train["Sex"], palette=["blue", "red"]
    )
plt.tight_layout()
plt.show()



## === cell 13
import pydicom as dicom
import cv2

data_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"
patients = [p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]
labels_df = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
labels_df.head()



## === cell 14
for patient in patients[:1]:
    label = labels_df.loc[labels_df.Patient == patient, "FVC"].iloc[0]
    path = os.path.join(data_dir, patient)
    slices = [
        dicom.dcmread(os.path.join(path, s))
        for s in os.listdir(path)
        if s.lower().endswith(".dcm")
    ]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    print("Patient:", patient, "Example FVC:", label)
    print("No. of scans:", len(slices))
    print("Height and width of the scan:", slices[0].pixel_array.shape)
    print("\nMetadata of the Dicom File:")
    print(slices[0])



## === cell 15
for patient in patients[:5]:
    label = labels_df.loc[labels_df.Patient == patient, "FVC"].iloc[0]
    path = os.path.join(data_dir, patient)
    slices = [
        dicom.dcmread(os.path.join(path, s))
        for s in os.listdir(path)
        if s.lower().endswith(".dcm")
    ]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    print(patient, len(slices), slices[0].pixel_array.shape, "example FVC", label)



## === cell 16
min_s = 999999
max_s = 0
for patient in patients:
    path = os.path.join(data_dir, patient)
    n_slices = len([s for s in os.listdir(path) if s.lower().endswith(".dcm")])
    min_s = min(min_s, n_slices)
    max_s = max(max_s, n_slices)

print("Minimum number of scans for any patient:", int(min_s))
print("Maximum number of scans for any patient:", int(max_s))



## === cell 17
for patient in patients[1:2]:
    path = os.path.join(data_dir, patient)
    slices = [
        dicom.dcmread(os.path.join(path, s))
        for s in os.listdir(path)
        if s.lower().endswith(".dcm")
    ]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    fig = plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title("CT Scan", size=15)
    plt.imshow(slices[0].pixel_array, cmap="gray")
    plt.show()



## === cell 18
for patient in patients:
    path = os.path.join(data_dir, patient)
    slices = [
        dicom.dcmread(os.path.join(path, s))
        for s in os.listdir(path)
        if s.lower().endswith(".dcm")
    ]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    img_px_size = 150

    if len(slices) <= 56:
        fig = plt.figure(figsize=(20, 40))
        for num, each_slice in enumerate(slices):
            fig.add_subplot(14, 4, num + 1)
            new_image = cv2.resize(
                np.array(each_slice.pixel_array), (img_px_size, img_px_size)
            )
            plt.axis("off")
            plt.title(num + 1, size=10)
            plt.imshow(new_image, cmap="gray")
        plt.show()
        break



## === cell 19
drop = train[train.duplicated(subset=["Patient", "Weeks"], keep="last")]
print("No. of rows to be dropped:", drop.shape[0])
train.drop_duplicates(subset=["Patient", "Weeks"], keep="last", inplace=True)



## === cell 20
sub[["Patient", "Weeks"]] = sub.Patient_Week.str.split("_", expand=True)
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]].copy()
sub["Weeks"] = sub["Weeks"].astype(int)
sub.head()



## === cell 21
sub = sub.merge(test.drop("Weeks", axis=1), on="Patient", how="left")
sub.head()



## === cell 22
train = train.copy()
sub = sub.copy()
train["Dataset"] = "train"
sub["Dataset"] = "test"

data = pd.concat([train, sub], axis=0, ignore_index=True)
data.head()



## === cell 23
data = pd.concat(
    [data, pd.get_dummies(data.Sex), pd.get_dummies(data.SmokingStatus)], axis=1
)
data.drop(["Sex", "SmokingStatus"], axis=1, inplace=True)

data["Weeks"] = data["Weeks"].astype("int64")
data.head()




## === cell 24
def get_baseline(df):
    _df = df.copy()

    _df["min_week"] = _df.groupby("Patient")["Weeks"].transform("min")
    _df.loc[_df.Dataset == "test", "min_week"] = 0

    _df["baselined_week"] = _df["Weeks"] - _df["min_week"]
    return _df


data = get_baseline(data)
data.head()




## === cell 25
def get_baseline_FVC(df):
    _df = df.copy()
    base = _df.loc[_df.Weeks == _df.min_week, ["Patient", "FVC"]].copy()
    base.columns = ["Patient", "base_FVC"]

    base["nb"] = 1
    base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
    base = base[base.nb == 1].drop("nb", axis=1)

    _df = _df.merge(base, on="Patient", how="left")
    return _df


data = get_baseline_FVC(data)
data.head()




## === cell 26
def scaling(series):
    denom = series.max() - series.min()
    if denom == 0:
        return series * 0.0
    return (series - series.min()) / denom


data["Age"] = scaling(data["Age"])
data["Percent"] = scaling(data["Percent"])
data["baselined_week"] = scaling(data["baselined_week"])
data["base_FVC"] = scaling(data["base_FVC"])
data.head()



## === cell 27
import tensorflow as tf

tf.random.set_seed(RANDOM_SEED)

from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras.models import Model

C1, C2 = tf.constant(70.0, dtype=tf.float32), tf.constant(1000.0, dtype=tf.float32)


def score(y_true, y_pred):
    """Internal form of competition metric used as a training term (lower is better)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, (-1, 3))
    y_pred = tf.reshape(y_pred, (-1, 3))

    sigma = y_pred[:, 2] - y_pred[:, 0]
    sigma = tf.where(tf.math.is_finite(sigma), sigma, C1)
    sigma = tf.maximum(sigma, tf.constant(1e-3, dtype=tf.float32))

    fvc_pred = y_pred[:, 1]
    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt(tf.constant(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    metric = tf.where(tf.math.is_finite(metric), metric, tf.zeros_like(metric))
    return tf.reduce_mean(metric)


def qloss(y_true, y_pred):
    """Pinball loss over three quantiles."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, (-1, 3))
    y_pred = tf.reshape(y_pred, (-1, 3))

    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1.0) * e)
    v = tf.where(tf.math.is_finite(v), v, tf.zeros_like(v))
    return tf.reduce_mean(v)


def mloss(_lambda):
    """Combine Score and qloss."""

    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1.0 - _lambda) * score(y_true, y_pred)

    return loss




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 28
def make_model(nh):
    z = Input((nh,), name="Patient")
    x = Dense(100, activation="relu", name="d1")(z)
    x = Dense(100, activation="relu", name="d2")(x)
    p1 = Dense(3, activation="linear", name="p1")(x)
    p2 = Dense(3, activation="relu", name="p2")(x)
    preds = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    model = Model(z, preds, name="CNN")

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    )
    model.compile(loss=mloss(0.8), optimizer=opt, metrics=[score])
    return model




## === cell 29
expected_cols = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
for c in expected_cols:
    if c not in data.columns:
        data[c] = 0

features_list = ["baselined_week", "Percent", "Age", "base_FVC"] + expected_cols

train_df = data.loc[data.Dataset == "train"].copy()
sub_df = data.loc[data.Dataset == "test"].copy()

y = train_df["FVC"].values.astype(np.float32)
y3 = np.stack([y, y, y], axis=1).astype(np.float32)

X_train = train_df[features_list].values.astype(np.float32)
X_test = sub_df[features_list].values.astype(np.float32)
nh = X_train.shape[1]

train_preds = np.zeros((X_train.shape[0], 3), dtype=np.float32)
test_preds = np.zeros((X_test.shape[0], 3), dtype=np.float32)

print(
    "X_train shape:",
    X_train.shape,
    "y3 shape:",
    y3.shape,
    "X_test shape:",
    X_test.shape,
)



## === cell 30
net = make_model(nh)
print(net.summary())
print("Params:", net.count_params())



## === cell 31
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=RANDOM_SEED)
OOF_val_score = []

cnt = 0
BATCH_SIZE = 128
EPOCHS = 800

last_history = None
for tr_idx, val_idx in kf.split(X_train):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)

    try:
        last_history = net.fit(
            X_train[tr_idx],
            y3[tr_idx],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_train[val_idx], y3[val_idx]),
            verbose=0,
        )

        print(
            "train",
            net.evaluate(X_train[tr_idx], y3[tr_idx], verbose=0, batch_size=BATCH_SIZE),
        )
        print(
            "val",
            net.evaluate(
                X_train[val_idx], y3[val_idx], verbose=0, batch_size=BATCH_SIZE
            ),
        )

        print("predict val...")
        train_preds[val_idx] = net.predict(
            X_train[val_idx], batch_size=BATCH_SIZE, verbose=0
        )

        OOF_val_score.append(
            net.evaluate(
                X_train[val_idx],
                y3[val_idx],
                verbose=0,
                batch_size=BATCH_SIZE,
                return_dict=True,
            )["score"]
        )

        print("predict test...")
        test_preds += net.predict(X_test, batch_size=BATCH_SIZE, verbose=0) / NFOLD
    except Exception as e:
        print("Fold failed with exception:", repr(e))
        continue



## === cell 32
history = last_history
if history is not None:
    score_hist = history.history.get("score", [])
    val_score_hist = history.history.get("val_score", [])

    loss_hist = history.history.get("loss", [])
    val_loss_hist = history.history.get("val_loss", [])

    epochs_range = range(len(loss_hist))

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, score_hist, label="Training score")
    plt.plot(epochs_range, val_score_hist, label="Validation score")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Score")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss_hist, label="Training Loss")
    plt.plot(epochs_range, val_loss_hist, label="Validation Loss")
    if len(val_loss_hist) > 0:
        plt.ylim(0.3 * np.mean(val_loss_hist), 1.8 * np.mean(val_loss_hist))
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()
else:
    print("No training history available to plot.")



## === cell 33
print(
    "Mean OOF score (lower is better in this internal metric):",
    float(np.mean(OOF_val_score)) if len(OOF_val_score) else float("nan"),
)



## === cell 34
sigma_opt = float(mean_absolute_error(y, train_preds[:, 1])) if y.size else 70.0
sigma_uncertain = train_preds[:, 2] - train_preds[:, 0]
sigma_uncertain = sigma_uncertain[np.isfinite(sigma_uncertain)]
sigma_mean = float(np.mean(sigma_uncertain)) if sigma_uncertain.size else float("nan")

if not np.isfinite(sigma_opt):
    sigma_opt = 70.0
sigma_opt = float(max(70.0, sigma_opt))

print("sigma_opt (clipped>=70):", sigma_opt, "sigma_mean:", sigma_mean)



## === cell 35
sub_df["FVC1"] = test_preds[:, 1]
sub_df["Confidence1"] = test_preds[:, 2] - test_preds[:, 0]

submission = sub_df[["Patient_Week"]].copy()

submission = submission.merge(
    pd.read_csv(
        "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
    )[["Patient_Week", "FVC", "Confidence"]],
    on="Patient_Week",
    how="left",
)

submission["FVC1"] = sub_df["FVC1"].values
submission["Confidence1"] = sub_df["Confidence1"].values
submission.loc[~submission.FVC1.isnull()].head(10)



## === cell 36
submission.loc[~submission.FVC1.isnull(), "FVC"] = submission.loc[
    ~submission.FVC1.isnull(), "FVC1"
]

conf = submission["Confidence1"].to_numpy(dtype=np.float64)
conf = np.where(np.isfinite(conf), conf, np.nan)
conf = np.abs(conf)

alpha = 0.30  # small, conservative weight toward per-row uncertainty
conf = (1.0 - alpha) * sigma_opt + alpha * conf

submission.loc[~submission.FVC1.isnull(), "Confidence"] = conf[
    ~submission.FVC1.isnull().to_numpy()
]

submission["Confidence"] = pd.to_numeric(submission["Confidence"], errors="coerce")
submission["Confidence"] = (
    submission["Confidence"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(sigma_opt)
    .astype(float)
    .clip(lower=70)
)



## === cell 37
print(submission[["FVC", "Confidence"]].describe().T)



## === cell 38
org_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

for i in range(len(org_test)):
    pw = org_test.Patient[i] + "_" + str(int(org_test.Weeks[i]))
    submission.loc[submission["Patient_Week"] == pw, "FVC"] = float(org_test.FVC[i])
    submission.loc[submission["Patient_Week"] == pw, "Confidence"] = 70.0



## === cell 39
submission[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    submission[["Patient_Week", "FVC", "Confidence"]].shape,
)
print(submission.head())
