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

3.9

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

-6.9198

# 6. Current score

-8.76207

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76207) has done: 'I fix the runtime errors caused by outdated seaborn/pandas/pydicom APIs so the notebook can execute end-to-end in the Kaggle environment. I also fix the broken dataframe construction (pandas `append` removal) so `data` is defined and all downstream feature engineering runs. For the model, I remove the incompatible `tensorflow_addons` import (which triggers the protobuf `MessageFactory` error) and add the missing Keras backend import before using `K.mean`, without changing the model architecture or the custom loss/metric logic. Finally, I ensure the output submission file is written as `submission.csv` with the exact required columns and that baseline test rows are set to the known FVC with confidence 70.'
- What this solution (achieved -8.76207) has done: 'I fix the two blockers that prevent training from running: the protobuf `MessageFactory.GetPrototype` crash (caused by importing the standalone `keras` package alongside `tensorflow.keras`) and the `Invalid dtype: object` error (caused by non-numeric columns in `Weeks` and/or dummy columns being `object`). The minimal fix is to use `tensorflow.keras.backend` consistently (no standalone `keras` import) and to force all model inputs/targets to numeric `float32` arrays after feature assembly. These are correctness/stability changes that also typically improve score slightly by ensuring the network actually trains on the intended numeric features. I also guard the plotting cell so it doesn’t crash if training fails early, while keeping the modeling/training logic unchanged.'
- What this solution (achieved -8.76207) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by avoiding the standalone `keras` package entirely and using only `tf.keras` (this is a known incompatibility in Kaggle TF environments). I also fix the `None values not supported` training crash by ensuring `y_true` has the correct shape `(N, 1)` inside the custom losses/metric and by defensively sanitizing all model inputs/targets for NaN/Inf right before fitting. These are minimal correctness changes that do not alter the model architecture or training loop, but ensure the network actually trains and should improve score toward the target. Finally, I keep the submission-writing logic intact but add small safety casts/clips so the produced `submission.csv` is always valid.'
- What this solution (achieved -8.76207) has done: 'I fix the two runtime blockers without changing the model architecture or training semantics: (1) avoid importing `pandas_profiling`/`ydata_profiling` at import-time because it triggers the protobuf `MessageFactory` crash in this environment, and (2) fix the `None values not supported` error by ensuring the feature matrix has no `None` objects (convert via `pd.DataFrame(...).apply(pd.to_numeric)` before `np.nan_to_num`). These changes are correctness/stability focused so training runs end-to-end, and should also improve score vs a non-training/partially-broken pipeline. I keep the same loss/metric/model and submission-writing logic, and still write `submission.csv` with the required columns.'
- What this solution (achieved -8.76207) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by removing the incompatible optional profiling imports and by enforcing `tf.keras` usage only, which is the root cause of the current cell-0 failure in this Kaggle environment. Then I fix the `None values not supported` error during `model.fit()` by sanitizing `X` and `y` right before training (including an explicit float32 cast and a final `np.nan_to_num` pass that also replaces any `None` that might have slipped in via object arrays). Finally, to move the score toward the target (higher is better), I make a minimal, metric-consistent calibration tweak: ensure predicted Confidence is always positive and set to a stable value derived from OOF MAE (clipped at 70), which usually improves Laplace log-likelihood without changing the model architecture or training loop.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ENABLE_PROFILING = False
ProfileReport = None

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras.models import Model

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
if ENABLE_PROFILING and ProfileReport is not None:
    _ = ProfileReport(train, progress_bar=False, minimal=True)



## === cell 3
if ENABLE_PROFILING and ProfileReport is not None:
    _ = ProfileReport(test, progress_bar=False, minimal=True)



## === cell 4
if ENABLE_PROFILING and ProfileReport is not None:
    _ = ProfileReport(sub, progress_bar=False, minimal=True)



## === cell 5
print("No of unique patients:", len(train.Patient.unique()))

readings = train.groupby("Patient").Weeks.count()
print("Min no. of readings for a patient:", int(min(readings)))
print("Max no. of readings for a patient:", int(max(readings)))

fig = plt.figure(figsize=(15, 5))
sns.barplot(x=readings.index, y=readings.values, color="#7AC8BE")
plt.title("Number of Readings per Patient", size=15)
plt.xlabel("Patient", size=12)
plt.ylabel("# Readings", size=12)
plt.xticks([])



## === cell 6
print("Minimum aged patient:", min(train["Age"]))
print("Maximum aged patient:", max(train["Age"]))

fig = plt.figure(figsize=(10, 5))
sns.histplot(train["Age"], kde=True)
plt.title("Age Distribution", size=15)
plt.xlabel("Age", size=12)



## === cell 7
sex = train.groupby("Patient").Sex.first()
print("Male Patients:", int((sex == "Male").sum()))
print("Female Patients:", int((sex == "Female").sum()))

fig = plt.figure(figsize=(5, 5))
sns.countplot(x=sex.values, order=["Male", "Female"])
plt.title("Sex Distribution", size=15)
plt.ylabel("# Patients", size=12)
plt.xlabel("Sex", size=12)



## === cell 8
smoke = train.groupby("Patient").SmokingStatus.first()
print("Ex-smokers:", int((smoke == "Ex-smoker").sum()))
print("Patients who never smoked:", int((smoke == "Never smoked").sum()))
print("Patients who currently smoke:", int((smoke == "Currently smokes").sum()))

fig = plt.figure(figsize=(6, 5))
sns.countplot(x=smoke.values, order=["Ex-smoker", "Never smoked", "Currently smokes"])
plt.title("Smoking Status", size=15)
plt.ylabel("# Patients", size=12)
plt.xlabel("Status", size=12)
plt.xticks(rotation=20)



## === cell 9
print("Maximum FVC value:", max(train["FVC"]))
print("Minimum FVC value:", min(train["FVC"]))

fig = plt.figure(figsize=(10, 5))
sns.histplot(train["FVC"], kde=True)
plt.title("FVC Value Distribution", size=15)
plt.xlabel("FVC Value", size=12)



## === cell 10
print("Maximum Percentage:", max(train["Percent"]))
print("Minimum Percentage:", min(train["Percent"]))

fig = plt.figure(figsize=(10, 5))
sns.histplot(train["Percent"], kde=True)
plt.title("Percentage Distribution", size=15)
plt.xlabel("Percent", size=12)



## === cell 11
a = train[["Age", "SmokingStatus", "Percent"]]
fig = plt.figure(figsize=(15, 5))
for i in range(len(a.columns)):
    fig.add_subplot(1, 3, i + 1)
    sns.scatterplot(
        x=a.iloc[:, i], y=train["FVC"], hue=train["Sex"], palette=["blue", "red"]
    )
plt.tight_layout()
plt.show()



## === cell 12
import pydicom as dicom
import cv2

data_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"
patients = os.listdir(data_dir)

labels_df = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
labels_df.head()



## === cell 13
for patient in patients[:1]:
    path = data_dir + "/" + patient
    slices = [dicom.dcmread(path + "/" + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    print("No. of scans:", len(slices))
    print("Height and width of the scan:", slices[0].pixel_array.shape)
    print("\nMetadata of the Dicom File:")
    print(slices[1])



## === cell 14
c = 0
for patient in patients:
    try:
        path = data_dir + "/" + patient
        slices = [dicom.dcmread(path + "/" + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        print(len(slices), slices[0].pixel_array.shape)
        c += 1
        if c == 5:
            break
    except Exception:
        continue



## === cell 15
min_s = 9999
max_s = 0
for patient in patients[:]:
    path = data_dir + "/" + patient
    n_scans = len(os.listdir(path))
    if n_scans < min_s:
        min_s = n_scans
    if n_scans > max_s:
        max_s = n_scans
print("Minimum number of scans for any patient:", min_s)
print("Maximum number of scans for any patient:", max_s)



## === cell 16
for patient in patients[1:2]:
    path = data_dir + "/" + patient
    slices = [dicom.dcmread(path + "/" + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    fig = plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title("CT Scan", size=15)
    plt.imshow(slices[0].pixel_array, cmap="gray")
    plt.show()



## === cell 17
for patient in patients:
    path = data_dir + "/" + patient
    slices = [dicom.dcmread(path + "/" + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    img_px_size = 150

    try:
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
    except Exception:
        continue




## === cell 18
def load_scan(path):
    slices = [dicom.dcmread(path + "/" + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except Exception:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices




## === cell 19
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    image[image == -2000] = 0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)




## === cell 20
first_patient = load_scan(data_dir + "/" + patients[0])
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color="c")
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

idx = min(80, first_patient_pixels.shape[0] - 1)
plt.imshow(first_patient_pixels[idx], cmap=plt.cm.gray)
plt.show()



## === cell 21
drop = train[train.duplicated(subset=["Patient", "Weeks"], keep="last")]
print("No. of rows to be dropped:", drop.shape[0])
train.drop_duplicates(subset=["Patient", "Weeks"], keep="last", inplace=True)



## === cell 22
sub[["Patient", "Weeks"]] = sub.Patient_Week.str.split("_", expand=True)
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub.head()



## === cell 23
sub = sub.merge(test.drop("Weeks", axis=1), on="Patient")
sub.head()



## === cell 24
train["Dataset"] = "train"
sub["Dataset"] = "test"

data = pd.concat([train, sub], axis=0, ignore_index=True)
data.head()



## === cell 25
data = pd.concat(
    [data, pd.get_dummies(data.Sex), pd.get_dummies(data.SmokingStatus)], axis=1
)

data.drop(["Sex", "SmokingStatus"], axis=1, inplace=True)

data["Weeks"] = pd.to_numeric(data["Weeks"], errors="coerce").fillna(0).astype("int64")
data.head()




## === cell 26
def get_baseline(df):
    _df = df.copy()
    _df["min_week"] = _df["Weeks"]
    _df.loc[_df.Dataset == "test", "min_week"] = 0
    _df["min_week"] = _df.groupby("Patient")["Weeks"].transform("min")
    _df["baselined_week"] = _df["Weeks"] - _df["min_week"]

    return _df


data["Weeks"] = data["Weeks"].astype("int64")
data = get_baseline(data)
data.head()




## === cell 27
def get_baseline_FVC(df):
    _df = df.copy()
    base = _df.loc[_df.Weeks == _df.min_week]
    base = base[["Patient", "FVC"]].copy()
    base.columns = ["Patient", "base_FVC"]

    base["nb"] = 1
    base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")

    base = base[base.nb == 1]
    base.drop("nb", axis=1, inplace=True)

    _df = _df.merge(base, on="Patient", how="left")
    _df = _df.drop(["min_week"], axis=1)

    return _df


data = get_baseline_FVC(data)
data.head()




## === cell 28
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



## === cell 29
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def _ensure_ytrue_2d(y_true):
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.reshape(y_true, (-1, 1))
    return y_true


def score(y_true, y_pred):
    """Competition metric surrogate (lower is better here because we return positive form)."""
    y_true = _ensure_ytrue_2d(y_true)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    """Pinball loss"""
    y_true = _ensure_ytrue_2d(y_true)
    y_pred = tf.cast(y_pred, tf.float32)
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    """Combine Score and qloss"""

    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss




## === cell 30
def make_model(nh):
    z = Input((nh,), name="Patient")
    x = Dense(100, activation="elu", name="d1")(z)
    x = Dense(100, activation="elu", name="d3")(x)
    p1 = Dense(3, activation="linear", name="p1")(x)
    p2 = Dense(3, activation="elu", name="p2")(x)
    preds = Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    model = Model(z, preds, name="CNN")
    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
        ),
        metrics=[score],
    )
    return model




## === cell 31
for col in ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]:
    if col not in data.columns:
        data[col] = 0

features_list = [
    "baselined_week",
    "Percent",
    "Age",
    "base_FVC",
    "Male",
    "Female",
    "Ex-smoker",
    "Never smoked",
    "Currently smokes",
]
train_df = data.loc[data.Dataset == "train"].copy()
sub_df = data.loc[data.Dataset == "test"].copy()

for c in features_list:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    sub_df[c] = pd.to_numeric(sub_df[c], errors="coerce")
train_df[features_list] = train_df[features_list].fillna(0.0)
sub_df[features_list] = sub_df[features_list].fillna(0.0)

y = pd.to_numeric(train_df["FVC"], errors="coerce").fillna(train_df["FVC"].median())
y = y.values.astype(np.float32).reshape(-1, 1)

X_train_df = (
    train_df[features_list].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
)
X_test_df = (
    sub_df[features_list].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
)

X_train = X_train_df.to_numpy(dtype=np.float32, copy=True)
X_test = X_test_df.to_numpy(dtype=np.float32, copy=True)
n_rows = X_train.shape[1]

X_train = np.asarray(X_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y_med = float(np.nanmedian(y)) if np.isfinite(np.nanmedian(y)) else 0.0
y = np.nan_to_num(y, nan=y_med, posinf=y_med, neginf=y_med).astype(np.float32)

train_preds = np.zeros((X_train.shape[0], 3), dtype=np.float32)
test_preds = np.zeros((X_test.shape[0], 3), dtype=np.float32)



## === cell 32
model = make_model(n_rows)
print(model.summary())



## === cell 33
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.4, patience=150, verbose=0, min_delta=1e-4, mode="min"
)

NFOLD = 6
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)
OOF_val_score = []

cnt = 0
BATCH_SIZE = 128
EPOCHS = 800

history = None
for tr_idx, val_idx in kf.split(X_train):
    cnt += 1
    print(f"FOLD {cnt}")
    model = make_model(n_rows)

    Xtr = np.ascontiguousarray(X_train[tr_idx], dtype=np.float32)
    Xva = np.ascontiguousarray(X_train[val_idx], dtype=np.float32)
    ytr = np.ascontiguousarray(y[tr_idx].reshape(-1, 1), dtype=np.float32)
    yva = np.ascontiguousarray(y[val_idx].reshape(-1, 1), dtype=np.float32)

    Xtr = np.nan_to_num(
        np.asarray(Xtr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    Xva = np.nan_to_num(
        np.asarray(Xva, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    ytr = np.nan_to_num(
        np.asarray(ytr, dtype=np.float32), nan=y_med, posinf=y_med, neginf=y_med
    )
    yva = np.nan_to_num(
        np.asarray(yva, dtype=np.float32), nan=y_med, posinf=y_med, neginf=y_med
    )

    history = model.fit(
        Xtr,
        ytr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(Xva, yva),
        verbose=0,
        callbacks=[reduce_lr_loss],
    )
    print(
        "train",
        model.evaluate(Xtr, ytr, verbose=0, batch_size=BATCH_SIZE),
    )
    print(
        "val",
        model.evaluate(Xva, yva, verbose=0, batch_size=BATCH_SIZE),
    )
    print("predict val...")
    train_preds[val_idx] = model.predict(Xva, batch_size=BATCH_SIZE, verbose=0)

    OOF_val_score.append(
        model.evaluate(
            Xva,
            yva,
            verbose=0,
            batch_size=BATCH_SIZE,
            return_dict=True,
        )["score"]
    )

    print("predict test...")
    test_preds += model.predict(X_test, batch_size=BATCH_SIZE, verbose=0) / NFOLD



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3823195515.py in <cell line: 0>()
     39     )
     40 
---> 41     history = model.fit(
     42         Xtr,
     43         ytr,

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
    plt.title("Training and Validation Score (lower is better)")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss_hist, label="Training Loss")
    plt.plot(epochs_range, val_loss_hist, label="Validation Loss")
    if len(val_loss_hist) > 0:
        plt.ylim(0.3 * np.mean(val_loss_hist), 1.8 * np.mean(val_loss_hist))
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()



## === cell 35
np.mean(OOF_val_score) if len(OOF_val_score) else None



## === cell 36
sigma_opt = mean_absolute_error(y, train_preds[:, 1].reshape(-1, 1))
sigma_uncertain = train_preds[:, 2] - train_preds[:, 0]
sigma_mean = np.mean(sigma_uncertain)
print(sigma_opt, sigma_mean)



## === cell 37
sub_df = sub_df.copy()
sub_df["FVC1"] = test_preds[:, 1]

sub_df["Confidence1"] = np.abs(test_preds[:, 2] - test_preds[:, 0])

submission = sub_df[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()
submission.loc[~submission.FVC1.isnull()].head(10)



## === cell 38
submission.loc[~submission.FVC1.isnull(), "FVC"] = submission.loc[
    ~submission.FVC1.isnull(), "FVC1"
]

calibrated_conf = float(max(70.0, sigma_opt))
submission.loc[~submission.FVC1.isnull(), "Confidence"] = calibrated_conf

submission["Confidence"] = pd.to_numeric(
    submission["Confidence"], errors="coerce"
).fillna(70.0)
submission["Confidence"] = submission["Confidence"].clip(lower=70)

submission["FVC"] = pd.to_numeric(submission["FVC"], errors="coerce")
submission["FVC"] = submission["FVC"].fillna(submission["FVC"].median())
submission["FVC"] = submission["FVC"].clip(lower=0)



## === cell 39
submission.describe().T



## === cell 40
org_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

for i in range(len(org_test)):
    key = org_test.Patient[i] + "_" + str(org_test.Weeks[i])
    submission.loc[submission["Patient_Week"] == key, "FVC"] = float(org_test.FVC[i])
    submission.loc[submission["Patient_Week"] == key, "Confidence"] = 70.0



## === cell 41
submission[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    submission[["Patient_Week", "FVC", "Confidence"]].shape,
)
print(submission[["Patient_Week", "FVC", "Confidence"]].head())
