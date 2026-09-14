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

-6.9239

# 6. Current score

-8.76207

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -8.76207) has done: 'I fix the runtime errors caused by newer seaborn/pandas/pydicom APIs (barplot/countplot argument changes, `DataFrame.append` removal, and `pydicom.read_file` deprecation) so the notebook runs end-to-end. I also remove the heavy `pandas_profiling` dependency (it is incompatible in this environment and not needed for training) and replace it with lightweight `DataFrame.describe()` calls to keep execution within the time limit. Next, I correct data preparation bugs that prevented `data` from being created (wrong index_col usage, incorrect baseline-week logic, and missing one-hot columns alignment), while preserving the core model and training loop semantics. Finally, I ensure the submission is written as a valid `submission.csv` with the required columns and that baseline test weeks are set to the known FVC with confidence 70 as intended.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



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
print("Train quick summary:")
print(train.describe(include="all").T.head(20))



## === cell 3
print("Test quick summary:")
print(test.describe(include="all").T.head(20))



## === cell 4
print("Submission template quick summary:")
print(sub.describe(include="all").T)



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
sns.countplot(x=sex.values, order=["Male", "Female"])
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
sns.countplot(x=smoke.values, order=["Ex-smoker", "Never smoked", "Currently smokes"])
plt.title("Smoking Status", size=15)
plt.ylabel("# Patients", size=12)
plt.xlabel("Status", size=12)
plt.xticks(rotation=20)
plt.show()



## === cell 10
print("Maximum FVC value:", float(train["FVC"].max()))
print("Minimum FVC value:", float(train["FVC"].min()))

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
a = train[["Age", "SmokingStatus", "Percent"]].copy()
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
    dcm_files = [
        os.path.join(path, s) for s in os.listdir(path) if s.lower().endswith(".dcm")
    ]
    slices = [dicom.dcmread(fp) for fp in dcm_files]
    slices.sort(
        key=lambda x: (
            float(x.ImagePositionPatient[2]) if "ImagePositionPatient" in x else 0.0
        )
    )
    print("No. of scans:", len(slices))
    print("Height and width of the scan:", slices[0].pixel_array.shape)
    print("\nMetadata of the Dicom File:")
    if len(slices) > 1:
        print(slices[1])



## === cell 15
c = 0
for patient in patients:
    try:
        path = os.path.join(data_dir, patient)
        dcm_files = [
            os.path.join(path, s)
            for s in os.listdir(path)
            if s.lower().endswith(".dcm")
        ]
        slices = [dicom.dcmread(fp) for fp in dcm_files]
        slices.sort(
            key=lambda x: (
                float(x.ImagePositionPatient[2]) if "ImagePositionPatient" in x else 0.0
            )
        )
        print(len(slices), slices[0].pixel_array.shape)
        c += 1
        if c == 5:
            break
    except Exception:
        continue



## === cell 16
min_s = 9999
max_s = 0
for patient in patients:
    path = os.path.join(data_dir, patient)
    files = [s for s in os.listdir(path) if s.lower().endswith(".dcm")]
    n = len(files)
    min_s = min(min_s, n)
    max_s = max(max_s, n)
print("Minimum number of scans for any patient:", int(min_s))
print("Maximum number of scans for any patient:", int(max_s))



## === cell 17
for patient in patients[1:2]:
    path = os.path.join(data_dir, patient)
    dcm_files = [
        os.path.join(path, s) for s in os.listdir(path) if s.lower().endswith(".dcm")
    ]
    slices = [dicom.dcmread(fp) for fp in dcm_files]
    slices.sort(
        key=lambda x: (
            float(x.ImagePositionPatient[2]) if "ImagePositionPatient" in x else 0.0
        )
    )

    fig = plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title("CT Scan", size=15)
    plt.imshow(slices[0].pixel_array, cmap="gray")
    plt.show()



## === cell 18
for patient in patients:
    path = os.path.join(data_dir, patient)
    dcm_files = [
        os.path.join(path, s) for s in os.listdir(path) if s.lower().endswith(".dcm")
    ]
    if len(dcm_files) <= 56:
        slices = [dicom.dcmread(fp) for fp in dcm_files]
        slices.sort(
            key=lambda x: (
                float(x.ImagePositionPatient[2]) if "ImagePositionPatient" in x else 0.0
            )
        )

        img_px_size = 150
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
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub.head()



## === cell 21
sub = sub.merge(test.drop("Weeks", axis=1), on="Patient")
sub.head()



## === cell 22
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
    _df.loc[_df["Dataset"] == "test", "min_week"] = 0

    _df["baselined_week"] = _df["Weeks"] - _df["min_week"]
    return _df


data = get_baseline(data)
data.head()




## === cell 25
def get_baseline_FVC(df):
    _df = df.copy()
    base = _df.loc[_df.Weeks == _df.min_week].copy()
    base = base[["Patient", "FVC"]].copy()
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
from keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras.models import Model

C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    """Calculate the competition metric (as loss-like quantity; lower is better)."""
    y_true = tf.cast(y_true, tf.float32)
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
    """Calculate Pinball loss"""
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




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 28
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




## === cell 29
expected_cols = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
for c in expected_cols:
    if c not in data.columns:
        data[c] = 0

features_list = ["baselined_week", "Percent", "Age", "base_FVC"] + expected_cols

train_df = data.loc[data.Dataset == "train"].copy()
sub_df = data.loc[data.Dataset == "test"].copy()

y = train_df["FVC"].values.astype(np.float32).reshape(-1, 1)

X_train = train_df[features_list].values.astype(np.float32)
X_test = sub_df[features_list].values.astype(np.float32)
n_rows = X_train.shape[1]

train_preds = np.zeros((X_train.shape[0], 3), dtype=np.float32)
test_preds = np.zeros((X_test.shape[0], 3), dtype=np.float32)

print("X_train:", X_train.shape, "y:", y.shape, "X_test:", X_test.shape)



## === cell 30
model = make_model(n_rows)
print(model.summary())



## === cell 31
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.4, patience=150, verbose=0, min_delta=1e-4, mode="min"
)

NFOLD = 6
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=RANDOM_STATE)
OOF_val_score = []

cnt = 0
BATCH_SIZE = 128
EPOCHS = 800

for tr_idx, val_idx in kf.split(X_train):
    cnt += 1
    print(f"FOLD {cnt}")
    model = make_model(n_rows)
    history = model.fit(
        X_train[tr_idx],
        y[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_train[val_idx], y[val_idx]),
        verbose=0,
        callbacks=[reduce_lr_loss],
    )
    print(
        "train",
        model.evaluate(X_train[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE),
    )
    print(
        "val",
        model.evaluate(X_train[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE),
    )
    print("predict val...")
    train_preds[val_idx] = model.predict(
        X_train[val_idx], batch_size=BATCH_SIZE, verbose=0
    )

    OOF_val_score.append(
        model.evaluate(
            X_train[val_idx],
            y[val_idx],
            verbose=0,
            batch_size=BATCH_SIZE,
            return_dict=True,
        )["score"]
    )

    print("predict test...")
    test_preds += model.predict(X_test, batch_size=BATCH_SIZE, verbose=0) / NFOLD



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2627933866.py in <cell line: 0>()
     19     print(f"FOLD {cnt}")
     20     model = make_model(n_rows)
---> 21     history = model.fit(
     22         X_train[tr_idx],
     23         y[tr_idx],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1190554999.py in loss(y_true, y_pred)
     39 
     40     def loss(y_true, y_pred):
---> 41         return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
     42 
     43     return loss

/tmp/ipykernel_11/1190554999.py in qloss(y_true, y_pred)
     32     e = y_true - y_pred
     33     v = tf.maximum(q * e, (q - 1) * e)
---> 34     return K.mean(v)
     35 
     36 

AttributeError: module 'keras.api.backend' has no attribute 'mean'

## === cell 32
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



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3826437612.py in <cell line: 0>()
      1 # Plot last fold history (as original intent)
----> 2 score_hist = history.history.get("score", [])
      3 val_score_hist = history.history.get("val_score", [])
      4 loss_hist = history.history.get("loss", [])
      5 val_loss_hist = history.history.get("val_loss", [])

NameError: name 'history' is not defined

## === cell 33
print(
    "Mean OOF score (custom metric-as-loss; lower better):",
    float(np.mean(OOF_val_score)),
)



## === cell 34
sigma_opt = mean_absolute_error(y.reshape(-1), train_preds[:, 1])
sigma_uncertain = train_preds[:, 2] - train_preds[:, 0]
sigma_mean = float(np.mean(sigma_uncertain))
print("sigma_opt (MAE):", float(sigma_opt), "sigma_mean (pred spread):", sigma_mean)



## === cell 35
sub_df = sub_df.copy()
sub_df["FVC1"] = test_preds[:, 1]
sub_df["Confidence1"] = test_preds[:, 2] - test_preds[:, 0]

submission = sub_df[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()
submission.loc[~submission.FVC1.isnull()].head(10)



## === cell 36
submission.loc[~submission.FVC1.isnull(), "FVC"] = submission.loc[
    ~submission.FVC1.isnull(), "FVC1"
]

if sigma_mean < 70:
    submission["Confidence"] = sigma_opt
else:
    submission.loc[~submission.FVC1.isnull(), "Confidence"] = submission.loc[
        ~submission.FVC1.isnull(), "Confidence1"
    ]

submission["Confidence"] = submission["Confidence"].clip(lower=1.0)



## === cell 37
print(submission[["FVC", "Confidence"]].describe().T)



## === cell 38
org_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

for i in range(len(org_test)):
    key = org_test.Patient[i] + "_" + str(org_test.Weeks[i])
    submission.loc[submission["Patient_Week"] == key, "FVC"] = org_test.FVC[i]
    submission.loc[submission["Patient_Week"] == key, "Confidence"] = 70.0



## === cell 39
submission_out = submission[["Patient_Week", "FVC", "Confidence"]].copy()
submission_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_out.shape)
print(submission_out.head())
