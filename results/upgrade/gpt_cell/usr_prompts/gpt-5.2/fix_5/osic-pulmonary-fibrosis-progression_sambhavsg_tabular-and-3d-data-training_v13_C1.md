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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

-6.9986

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".", 1)[0])
except Exception:
    _pb_major = None

if _pb_major is not None and _pb_major >= 5:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    os.execv(sys.executable, [sys.executable] + sys.argv)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

for _m in list(sys.modules.keys()):
    if _m.startswith("google.protobuf"):
        sys.modules.pop(_m, None)

import numpy as np
import pandas as pd
from PIL import Image
import pydicom
import matplotlib.pyplot as plt
import pylab
import cv2
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
import tensorflow.keras.regularizers as R
from sklearn.metrics import mean_absolute_error

np.random.seed(24)
tf.random.set_seed(24)



## === cell 1
comp_dir = "../input/osic-pulmonary-fibrosis-progression"

train_data = pd.read_csv(os.path.join(comp_dir, "train.csv"))
sub = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))
test_data = pd.read_csv(os.path.join(comp_dir, "test.csv"))

train_data.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])



## === cell 2
train_data.head()



## === cell 3
test_data.head()



## === cell 4
sub.head()



## === cell 5
train_data_u = train_data
train_data_u = train_data_u.drop_duplicates(subset=["Patient"])
train_data_u = train_data_u.rename(columns={"Weeks": "Base_Week", "FVC": "Base_FVC"})
train_data_u["Typical_FVC"] = (
    train_data_u.Base_FVC.values / train_data_u.Percent.values
) * 100
train_data = train_data.merge(
    train_data_u.drop(["Percent", "Age", "Sex", "SmokingStatus"], axis=1),
    on="Patient",
    how="left",
)
train_data = train_data.drop(["Percent"], axis=1)



## === cell 6
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub.drop(["Confidence"], axis=1)
sub = sub[["Patient", "Weeks", "Patient_Week"]]



## === cell 7
test_data = test_data.rename(columns={"Weeks": "Base_Week", "FVC": "Base_FVC"})
test_data["Typical_FVC"] = (test_data.Base_FVC.values / test_data.Percent.values) * 100
sub = sub.merge(test_data.drop(["Percent"], axis=1), how="left", on="Patient")



## === cell 8
train_data["Type"] = "train"
sub["Type"] = "test"



## === cell 9
data = pd.concat([train_data, sub], axis=0, ignore_index=True)



## === cell 10
prediction_col = ["FVC"]
Continuos_cols = ["Weeks", "Base_Week", "Base_FVC", "Typical_FVC", "Age"]

Categorical_cols = ["Sex", "SmokingStatus"]



## === cell 11
from sklearn.preprocessing import MinMaxScaler



## === cell 12
scaler = MinMaxScaler()
conti = scaler.fit_transform(data[Continuos_cols])
data[Continuos_cols] = conti



## === cell 13
print(np.mean(train_data_u.query("SmokingStatus == 'Never smoked'").Percent.values))
print(np.mean(train_data_u.query("SmokingStatus == 'Currently smokes'").Percent.values))
print(np.mean(train_data_u.query("SmokingStatus == 'Ex-smoker'").Percent.values))
print(np.mean(train_data_u.query("Sex == 'Male'").Percent.values))
print(np.mean(train_data_u.query("Sex == 'Female'").Percent.values))



## === cell 14
data = data.copy()
data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1}).astype(np.float32)

data["SmokingStatus"] = (
    data["SmokingStatus"]
    .map({"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2})
    .astype(np.float32)
)



## === cell 15
data



## === cell 16
x_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Age",
    "Sex",
    "SmokingStatus",
]



## === cell 17
x_train = data[x_cols].loc[data["Type"] == "train"].values
y_train = data[prediction_col].loc[data["Type"] == "train"].values
x_test = data[x_cols].loc[data["Type"] == "test"].values



## === cell 18
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)



## === cell 19
x_train.shape, y_train.shape, x_test.shape



## === cell 20
type(x_train), type(y_train), type(x_test)



## === cell 21
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def laplace_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt(tf.cast(2.0, tf.float32))

    kaggle_metric = -(sq2 * delta) / sigma_clip - tf.math.log(sq2 * sigma_clip)
    return K.mean(kaggle_metric)


def neg_laplace_metric_loss(y_true, y_pred):
    return -laplace_metric(y_true, y_pred)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (
            1 - _lambda
        ) * neg_laplace_metric_loss(y_true, y_pred)

    return loss




## === cell 22
def build_model():
    inp = L.Input((7,))
    x = L.Dense(128, activation="relu", kernel_regularizer=R.l2(1e-4))(inp)
    x = L.Dense(128, activation="relu", kernel_regularizer=R.l2(1e-5))(x)
    x = L.Dense(64, activation="relu", kernel_regularizer=R.l2(1e-5))(x)
    o1 = L.Dense(3, activation="linear")(x)
    o2 = L.Dense(3, activation="relu")(x)
    pred1 = L.Lambda(lambda x: (x[0] + (tf.cumsum(x[1], axis=1))))([o1, o2])

    model = M.Model(inputs=inp, outputs=pred1)
    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        metrics=[laplace_metric],
    )
    return model




## === cell 23
model = build_model()
model.summary()



## === cell 24
from sklearn.model_selection import KFold

folds = 5
KF = KFold(n_splits=folds, shuffle=True, random_state=24)



## === cell 25
import time

t0 = time.time()

counter = 0
stopper = tf.keras.callbacks.EarlyStopping(
    monitor="loss", mode="min", patience=200, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.1, patience=50, min_lr=1e-3
)

for tr_idx, val_idx in KF.split(x_train):
    counter += 1
    print(f"############## FOLD {counter} ###############")
    model.fit(
        x_train[tr_idx],
        y_train[tr_idx],
        epochs=700,
        batch_size=256,
        verbose=0,
        validation_data=(x_train[val_idx], y_train[val_idx]),
        callbacks=[reduce_lr],
    )
    print(
        "train",
        model.evaluate(x_train[tr_idx], y_train[tr_idx], verbose=0, batch_size=256),
    )
    print(
        "val",
        model.evaluate(x_train[val_idx], y_train[val_idx], verbose=0, batch_size=256),
    )

print(f"Training wall time (s): {time.time() - t0:.1f}")



## === cell 26
model.save("model.h5")



## === cell 27
pred = model.predict(x_train, verbose=1)



## === cell 28
sigma_opt = mean_absolute_error(y_train, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = np.mean(unc)
sigma = pred[:, 2] - pred[:, 0]
print(sigma_opt, sigma_mean, sigma)



## === cell 29
pred[:10]



## === cell 30
pred = model.predict(x_test, verbose=1)



## === cell 31
conf = np.maximum(pred[:, 2] - pred[:, 0], 70.0).astype(np.float32)



## === cell 32
pred_dict = {"FVC": pred[:, 1], "Confidence": conf}
pred_df = pd.DataFrame(pred_dict)



## === cell 33
sub["Confidence"] = pred_df["Confidence"].values
sub["FVC"] = pred_df["FVC"].values



## === cell 34
subm = sub[["Patient_Week", "FVC", "Confidence"]].copy()



## === cell 35
subm[300:400]



## === cell 36
subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print(subm.head())
