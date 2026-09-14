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

-6.9867

# 6. Current score

-8.10089

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -7.73127) has done: 'I fix the TensorFlow/protobuf import crash by setting a safe protobuf implementation before importing TensorFlow, and I update deprecated pandas `.append()` to `pd.concat()` so the combined dataset `data` is created correctly. Then I fix column name mismatches (`Smoking_status` vs `SmokingStatus`, and missing `Weeks` in the merged test rows) and ensure scaling/encoding happens on the same combined dataframe used for train/test splits. Finally, I remove notebook-only magics (`%%time`) and make sure the produced `submission.csv` matches `sample_submission.csv` ordering and required columns so Kaggle accepts it and yields a score.'
- What this solution (achieved -7.88972) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation and also pinning protobuf’s internal API expectation via a safe import order, which resolves the `MessageFactory.GetPrototype` error in this Kaggle environment. Then I make the output confidence consistent with the Laplace metric by using a single calibrated constant sigma (derived from train MAE) instead of the model’s raw quantile spread, which tends to be miscalibrated and is a minimal post-processing change that usually improves this competition score. Finally, I keep the submission formatting/ordering aligned to `sample_submission.csv` and ensure we always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.33058) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from even importing `tensorflow` by forcing a compatible protobuf stack before TensorFlow loads (and falling back cleanly if the fast C++ implementation triggers the `MessageFactory.GetPrototype` error). I keep the model/training/prediction logic identical, but ensure the script is robust in this Kaggle environment by setting TF flags and import order safely. I also add a small safety check to guarantee the produced `submission.csv` exactly matches `sample_submission.csv` row ordering and required columns (score-neutral, but prevents silent misalignment). No score-tuning changes beyond that, since your current score (-7.88972) is still outside the ±10% band around the target (-6.9867), and the main blocker here is the runtime error.'
- What this solution (achieved -8.99647) has done: 'I fix the TensorFlow/protobuf crash by forcing a protobuf version that still exposes `MessageFactory.GetPrototype` before importing TensorFlow (this is the root cause of the current runtime error). I keep your model, loss, training loop, and post-processing intact, only changing the import/bootstrap logic so it runs reliably in this Kaggle environment. I also ensure the script consistently finds the competition data directory whether it’s under `/kaggle/input/...` or `../input/...` (score-neutral but prevents path-related failures). With these fixes, the notebook should run end-to-end and write a valid `submission.csv` in the correct format.'
- What this solution (achieved -8.10089) has done: 'The runtime crash happens before any training because TensorFlow 2.18 + protobuf 6 uses a `MessageFactory` object that no longer has `GetPrototype`, and patching the class doesn’t affect the already-created instance used during TF import. I fix this by forcing the pure-Python protobuf backend and pre-creating a `GetPrototype` attribute on the default factory instance before importing TensorFlow, which reliably avoids the import-time AttributeError in this environment. To move the score toward your target (higher is better), I keep the exact same model/training but calibrate the constant submission confidence using an out-of-fold MAE computed across the 5 CV folds (instead of in-sample MAE), which is a minimal, metric-aligned change that typically improves OSIC’s Laplace score. Everything else (features, architecture, loss, epochs, submission ordering/format) remains unchanged and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import importlib

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import message_factory as _mf

    if hasattr(_mf, "MessageFactory") and hasattr(
        _mf.MessageFactory, "GetMessageClass"
    ):
        if not hasattr(_mf.MessageFactory, "GetPrototype"):
            _mf.MessageFactory.GetPrototype = _mf.MessageFactory.GetMessageClass

    try:
        _default_factory = _mf.GetMessageFactory()
        if not hasattr(_default_factory, "GetPrototype") and hasattr(
            _default_factory, "GetMessageClass"
        ):
            _default_factory.GetPrototype = _default_factory.GetMessageClass
    except Exception:
        pass
except Exception:
    pass

import numpy as np
import pandas as pd

from PIL import Image  # noqa: F401
import pydicom  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import pylab  # noqa: F401
import cv2  # noqa: F401
from tensorflow.keras.utils import Sequence  # noqa: F401
from tqdm import tqdm  # noqa: F401

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
import tensorflow.keras.regularizers as R
from sklearn.metrics import mean_absolute_error

np.random.seed(24)
tf.random.set_seed(24)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _resolve_comp_dir():
    candidates = [
        "../input/osic-pulmonary-fibrosis-progression",
        "/kaggle/input/osic-pulmonary-fibrosis-progression",
        "../input/osic-pulmonary-fibrosis-progression/osic-pulmonary-fibrosis-progression",
        "/kaggle/input/osic-pulmonary-fibrosis-progression/osic-pulmonary-fibrosis-progression",
        "/kaggle/data/osic-pulmonary-fibrosis-progression",
        "/kaggle/data/osic-pulmonary-fibrosis-progression/osic-pulmonary-fibrosis-progression",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "sample_submission.csv")
        ):
            return c
    return "../input/osic-pulmonary-fibrosis-progression"


comp_dir = _resolve_comp_dir()

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
train_data_u = train_data.copy()
train_data_u = train_data_u.drop_duplicates(subset=["Patient"])
train_data_u = train_data_u.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
train_data_u["Typical_FVC"] = (
    train_data_u.Base_FVC.values / train_data_u.Base_Percent.values
) * 100
train_data = train_data.merge(
    train_data_u.drop(["Age", "Sex", "SmokingStatus"], axis=1), on="Patient", how="left"
)



## === cell 6
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub.drop(["Confidence"], axis=1)
sub = sub[["Patient", "Weeks", "Patient_Week"]]



## === cell 7
test_data = test_data.rename(
    columns={"Weeks": "Base_Week", "FVC": "Base_FVC", "Percent": "Base_Percent"}
)
test_data["Typical_FVC"] = (
    test_data.Base_FVC.values / test_data.Base_Percent.values
) * 100

sub = sub.merge(test_data, how="left", on="Patient")



## === cell 8
train_data["Type"] = "train"
sub["Type"] = "test"



## === cell 9
data = pd.concat([train_data, sub], axis=0, ignore_index=True)



## === cell 10
prediction_col = ["FVC"]
Continuos_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Typical_FVC",
    "Age",
    "Percent",
    "Base_Percent",
]

Categorical_cols = ["Sex", "SmokingStatus"]



## === cell 11
from sklearn.preprocessing import MinMaxScaler



## === cell 12
missing_cols = [
    c
    for c in (Continuos_cols + Categorical_cols + prediction_col + ["Type"])
    if c not in data.columns
]
if missing_cols:
    raise KeyError(f"Missing required columns in combined data: {missing_cols}")

scaler = MinMaxScaler()
data[Continuos_cols] = scaler.fit_transform(data[Continuos_cols].astype(float))



## === cell 13
print(
    np.mean(train_data_u.query("SmokingStatus == 'Never smoked'").Base_Percent.values)
)
print(
    np.mean(
        train_data_u.query("SmokingStatus == 'Currently smokes'").Base_Percent.values
    )
)
print(np.mean(train_data_u.query("SmokingStatus == 'Ex-smoker'").Base_Percent.values))
print(np.mean(train_data_u.query("Sex == 'Male'").Base_Percent.values))
print(np.mean(train_data_u.query("Sex == 'Female'").Base_Percent.values))



## === cell 14
data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1}).astype(np.float32)

smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
data["SmokingStatus"] = data["SmokingStatus"].map(smoke_map).astype(np.float32)



## === cell 15
data.head()



## === cell 16
x_cols = [
    "Weeks",
    "Base_Week",
    "Base_FVC",
    "Sex",
    "SmokingStatus",
    "Age",
    "Typical_FVC",
]



## === cell 17
data[x_cols + prediction_col + ["Type"]].head()



## === cell 18
x_train = data.loc[data["Type"] == "train", x_cols].values
y_train = data.loc[data["Type"] == "train", prediction_col].values
x_test = data.loc[data["Type"] == "test", x_cols].values



## === cell 19
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)



## === cell 20
x_train.shape, y_train.shape, x_test.shape



## === cell 21
type(x_train), type(y_train), type(x_test)



## === cell 22
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
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
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss




## === cell 23
def build_model():
    inp = L.Input((7,))
    x = L.Dense(128, activation="relu", kernel_regularizer=R.l2(1e-6))(inp)
    x = L.Dense(128, activation="relu", kernel_regularizer=R.l2(1e-6))(x)
    x = L.Dense(64, activation="relu", kernel_regularizer=R.l2(1e-6))(x)
    o1 = L.Dense(3, activation="linear")(x)
    o2 = L.Dense(3, activation="relu")(x)
    pred1 = L.Lambda(lambda x: (x[0] + (tf.cumsum(x[1], axis=1))))([o1, o2])

    model = M.Model(inputs=inp, outputs=pred1)
    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        metrics=[score],
    )
    return model




## === cell 24
model = build_model()
model.summary()



## === cell 25
from sklearn.model_selection import KFold

folds = 5
KF = KFold(n_splits=folds, shuffle=True, random_state=24)



## === cell 26
counter = 0
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.1, patience=50, min_lr=1e-3
)

oof_pred = np.zeros((x_train.shape[0], 3), dtype=np.float32)

for tr_idx, val_idx in KF.split(x_train):
    counter += 1
    print(f"############## FOLD {counter} ###############")
    model.fit(
        x_train[tr_idx],
        y_train[tr_idx],
        epochs=800,
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
    oof_pred[val_idx] = model.predict(x_train[val_idx], verbose=0, batch_size=256)



## === cell 27
model.save("model.h5")



## === cell 28
pred = model.predict(x_train, verbose=0)



## === cell 29
sigma_opt = mean_absolute_error(y_train, oof_pred[:, 1])
unc = oof_pred[:, 2] - oof_pred[:, 0]
sigma_mean = float(np.mean(unc))
sigma = unc
print("sigma_opt(oof MAE), sigma_mean(oof unc mean):", sigma_opt, sigma_mean)
print("unc head:", sigma[:10])



## === cell 30
pred[:10]



## === cell 31
pred = model.predict(x_test, verbose=0)



## === cell 32
pred[:5]



## === cell 33
sigma_cal = float(np.maximum(sigma_opt, 70.0))
conf = np.full(shape=(pred.shape[0],), fill_value=sigma_cal, dtype=np.float32)



## === cell 34
pred_df = pd.DataFrame({"FVC": pred[:, 1], "Confidence": conf})



## === cell 35
sub = sub.copy()
sub["Confidence"] = pred_df["Confidence"].values
sub["FVC"] = pred_df["FVC"].values



## === cell 36
subm = sub[["Patient_Week", "FVC", "Confidence"]].copy()



## === cell 37
subm.head()



## === cell 38
sample_sub = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))
subm = sample_sub[["Patient_Week"]].merge(subm, on="Patient_Week", how="left")
if subm[["FVC", "Confidence"]].isna().any().any():
    raise ValueError(
        "Submission has missing predictions after aligning to sample_submission order."
    )
subm = subm[["Patient_Week", "FVC", "Confidence"]]

subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print(subm.head())
