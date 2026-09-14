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

-6.9319

# 6. Current score

-8.15893

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.34788) has done: 'I fix the runtime crash at import time by pinning protobuf to a compatible implementation mode before TensorFlow loads, which resolves the `MessageFactory.GetPrototype` error in this environment. Then I update the deprecated `DataFrame.append` usage to `pd.concat`, which unblocks feature engineering so `FE`, `z`, `pred`, and the rest of the pipeline are defined. I also make the KFold split deterministic and correct a couple of TensorFlow API incompatibilities (Adam `lr` → `learning_rate`, remove deprecated `decay`) to ensure training runs end-to-end. Finally, I ensure the submission is created with the exact required columns and written to `submission.csv`.'
- What this solution (achieved -8.34788) has done: 'I fix the import-time crash by ensuring the protobuf implementation mode is set before any TensorFlow-related imports and by forcing the pure-Python protobuf backend early (this resolves the `MessageFactory.GetPrototype` issue in this environment). Then I make a minimal, score-improving calibration fix: your submission currently sets `Confidence=0.1` for the baseline test rows, which is extremely penalized by the metric (because σ is clipped to 70 but your prediction error at baseline can still be non-zero due to scaling/float), so I set baseline confidence to 70 (the metric’s clip floor) while keeping the baseline FVC override intact. Everything else (features, model, training loop, and prediction logic) stays the same, and the script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.34788) has done: 'I fix the import-time protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow-related imports, and by importing TensorFlow before `pydicom` to avoid protobuf symbol clashes in this environment. Then I make one minimal, score-improving calibration fix: ensure `Confidence` is always finite and at least 70 everywhere (your metric clips at 70 anyway, but negative/NaN confidences can silently degrade score). Everything else (feature engineering, model, loss, training loop, and prediction logic) stays the same, and the script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.15893) has done: 'We fix the import-time protobuf/TensorFlow crash by forcing the Python protobuf backend early and additionally disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus importing TensorFlow before `pydicom` (this environment can otherwise hit `MessageFactory.GetPrototype`). Then we correct the metric implementation used inside training: your `score()` currently returns the *negative* of the competition metric (missing the leading minus), which pushes training in the wrong direction; fixing this should move the public score upward toward the target while keeping the same model and training loop. Finally, we keep your submission logic but ensure `Confidence` is always finite and clipped to at least 70 everywhere (score-stable) and that the `submission.csv` is written with the exact required columns.'
- What this solution (achieved -8.15893) has done: 'I fix the import-time crash (`MessageFactory` has no `GetPrototype`) by pinning protobuf to the pure-Python backend *and* forcing the legacy Python implementation version **before any TF/protobuf modules load**, then importing `google.protobuf` once to lock it in. I keep the model/training logic unchanged, but I make the score/loss numerically safer by ensuring the predicted sigma is strictly positive before clipping (prevents NaNs/inf that can silently hurt training and public score). Finally, I keep your existing submission construction, but add a last-mile validation step to guarantee the output CSV has the exact required columns, dtypes are numeric, and confidences are finite and >= 70.'
- What this solution (achieved -8.15893) has done: 'I fix the import-time crash (`MessageFactory` has no `GetPrototype`) by ensuring the protobuf runtime stays on a TensorFlow-compatible implementation before TensorFlow loads, without changing your modeling/training logic. Concretely, I switch to the C++ protobuf backend when available (and avoid forcing the legacy python backend), then import TensorFlow first, and only then import `pydicom`. Everything else (feature engineering, model, loss/metric, training loop, and submission formatting) is kept identical so score behavior should remain consistent while the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -8.15893) has done: 'We need to fix the import-time crash coming from a protobuf/TensorFlow/pydicom interaction (`MessageFactory.GetPrototype`), by forcing a TensorFlow-compatible protobuf implementation *before* importing TensorFlow and pydicom. Then we keep your model/training logic and feature pipeline unchanged, only making the environment/import ordering robust so the script runs end-to-end. Finally, we keep the submission generation identical but add a small last-mile guard that ensures the CSV has the required columns and numeric dtypes (score-neutral, prevents invalid submission issues). This should restore execution and allow your existing training/metric fix to realize the intended score improvement toward the target.'
- What this solution (achieved -8.15893) has done: 'I fix the runtime crash happening before training by adjusting the protobuf/TensorFlow/pydicom import order and removing the protobuf “python backend + v2” forcing that triggers `MessageFactory.GetPrototype` in this environment. This is a correctness-only fix to unblock execution end-to-end; the model, features, training loop, loss, and submission logic remain the same. I also add a tiny guard so DICOM visualization doesn’t ever break the run (it’s not needed for training/inference). The submission still be written as `submission.csv` with the required columns and with Confidence safely clipped to at least 70.'
- What this solution (achieved -8.15893) has done: 'I fix the import-time crash by forcing a TensorFlow-compatible protobuf runtime *before* importing TensorFlow (this environment is hitting the `MessageFactory.GetPrototype` issue). To avoid the TF/pydicom protobuf symbol clash, I also keep TensorFlow imported before `pydicom` and avoid clearing the protobuf env vars that were previously stabilizing execution. Then I keep your model/feature/training logic unchanged, but add a tiny deterministic guard and a final submission integrity check so the run always completes and writes a valid `submission.csv` with correct columns and numeric, finite `Confidence >= 70`. These changes should unblock execution and keep (or improve) score behavior without altering the core approach.'
- What this solution (achieved -8.15893) has done: 'We fix the import-time crash by removing the forced legacy protobuf environment variables that are triggering the `MessageFactory.GetPrototype` error in this Kaggle TensorFlow 2.18 + protobuf 6.x environment, and by importing TensorFlow before pydicom to avoid protobuf symbol clashes. This change is execution-only and keeps your model, features, training loop, and loss/metric semantics intact. After that, the pipeline run end-to-end and still produce a valid `submission.csv` with the required columns and safe `Confidence >= 70`. No score-tuning changes are made beyond restoring successful execution so your existing metric-aligned training can realize its current performance.'
- What this solution (achieved -8.15893) has done: 'I fix the import-time crash by enforcing a TensorFlow-compatible protobuf runtime *before* importing TensorFlow and by importing TensorFlow before `pydicom`, which avoids the `MessageFactory.GetPrototype` conflict in this environment. I keep your feature engineering, model, folds, epochs, and training loop unchanged, but add two small stability guards that are score-positive: ensure the predicted uncertainty used for `Confidence` is always finite and non-negative before the final `>=70` clip, and ensure we never accidentally keep an uncalibrated confidence for test rows. These are minimal changes aligned with the competition metric (it heavily penalizes bad/invalid sigma). The script still write a valid `submission.csv` with exactly the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import google.protobuf  # noqa: F401

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import pydicom




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)



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
        if ".dcm" in filename.lower():
            image_files_list.append(os.path.join(dirName, filename))

if len(image_files_list) > 0:
    try:
        image = pydicom.dcmread(image_files_list[0])
        plt.figure()
        plt.imshow(image.pixel_array, cmap=plt.cm.bone)
        plt.axis("off")
        plt.show()
    except Exception as e:
        print("DICOM preview skipped due to:", repr(e))
else:
    print("No DICOM files found under:", image_path)



## === cell 6
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([train, test, sub], axis=0, ignore_index=True)

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
print("Features:", FE)

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

print(train.shape, test.shape, sub.shape)



## === cell 7
C1, C2 = tf.constant(70.0, dtype="float32"), tf.constant(1000.0, dtype="float32")


def score(y_true, y_pred):
    """
    Competition metric (higher is better):
      -sqrt(2)*delta/sigma_clipped - log(sqrt(2)*sigma_clipped)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    sigma = tf.maximum(sigma, tf.constant(1e-3, dtype=tf.float32))

    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))

    metric = -((delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2))
    return tf.keras.backend.mean(metric)


def qloss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return tf.keras.backend.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model(nh):
    z = tf.keras.layers.Input((nh,), name="Patient")
    x = tf.keras.layers.Dense(100, activation="relu", name="d1")(z)
    x = tf.keras.layers.Dense(100, activation="relu", name="d2")(x)
    p1 = tf.keras.layers.Dense(3, activation="linear", name="p1")(x)
    p2 = tf.keras.layers.Dense(3, activation="relu", name="p2")(x)
    preds = tf.keras.layers.Lambda(
        lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds"
    )([p1, p2])

    model = tf.keras.models.Model(z, preds, name="definitely_not_a_CNN")

    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False
        ),
        metrics=[score],
    )
    return model




## === cell 8
y = train["FVC"].values.reshape(-1, 1).astype(np.float32)
z = train[FE].values.astype(np.float32)
ze = sub[FE].values.astype(np.float32)

nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

net = make_model(nh)
print(net.summary())
print("Params:", net.count_params())



## === cell 9
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 10
cnt = 0
EPOCHS = 1000
BATCH_SIZE = 48

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)
    net.fit(
        z[tr_idx],
        y[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(z[val_idx], y[val_idx]),
        verbose=0,
    )
    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))
    print("predict val...")
    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print("predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD



## === cell 11
sigma_opt = mean_absolute_error(y[:, 0], pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))

sub["FVC1"] = 0.996 * pe[:, 1]

conf1 = (pe[:, 2] - pe[:, 0]).astype(np.float32)
conf1 = np.where(np.isfinite(conf1), conf1, np.nan)
conf1 = np.maximum(conf1, 0.0)
sub["Confidence1"] = conf1

subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

subm["FVC"] = pd.to_numeric(subm["FVC"], errors="coerce")
subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce")

subm["FVC"] = subm["FVC"].replace([np.inf, -np.inf], np.nan)
subm["Confidence"] = subm["Confidence"].replace([np.inf, -np.inf], np.nan)

subm["FVC"] = subm["FVC"].fillna(subm["FVC"].median())
subm["Confidence"] = subm["Confidence"].fillna(70.0)

subm["Confidence"] = np.maximum(subm["Confidence"].values.astype(np.float32), 70.0)



## === cell 12
otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission["FVC"] = pd.to_numeric(submission["FVC"], errors="coerce").fillna(
    submission["FVC"].median()
)
submission["Confidence"] = pd.to_numeric(submission["Confidence"], errors="coerce")
submission["Confidence"] = (
    submission["Confidence"].replace([np.inf, -np.inf], np.nan).fillna(70.0)
)
submission["Confidence"] = np.maximum(
    submission["Confidence"].values.astype(np.float32), 70.0
)

submission = submission[["Patient_Week", "FVC", "Confidence"]]
assert (
    submission.shape[0]
    == pd.read_csv(
        "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
    ).shape[0]
), "Row count mismatch vs sample_submission"
assert submission["Patient_Week"].isnull().sum() == 0, "Null Patient_Week found"
assert submission["FVC"].isnull().sum() == 0, "Null FVC found"
assert submission["Confidence"].isnull().sum() == 0, "Null Confidence found"

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Confidence min/max:",
    float(submission["Confidence"].min()),
    float(submission["Confidence"].max()),
)
print("Any nulls:", submission.isnull().any().to_dict())
