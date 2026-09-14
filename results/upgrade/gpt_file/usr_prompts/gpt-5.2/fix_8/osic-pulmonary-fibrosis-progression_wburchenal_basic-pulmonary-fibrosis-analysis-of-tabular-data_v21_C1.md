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

-6.9488

# 6. Current score

-8.76216

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I fix the two runtime blockers: the protobuf/TensorFlow import crash and the pandas `.append` removal (replacing it with `pd.concat`), so the notebook runs end-to-end. Then I fix the downstream `NameError`s that were only happening because feature engineering never completed, and ensure the feature columns exist consistently in train/test/sub. Finally, I keep your model/training core logic intact but make the optimizer args compatible with TF 2.18 (use `learning_rate` and remove deprecated `decay`) so training runs and a valid `submission.csv` is written with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf setting, which is what triggers the `MessageFactory.GetPrototype` error under the current library versions. Then I fix the training-time `ValueError: None values not supported` by ensuring the engineered feature matrix and target contain no NaNs/Infs (imputing numeric columns and filling any missing one-hot columns), without changing the model architecture or training loop. Finally, I keep your submission logic intact but make sure `FVC`/`Confidence` are numeric and the output CSV is written correctly as `submission.csv`. These changes are score-neutral-to-positive and should move you toward the target by allowing training to complete and by removing pathological NaNs that harm optimization.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf import crash by setting a safe protobuf implementation before importing TensorFlow, which is the root cause of the `MessageFactory.GetPrototype` error in this environment. Then I eliminate the `None values not supported` training failure by ensuring the engineered feature set is created consistently (including one-hot columns) and by coercing/imputing any remaining non-numeric/NaN values in both `X` and `y` right before model fitting. Finally, I make the prediction-to-submission post-processing more metric-consistent by forcing predicted confidence to be positive and clipped to the competition minimum (70), which should improve score toward your target without changing the model architecture or training loop.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf setting (it triggers the `MessageFactory.GetPrototype` error with these library versions) and importing TensorFlow normally. Then I eliminate the `None values not supported` training failure by ensuring the feature matrix `z` and target `y` are strictly numeric `float32` with no object dtype and no NaN/Inf anywhere (including a safe `np.nanmedian` fallback when the target contains all-NaNs). Finally, I keep your model/training and submission logic intact while adding a minimal sanity-cast right before `fit()` so Keras never sees `None`/object arrays, and the pipeline reliably writes a valid `submission.csv`.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the safest workaround for the `MessageFactory.GetPrototype` incompatibility in this environment. Then I fix the `ValueError: None values not supported` at `model.fit()` by making the custom losses robust to the model’s 3-output prediction shape (broadcast `y_true` properly and ensure all tensors are finite), without changing your model architecture or training loop. Finally, I make the confidence post-processing metric-consistent by ensuring it’s positive and clipped to the competition minimum (70), and keep the submission format exactly as required.'
- What this solution (achieved -8.76216) has done: 'I first fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf environment override, which is incompatible in this Kaggle environment. Next, I eliminate the `None values not supported` training error by making the custom losses robust: ensure all intermediate tensors are finite and, crucially, force the predicted quantiles to be ordered so `sigma = q80-q20` is never negative/NaN during training (this keeps your same 3-output model but stabilizes the loss). Finally, I keep your submission logic intact but make the baseline rows use a metric-safe confidence (>=70), since `0.1` gets clipped to 70 anyway and can hurt calibration; this should improve score toward the target without changing the modeling approach.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf implementation safely *before* importing TensorFlow, which prevents the `MessageFactory.GetPrototype` error in this environment. Then I fix the `None values not supported` training crash by (1) ensuring the feature list uses stable, unique, valid column names (no spaces/special chars collisions) and (2) forcing every input/output array passed to Keras to be finite `float32` with no object dtype and no `None`/NaN/Inf. Finally, I keep your model architecture/training loop intact but make a small metric-consistent improvement by setting a more sensible default confidence derived from out-of-fold uncertainty (still clipped at 70), which should nudge the score upward toward your target without changing the learning setup.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import pydicom

print("TF version:", tf.__version__)




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
        if filename.lower().endswith(".dcm"):
            image_files_list.append(os.path.join(dirName, filename))

image = pydicom.dcmread(image_files_list[0])

plt.figure()
plt.imshow(image.pixel_array, cmap=plt.cm.bone)
plt.axis("off")
plt.show()



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


def _safe_colname(prefix: str, val: str) -> str:
    s = str(val)
    s = s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    s = s.replace("-", "_").replace(".", "_").replace("(", "").replace(")", "")
    return f"{prefix}__{s}"


COLS = ["Sex", "SmokingStatus"]  # ,'Age'
FE = []
for col in COLS:
    for mod in data[col].dropna().unique():
        cname = _safe_colname(col, mod)
        FE.append(cname)
        data[cname] = (data[col] == mod).astype(int)

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
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def _ordered_pred(y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    q20 = y_pred[:, 0]
    q50 = y_pred[:, 1]
    q80 = y_pred[:, 2]
    q20o = tf.minimum(q20, q80)
    q80o = tf.maximum(q20, q80)
    q50o = tf.clip_by_value(q50, q20o, q80o)
    return tf.stack([q20o, q50o, q80o], axis=1)


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = _ordered_pred(y_pred)

    y_true = tf.reshape(y_true, (-1, 1))
    y_true_fvc = y_true[:, 0]

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true_fvc - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.constant(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)

    metric = tf.where(tf.math.is_finite(metric), metric, tf.zeros_like(metric))
    return tf.keras.backend.mean(metric)


def qloss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = _ordered_pred(y_pred)

    y_true = tf.reshape(y_true, (-1, 1))  # (N,1) -> broadcast to (N,3)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)

    v = tf.where(tf.math.is_finite(v), v, tf.zeros_like(v))
    return tf.keras.backend.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model(nh):
    z = tf.keras.layers.Input((nh,), name="Patient")
    x = tf.keras.layers.Dense(120, activation="relu", name="d1")(z)
    x = tf.keras.layers.Dense(120, activation="relu", name="d2")(x)
    p1 = tf.keras.layers.Dense(3, activation="linear", name="p1")(x)
    p2 = tf.keras.layers.Dense(3, activation="relu", name="p2")(x)
    preds = tf.keras.layers.Lambda(
        lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds"
    )([p1, p2])

    model = tf.keras.models.Model(z, preds, name="definitely_not_a_CNN")
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
    )
    model.compile(loss=mloss(0.8), optimizer=opt, metrics=[score])
    return model




## === cell 8
for col in FE:
    if col not in train.columns:
        train[col] = 0
    if col not in sub.columns:
        sub[col] = 0

train[FE] = train[FE].apply(pd.to_numeric, errors="coerce")
sub[FE] = sub[FE].apply(pd.to_numeric, errors="coerce")

med = train[FE].median(numeric_only=True)
train[FE] = train[FE].fillna(med).replace([np.inf, -np.inf], 0.0)
sub[FE] = sub[FE].fillna(med).replace([np.inf, -np.inf], 0.0)

train["FVC"] = pd.to_numeric(train["FVC"], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
fvc_median = float(np.nanmedian(train["FVC"].values.astype(np.float64)))
if not np.isfinite(fvc_median):
    fvc_median = 0.0
train["FVC"] = train["FVC"].fillna(fvc_median)

train[FE] = train[FE].astype(np.float32)
sub[FE] = sub[FE].astype(np.float32)
train["FVC"] = train["FVC"].astype(np.float32)

y = train["FVC"].values.reshape(-1, 1).astype(np.float32)
z = train[FE].values.astype(np.float32)
ze = sub[FE].values.astype(np.float32)

z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
ze = np.nan_to_num(ze, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

y_med = float(np.nanmedian(y.astype(np.float64)))
if not np.isfinite(y_med):
    y_med = 0.0
y = np.nan_to_num(y, nan=y_med, posinf=y_med, neginf=y_med).astype(np.float32)

z = np.ascontiguousarray(z, dtype=np.float32)
ze = np.ascontiguousarray(ze, dtype=np.float32)
y = np.ascontiguousarray(y, dtype=np.float32)

nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

net = make_model(nh)
print(net.summary())
print("Params:", net.count_params())

assert (
    np.isfinite(z).all() and np.isfinite(ze).all() and np.isfinite(y).all()
), "Non-finite values remain in inputs."



## === cell 9
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 10
cnt = 0
EPOCHS = 800
BATCH_SIZE = 64
diff_sum = 0.0

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)

    X_tr = np.ascontiguousarray(z[tr_idx], dtype=np.float32)
    y_tr = np.ascontiguousarray(y[tr_idx], dtype=np.float32)
    X_va = np.ascontiguousarray(z[val_idx], dtype=np.float32)
    y_va = np.ascontiguousarray(y[val_idx], dtype=np.float32)

    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y_tr = np.nan_to_num(y_tr, nan=y_med, posinf=y_med, neginf=y_med).astype(np.float32)
    y_va = np.nan_to_num(y_va, nan=y_med, posinf=y_med, neginf=y_med).astype(np.float32)

    net.fit(
        X_tr,
        y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_va, y_va),
        verbose=0,
    )
    train_loss, train_score = net.evaluate(X_tr, y_tr, verbose=0, batch_size=BATCH_SIZE)
    print(f"Train Loss: {train_loss}  Score: {train_score}")
    val_loss, val_score = net.evaluate(X_va, y_va, verbose=0, batch_size=BATCH_SIZE)
    print(f"Val Loss: {val_loss}  Score: {val_score}")
    score_diff = float(val_score - train_score)
    diff_sum += score_diff
    print(f"Score diff: {score_diff}")

    print("Predict val...")
    pred[val_idx] = net.predict(X_va, batch_size=BATCH_SIZE, verbose=0)
    print("Predict test...")
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD

print(f"Score diff sum : {diff_sum}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/38774423.py in <cell line: 0>()
     20     y_va = np.nan_to_num(y_va, nan=y_med, posinf=y_med, neginf=y_med).astype(np.float32)
     21 
---> 22     net.fit(
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

## === cell 11
sigma_opt = mean_absolute_error(y[:, 0], pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))

sub["FVC1"] = 0.996 * pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

subm["Confidence1"] = pd.to_numeric(subm["Confidence1"], errors="coerce")
subm["Confidence1"] = subm["Confidence1"].abs()

robust_conf = float(np.nanmedian(np.abs(unc)))
if not np.isfinite(robust_conf) or robust_conf <= 0:
    robust_conf = float(sigma_opt)

if sigma_mean < 70:
    subm["Confidence"] = robust_conf
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

subm["FVC"] = pd.to_numeric(subm["FVC"], errors="coerce").fillna(0.0)
subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce").fillna(
    robust_conf
)
subm["Confidence"] = subm["Confidence"].abs().clip(lower=70.0)



## === cell 12
otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    subm[["Patient_Week", "FVC", "Confidence"]].shape,
)
print(subm[["Patient_Week", "FVC", "Confidence"]].head())
