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

-6.9355

# 6. Current score

-8.16252

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I fix the environment import crash by switching protobuf to the pure-Python implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error. I update pandas usage to replace the removed `DataFrame.append` with `pd.concat`, which unblocks feature engineering and ensures `FE` is defined for later cells. I also make the TensorFlow compile call compatible with TF 2.18 by using `learning_rate` (not deprecated `lr`) and removing the deprecated `decay` argument, keeping the same optimizer and model structure. Finally, I remove the notebook-only `%%time` magic so the script runs as a `.py`, and ensure a valid `submission.csv` is always written with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf crash by setting both protobuf environment variables before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` incompatibility seen in some Kaggle images. Then I fix the training-time shape error by making the loss/metric functions accept `y_true` as either shape `(batch,)` or `(batch,1)` (Keras is currently feeding a 1D target), without changing the model outputs or the intended metric math. Finally, I keep the existing training/prediction pipeline intact and ensure the script always writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *and* disabling the C++ descriptors, which is the common root cause of the `MessageFactory.GetPrototype` error in Kaggle images. Then I fix the training-time `OperatorNotAllowedInGraphError` by replacing the Python `if tf.rank(...) == 1` with a graph-safe reshape that works for both `(batch,)` and `(batch,1)` targets without changing the loss/metric math. Finally, I keep the same model/training loop and submission logic, only ensuring the pipeline runs end-to-end and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix two execution blockers that prevent your current pipeline from running: the TensorFlow/protobuf import crash and the `None values not supported` error during `model.fit`. The protobuf issue is resolved by importing TensorFlow first (in this Kaggle image TF 2.18 works cleanly with the installed protobuf), then importing everything else; this is a minimal change that restores runtime. The `None` error comes from `min_FVC` being missing for test/sub rows because the baseline merge uses only `Weeks == min_week` and `min_week` is set to `NaN` for `WHERE=="test"`; I compute baseline from each patient’s earliest available record (train/test) and merge that into all rows so no feature columns contain NaNs. This should also improve score versus the current broken run by restoring the intended features and letting training complete, without changing the model or loss/metric logic.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which addresses the `MessageFactory.GetPrototype` error. Then I fix the `None values not supported` training crash by ensuring the training target `y` has no missing values (drop rows with missing `FVC` after concatenation/merge), and by filling any remaining feature NaNs defensively. Finally, I keep your model/training and submission logic unchanged, only adding a couple of safety assertions so the pipeline reliably trains and always writes a valid `submission.csv` in the required format.'
- What this solution (achieved -8.76216) has done: 'I fix the two execution blockers causing your run to fail: the TensorFlow/protobuf import crash and the `None values not supported` error during `model.fit`. The protobuf crash is handled by setting the additional environment flag that disables C++ descriptors before importing TensorFlow, which is the common root cause of `MessageFactory.GetPrototype` in Kaggle images. The `None` error is fixed by making sure all training features are numeric and non-missing (coercing to numeric, then filling NaNs/inf), and by forcing `y` to be a dense float32 array with no missing values. These changes are score-neutral-to-positive (they restore the intended features/training) and keep the model, loss, CV loop, and submission logic unchanged.'
- What this solution (achieved -8.76216) has done: 'I fix the two execution blockers shown: the protobuf/TensorFlow import crash and the `None values not supported` error during `model.fit`. The protobuf issue is addressed by avoiding the forced pure-Python protobuf settings (they trigger the `MessageFactory.GetPrototype` mismatch here) and importing TensorFlow cleanly first, which is score-neutral and unblocks runtime. The `None` error is fixed by ensuring `y` is fully numeric and non-missing *after* all merges/concats, and by coercing/filling any remaining feature NaNs so Keras never receives `None` objects. These changes preserve your model, loss, CV loop, and submission logic, and should also improve score versus the broken run by restoring intended training.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf implementation env vars before importing TensorFlow (this directly addresses the `MessageFactory.GetPrototype` error in this environment). Then I fix the `None values not supported` crash during `model.fit` by ensuring both `z` and `y` are strictly finite numeric arrays after all merges/feature engineering, and by explicitly dropping any rows that still have non-finite features/targets (this preserves the same model and training loop). Finally, I make the submission-writing step robust by clipping Confidence to be at least 70 (matching the metric’s clipping and typically improving score slightly without changing core modeling), while keeping the rest of the post-processing identical and still writing `submission.csv`.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf settings and (instead) setting `TF_USE_LEGACY_KERAS=1` before importing TensorFlow, which is a common compatibility requirement in TF 2.18 Kaggle images and keeps your core model/training intact. Then I fix the `None values not supported` error by ensuring `SmokingStatus`/`Sex` are filled before one-hot feature creation and by making `sub["FVC"]` a fully numeric (non-None) column after merging so the later `subm = sub[...]` selection never injects object/None into the training graph. Finally, I keep your architecture/loss/training loop the same, only adding strict numeric coercion and finite checks right before `model.fit`, and still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.16252) has done: 'I fix the TensorFlow/protobuf import crash by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the `None values not supported` failure during `model.fit` by explicitly setting a valid Adam `epsilon` (TF 2.18 can otherwise pass `None` into the update step under XLA) while keeping the same optimizer type and learning rate. Finally, I make the submission confidence handling consistent with the competition’s clipping rule by ensuring the produced `Confidence` is always finite and clipped to at least 70 (score-positive calibration, minimal post-processing), and still write `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import pydicom

print("TF version:", tf.__version__)
print("NumPy:", np.__version__, "Pandas:", pd.__version__)




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

for c in ["Sex", "SmokingStatus"]:
    if c in data.columns:
        data[c] = data[c].fillna("Unknown").astype(str)

data["min_week"] = data.groupby("Patient")["Weeks"].transform("min")

base = (
    data.sort_values(["Patient", "Weeks"], ascending=[True, True])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC"]]
    .copy()
)
base.columns = ["Patient", "min_week_base", "min_FVC"]

data = data.merge(base[["Patient", "min_FVC"]], on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]

COLS = ["Sex", "SmokingStatus"]  # ,'Age'
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
print(FE)

data[FE] = data[FE].apply(pd.to_numeric, errors="coerce")
data[FE] = data[FE].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

if "FVC" in data.columns:
    data["FVC"] = pd.to_numeric(data["FVC"], errors="coerce")

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data, base

print(train.shape, test.shape, sub.shape)



## === cell 7
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def _ensure_2d_y(y_true):
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.reshape(y_true, (-1, 1))
    return y_true


def score(y_true, y_pred):
    y_true = _ensure_2d_y(y_true)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return tf.keras.backend.mean(metric)


def qloss(y_true, y_pred):
    y_true = _ensure_2d_y(y_true)
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

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    )
    model.compile(loss=mloss(0.8), optimizer=opt, metrics=[score])
    return model




## === cell 8
train = train.loc[~train["FVC"].isna()].copy()
train = train.reset_index(drop=True)

y = pd.to_numeric(train["FVC"], errors="coerce").astype(np.float32).values
z = train[FE].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
ze = sub[FE].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
ze = np.nan_to_num(ze, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y = np.nan_to_num(y, nan=np.nan, posinf=np.nan, neginf=np.nan).astype(np.float32)

good_rows = np.isfinite(y) & np.isfinite(z).all(axis=1)
if not np.all(good_rows):
    train = train.loc[good_rows].reset_index(drop=True)
    y = y[good_rows]
    z = z[good_rows]

nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

net = make_model(nh)
print(net.summary())
print(net.count_params())

assert z.shape[0] == y.shape[0]
assert np.isfinite(z).all() and np.isfinite(y).all() and np.isfinite(ze).all()



## === cell 9
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)



## === cell 10
cnt = 0
EPOCHS = 1000
BATCH_SIZE = 64

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)

    x_tr = np.asarray(z[tr_idx], dtype=np.float32)
    y_tr = np.asarray(y[tr_idx], dtype=np.float32)
    x_va = np.asarray(z[val_idx], dtype=np.float32)
    y_va = np.asarray(y[val_idx], dtype=np.float32)

    x_tr = np.nan_to_num(x_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x_va = np.nan_to_num(x_va, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y_tr = np.nan_to_num(y_tr, nan=np.nan, posinf=np.nan, neginf=np.nan).astype(
        np.float32
    )
    y_va = np.nan_to_num(y_va, nan=np.nan, posinf=np.nan, neginf=np.nan).astype(
        np.float32
    )

    ok_tr = np.isfinite(y_tr) & np.isfinite(x_tr).all(axis=1)
    ok_va = np.isfinite(y_va) & np.isfinite(x_va).all(axis=1)
    x_tr, y_tr = x_tr[ok_tr], y_tr[ok_tr]
    x_va, y_va = x_va[ok_va], y_va[ok_va]

    assert np.isfinite(x_tr).all() and np.isfinite(y_tr).all()
    assert np.isfinite(x_va).all() and np.isfinite(y_va).all()

    net.fit(
        x_tr,
        y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_va, y_va),
        verbose=0,
    )
    print("train", net.evaluate(x_tr, y_tr, verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate(x_va, y_va, verbose=0, batch_size=BATCH_SIZE))
    print("predict val...")
    pred[val_idx] = net.predict(x_va, batch_size=BATCH_SIZE, verbose=0)
    print("predict test...")
    pe += (
        net.predict(np.asarray(ze, dtype=np.float32), batch_size=BATCH_SIZE, verbose=0)
        / NFOLD
    )



## === cell 11
sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))

sub["FVC1"] = 0.996 * pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]
if sigma_mean < 70:
    subm["Confidence"] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce")
subm["Confidence"] = subm["Confidence"].replace([np.inf, -np.inf], np.nan).fillna(70.0)
subm["Confidence"] = subm["Confidence"].clip(lower=70.0)



## === cell 12
otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(int(otest.Weeks[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1

out = subm[["Patient_Week", "FVC", "Confidence"]].copy()
out["FVC"] = pd.to_numeric(out["FVC"], errors="coerce").astype(float)
out["Confidence"] = pd.to_numeric(out["Confidence"], errors="coerce").astype(float)

out["FVC"] = out["FVC"].fillna(out["FVC"].median())
out["Confidence"] = out["Confidence"].fillna(70.0)

out["Confidence"] = out["Confidence"].clip(lower=70.0)

out.to_csv("submission.csv", index=False)
print(out.head())
print("Wrote submission.csv with shape:", out.shape)
print("Columns:", list(out.columns))
assert list(out.columns) == ["Patient_Week", "FVC", "Confidence"]
assert out.shape[0] > 0
assert out["Patient_Week"].isna().sum() == 0
