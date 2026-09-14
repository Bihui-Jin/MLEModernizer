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

- What this solution (achieved -8.76216) has done: 'I fix the runtime crash at import time by forcing a compatible pure-Python protobuf implementation before TensorFlow loads (this avoids the `MessageFactory.GetPrototype` error). Then I fix the pandas 2.x breakage by replacing deprecated `DataFrame.append` with `pd.concat`, which also restore the creation of `FE`, `z`, `pred`, and downstream variables. Finally, I make TensorFlow’s optimizer arguments compatible with TF 2.18 (`learning_rate` instead of `lr`, remove deprecated `decay`) so training runs end-to-end and a valid `submission.csv` with the exact required columns is always written.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf import crash by ensuring the pure-Python protobuf runtime is forced before TensorFlow loads and by falling back to disabling C++ protos if needed. Then I fix the training crash caused by a shape mismatch: your custom loss expects `y_true` to have 3 columns (`y_true[:,0]`) but you currently pass a 1D `FVC` target, so I create a 3-column target tensor (`[FVC, FVC, FVC]`) for training/validation while keeping the model, loss, and metric logic unchanged. Finally, I keep the submission writing logic intact but make sure it always writes a valid `submission.csv` with the required columns even if inference produces NaNs.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before TensorFlow loads and by disabling the C++ protobuf backend as a safe fallback in this Kaggle environment. Then I fix the training crash (`None values not supported`) by ensuring all engineered feature columns in `FE` are numeric floats with no NaNs/Infs in train/val/test, which currently can happen when a category exists only in one split or when merges introduce missing values. Finally, I keep the model/loss/training loop unchanged, but make submission writing more robust (clip/clean confidence and keep required columns) so it always produces a valid `submission.csv`.'
- What this solution (achieved -8.76216) has done: 'I fix two blockers that prevent the notebook from running end-to-end: the TensorFlow/protobuf import crash and the `None values not supported` training crash. For protobuf, I keep your intent (force pure-Python protobuf) but add a safe compatibility patch that restores `MessageFactory.GetPrototype` when newer protobuf versions remove it, which avoids changing your ML logic. For training, I ensure the label tensor passed to `fit()` contains no NaNs/Infs (these can arise from rare missing/invalid FVC rows after dedup/merge), while keeping the same 3-column target and model/loss unchanged. These changes are score-neutral-to-positive (they restore training completion and prevent silent NaN propagation) and still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring every tensor passed into `model.fit()` is a pure numeric `float32` NumPy array with no `None`/object dtypes and no NaNs/Infs, including a hard cast of the engineered one-hot columns and the `FVC` label before building `y3`. This is a minimal correctness fix that preserves your model/loss/training loop exactly, but prevents Keras from encountering `None` values originating from mixed dtypes after merges/one-hot creation. I also add a small defensive cleanup right before fitting (per-fold) to guarantee the split arrays are clean without changing the training semantics. The submission-writing logic and column names stay the same, and the script always write `submission.csv`.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring every feature column in `FE` is created with safe, unique column names (your current one-hot encoding can create duplicate column names like `Male` appearing for both `Sex` and `SmokingStatus`, which leads to object/None values when selecting `train[FE]`). I also force all model inputs/targets to be finite `float32` right before training, and add a single defensive cleanup that fails fast if any non-numeric slips through. These changes are minimal, keep your model/loss/training loop intact, and should also improve score by preventing corrupted features from silently degrading training. The submission writing remains the same and still produce a valid `submission.csv`.'
- What this solution (achieved -8.76216) has done: 'I fix the training crash (`ValueError: None values not supported`) by ensuring there are no `None` values (from object dtypes) anywhere in the engineered feature matrix or labels right before `model.fit()`. The root cause is that, with mixed train/test/sub merges, some numeric columns (especially `Percent`, `Age`, and the merged `min_FVC`) can become `object` with actual `None` values, and `pd.to_numeric(..., errors="coerce")` converts those to `NaN` but not always if the column already contains Python `None` inside arrays; we explicitly coerce the source numeric columns early and use `np.asarray(..., dtype=np.float32)` + `np.nan_to_num` after feature construction. This is a correctness/stability fix that preserves your model, loss, folds, epochs, and submission logic, and it should also slightly improve score by preventing corrupted inputs. Finally, I add a single assert-like diagnostic to fail fast if any non-finite values remain (should not trigger after the fix).'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring every feature column used for training/inference is strictly numeric `float32` and contains no Python `None` values (which can still slip through via mixed-dtype DataFrames even after `to_numeric`). I add a single, centralized “make matrix” helper that converts `train[FE]`/`sub[FE]` into clean NumPy arrays and use it both before CV and inside each fold to guarantee tensors are safe for Keras. This is a correctness/stability fix that keeps the same features, model, loss, folds, and epochs, and should also improve score slightly by preventing corrupted inputs from degrading training. The submission-writing logic remains the same and always produce a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the remaining `None values not supported` crash by ensuring the feature/label arrays passed into Keras are strictly finite `float32` NumPy arrays with no object dtype and no hidden `None` values, using a centralized conversion that also forces the overall array dtype before fitting. I also ensure TensorFlow runs deterministically in this environment and explicitly disable XLA JIT to avoid sporadic conversion edge cases that can surface as `None` during graph tracing (score-neutral, stability-only). Finally, I keep your model, loss, folds, epochs, and submission logic unchanged, only making the input pipeline robust so training completes and a valid `submission.csv` is always written.'
- What this solution (achieved -8.76216) has done: 'I fix the training crash (`ValueError: None values not supported`) by ensuring that *all* model inputs and targets passed to `fit/evaluate/predict` are guaranteed numeric `float32` and contain no hidden `None` values, by adding a final centralized sanitization step right before each fold’s training (this preserves your model, loss, folds, and epochs). I also make the submission step robust by guaranteeing `Confidence` is always finite and applying the competition-required lower clip at 70 only via the metric (keeping your existing logic unchanged) while preventing negative/NaN confidence values from being written. These are correctness/stability fixes (score-neutral-to-slightly-positive because they prevent silent bad tensors), and the script always write a valid `submission.csv` with the exact required columns. No changes are made to the core model architecture, loss, or training schedule.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_CPP", "0")

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import pydicom

print("TF:", tf.__version__)
print("Pandas:", pd.__version__)




## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


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
    image = pydicom.dcmread(image_files_list[0])
    plt.figure()
    plt.imshow(image.pixel_array, cmap=plt.cm.bone)
    plt.axis("off")
    plt.show()
else:
    print("No DICOM files found under:", image_path)



## === cell 6
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([train, test, sub], axis=0, ignore_index=True)

for col in ["Weeks", "FVC", "Percent", "Age"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

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

data["min_FVC"] = pd.to_numeric(data["min_FVC"], errors="coerce")
data["base_week"] = pd.to_numeric(data["base_week"], errors="coerce")

COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    data[col] = data[col].fillna("Unknown").astype(str)
    for mod in data[col].unique():
        feat = f"{col}__{mod}"
        FE.append(feat)
        data[feat] = (data[col] == mod).astype(np.float32)


def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    arr = s.values.astype(np.float64, copy=False)
    finite = np.isfinite(arr)
    mn = float(np.min(arr[finite])) if finite.any() else 0.0
    mx = float(np.max(arr[finite])) if finite.any() else 1.0
    denom = (mx - mn) if (mx - mn) != 0 else 1.0
    return (s - mn) / denom


data["age"] = _minmax(data["Age"])
data["BASE"] = _minmax(data["min_FVC"])
data["week"] = _minmax(data["base_week"])
data["percent"] = _minmax(data["Percent"])

FE += ["age", "percent", "week", "BASE"]
print("Features:", FE)
print("Num features:", len(FE), "Unique features:", len(set(FE)))

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

for df in (train, test, sub):
    for c in FE:
        if c not in df.columns:
            df[c] = 0.0
    df[FE] = df[FE].apply(pd.to_numeric, errors="coerce")
    df[FE] = df[FE].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df[FE] = df[FE].astype(np.float32)

print(train.shape, test.shape, sub.shape)



## === cell 7
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
            learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
        ),
        metrics=[score],
    )
    return model




## === cell 8
def make_matrix(df: pd.DataFrame, fe_cols):
    x = df.reindex(columns=fe_cols)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = x.to_numpy(dtype=np.float32, copy=True)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    return arr


def ensure_float32_matrix(a, name):
    a = np.asarray(a)
    if a.dtype == object:
        a = (
            pd.DataFrame(a)
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
    a = np.asarray(a, dtype=np.float32)
    a = np.ascontiguousarray(a, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if not np.isfinite(a).all():
        raise ValueError(f"Non-finite values remain in {name}.")
    return a


train["FVC"] = pd.to_numeric(train["FVC"], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
valid_mask = train["FVC"].notna()
if valid_mask.mean() < 1.0:
    print(
        f"Dropping {int((~valid_mask).sum())} train rows with invalid FVC to avoid NaN/None tensors."
    )
train = train.loc[valid_mask].reset_index(drop=True)

y = train["FVC"].values.astype(np.float32)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y3 = np.repeat(y.reshape(-1, 1), 3, axis=1).astype(np.float32)

z = make_matrix(train, FE)
ze = make_matrix(sub, FE)

z = ensure_float32_matrix(z, "z")
ze = ensure_float32_matrix(ze, "ze")
y3 = ensure_float32_matrix(y3, "y3")

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
EPOCHS = 800
BATCH_SIZE = 64
diff_sum = 0.0

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)

    x_tr = ensure_float32_matrix(z[tr_idx], "x_tr")
    y_tr = ensure_float32_matrix(y3[tr_idx], "y_tr")
    x_va = ensure_float32_matrix(z[val_idx], "x_va")
    y_va = ensure_float32_matrix(y3[val_idx], "y_va")

    net.fit(
        x_tr,
        y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_va, y_va),
        verbose=0,
    )
    train_loss, train_score = net.evaluate(x_tr, y_tr, verbose=0, batch_size=BATCH_SIZE)
    print(f"Train Loss: {train_loss}  Score: {train_score}")
    val_loss, val_score = net.evaluate(x_va, y_va, verbose=0, batch_size=BATCH_SIZE)
    print(f"Val Loss: {val_loss}  Score: {val_score}")
    score_diff = val_score - train_score
    diff_sum += score_diff
    print(f"Score diff: {score_diff}")
    print("Predict val...")
    pred[val_idx] = ensure_float32_matrix(
        net.predict(x_va, batch_size=BATCH_SIZE, verbose=0), "pred_val"
    )
    print("Predict test...")
    pe += (
        ensure_float32_matrix(
            net.predict(ze, batch_size=BATCH_SIZE, verbose=0), "pred_test"
        )
        / NFOLD
    )

print(f"Score diff sum : {diff_sum}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1918503308.py in <cell line: 0>()
     16     y_va = ensure_float32_matrix(y3[val_idx], "y_va")
     17 
---> 18     net.fit(
     19         x_tr,
     20         y_tr,

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

subm["FVC"] = (
    pd.to_numeric(subm["FVC"], errors="coerce")
    .astype(np.float32)
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
subm["Confidence"] = subm["Confidence"].fillna(float(sigma_opt)).astype(np.float32)

subm["Confidence"] = subm["Confidence"].clip(lower=0.1)



## === cell 12
otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1

subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    subm[["Patient_Week", "FVC", "Confidence"]].shape,
)
print(subm[["Patient_Week", "FVC", "Confidence"]].head())
