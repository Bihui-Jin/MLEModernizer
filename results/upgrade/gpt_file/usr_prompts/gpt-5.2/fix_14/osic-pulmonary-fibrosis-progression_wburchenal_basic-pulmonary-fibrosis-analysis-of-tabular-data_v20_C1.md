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

-6.9308

# 6. Current score

-8.76198

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I fix the two runtime blockers first: the `protobuf`/TensorFlow import error by forcing the pure‑Python protobuf implementation before importing TensorFlow, and the pandas 2.x removal of `DataFrame.append` by switching to `pd.concat`. Next I ensure the feature list `FE` is always defined and stable (handle NaNs and keep one-hot columns consistent) so later cells can build `z/ze` without `NameError`. Finally I keep your model, losses, and training loop intact, but update the Adam arguments to TensorFlow 2.18 (use `learning_rate` instead of deprecated `lr/decay`) and guarantee a correctly formatted `submission.csv` is written.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure‑Python protobuf implementation *and* pinning TensorFlow to use the legacy protobuf code path before importing TensorFlow. Then I fix the training crash by making the target `y` match what your custom `score()`/`qloss()` expect (a 3-column tensor), without changing the model or loss definitions. Finally, I keep the existing post-processing and ensure the pipeline always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by forcing the pure-Python protobuf implementation and downgrading protobuf at runtime (Kaggle allows pip installs) before importing TensorFlow. Then I fix the `None values not supported` training error by ensuring all model inputs/features are finite float32 arrays (including coercing any accidental object dtype and replacing NaN/inf right before training). Finally, I keep your model/loss/training loop intact and only make the confidence post-processing consistent and clipped (>=70) to better match the competition metric, which should nudge the score upward toward the target while remaining minimal.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring the training targets match what `score()` expects (first column must be the true FVC) and by forcing all model inputs/targets to be finite `float32` right before `fit()`. This keeps your model architecture, loss definitions, and training loop intact, but removes the silent NaN/None pathway that triggers Keras tensor conversion errors. I also make the confidence handling safer by clipping the model-derived confidence to be finite and ≥70 (as the metric clips), which should improve calibration slightly and move the score upward toward the target. Finally, I keep the submission-writing logic the same but add alignment checks so a valid `submission.csv` is always produced.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring every array fed to Keras (`x_tr/x_va/y_tr/y_va`) is strictly numeric `float32` and contains only finite values, and by adding a hard assertion right before `fit()` to catch any remaining non-finite/None issues early. This is a minimal, score-neutral stability change that preserves your model, loss, and training loop. I also make the confidence/FVC post-processing more robust by forcing predictions to finite numbers before computing uncertainty and writing the submission, preventing NaNs/infs from leaking into `submission.csv`. The rest of the pipeline (features, architecture, epochs, folds, and metric semantics) remains unchanged.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` training crash by locating and removing any `None`/NaN values in the combined dataframe *before* building `z/ze/y`, because Keras can still receive `None` via object-dtype arrays even if `np.nan_to_num` is applied later. I also ensure the one-hot feature columns are created with safe, non-colliding names (prefixing with the original column) to prevent accidental overwrites that can introduce mixed dtypes. These changes are minimal and keep your model/loss/training loop intact, but should both unblock training and slightly improve stability/score by ensuring inputs are truly numeric. Finally, I keep the same submission logic but add a last-mile numeric coercion so `submission.csv` is always valid.'
- What this solution (achieved -8.76216) has done: 'I fix the training-time `None values not supported` error by ensuring the combined dataframe has no object/None leakage in the model inputs/targets and by hard-coercing the `FVC` target to numeric (some folds can still pick up `None` via mixed dtypes). I also make the Keras `fit()` call use explicit numpy float32 arrays right at the call site (even if upstream looked clean), which reliably prevents Keras tensor conversion from seeing `None`. These are stability-only changes that keep your model, loss, folds, epochs, and post-processing intact, and should let the pipeline finish and generate `submission.csv`. No score-tuning beyond correctness is introduced (your existing confidence clipping remains as-is).'
- What this solution (achieved -8.76216) has done: 'I fix the training crash (`None values not supported`) by removing the one remaining source of `None`: the target `y` is being built from `train["FVC"]` before ensuring `train["FVC"]` is fully numeric/filled, so we hard-coerce `FVC` for the train split right after the combined preprocessing and then rebuild `y` from that clean column. I also add a small safety net right before `fit()` to assert there are no `None` values hiding in object arrays (this is score-neutral but prevents the same crash). Finally, I keep your model/loss/loop/post-processing intact and only ensure the pipeline completes and writes a valid `submission.csv`.'
- What this solution (achieved -8.76198) has done: 'I fix the training crash (`ValueError: None values not supported`) by removing the only remaining source of `None` leaking into tensors: `train.drop_duplicates(keep=False, ...)` can drop all rows for duplicated (Patient, Weeks) pairs, which can later create empty/NaN baseline joins and object-typed arrays that still contain Python `None` despite numeric coercion. I change that to a stable dedup (`keep="first"`) and add a hard cleanup of the exact columns used by the model (`FE` and `FVC`) after the train/test/sub split to guarantee pure numeric `float32` arrays reach Keras. This is minimal and keeps your model, loss, folds, epochs, and post-processing intact; it should also improve score slightly by preserving more training rows instead of discarding them. The script then run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -8.76198) has done: 'I fix the `None values not supported` crash by eliminating the remaining path where `None` can enter tensors: `data.replace({None: np.nan})` doesn’t remove Python `None` inside object columns reliably, so I normalize all object columns with `replace(..., regex=True)` and then hard-coerce the exact arrays fed into Keras using `np.nan_to_num` plus a strict `astype(np.float32)` check. I also add a small diagnostic that pinpoints which column/row is still producing `None` if it happens again, without changing the model, loss, folds, epochs, or prediction post-processing. These changes are score-neutral in intent (they only prevent invalid tensors), and your existing confidence clipping/calibration is kept intact to preserve current scoring behavior while allowing the run to complete and generate `submission.csv`. Finally, I ensure the written submission is correctly aligned to `sample_submission.csv` and contains no NaNs.'
- What this solution (achieved -8.76198) has done: 'I fix the `None values not supported` crash at `net.fit()` by identifying and removing the remaining source of `None`: the model inputs/targets can still become `object` arrays due to mixed dtypes from pandas, even after earlier coercions. The minimal fix is to rebuild `z/ze/y` from the cleaned dataframes using a strict numeric conversion path, and add a hard assert that no `None` exists anywhere before Keras sees the arrays (so the failure is prevented, not just masked). This keeps your model architecture, custom losses, folds, and training loop intact while unblocking training and producing a valid `submission.csv`. As a small, metric-aligned and minimal calibration nudge toward the target, I also ensure the predicted confidence is always finite and clipped to ≥70 exactly as the evaluation does (your code already mostly does this; we just make it airtight).'
- What this solution (achieved -8.76198) has done: 'I fix the `None values not supported` crash by finding where `None` is still entering the arrays (it can happen via pandas `object` columns or a stray `None` inside a numpy array) and coercing everything through a strict numeric pipeline right before `fit()`, with an explicit scan that pinpoints any remaining offending indices. This is a stability fix only (no architecture/loss/training-loop changes) and unblock training so you can generate a valid `submission.csv`. I also make the tensor conversion path in `_ensure_float32_finite` robust for both 1D and 2D inputs (the current `pd.DataFrame(arr)` approach can silently keep `None` in some cases). Finally, I keep your existing confidence clipping/post-processing intact to preserve score behavior while letting the run complete.'
- What this solution (achieved -8.76198) has done: 'I fix the remaining `ValueError: None values not supported` by ensuring we never pass a `None`/object-containing numpy array into Keras: the issue can still happen because Keras may see a view with object dtype or a hidden `None` even after earlier coercions. The minimal, score-neutral fix is to rebuild all fold arrays using a strict `np.asarray(..., dtype=np.float32)` path plus a hard scan for `None` before conversion, and to pass inputs/targets to `fit()` as `tf.constant` tensors (which guarantees TensorFlow sees dense numeric tensors). I also make the same strict conversion for `validation_data` and prediction inputs to avoid intermittent failures. No model architecture, loss definitions, epochs, folds, or post-processing logic are changed, so the score behavior should stay similar while allowing the run to complete and generate `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf
        from packaging.version import Version

        v = Version(google.protobuf.__version__)
        if v.major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            import importlib

            importlib.invalidate_caches()
    except Exception as e:
        print("Warning: protobuf compatibility step failed:", repr(e))


_ensure_protobuf_compat()

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import pydicom

print("TF:", tf.__version__)
print("pandas:", pd.__version__)




## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)



## === cell 2
DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(train.head())
print(test.head())
print(sub.head())



## === cell 3
train.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(test.drop("Weeks", axis=1), on="Patient")

print(train.shape, test.shape, sub.shape)



## === cell 4
print(train.info())



## === cell 5
image_path = DATA_DIR
image_files_list = []
for dirName, subdirList, fileList in os.walk(image_path):
    for filename in fileList:
        if filename.lower().endswith(".dcm"):
            image_files_list.append(os.path.join(dirName, filename))
            break
    if len(image_files_list) > 0:
        break

if len(image_files_list) > 0:
    image = pydicom.dcmread(image_files_list[0])
    plt.figure()
    plt.imshow(image.pixel_array, cmap=plt.cm.bone)
    plt.title("Example DICOM slice")
    plt.axis("off")
    plt.show()
else:
    print("No DICOMs found under", image_path)



## === cell 6
train["WHERE"] = "train"
test["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([train, test, sub], axis=0, ignore_index=True)

data = data.replace({None: np.nan})
data = data.replace(to_replace=r"^\s*None\s*$", value=np.nan, regex=True)

obj_cols = data.select_dtypes(include=["object"]).columns.tolist()
for c in obj_cols:
    s = data[c]
    s = s.replace(to_replace=r"^\s*None\s*$", value=np.nan, regex=True)
    s = s.where(pd.notna(s), np.nan)
    data[c] = s

for _c in ["FVC", "Percent", "Age", "Weeks"]:
    if _c in data.columns:
        data[_c] = pd.to_numeric(data[_c], errors="coerce")

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

data["Sex"] = data["Sex"].fillna("Unknown")
data["SmokingStatus"] = data["SmokingStatus"].fillna("Unknown")

COLS = ["Sex", "SmokingStatus"]
FE = []

for col in COLS:
    for mod in sorted(data[col].astype(str).unique()):
        cname = f"{col}__{mod}"
        FE.append(cname)
        data[cname] = (data[col].astype(str) == mod).astype(np.int8)

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

data[FE] = data[FE].apply(pd.to_numeric, errors="coerce")
data[FE] = data[FE].replace([np.inf, -np.inf], np.nan).fillna(0.0)

print("Feature columns:", FE)

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

train["FVC"] = (
    pd.to_numeric(train["FVC"], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
train[FE] = (
    train[FE]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
sub[FE] = (
    sub[FE]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

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
        learning_rate=0.05, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
    )
    model.compile(loss=mloss(0.8), optimizer=opt, metrics=[score])
    return model




## === cell 8
def _df_to_float32_matrix(df, cols, name):
    m = df[cols].copy()
    for c in cols:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = m.to_numpy(dtype=np.float32, copy=True)
    if arr.dtype == object:
        raise ValueError(f"{name} still object dtype after coercion")
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )
    return arr


y1 = (
    pd.to_numeric(train["FVC"], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .to_numpy(dtype=np.float32, copy=True)
)
y = np.repeat(y1.reshape(-1, 1), 3, axis=1).astype(np.float32, copy=False)

z = _df_to_float32_matrix(train, FE, "z")
ze = _df_to_float32_matrix(sub, FE, "ze")

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
def _ensure_float32_finite(a, name):
    """
    Bugfix: Keras/TF can error with 'None values not supported' if an array contains
    Python None (often via object dtype views). We (1) scan for None early, and
    (2) force a strict float32 dense array.
    """
    arr = np.asarray(a)

    try:
        flat = arr.ravel()
        if flat.dtype == object:
            if any(x is None for x in flat[: min(flat.size, 5000)]):
                none_mask = np.fromiter(
                    (x is None for x in flat), dtype=bool, count=flat.size
                )
                idx = np.where(none_mask)[0][:10]
                raise ValueError(
                    f"{name} contains Python None at flattened indices {idx.tolist()} (shape {arr.shape})."
                )
    except ValueError:
        raise
    except Exception:
        pass

    if arr.dtype == object:
        if arr.ndim == 1:
            s = pd.Series(arr)
            s = s.replace(to_replace=r"^\s*None\s*$", value=np.nan, regex=True)
            s = pd.to_numeric(s, errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out = s.to_numpy(dtype=np.float32, copy=True)
        else:
            df = pd.DataFrame(arr)
            df = df.replace(to_replace=r"^\s*None\s*$", value=np.nan, regex=True)
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out = df.to_numpy(dtype=np.float32, copy=True)
    else:
        out = np.asarray(arr, dtype=np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    if not np.isfinite(out).all():
        bad = np.where(~np.isfinite(out))
        raise ValueError(
            f"{name} contains non-finite values at indices like {bad[0][:5]}"
        )
    return out


def _to_tf(a, name):
    """
    Bugfix: ensure TF sees a dense numeric Tensor (never a numpy object array).
    This prevents the backend conversion path from tripping on hidden None values.
    """
    a = _ensure_float32_finite(a, name)
    return tf.constant(a, dtype=tf.float32)


cnt = 0
EPOCHS = 800
BATCH_SIZE = 64
diff_sum = 0.0

z = _ensure_float32_finite(z, "z")
ze = _ensure_float32_finite(ze, "ze")
y = _ensure_float32_finite(y, "y")

assert z.dtype == np.float32 and ze.dtype == np.float32 and y.dtype == np.float32
assert np.isfinite(z).all() and np.isfinite(ze).all() and np.isfinite(y).all()

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)

    x_tr = _to_tf(z[tr_idx], "x_tr")
    y_tr = _to_tf(y[tr_idx], "y_tr")
    x_va = _to_tf(z[val_idx], "x_va")
    y_va = _to_tf(y[val_idx], "y_va")

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
    diff_sum += float(score_diff)
    print(f"Score diff: {score_diff}")

    print("Predict val...")
    pred[val_idx] = net.predict(x_va, batch_size=BATCH_SIZE, verbose=0)
    print("Predict test...")
    pe += (
        net.predict(tf.constant(ze, dtype=tf.float32), batch_size=BATCH_SIZE, verbose=0)
        / NFOLD
    )

print(f"Score diff sum : {diff_sum}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2619286699.py in <cell line: 0>()
     85     y_va = _to_tf(y[val_idx], "y_va")
     86 
---> 87     net.fit(
     88         x_tr,
     89         y_tr,

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
pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(
    np.float32, copy=False
)
pe = np.nan_to_num(pe, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

sigma_opt = mean_absolute_error(y1, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))

sub["FVC1"] = 0.996 * pe[:, 1]
sub["Confidence1"] = pe[:, 2] - pe[:, 0]
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

subm["Confidence1"] = pd.to_numeric(subm["Confidence1"], errors="coerce")
subm["Confidence1"] = subm["Confidence1"].replace([np.inf, -np.inf], np.nan)

if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
subm["Confidence"] = subm["Confidence"].fillna(float(sigma_opt))
subm["Confidence"] = subm["Confidence"].clip(lower=70.0)

subm["FVC"] = (
    pd.to_numeric(subm["FVC"], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(2000.0)
)



## === cell 12
otest = pd.read_csv(f"{DATA_DIR}/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = otest.FVC[i]
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

final_sub = subm[["Patient_Week", "FVC", "Confidence"]].copy()
final_sub["FVC"] = pd.to_numeric(final_sub["FVC"], errors="coerce").astype(float)
final_sub["Confidence"] = pd.to_numeric(
    final_sub["Confidence"], errors="coerce"
).astype(float)

sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")[["Patient_Week"]]
final_sub = sample.merge(final_sub, on="Patient_Week", how="left")
final_sub["FVC"] = final_sub["FVC"].fillna(2000.0)
final_sub["Confidence"] = final_sub["Confidence"].fillna(100.0).clip(lower=70.0)

final_sub["FVC"] = final_sub["FVC"].astype(float)
final_sub["Confidence"] = final_sub["Confidence"].astype(float)

final_sub.to_csv("submission.csv", index=False)

print(final_sub.head())
print("Wrote submission.csv with shape:", final_sub.shape)
print("Columns:", list(final_sub.columns))
print("Any nulls:", final_sub.isnull().any().to_dict())
