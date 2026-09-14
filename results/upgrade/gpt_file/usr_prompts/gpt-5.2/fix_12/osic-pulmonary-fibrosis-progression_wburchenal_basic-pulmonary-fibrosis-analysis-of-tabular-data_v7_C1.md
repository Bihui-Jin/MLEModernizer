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

-6.9586

# 6. Current score

-8.76216

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76216) has done: 'I fix the two blockers preventing the notebook from running: (1) the TensorFlow/protobuf `MessageFactory.GetPrototype` crash triggered at import time by adjusting the protobuf implementation before importing TensorFlow, and (2) pandas 2.x removing `DataFrame.append` by replacing it with `pd.concat`. Those fixes allow feature engineering to define `FE`, so downstream cells (`z`, `pred`, `subm`) exist and training/inference run end-to-end. I also make one compatibility fix for TF 2.18 by using `learning_rate=` instead of the deprecated `lr=` argument in Adam. The core model/loss/training loop remains unchanged, and the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the two runtime blockers without changing the model architecture or training approach: (1) the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, and (2) the loss/metric shape bug by making `y` 2D so your existing `y_true[:, 0]` indexing works. I also correct the no-op casts in `score()` so tensors are actually casted (score-neutral but prevents dtype issues), and make KFold deterministic via `shuffle=True, random_state=...` (small, safe score improvement toward the target). Finally, I ensure the submission is written with the required columns and `.csv` suffix.'
- What this solution (achieved -8.76216) has done: 'I fix the two runtime blockers shown in your tracebacks without changing the model/loss/training loop semantics: (1) the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation early (including the required `_PREFERRED` env var for newer protobuf builds), and (2) the `None values not supported` crash by ensuring all engineered features in `FE` are fully numeric with no NaNs/Infs (fill + safe dtype conversion). This keeps the same feature set and model architecture, but makes the pipeline robust under pandas 2.x / TF 2.18. I also add a small safety clamp for predicted confidence (to avoid pathological negative/NaN sigma) which is aligned with the metric and should modestly improve score toward your target without changing the core approach. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the two runtime blockers shown: the TensorFlow/protobuf import crash and the “None values not supported” error during `model.fit()`. The protobuf crash is addressed by forcing the pure-Python protobuf implementation *and* applying a small compatibility shim so TF can still call `MessageFactory.GetPrototype` under newer protobuf versions. The `None` error is resolved by ensuring all feature columns used in `z/ze` are strictly numeric, finite, and contain no `None`/`NaN` (including handling rare missing `Sex/SmokingStatus` categories consistently). These are stability fixes and should also slightly improve score by preventing bad rows/NaNs from corrupting training.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring every feature column in `FE` is strictly numeric and present for all rows, even when category values contain spaces/symbols or collide with existing column names. Specifically, I create safe one-hot column names (prefixed and sanitized), build them using a stable category vocabulary from train+test, and then coerce/fill all engineered features plus `min_FVC` safely before creating `z/ze`. This is a runtime/stability fix and should also modestly improve score versus silently broken/partially-missing features. I keep the model, loss, folds, epochs, and submission post-processing unchanged.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring the training target `y` is a dense 2D float32 array with no missing values (your current `y` is built from `train["FVC"]` after concatenation/merging, which can contain NaNs and triggers this error inside `model.fit`). I also harden the feature matrix creation by enforcing finite float32 for all FE columns right before converting to NumPy, to prevent any stray non-finite/NA from slipping through. These changes are execution/stability fixes and keep the model, loss, folds, epochs, and submission logic the same; they should also slightly improve the score by avoiding corrupted rows during training. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash in `model.fit()` by ensuring the feature matrices and target are strictly numeric `float32` with all missing/infinite values removed, and by resetting indices so KFold indexing cannot pick up misaligned/invalid rows. I also make sure `sub[FE]` exists and is fully filled (the current code only hardens `train[FE]`, not `sub[FE]`, which is a common source of hidden `None`/object values). These are execution/stability fixes that preserve your model architecture, loss, folds, epochs, and submission logic; they should also modestly improve score by preventing corrupted batches. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring the training target `y` has the correct shape `(n, 3)` expected by your custom loss (which uses quantiles) while still preserving the same semantics by repeating the true FVC across the three outputs. I also harden `z`, `ze`, and `y` right before fitting so they are strictly finite `float32` arrays, eliminating any remaining hidden `object`/`None`/NaN pathways into Keras. These are minimal, execution-blocking fixes and should also improve score (your previous run likely trained incorrectly/unstably due to target mismatch). The model architecture, loss definition, folds, epochs, and submission post-processing are otherwise unchanged, and the script still write `submission.csv` with the required columns.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring every input to `model.fit()` is a dense, finite `float32` NumPy array (the current failure is caused by `train[FE]` containing object/extension dtypes that can carry `pd.NA` through despite the earlier cleaning). I do this by rebuilding `z`, `ze`, and `y` from `train/sub` after forcing all feature columns to plain numpy floats via `to_numpy()` + `np.nan_to_num()`, and I also ensure `FE` columns exist in both train and sub (missing one-hot columns can otherwise introduce Nones/NA). These are execution/stability fixes and are score-neutral in intent, but they should also modestly improve score by preventing corrupted batches/rows during training. No model architecture, loss, folds, epochs, or submission post-processing logic is changed.'
- What this solution (achieved -8.76216) has done: 'I fix the training crash (“None values not supported”) by enforcing that every feature/target array passed into `model.fit()` is a plain dense `float32` NumPy array with finite values, and by rebuilding `y` from the cleaned `train["FVC"]` after all merges (so no hidden `pd.NA`/object dtypes slip through). I also remove the deprecated/unsupported `decay=` argument from the Adam optimizer for TF 2.18 compatibility (score-neutral, but avoids runtime issues). These changes keep your model architecture, loss, folds, epochs, and submission post-processing intact while making training run end-to-end and produce a valid `submission.csv`. This should also modestly improve the score versus a run that fails or trains on corrupted batches.'
- What this solution (achieved -8.76216) has done: 'I fix the `None values not supported` crash by ensuring every feature column in `train[FE]` and `sub[FE]` is a plain dense numeric numpy array (not pandas nullable/extension dtype that can carry `pd.NA` even after `astype(np.float32)`). To keep the core model/training loop unchanged, I only add a final, strict conversion step right before KFold training: `to_numpy(dtype=np.float32, na_value=0.0)` plus `np.nan_to_num(...)`. I also make sure `FE` has no duplicates (can happen if category strings sanitize to the same name) to avoid accidental object columns and hidden missing values. These are execution/stability fixes and should also slightly improve score by preventing corrupted batches.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_PREFERRED", "python")

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

print(
    "Python/TensorFlow/Pandas versions:",
    os.sys.version.split()[0],
    tf.__version__,
    pd.__version__,
)




## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(6942069)



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
    plt.imshow(image.pixel_array, cmap=plt.cm.inferno)
    plt.axis("off")
    plt.show()
else:
    print("No DICOM files found under:", image_path)




## === cell 6
def _safe_cat_colname(prefix: str, value: str) -> str:
    v = str(value)
    v = v.strip()
    v = v.replace(" ", "_")
    v = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in v)
    if v == "":
        v = "Unknown"
    return f"{prefix}__{v}"


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
    data[col] = data[col].fillna("Unknown").astype(str)
    vocab = pd.Index(data[col].unique()).tolist()
    for mod in vocab:
        onehot_name = _safe_cat_colname(col, mod)
        FE.append(onehot_name)
        data[onehot_name] = (data[col] == mod).astype(np.int8)

for c in ["Age", "Percent", "base_week", "min_FVC"]:
    data[c] = pd.to_numeric(data[c], errors="coerce")

data[["Age", "Percent", "base_week", "min_FVC"]] = data[
    ["Age", "Percent", "base_week", "min_FVC"]
].replace([np.inf, -np.inf], np.nan)

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Percent"] = data["Percent"].fillna(data["Percent"].median())
data["base_week"] = data["base_week"].fillna(0.0)
data["min_FVC"] = data["min_FVC"].fillna(data["min_FVC"].median())

age_den = (data["Age"].max() - data["Age"].min()) or 1.0
minfvc_den = (data["min_FVC"].max() - data["min_FVC"].min()) or 1.0
week_den = (data["base_week"].max() - data["base_week"].min()) or 1.0
pct_den = (data["Percent"].max() - data["Percent"].min()) or 1.0

data["age"] = (data["Age"] - data["Age"].min()) / age_den
data["BASE"] = (data["min_FVC"] - data["min_FVC"].min()) / minfvc_den
data["week"] = (data["base_week"] - data["base_week"].min()) / week_den
data["percent"] = (data["Percent"] - data["Percent"].min()) / pct_den

FE += ["age", "percent", "week", "BASE"]

FE = list(dict.fromkeys(FE))

print("Feature columns (FE):", FE)
print("Number of features:", len(FE))

for col in FE:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data[FE] = data[FE].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

data["FVC"] = pd.to_numeric(data["FVC"], errors="coerce")

train = data.loc[data.WHERE == "train"].copy()
test = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data

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

    model.compile(
        loss=mloss(0.8),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            amsgrad=False,
        ),
        metrics=[score],
    )
    return model




## === cell 8
train = train.copy().reset_index(drop=True)
sub = sub.copy().reset_index(drop=True)

train["FVC"] = pd.to_numeric(train["FVC"], errors="coerce")
mask = train["FVC"].notna()
train = train.loc[mask].reset_index(drop=True)

for col in FE:
    if col not in train.columns:
        train[col] = 0.0
    if col not in sub.columns:
        sub[col] = 0.0

train[FE] = (
    train[FE].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
)
sub[FE] = (
    sub[FE].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
)

z = train[FE].to_numpy(dtype=np.float32, na_value=0.0)
ze = sub[FE].to_numpy(dtype=np.float32, na_value=0.0)
z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
ze = np.nan_to_num(ze, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

y_scalar = train["FVC"].to_numpy(dtype=np.float32, na_value=0.0).reshape(-1, 1)
y_scalar = np.nan_to_num(y_scalar, nan=0.0, posinf=0.0, neginf=0.0).astype(
    np.float32, copy=False
)
y = np.repeat(y_scalar, 3, axis=1).astype(np.float32, copy=False)

nh = z.shape[1]
pe = np.zeros((ze.shape[0], 3), dtype=np.float32)
pred = np.zeros((z.shape[0], 3), dtype=np.float32)

net = make_model(nh)
print(net.summary())
print("Params:", net.count_params())
print("Shapes:", "z", z.shape, "y", y.shape, "ze", ze.shape)



## === cell 9
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=6942069)



## === cell 10
cnt = 0
EPOCHS = 800
BATCH_SIZE = 128

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model(nh)

    x_tr = np.asarray(z[tr_idx], dtype=np.float32)
    y_tr = np.asarray(y[tr_idx], dtype=np.float32)
    x_va = np.asarray(z[val_idx], dtype=np.float32)
    y_va = np.asarray(y[val_idx], dtype=np.float32)

    x_tr = np.nan_to_num(x_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    y_tr = np.nan_to_num(y_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    x_va = np.nan_to_num(x_va, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    y_va = np.nan_to_num(y_va, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )

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
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2496421194.py in <cell line: 0>()
     26     )
     27 
---> 28     net.fit(
     29         x_tr,
     30         y_tr,

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
sigma_opt = mean_absolute_error(y_scalar.ravel(), pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = np.mean(unc)

sub["FVC1"] = 0.996 * pe[:, 1]
sub["Confidence1"] = np.maximum(pe[:, 2] - pe[:, 0], 70.0)

subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]
if sigma_mean < 70:
    subm["Confidence"] = max(float(sigma_opt), 70.0)
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    subm.loc[
        subm["Patient_Week"] == otest.Patient[i] + "_" + str(otest.Weeks[i]), "FVC"
    ] = otest.FVC[i]
    subm.loc[
        subm["Patient_Week"] == otest.Patient[i] + "_" + str(otest.Weeks[i]),
        "Confidence",
    ] = 0.1

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
