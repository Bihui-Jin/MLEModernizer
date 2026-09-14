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

No external packages required in the script and installed.

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

-6.982605741871804

# 6. Current score

-8.76189

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76189) has done: 'I fix the immediate runtime/import failure by forcing TensorFlow to use the pure-Python protobuf implementation (this avoids the `MessageFactory.GetPrototype` crash in some Kaggle images). Then I correct all broken Kaggle input paths and remove dependencies on missing external pickle files by fitting `data_preparation` and building the patient-fold assignment directly from the provided `train.csv`. Finally, I update the Adam optimizer arguments to the current Keras API (use `learning_rate` instead of `lr`) and ensure the pipeline always writes a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -8.76189) has done: 'I fix the two blockers that prevent this notebook from running: the protobuf/TensorFlow import crash and the Keras `compile(metrics=...)` API error. The protobuf fix is made more robust by forcing the pure-Python protobuf implementation and disabling the C++ fast path before importing TensorFlow. The Keras error is fixed by passing `metrics=[score]` (list) instead of a bare function, which is required in newer Keras. These changes are execution/stability-only and should keep the model/training logic the same while allowing you to generate a valid `submission.csv` end-to-end (and likely improve from “no run” to your previous scored pipeline behavior).'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by switching to a robust fallback: attempt to import TensorFlow normally, and if it fails with the known protobuf `MessageFactory.GetPrototype` error, retry after forcing the pure-Python protobuf implementation (this keeps behavior stable while unblocking execution). Then I fix the `model.fit(... ValueError: None values not supported)` by ensuring all feature columns used by the model are numeric and have no missing values after merges/encoding (specifically: fill missing clinical fields and ensure one-hot columns exist for all categories). Finally, I keep the modeling/training logic intact but make the submission generation deterministic and valid by clipping confidence to Kaggle’s required minimum (70) and ensuring output columns exactly match the sample submission format.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import and removing the brittle “retry after failure” path that can’t recover once protobuf is already loaded. Then I fix the `None values not supported` training crash by guaranteeing all model inputs and sample weights are finite `float32` numpy arrays (no pandas nullable dtypes / object / None), and by making the training labels shape consistent with the model’s 3-quantile output (tile FVC to shape `(n,3)`). These changes keep your model architecture and loss/metric logic intact while unblocking training/inference end-to-end. Finally, I keep the submission logic the same but ensure confidence/FVC columns are always numeric and the CSV is written correctly.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf version/implementation before importing TensorFlow, with a safe fallback so the notebook imports reliably in Kaggle. Then I fix the `None values not supported` training error by ensuring `sample_weight` is a pure float32 NumPy array with no missing/None values (and by aligning the boolean masks to the DataFrame index so we never accidentally create misaligned/empty selections). These are stability fixes that keep your model, loss, and training loop intact while allowing end-to-end execution. Finally, I keep your submission logic but ensure the file is always written as a valid `submission.csv` with correct columns and numeric types.'
- What this solution (achieved -8.76189) has done: 'I fix the `None values not supported` crash by ensuring the training targets are pulled from the same preprocessed dataframe/mask as the inputs (the current code incorrectly indexes `train` with a boolean mask built on `train_prep`, which can misalign and introduce `None/NaN` into `y`). I also harden the fold loop by explicitly coercing `x/y/sample_weight` to finite `float32` arrays and adding a safe fallback if a fold ends up empty. These are execution/correctness fixes that preserve your model architecture and training loop, but should also improve score because the model finally train on the correct aligned labels instead of corrupted/misaligned targets. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76189) has done: 'I fix the `None values not supported` crash by ensuring every array passed into `model.fit()` (x, y, validation data, and sample_weight) is a plain finite `float32` NumPy array with no `None`/NaN/inf, and by guaranteeing the required `FVC` target exists and is numeric after preprocessing. This is a correctness/stability fix (no architecture/training loop changes) and should also nudge score upward because the model actually train on valid labels instead of failing or silently carrying invalid values. I also make `pred` assignment robust by writing fold predictions only for the validation indices, and keep the submission logic/format identical while ensuring Confidence is always finite and clipped to Kaggle’s minimum. All paths and core modeling logic remain unchanged.'
- What this solution (achieved -8.76189) has done: 'I fix the `None values not supported` crash by ensuring we always feed Keras plain NumPy `float32` arrays for `sample_weight` (some Keras versions can choke on pandas objects even if they look numeric). I also harden the fold training inputs by forcing contiguous arrays and sanitizing any non-finite values right before `model.fit`, without changing the model, loss, or training loop. This should unblock end-to-end execution and, since the model actually train in all folds, it should improve the score toward your target. The submission-writing logic and required column names/paths remain unchanged, and the script always write `submission.csv`.'
- What this solution (achieved -8.76189) has done: 'I fix the `None values not supported` crash during `model.fit()` by ensuring every tensor-like input to Keras (x, y, validation data, and sample_weight) is a plain finite `float32` NumPy array, and by sanitizing the *post-model* predictions to remove any non-finite values before using them (non-finite predictions can propagate into later folds/metrics and trigger conversion errors in some Keras builds). I also harden the custom metric/loss functions against dtype/shape edge cases by explicitly casting inputs to `float32` and ensuring `y_true` has the expected 2D shape. These changes preserve your model architecture and training loop while making execution reliable end-to-end. Finally, I keep your submission logic intact but make sure the written `submission.csv` is always valid and numeric.'
- What this solution (achieved -8.76189) has done: 'I fix the `None values not supported` crash by ensuring the custom loss/metric never produces `None`/NaN/inf tensors (this is typically triggered when `sigma` becomes non-finite or zero and then hits `log()`), while keeping the same model and training loop. Concretely, I make `score()` numerically safe by forcing `sigma` to be finite and strictly positive via `tf.where` and a small epsilon, without changing the overall metric semantics. I also make the training targets explicitly `float32` and add a final safety cast for `sample_weight` right before `fit()` to avoid any backend dtype edge cases. These changes are execution/correctness-focused and should also nudge the score upward (training no longer breaks/poisons gradients due to invalid logs/divisions).'
- What this solution (achieved -8.76189) has done: 'I fix the `None values not supported` crash by ensuring the model’s output is a single tensor (not a 1-item list), because Keras can propagate `None` when losses/metrics receive unexpected nested structures. Then I make the custom `score/qloss` functions robust to both tensor and list/tuple predictions by unwrapping `y_pred` if needed, without changing the math. Finally, I keep the training loop and submission logic identical, only making the minimum dtype/shape adjustments needed so training runs end-to-end and produces `submission.csv` reliably, which should also improve score by actually training across folds.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION"] = "1"

try:
    import pkgutil, subprocess, sys

    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version as _pkg_version

        pb_ver = _pkg_version("protobuf")
        major = int(pb_ver.split(".")[0])
        if major >= 4:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<4"]
            )
    except Exception:
        pass
except Exception:
    pass

import random
import time
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L

from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)



## === cell 1
pass




## === cell 2
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_all(20)



## === cell 3
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
    "/kaggle/input/data/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression",
]

DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find osic-pulmonary-fibrosis-progression dataset folder in expected locations."
    )

print("Using DATA_ROOT:", DATA_ROOT)



## === cell 4
train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
raw_test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
X_prediction = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

print(train.shape, raw_test.shape, X_prediction.shape)



## === cell 5
ID = "Patient_Week"
PINBALL_QUANTILE = [0.15, 0.50, 0.85]
LAMBDA_LOSS = 0.585
EPOCH = [54, 55, 20, 60, 23]
BATCH_SIZE = 128

NFOLD = 5



## === cell 6
pass



## === cell 7
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")
EPS = tf.constant(1e-6, dtype=tf.float32)


def _ensure_2d_f32(y):
    y = tf.cast(y, tf.float32)
    if y.shape.rank == 1:
        y = tf.reshape(y, (-1, 1))
    return y


def _unwrap_pred(y_pred):
    if isinstance(y_pred, (list, tuple)):
        if len(y_pred) != 1:
            raise ValueError(f"Unexpected y_pred list length: {len(y_pred)}")
        y_pred = y_pred[0]
    return y_pred


def score(y_true, y_pred):
    y_true = _ensure_2d_f32(y_true)
    y_pred = _unwrap_pred(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma = tf.where(tf.math.is_finite(sigma), sigma, C1)
    sigma_clip = tf.maximum(sigma, C1)
    sigma_clip = tf.maximum(sigma_clip, EPS)

    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    metric = tf.where(tf.math.is_finite(metric), metric, tf.zeros_like(metric))
    return K.backend.mean(metric)


def qloss(y_true, y_pred):
    y_true = _ensure_2d_f32(y_true)
    y_pred = _unwrap_pred(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)

    qs = PINBALL_QUANTILE
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    v = tf.where(tf.math.is_finite(v), v, tf.zeros_like(v))
    return K.backend.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss




## === cell 8
def eval_score(y_true, y_pred):
    y_pred = _unwrap_pred(y_pred)
    y_true = (
        tf.cast(y_true, tf.float32) * (data_prep.fvc_max - data_prep.fvc_min)
        + data_prep.fvc_min
    )
    y_pred = (
        tf.cast(y_pred, tf.float32) * (data_prep.fvc_max - data_prep.fvc_min)
        + data_prep.fvc_min
    )
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma = tf.where(tf.math.is_finite(sigma), sigma, C1)
    sigma_clip = tf.maximum(sigma, C1)
    sigma_clip = tf.maximum(sigma_clip, EPS)

    delta = tf.abs(y_true - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    metric = tf.where(tf.math.is_finite(metric), metric, tf.zeros_like(metric))
    return -K.backend.mean(metric)




## === cell 9
SELECTED_COLUMNS = [
    "Weeks",
    "Percent",
    "Age",
    "Sex",
    "Min_week",
    "Base_FVC",
    "Base_week",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]


def create_model(lambda_loss):
    model_input = K.Input(shape=(len(SELECTED_COLUMNS),))
    x = L.Dense(500, activation="selu", name="dense_to_freeze1")(model_input)
    x = L.Dense(100, activation="selu", name="dense_to_freeze2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="selu", name="p2")(x)
    FVC = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="FVC")([p1, p2])

    model = K.Model(inputs=model_input, outputs=FVC)

    model.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
        ),
        loss=mloss(lambda_loss),
        metrics=[score],
    )
    return model


model = create_model(LAMBDA_LOSS)
model.summary()



## === cell 10
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

rename_cols = {"Weeks_y": "Min_week", "Weeks_x": "Weeks", "FVC": "Base_FVC"}
X_prediction = (
    X_prediction.merge(raw_test, how="left", on="Patient")
    .rename(columns=rename_cols)[
        [
            "Patient",
            "Min_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ]
    ]
    .reset_index(drop=True)
)

for col, default in [
    ("Min_week", 0.0),
    (
        "Base_FVC",
        X_prediction["Base_FVC"].median() if "Base_FVC" in X_prediction else 0.0,
    ),
    ("Percent", X_prediction["Percent"].median() if "Percent" in X_prediction else 0.0),
    ("Age", X_prediction["Age"].median() if "Age" in X_prediction else 0.0),
    ("Sex", "Male"),
    ("SmokingStatus", "Never smoked"),
]:
    if col in X_prediction.columns:
        X_prediction[col] = X_prediction[col].fillna(default)



## === cell 11
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]



## === cell 12
train_sorted = train.sort_values(["Patient", "Weeks"]).reset_index(drop=True)
base = (
    train_sorted.groupby("Patient").first().reset_index()[["Patient", "Weeks", "FVC"]]
)
base = base.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
train = train.merge(base, on="Patient", how="left")
train["Base_week"] = train["Weeks"] - train["Min_week"]



## === cell 13
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        if "sparse" in kwargs:
            kwargs.pop("sparse")
        super(OneHotEncoder, self).__init__(
            sparse_output=True, handle_unknown="ignore", **kwargs
        )
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index="", name="", **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        new_columns = []
        for j in range(len(categories)):
            new_columns.append("{}_{}".format(name, categories[j]))
        return new_columns


def standardisation(x, u, s):
    return (x - u) / s


def normalization(x, ma, mi):
    return (x - mi) / (ma - mi)


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()
        self.standardisation = bool_standard
        self.normalization = bool_normalization

    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)

        if "Sex" in data.columns:
            data["Sex"] = data["Sex"].fillna("Male")
        if "SmokingStatus" in data.columns:
            data["SmokingStatus"] = data["SmokingStatus"].fillna("Never smoked")

        try:
            data["Sex"] = self.enc_sex.transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ).astype(int),
                ],
                axis=1,
            )

            if self.standardisation:
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )
                data["Base_percent"] = standardisation(
                    data["Base_percent"], self.base_percent_mean, self.base_percent_std
                )
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
                )

        except NotFittedError:
            data["Sex"] = self.enc_sex.fit_transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.fit_transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.fit_transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ).astype(int),
                ],
                axis=1,
            )

            self.fvc_min = (
                data["FVC"].min() if "FVC" in data.columns else train["FVC"].min()
            )
            self.fvc_max = (
                data["FVC"].max() if "FVC" in data.columns else train["FVC"].max()
            )

            if self.standardisation:
                self.base_week_mean = data["Base_week"].mean()
                self.base_week_std = data["Base_week"].std()
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )

                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = data["Base_FVC"].std()
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )

                self.base_percent_mean = data["Base_percent"].mean()
                self.base_percent_std = data["Base_percent"].std()
                data["Base_percent"] = standardisation(
                    data["Base_percent"], self.base_percent_mean, self.base_percent_std
                )

                self.age_mean = data["Age"].mean()
                self.age_std = data["Age"].std()
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)

                self.weeks_mean = data["Weeks"].mean()
                self.weeks_std = data["Weeks"].std()
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                self.base_week_min = data["Base_week"].min()
                self.base_week_max = data["Base_week"].max()
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )

                self.base_fvc_min = data["Base_FVC"].min()
                self.base_fvc_max = data["Base_FVC"].max()
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )

                self.base_percent_min = data["Percent"].min()
                self.base_percent_max = data["Percent"].max()
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )

                self.age_min = data["Age"].min()
                self.age_max = data["Age"].max()
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)

                self.weeks_min = data["Weeks"].min()
                self.weeks_max = data["Weeks"].max()
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )

                self.min_week_min = data["Min_week"].min()
                self.min_week_max = data["Min_week"].max()
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
                )

        return data




## === cell 14
data_prep = data_preparation(bool_normalization=True, bool_standard=False)

train_for_prep = train[
    [
        "Patient",
        "Weeks",
        "FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Min_week",
        "Base_FVC",
        "Base_week",
    ]
].copy()

for col, default in [
    ("Percent", train_for_prep["Percent"].median()),
    ("Age", train_for_prep["Age"].median()),
    ("Sex", "Male"),
    ("SmokingStatus", "Never smoked"),
    ("Min_week", train_for_prep["Min_week"].median()),
    ("Base_FVC", train_for_prep["Base_FVC"].median()),
    ("Base_week", train_for_prep["Base_week"].median()),
]:
    train_for_prep[col] = train_for_prep[col].fillna(default)

_ = data_prep(train_for_prep)



## === cell 15
train_prep = data_prep(train_for_prep)
train_prep["Patient"] = train["Patient"].values

X_prediction = data_prep(X_prediction)

for c in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if c not in train_prep.columns:
        train_prep[c] = 0
    if c not in X_prediction.columns:
        X_prediction[c] = 0

for df in [train_prep, X_prediction]:
    for c in SELECTED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[SELECTED_COLUMNS] = df[SELECTED_COLUMNS].replace([np.inf, -np.inf], np.nan)
    df[SELECTED_COLUMNS] = df[SELECTED_COLUMNS].fillna(df[SELECTED_COLUMNS].median())

train_prep["FVC"] = pd.to_numeric(train_prep["FVC"], errors="coerce")
train_prep["FVC"] = train_prep["FVC"].replace([np.inf, -np.inf], np.nan)
train_prep["FVC"] = train_prep["FVC"].fillna(train_prep["FVC"].median())



## === cell 16
missing = [c for c in SELECTED_COLUMNS if c not in train_prep.columns]
if missing:
    raise ValueError(f"Missing required columns after preprocessing: {missing}")

missing_test = [c for c in SELECTED_COLUMNS if c not in X_prediction.columns]
if missing_test:
    raise ValueError(
        f"Missing required columns in X_prediction after preprocessing: {missing_test}"
    )

if "FVC" not in train_prep.columns:
    raise ValueError("FVC missing from train_prep; cannot train.")
if train_prep["FVC"].isna().any():
    raise ValueError(
        "NaNs remain in train_prep['FVC'] after coercion/fill; cannot train."
    )



## === cell 17
patients = train_prep["Patient"].unique()
patients = np.array(sorted(patients))

list_patient_score = []
for p in patients:
    fvc_vals = train.loc[train.Patient == p, "FVC"].values
    if len(fvc_vals) < 2:
        proxy = 0.0
    else:
        proxy = float(np.std(fvc_vals))
    list_patient_score.append([p, [proxy]])

list_patient_KFOLD = []
for idx, p in enumerate(patients):
    list_patient_KFOLD.append([p, int(idx % NFOLD)])



## === cell 18
with open("list_patient_score", "wb") as f:
    pickle.dump(list_patient_score, f)



## === cell 19
pass



## === cell 20
pass



## === cell 21
train_prep["Weight"] = 1.0



## === cell 22
with open("list_patient_weight", "wb") as f:
    pickle.dump([[p, 1.0] for p in patients], f)



## === cell 23
list_patient_KFOLD = list_patient_KFOLD



## === cell 24
pass



## === cell 25
pass



## === cell 26
pass



## === cell 27
KFOLD_confidence = [0.05, 0.15, 0.2, 0.25, 0.35]




## === cell 28
def to_float32_matrix(df, cols):
    arr = df[cols].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(arr).all():
        col_medians = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan), axis=0)
        inds = ~np.isfinite(arr)
        arr[inds] = np.take(col_medians, np.where(inds)[1])
    return np.ascontiguousarray(arr, dtype=np.float32)


def to_float32_vector(s):
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(arr).all():
        med = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan))
        arr = np.nan_to_num(arr, nan=med, posinf=med, neginf=med).astype(np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def to_float32_weight(s, default=1.0):
    arr = pd.to_numeric(s, errors="coerce").to_numpy(copy=True)
    arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
    if np.isnan(arr).any():
        arr = np.nan_to_num(
            arr,
            nan=np.float32(default),
            posinf=np.float32(default),
            neginf=np.float32(default),
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


def sanitize_pred(a, fill_value=0.0):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Unexpected pred shape: {a.shape}")
    if not np.isfinite(a).all():
        a = np.nan_to_num(
            a,
            nan=np.float32(fill_value),
            posinf=np.float32(fill_value),
            neginf=np.float32(fill_value),
        ).astype(np.float32)
    return np.ascontiguousarray(a, dtype=np.float32)


train_prep = train_prep.reset_index(drop=True)
train = train.reset_index(drop=True)

pe = np.zeros((X_prediction.shape[0], 3), dtype=np.float32)
pred = np.zeros((train_prep.shape[0], 3), dtype=np.float32)

X_pred_mat = to_float32_matrix(X_prediction, SELECTED_COLUMNS)

for i in range(NFOLD):
    print(f"FOLD {i}")
    model = create_model(LAMBDA_LOSS)

    list_patients = [j[0] for j in list_patient_KFOLD if j[1] == i]

    val_mask = train_prep["Patient"].isin(list_patients).to_numpy(dtype=bool)
    trn_mask = ~val_mask

    if trn_mask.sum() == 0 or val_mask.sum() == 0:
        print(
            f"Warning: empty split in fold {i} (trn={trn_mask.sum()}, val={val_mask.sum()}); skipping fit."
        )
        fold_pred = model.predict(X_pred_mat, batch_size=BATCH_SIZE, verbose=0)
        fold_pred = sanitize_pred(fold_pred, fill_value=0.0)
        pe += fold_pred * KFOLD_confidence[i]
        continue

    x_trn = to_float32_matrix(train_prep.loc[trn_mask], SELECTED_COLUMNS)
    x_val = to_float32_matrix(train_prep.loc[val_mask], SELECTED_COLUMNS)

    y_trn_1d = to_float32_vector(train_prep.loc[trn_mask, "FVC"])
    y_val_1d = to_float32_vector(train_prep.loc[val_mask, "FVC"])

    y_trn = np.repeat(y_trn_1d.reshape(-1, 1), 3, axis=1).astype(np.float32)
    y_val = np.repeat(y_val_1d.reshape(-1, 1), 3, axis=1).astype(np.float32)
    y_trn = np.ascontiguousarray(y_trn, dtype=np.float32)
    y_val = np.ascontiguousarray(y_val, dtype=np.float32)

    sw = to_float32_weight(train_prep.loc[trn_mask, "Weight"], default=1.0)
    sw = np.asarray(sw, dtype=np.float32)

    if (
        (not np.isfinite(x_trn).all())
        or (not np.isfinite(y_trn).all())
        or (not np.isfinite(sw).all())
    ):
        raise ValueError("Non-finite values in training tensors after sanitation.")
    if (not np.isfinite(x_val).all()) or (not np.isfinite(y_val).all()):
        raise ValueError("Non-finite values in validation tensors after sanitation.")

    history = model.fit(
        x=x_trn,
        y=y_trn,
        validation_data=(x_val, y_val),
        epochs=EPOCH[i],
        sample_weight=sw,
        verbose=0,
        batch_size=BATCH_SIZE,
    )

    print("train", model.evaluate(x_trn, y_trn, verbose=0, batch_size=BATCH_SIZE))
    print("val", model.evaluate(x_val, y_val, verbose=0, batch_size=BATCH_SIZE))

    val_pred = model.predict(x_val, batch_size=BATCH_SIZE, verbose=0)
    val_pred = sanitize_pred(val_pred, fill_value=float(np.nanmedian(y_trn_1d)))
    pred[np.where(val_mask)[0]] = val_pred

    test_pred = model.predict(X_pred_mat, batch_size=BATCH_SIZE, verbose=0)
    test_pred = sanitize_pred(test_pred, fill_value=float(np.nanmedian(y_trn_1d)))
    pe += test_pred * KFOLD_confidence[i]

    model.save("model_" + str(i))



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3441116957.py in <cell line: 0>()
     92         raise ValueError("Non-finite values in validation tensors after sanitation.")
     93 
---> 94     history = model.fit(
     95         x=x_trn,
     96         y=y_trn,

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

## === cell 29
pass



## === cell 30
pass



## === cell 31
sigma_opt = mean_absolute_error(train_prep[["FVC"]], pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt (MAE):", sigma_opt, "sigma_mean (predicted):", sigma_mean)

X_pred_df = X_prediction.copy()
X_pred_df["FVC1"] = 0.996 * pe[:, 1]
X_pred_df["Confidence1"] = pe[:, 2] - pe[:, 0]



## === cell 32
subm = X_pred_df.copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

subm["FVC"] = pd.to_numeric(subm["FVC"], errors="coerce").astype(float)
subm["Confidence"] = pd.to_numeric(subm["Confidence"], errors="coerce").astype(float)

subm["FVC"] = subm["FVC"].replace([np.inf, -np.inf], np.nan).fillna(3020.0)
subm["Confidence"] = (
    subm["Confidence"].replace([np.inf, -np.inf], np.nan).fillna(100.0).clip(lower=70.0)
)

otest = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(int(otest.Weeks[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

subm_out = subm[["Patient_Week", "FVC", "Confidence"]].copy()
subm_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm_out.shape)
print(subm_out.head())
