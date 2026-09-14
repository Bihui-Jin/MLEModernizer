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

-6.989099049036112

# 6. Current score

-8.76003

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76003) has done: 'I fix the immediate runtime/import crash by forcing TensorFlow to use the Python protobuf implementation (avoids the `MessageFactory.GetPrototype` error seen in some Kaggle TF/protobuf combos). Then I remove dependencies on missing Kaggle “input” datasets (`data-preparation-for-osic`, `simple-mlp-training`) by fitting the existing `data_preparation` on the provided train.csv and by creating a deterministic patient-level KFold split directly from the training data. Finally, I correct file paths to the OSIC dataset you actually have, update the Adam optimizer argument names to match the installed Keras, and ensure a valid `submission.csv` with columns `Patient_Week,FVC,Confidence` is always written.'
- What this solution (achieved -8.76003) has done: 'I fix the TensorFlow/protobuf crash by forcing protobuf 3 compatibility before importing TensorFlow, since your current environment is using newer protobuf where the old `MessageFactory.GetPrototype` call breaks. Then I fix the Keras compile error by passing `metrics` as a list (Keras 3 requires list/tuple/dict), which unblocks model creation and therefore training/inference and submission generation. I also make the custom `OneHotEncoder` compatible with both older and newer scikit-learn (`sparse` vs `sparse_output`) to avoid runtime errors depending on the exact sklearn version. These changes are execution/stability fixes and preserve the model/training logic; once unblocked, the script run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -8.76003) has done: 'I fix the TensorFlow/protobuf import crash by switching to the safe environment flag (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) and avoiding the incompatible extra protobuf env vars that still trigger `MessageFactory.GetPrototype` failures in some Kaggle images. Then I fix the `None values not supported` training error by ensuring all model input columns are numeric and contain no missing values (most commonly from the `Min_week/Base_FVC/Base_week` merge in `X_prediction`), using train-derived fallbacks per-patient and finally global medians (score-neutral vs. crashing). Finally, I keep the existing model/training logic intact and ensure the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76003) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf env vars *before* any TensorFlow-related imports and importing TensorFlow only after that, which resolves the `MessageFactory.GetPrototype` error in Kaggle images. Then I fix the `None values not supported` training failure by ensuring every model input column is strictly numeric and contains no NaN/None both in train and prediction frames (the most common source is the `Min_week/Base_FVC` merge for test and the one-hot columns). Finally, I keep the model/training logic intact and ensure the script always writes a valid `submission.csv` with the required columns; these fixes should unblock training and, compared to the current crash, move the score toward the target by allowing the intended model predictions to be used.'
- What this solution (achieved -8.76003) has done: 'I fix the TensorFlow/protobuf crash by removing the incompatible protobuf env var and instead importing TensorFlow after setting only the safe fallback flag; this directly addresses the `MessageFactory.GetPrototype` error. Next, I fix the `None values not supported` crash in `model.fit` by ensuring `sample_weight` is passed as a pure `float32` NumPy array with no missing values, and by hardening all training feature/label arrays against NaN/None right before fitting. These changes preserve your model architecture and training loop while making the notebook run end-to-end reliably. With training actually completing (instead of crashing), the resulting predictions should move the score upward from the current -8.76003 toward the target band.'
- What this solution (achieved -8.76003) has done: 'I first fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend *and* disabling the C++ implementation in a way that is compatible with newer protobuf builds. Next, I fix the `None values not supported` crash in `model.fit` by ensuring the loss/metric always receive tensors with consistent shapes (y_true as `(batch, 3)` to match the model output), which is the real source of the `None` conversion failure in Keras 3 when broadcasting breaks. Finally, I keep your model/training loop intact but make the submission confidence strictly positive and clipped to the competition’s effective minimum (70) to improve calibration and move the score upward toward the target without changing the architecture.'
- What this solution (achieved -8.76003) has done: 'I fix the TensorFlow/protobuf import crash by setting only the safe protobuf env flag (and not forcing an incompatible implementation version) before importing TensorFlow. Then I fix the `None values not supported` error during `model.fit` by ensuring the training/validation inputs, targets, and `sample_weight` are strictly finite `float32` NumPy arrays (no `None`/object dtype) right before fitting. Finally, I keep your model/loss logic unchanged and only harden the data pipeline and submission writing so it always completes end-to-end and produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import random
import time

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_all(20)



## === cell 2
CANDIDATE_BASES = [
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/input/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression/osic-pulmonary-fibrosis-progression",
]
BASE_PATH = None
for p in CANDIDATE_BASES:
    if os.path.exists(os.path.join(p, "train.csv")):
        BASE_PATH = p
        break
if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate OSIC dataset base path in expected locations."
    )

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(train_path), f"Missing: {train_path}"
assert os.path.exists(test_path), f"Missing: {test_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"

print("Using BASE_PATH:", BASE_PATH)



## === cell 3
train = pd.read_csv(train_path)
raw_test = pd.read_csv(test_path)
X_prediction = pd.read_csv(sample_sub_path)

print(train.shape, raw_test.shape, X_prediction.shape)
print("train cols:", train.columns.tolist())
print("test cols:", raw_test.columns.tolist())
print("sample sub cols:", X_prediction.columns.tolist())



## === cell 4
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.8
EPOCH = [54, 55, 20, 60, 23]
BATCH_SIZE = 128
NFOLD = 5



## === cell 5
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

rename_cols = {"Weeks_y": "Min_week", "Weeks_x": "Weeks", "FVC": "Base_FVC"}
X_prediction = (
    X_prediction.merge(raw_test, how="left", on="Patient", suffixes=("_x", "_y"))
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

X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]



## === cell 6
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.backend.mean(metric)


def qloss(y_true, y_pred):
    qs = PINBALL_QUANTILE
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.backend.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss




## === cell 7
def eval_score(y_true, y_pred):
    y_true = tf.dtypes.cast(y_true, tf.float32)
    y_pred = tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return -K.backend.mean(metric)




## === cell 8
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

    model = K.Model(inputs=model_input, outputs=[FVC])

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



## === cell 9
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        kwargs.setdefault("handle_unknown", "ignore")
        if "sparse_output" in SklearnOneHotEncoder.__init__.__code__.co_varnames:
            kwargs.setdefault("sparse_output", True)
        else:
            kwargs.setdefault("sparse", True)
        super().__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index="", name="", **kwargs):
        sparse_matrix = super().transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        return [f"{name}_{c}" for c in categories]


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




## === cell 10
train_feat = train.copy()
grp = train_feat.groupby("Patient")["Weeks"].min().rename("Min_week")
train_feat = train_feat.merge(grp, on="Patient", how="left")

base = train_feat.loc[
    train_feat["Weeks"] == train_feat["Min_week"],
    ["Patient", "FVC", "Weeks", "Percent"],
].copy()
base = base.rename(
    columns={"FVC": "Base_FVC", "Weeks": "Base_week_raw", "Percent": "Base_percent"}
)

train_feat = train_feat.merge(
    base[["Patient", "Base_FVC", "Base_week_raw", "Base_percent"]],
    on="Patient",
    how="left",
)
train_feat["Base_week"] = train_feat["Weeks"] - train_feat["Min_week"]

train_feat["Weight"] = 1.0

X_prediction = X_prediction.merge(
    raw_test[["Patient", "Percent"]].rename(columns={"Percent": "Base_percent"}),
    on="Patient",
    how="left",
)



## === cell 11
train_patient_minweek = train.groupby("Patient")["Weeks"].min()
global_min_week = float(train["Weeks"].min())
global_base_fvc = float(
    train.sort_values(["Patient", "Weeks"]).groupby("Patient")["FVC"].first().median()
)

if "Min_week" in X_prediction.columns:
    X_prediction["Min_week"] = pd.to_numeric(X_prediction["Min_week"], errors="coerce")
else:
    X_prediction["Min_week"] = np.nan

min_week_map = train_patient_minweek.to_dict()
X_prediction["Min_week"] = X_prediction["Min_week"].fillna(
    X_prediction["Patient"].map(min_week_map)
)
X_prediction["Min_week"] = X_prediction["Min_week"].fillna(global_min_week)

if "Base_FVC" in X_prediction.columns:
    X_prediction["Base_FVC"] = pd.to_numeric(X_prediction["Base_FVC"], errors="coerce")
else:
    X_prediction["Base_FVC"] = np.nan

train_baseline_fvc = (
    train.sort_values(["Patient", "Weeks"]).groupby("Patient")["FVC"].first().to_dict()
)
X_prediction["Base_FVC"] = X_prediction["Base_FVC"].fillna(
    X_prediction["Patient"].map(train_baseline_fvc)
)
X_prediction["Base_FVC"] = X_prediction["Base_FVC"].fillna(global_base_fvc)

for col in ["Percent", "Age", "Weeks", "Base_week", "Base_percent"]:
    if col in X_prediction.columns:
        X_prediction[col] = pd.to_numeric(X_prediction[col], errors="coerce")

X_prediction["Base_week"] = pd.to_numeric(
    X_prediction["Weeks"], errors="coerce"
) - pd.to_numeric(X_prediction["Min_week"], errors="coerce")

train_medians = {
    "Percent": float(train["Percent"].median()),
    "Age": float(train["Age"].median()),
    "Weeks": float(train["Weeks"].median()),
    "Base_week": float((train_feat["Base_week"]).median()),
    "Base_percent": float(train_feat["Base_percent"].median()),
}
for col, med in train_medians.items():
    if col in X_prediction.columns:
        X_prediction[col] = X_prediction[col].fillna(med)

for col in ["Sex", "SmokingStatus"]:
    if col in X_prediction.columns:
        if X_prediction[col].isna().any():
            mode = train[col].mode(dropna=True)
            fillv = (
                mode.iloc[0]
                if len(mode)
                else ("Male" if col == "Sex" else "Never smoked")
            )
            X_prediction[col] = X_prediction[col].fillna(fillv)

for col in ["Sex", "SmokingStatus"]:
    if train_feat[col].isna().any():
        mode = train_feat[col].mode(dropna=True)
        fillv = (
            mode.iloc[0] if len(mode) else ("Male" if col == "Sex" else "Never smoked")
        )
        train_feat[col] = train_feat[col].fillna(fillv)



## === cell 12
data_prep = data_preparation(bool_normalization=True, bool_standard=False)

train_prep_input = train_feat[
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
train_transformed = data_prep(train_prep_input)

X_prediction_transformed = data_prep(X_prediction)

for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if col not in train_transformed.columns:
        train_transformed[col] = 0
    if col not in X_prediction_transformed.columns:
        X_prediction_transformed[col] = 0

missing_train = [c for c in SELECTED_COLUMNS if c not in train_transformed.columns]
missing_test = [
    c for c in SELECTED_COLUMNS if c not in X_prediction_transformed.columns
]
assert not missing_train, f"Missing train columns after prep: {missing_train}"
assert not missing_test, f"Missing test columns after prep: {missing_test}"

train_transformed["Patient"] = train_feat["Patient"].values
train_transformed["FVC"] = train_feat["FVC"].values
train_transformed["Weight"] = train_feat["Weight"].values

for df_name, df in [
    ("train_transformed", train_transformed),
    ("X_prediction_transformed", X_prediction_transformed),
]:
    for c in SELECTED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[SELECTED_COLUMNS] = df[SELECTED_COLUMNS].replace([np.inf, -np.inf], np.nan)
    df[SELECTED_COLUMNS] = df[SELECTED_COLUMNS].fillna(0.0).astype(np.float32)

train_transformed["FVC"] = (
    pd.to_numeric(train_transformed["FVC"], errors="coerce")
    .fillna(float(train["FVC"].median()))
    .astype(np.float32)
)

train_transformed["Weight"] = (
    pd.to_numeric(train_transformed["Weight"], errors="coerce")
    .fillna(1.0)
    .astype(np.float32)
)



## === cell 13
patients = train_transformed["Patient"].unique()
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)

patient_to_fold = {}
for fold, (_, val_idx) in enumerate(kf.split(patients)):
    for p in patients[val_idx]:
        patient_to_fold[p] = fold

list_patient_KFOLD = [[p, patient_to_fold[p]] for p in patients]



## === cell 14
pe = np.zeros((X_prediction_transformed.shape[0], 3), dtype=np.float32)
pred = np.zeros((train_transformed.shape[0], 3), dtype=np.float32)

X_pred_all = X_prediction_transformed[SELECTED_COLUMNS].to_numpy(dtype=np.float32)
X_pred_all = np.nan_to_num(X_pred_all, nan=0.0, posinf=0.0, neginf=0.0).astype(
    np.float32, copy=False
)

fvc_median = float(np.nanmedian(train_transformed["FVC"].to_numpy()))
y_all_fvc = train_transformed[["FVC"]].to_numpy(dtype=np.float32)
y_all_fvc = np.nan_to_num(
    y_all_fvc, nan=fvc_median, posinf=fvc_median, neginf=fvc_median
).astype(np.float32, copy=False)

for i in range(NFOLD):
    print(f"FOLD {i}")
    model = create_model(LAMBDA_LOSS)

    list_patients = [j[0] for j in list_patient_KFOLD if j[1] == i]
    tr_mask = ~train_transformed.Patient.isin(list_patients)
    va_mask = ~tr_mask

    X_tr = train_transformed.loc[tr_mask, SELECTED_COLUMNS].to_numpy(dtype=np.float32)
    X_va = train_transformed.loc[va_mask, SELECTED_COLUMNS].to_numpy(dtype=np.float32)

    y_tr_1 = y_all_fvc[tr_mask.values]
    y_va_1 = y_all_fvc[va_mask.values]

    y_tr = np.repeat(y_tr_1, 3, axis=1).astype(np.float32, copy=False)
    y_va = np.repeat(y_va_1, 3, axis=1).astype(np.float32, copy=False)

    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )

    sw = train_transformed.loc[tr_mask, "Weight"].to_numpy()
    sw = pd.to_numeric(sw, errors="coerce")
    sw = np.asarray(sw, dtype=np.float32)
    sw = np.nan_to_num(sw, nan=1.0, posinf=1.0, neginf=1.0).astype(
        np.float32, copy=False
    )

    history = model.fit(
        x=X_tr,
        y=y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCH[i],
        sample_weight=sw,
        verbose=0,
        batch_size=BATCH_SIZE,
    )

    print("train", model.evaluate(X_tr, y_tr, verbose=0, batch_size=BATCH_SIZE))
    print("val", model.evaluate(X_va, y_va, verbose=0, batch_size=BATCH_SIZE))

    pred[train_transformed.loc[va_mask].index.values] = model.predict(
        X_va,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    pe += model.predict(X_pred_all, batch_size=BATCH_SIZE, verbose=0) / NFOLD

    model.save(f"model_{i}.keras")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/257452839.py in <cell line: 0>()
     47     )
     48 
---> 49     history = model.fit(
     50         x=X_tr,
     51         y=y_tr,

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

## === cell 15
sigma_opt = mean_absolute_error(
    np.repeat(train_transformed[["FVC"]].to_numpy(), 3, axis=1)[:, 1], pred[:, 1]
)
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt:", sigma_opt, "sigma_mean:", sigma_mean)

X_prediction_transformed["FVC1"] = 0.996 * pe[:, 1]
X_prediction_transformed["Confidence1"] = pe[:, 2] - pe[:, 0]

X_prediction_transformed["Confidence1"] = (
    pd.to_numeric(X_prediction_transformed["Confidence1"], errors="coerce")
    .fillna(float(sigma_opt))
    .astype(np.float32)
)
X_prediction_transformed["Confidence1"] = np.clip(
    X_prediction_transformed["Confidence1"], 70.0, 1e9
)

subm = X_prediction_transformed.copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = float(max(sigma_opt, 70.0))
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

otest = raw_test.copy()
for i in range(len(otest)):
    key = otest.Patient.iloc[i] + "_" + str(int(otest.Weeks.iloc[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC.iloc[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1

submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
