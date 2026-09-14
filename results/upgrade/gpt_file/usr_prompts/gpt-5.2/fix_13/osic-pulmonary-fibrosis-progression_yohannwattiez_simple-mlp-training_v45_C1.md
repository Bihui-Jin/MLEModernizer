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

-7.122647613219701

# 6. Current score

-8.5747

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/Keras import crash by forcing the pure-Python protobuf implementation early, which resolves the `MessageFactory.GetPrototype` error in this environment. Then I correct all broken Kaggle input paths (the script references non-existent datasets) to use the provided `/kaggle/input/osic-pulmonary-fibrosis-progression/*.csv`. Next I remove dependencies on missing pickles (`data_prep`, `list_patient_score`) by fitting `data_preparation()` on-the-fly from the available training data and using uniform sample weights, preserving the same feature columns and model architecture. Finally I fix KFold indexing and the optimizer argument error (`lr`→`learning_rate`) so training/inference run end-to-end and a valid `submission.csv` is written.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation env vars before any TensorFlow import and by disabling C++ protobuf in a way that works with this Kaggle runtime. Then I fix the Keras `compile()` error by providing `metrics` as a list (Keras 3 requirement), which unblocks model creation and training. Finally, I keep the same model/training logic but ensure the script completes inference and always writes a valid `submission.csv` with the exact required columns and types; this should also improve your score versus the current “mostly-constant fallback” behavior since training actually run.'
- What this solution (achieved -8.76189) has done: 'I fix the early TensorFlow/protobuf crash by setting the additional protobuf env var before importing TensorFlow, which is the root cause of the `MessageFactory.GetPrototype` error in this runtime. Then I correct the test merge logic that accidentally drops/renames the wrong `Weeks` column, which currently causes the `KeyError: 'Weeks'` cascade and prevents `X_prediction_proc` from being created. Finally, I ensure all model inputs are numeric and NaN-free (including `Sex`/`SmokingStatus` and derived baseline fields) so `model.fit()` doesn’t error with “None values not supported”, and I keep the same training/inference flow so a valid `submission.csv` is always written.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before any TensorFlow import, which resolves the `MessageFactory.GetPrototype` error in this Kaggle runtime. Then I fix the `None values not supported` training crash by (1) ensuring the target `y` has the expected shape `(n, 3)` for the quantile + sigma model and (2) removing any remaining `None`/object dtypes in both features and labels by explicit numeric coercion and NaN handling. These changes preserve your model architecture and training loop but allow training/inference to run end-to-end, so predictions replace the current constant fallback and should move the score upward toward the target band. Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf crash by switching to a TensorFlow import strategy that works in this Kaggle runtime (stop forcing pure-Python protobuf, which triggers the `MessageFactory.GetPrototype` failure). Then I fix the `None values not supported` training error by ensuring `sample_weight` is passed as a plain numeric NumPy array with no NaNs/None and by using position-based indexing (`.iloc`) for `y_all`/`sample_weight` to avoid index-misalignment creating invalid tensors. These are execution/stability fixes that preserve your model architecture and training loop, but they also prevent the current fallback-to-constant submission behavior, which should improve the score toward the target. Finally, I keep the submission-writing logic intact and ensure a valid `submission.csv` is always produced.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf env vars before importing TensorFlow, which is the root cause of the `MessageFactory.GetPrototype` error in this runtime. Then I fix the `None values not supported` training crash by ensuring the model always receives fully numeric, NaN/None-free `x`, `y`, and `sample_weight` arrays with aligned positional indexing (using `.iloc`-based slices for `y_all`/weights). These are execution/correctness fixes that preserve your model architecture, loss, and training loop, but they also ensure training actually runs so predictions replace the current constant fallback, which should improve the score toward the target band. Finally, I keep the submission formatting intact and guarantee `submission.csv` is written.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf environment variables that trigger the `MessageFactory.GetPrototype` error in this Kaggle runtime. Then I fix the `None values not supported` training failure by ensuring the KFold indices are strictly positional (so `y_all[...]` and `sample_weight` align) and by guaranteeing the arrays passed into `fit()` contain only finite float32 values with no `None`. These are execution/correctness fixes that preserve your model, loss, and training loop semantics while allowing training to complete, so predictions replace the constant fallback and should improve the score toward the target band. Finally, I keep the same submission formatting logic and ensure `submission.csv` is always written with the required columns.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the reliable workaround for the `MessageFactory.GetPrototype` error in this environment. Then I fix the `None values not supported` training crash by (a) ensuring the training labels passed to `fit()` have the exact `(n, 3)` float32 shape the loss expects and (b) passing `sample_weight` as a plain 1D float32 NumPy array (not a pandas object) with guaranteed finite values. These changes preserve your model, loss, and training loop semantics, but they unblock training so the model predictions replace the constant fallback, which should improve the score toward the target. Finally, I keep the same submission formatting and ensure a valid `submission.csv` is always written.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf environment variables that trigger the `MessageFactory.GetPrototype` error in this Kaggle runtime. Then I fix the `None values not supported` training failure by ensuring `x`, `y`, and `sample_weight` passed to `model.fit()` are strictly finite `np.float32` NumPy arrays, and by making the validation sample weights explicit as well (to avoid any Keras 3 edge-cases with missing weights). These changes preserve your exact model architecture, loss, and training loop semantics, but unblock training so the model produces non-constant predictions and should improve the score toward the target band. Finally, I keep the same submission formatting and guarantee `submission.csv` is written with the required columns.'
- What this solution (achieved -8.5747) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf environment variables *before* importing TensorFlow, which is the root cause of your cell 0 failure. Then I fix the `None values not supported` training crash by ensuring Keras receives `validation_data` in the correct format (Keras 3 expects `(x_val, y_val)` and `validation_sample_weight` passed separately), while keeping your model, loss, and training loop semantics unchanged. Finally, I make the prediction post-processing consistent with your training-time FVC normalization by de-normalizing `pe/pred` back to ml before writing the submission; this is a minimal calibration fix that should improve the score toward your target without changing the core model.'
- What this solution (achieved -8.5747) has done: 'I fix the TensorFlow/protobuf import crash by removing the environment forcing of the pure-Python protobuf backend, which is what triggers the `MessageFactory.GetPrototype` failure in this Kaggle runtime. Then I fix the Keras 3 `fit()` runtime error by removing the unsupported `validation_sample_weight` argument (keeping your training loop intact by simply validating without weights). Finally, I keep your existing de-normalization and submission formatting, ensuring the script trains, predicts, and always writes a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)




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
BASE_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression"

train = pd.read_csv(f"{BASE_PATH}/train.csv")
raw_test = pd.read_csv(f"{BASE_PATH}/test.csv")
X_prediction = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

print(train.shape, raw_test.shape, X_prediction.shape)



## === cell 3
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.8
EPOCH = 250
BATCH_SIZE = 128



## === cell 4
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
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




## === cell 5
def eval_score(y_true, y_pred):
    y_true = (
        tf.dtypes.cast(y_true, tf.float32) * (data_prep.fvc_max - data_prep.fvc_min)
        + data_prep.fvc_min
    )
    y_pred = (
        tf.dtypes.cast(y_pred, tf.float32) * (data_prep.fvc_max - data_prep.fvc_min)
        + data_prep.fvc_min
    )
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return -K.backend.mean(metric)




## === cell 6
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



## === cell 7
X_prediction = X_prediction.copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

base = raw_test[
    ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
].copy()
base = base.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})

X_prediction = X_prediction.merge(base, how="left", on="Patient")

X_prediction = X_prediction[
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
].reset_index(drop=True)

X_prediction["Min_week"] = X_prediction["Min_week"].fillna(0).astype(int)
X_prediction["Base_FVC"] = X_prediction["Base_FVC"].fillna(
    X_prediction["Base_FVC"].median()
)
X_prediction["Percent"] = X_prediction["Percent"].fillna(
    X_prediction["Percent"].median()
)
X_prediction["Age"] = X_prediction["Age"].fillna(X_prediction["Age"].median())
X_prediction["Sex"] = X_prediction["Sex"].fillna("Male")
X_prediction["SmokingStatus"] = X_prediction["SmokingStatus"].fillna("Never smoked")

X_prediction.head()



## === cell 8
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]
X_prediction[["Patient", "Weeks", "Min_week", "Base_week"]].head()



## === cell 9
train = train.copy()
train["Min_week"] = train.groupby("Patient")["Weeks"].transform("min")
base_map = (
    train.loc[train["Weeks"] == train["Min_week"], ["Patient", "FVC"]]
    .drop_duplicates("Patient")
    .set_index("Patient")["FVC"]
)
train["Base_FVC"] = train["Patient"].map(base_map)
train["Base_week"] = train["Weeks"] - train["Min_week"]

train["Base_FVC"] = train["Base_FVC"].fillna(
    train.groupby("Patient")["FVC"].transform("median")
)
train["Percent"] = train["Percent"].fillna(train["Percent"].median())
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Sex"] = train["Sex"].fillna("Male")
train["SmokingStatus"] = train["SmokingStatus"].fillna("Never smoked")
train.isna().sum()



## === cell 10
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
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


from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


def standardisation(x, u, s):
    s = s if (s is not None and float(s) != 0.0) else 1.0
    return (x - u) / s


def normalization(x, ma, mi):
    denom = ma - mi
    denom = (
        denom if (ma is not None and mi is not None and float(denom) != 0.0) else 1.0
    )
    return (x - mi) / denom


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder(sparse=True, handle_unknown="ignore")
        self.standardisation = bool_standard
        self.normalization = bool_normalization

        self.fvc_min = None
        self.fvc_max = None

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

            self.fvc_min = (
                float(data_untransformed["FVC"].min())
                if "FVC" in data_untransformed.columns
                else 0.0
            )
            self.fvc_max = (
                float(data_untransformed["FVC"].max())
                if "FVC" in data_untransformed.columns
                else 1.0
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




## === cell 11
data_prep = data_preparation(bool_normalization=True, bool_standard=False)

_ = data_prep(
    train[
        [
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Percent",
            "Age",
            "Min_week",
            "Base_FVC",
            "Base_week",
            "FVC",
        ]
    ].copy()
)
print("data_prep fitted. fvc_min/max:", data_prep.fvc_min, data_prep.fvc_max)



## === cell 12
train_proc = data_prep(
    train[
        [
            "Patient",
            "Weeks",
            "Percent",
            "Age",
            "Sex",
            "Min_week",
            "Base_FVC",
            "Base_week",
            "SmokingStatus",
            "FVC",
        ]
    ].copy()
)
X_prediction_proc = data_prep(
    X_prediction[
        [
            "Patient",
            "Weeks",
            "Percent",
            "Age",
            "Sex",
            "Min_week",
            "Base_FVC",
            "Base_week",
            "SmokingStatus",
            "Patient_Week",
        ]
    ].copy()
)

for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if col not in train_proc.columns:
        train_proc[col] = 0
    if col not in X_prediction_proc.columns:
        X_prediction_proc[col] = 0

train_proc["Patient"] = train["Patient"].values
train_proc["FVC"] = train["FVC"].values
X_prediction_proc["Patient_Week"] = X_prediction["Patient_Week"].values
X_prediction_proc["Patient"] = X_prediction["Patient"].values

train_proc[SELECTED_COLUMNS] = (
    train_proc[SELECTED_COLUMNS]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .astype(np.float32)
)
X_prediction_proc[SELECTED_COLUMNS] = (
    X_prediction_proc[SELECTED_COLUMNS]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .astype(np.float32)
)

train_proc["FVC"] = pd.to_numeric(train_proc["FVC"], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
train_proc["FVC"] = (
    train_proc["FVC"].fillna(train_proc["FVC"].median()).astype(np.float32)
)

train_proc[SELECTED_COLUMNS].head()



## === cell 13
list_patient_score = None
train_proc["Weight"] = 1.0



## === cell 14
list_patient_weight = [[p, 1.0] for p in train_proc["Patient"].unique()]



## === cell 15
train_proc["Weight"] = (
    train_proc["Patient"].map(dict(list_patient_weight)).astype(float)
)
train_proc["Weight"].fillna(1.0, inplace=True)

train_proc["Weight"] = (
    pd.to_numeric(train_proc["Weight"], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(1.0)
    .astype(np.float32)
)



## === cell 16
from sklearn.cluster import KMeans

KMEANS_COLUMNS = SELECTED_COLUMNS
number_cluster = 2

kmeans = KMeans(n_clusters=number_cluster, random_state=0, n_init=10).fit(
    train_proc[KMEANS_COLUMNS]
)
train_proc["Patient_Class"] = kmeans.predict(train_proc[KMEANS_COLUMNS])
train_proc["Patient_Class"] = train_proc["Patient"].map(
    train_proc.groupby(["Patient"])["Patient_Class"].agg(
        lambda x: x.value_counts().index[0]
    )
)

X_prediction_proc["Patient_Class"] = kmeans.predict(X_prediction_proc[KMEANS_COLUMNS])
X_prediction_proc["Patient_Class"] = X_prediction_proc["Patient"].map(
    X_prediction_proc.groupby(["Patient"])["Patient_Class"].agg(
        lambda x: x.value_counts().index[0]
    )
)



## === cell 17
train_proc.groupby(["Patient_Class", "Patient"]).size().reset_index(name="n")[
    "Patient_Class"
].value_counts()



## === cell 18
NFOLD = 3
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)

pe = np.zeros((X_prediction_proc.shape[0], 3), dtype=np.float32)
pred = np.zeros((train_proc.shape[0], 3), dtype=np.float32)



## === cell 19
val_idx = train_proc.index[:5]
train_proc.loc[val_idx, SELECTED_COLUMNS]



## === cell 20
cnt = 0
EPOCHS = 50

y_base_ml = train_proc["FVC"].to_numpy(dtype=np.float32).reshape(-1, 1)
den = (
    float(data_prep.fvc_max - data_prep.fvc_min)
    if float(data_prep.fvc_max - data_prep.fvc_min) != 0.0
    else 1.0
)
y_base = (y_base_ml - float(data_prep.fvc_min)) / den
y_all = np.repeat(y_base, 3, axis=1).astype(np.float32)

y_all = np.nan_to_num(
    y_all,
    nan=np.nanmedian(y_all),
    posinf=np.nanmedian(y_all),
    neginf=np.nanmedian(y_all),
).astype(np.float32)
y_all = np.ascontiguousarray(y_all, dtype=np.float32)

for i in range(number_cluster):
    subset_pos = np.where(train_proc["Patient_Class"].to_numpy() == i)[0]
    if len(subset_pos) < NFOLD:
        continue

    for tr_subpos, val_subpos in kf.split(subset_pos):
        tr_pos = subset_pos[tr_subpos]
        val_pos = subset_pos[val_subpos]

        cnt += 1
        print(f"FOLD {cnt} (cluster {i})")

        net = create_model(LAMBDA_LOSS)

        x_tr = train_proc.iloc[tr_pos][SELECTED_COLUMNS].to_numpy(dtype=np.float32)
        x_val = train_proc.iloc[val_pos][SELECTED_COLUMNS].to_numpy(dtype=np.float32)

        x_tr = np.nan_to_num(x_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        x_val = np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        x_tr = np.ascontiguousarray(x_tr, dtype=np.float32)
        x_val = np.ascontiguousarray(x_val, dtype=np.float32)

        sw_tr = train_proc.iloc[tr_pos]["Weight"].to_numpy(dtype=np.float32).reshape(-1)
        sw_tr = np.nan_to_num(sw_tr, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
        sw_tr = np.ascontiguousarray(sw_tr, dtype=np.float32)

        sw_val = (
            train_proc.iloc[val_pos]["Weight"].to_numpy(dtype=np.float32).reshape(-1)
        )
        sw_val = np.nan_to_num(sw_val, nan=1.0, posinf=1.0, neginf=1.0).astype(
            np.float32
        )
        sw_val = np.ascontiguousarray(sw_val, dtype=np.float32)

        assert x_tr.dtype == np.float32 and x_val.dtype == np.float32
        assert (
            y_all.dtype == np.float32
            and sw_tr.dtype == np.float32
            and sw_val.dtype == np.float32
        )
        assert np.isfinite(x_tr).all() and np.isfinite(x_val).all()
        assert np.isfinite(y_all[tr_pos]).all() and np.isfinite(y_all[val_pos]).all()
        assert np.isfinite(sw_tr).all() and np.isfinite(sw_val).all()
        assert sw_tr.ndim == 1 and sw_tr.shape[0] == x_tr.shape[0]
        assert sw_val.ndim == 1 and sw_val.shape[0] == x_val.shape[0]
        assert y_all[tr_pos].shape == (x_tr.shape[0], 3)

        net.fit(
            x_tr,
            y_all[tr_pos],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_all[val_pos]),
            sample_weight=sw_tr,
            verbose=0,
        )

        print(
            "train",
            net.evaluate(
                x_tr,
                y_all[tr_pos],
                verbose=0,
                batch_size=BATCH_SIZE,
                sample_weight=sw_tr,
            ),
        )
        print(
            "val  ",
            net.evaluate(
                x_val,
                y_all[val_pos],
                verbose=0,
                batch_size=BATCH_SIZE,
            ),
        )

        pred[val_pos] = net.predict(
            x_val,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        test_mask = X_prediction_proc["Patient_Class"].to_numpy() == i
        test_pos = np.where(test_mask)[0]

        x_test = X_prediction_proc.iloc[test_pos][SELECTED_COLUMNS].to_numpy(
            dtype=np.float32
        )
        x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        x_test = np.ascontiguousarray(x_test, dtype=np.float32)

        pe[test_pos] += (
            net.predict(
                x_test,
                batch_size=BATCH_SIZE,
                verbose=0,
            )
            / NFOLD
        )

        net.save(f"model_{cnt}.keras")



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2745215116.py in <cell line: 0>()
     68         # Fix: keras in this runtime (TFTrainer) does not accept validation_sample_weight.
     69         # Minimal change: keep sample_weight for training, validate unweighted.
---> 70         net.fit(
     71             x_tr,
     72             y_all[tr_pos],

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

## === cell 21
pred_ml = pred * den + float(data_prep.fvc_min)
pe_ml = pe * den + float(data_prep.fvc_min)

sigma_opt = mean_absolute_error(train_proc[["FVC"]].values, pred_ml[:, 1])
unc = pred_ml[:, 2] - pred_ml[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt:", sigma_opt, "sigma_mean:", sigma_mean)



## === cell 22
X_prediction_proc["FVC1"] = 0.996 * pe_ml[:, 1]
X_prediction_proc["Confidence1"] = pe_ml[:, 2] - pe_ml[:, 0]



## === cell 23
subm = X_prediction_proc[["Patient_Week", "FVC1", "Confidence1"]].copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0



## === cell 24
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

subm["Confidence"] = subm["Confidence"].abs().clip(lower=70.0)



## === cell 25
otest = pd.read_csv(f"{BASE_PATH}/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = (
        0.1  # clipped to 70 by metric anyway
    )



## === cell 26
out = subm[["Patient_Week", "FVC", "Confidence"]].copy()
out["FVC"] = out["FVC"].astype(float)
out["Confidence"] = out["Confidence"].astype(float)

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
