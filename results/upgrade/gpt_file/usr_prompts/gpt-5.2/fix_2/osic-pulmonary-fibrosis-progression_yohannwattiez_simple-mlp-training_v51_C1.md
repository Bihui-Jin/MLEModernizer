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

-8.241654740851187

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import time
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

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
BASE1 = "/kaggle/input/osic-pulmonary-fibrosis-progression"
BASE2 = "/kaggle/input/data"  # fallback in this environment


def _resolve(path1, path2):
    if os.path.exists(path1):
        return path1
    if os.path.exists(path2):
        return path2
    raise FileNotFoundError(f"Neither exists: {path1} nor {path2}")


train_path = _resolve(f"{BASE1}/train.csv", f"{BASE2}/train.csv")
test_path = _resolve(f"{BASE1}/test.csv", f"{BASE2}/test.csv")
sample_path = _resolve(
    f"{BASE1}/sample_submission.csv", f"{BASE2}/sample_submission.csv"
)

train_raw = pd.read_csv(train_path)
raw_test = pd.read_csv(test_path)
X_prediction_raw = pd.read_csv(sample_path)

print(train_raw.shape, raw_test.shape, X_prediction_raw.shape)
print(train_raw.columns.tolist())



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
    sq2 = tf.sqrt(tf.cast(2.0, tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return keras.backend.mean(metric)


def qloss(y_true, y_pred):
    qs = PINBALL_QUANTILE
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return keras.backend.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss




## === cell 5
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


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
        return [f"{name}_{c}" for c in categories]


def standardisation(x, u, s):
    return (x - u) / s


def normalization(x, ma, mi):
    denom = ma - mi
    denom = denom if denom != 0 else 1.0
    return (x - mi) / denom


class data_preparation:
    """
    Fix: original notebook attempted to load a pre-fitted pickled object from a missing dataset.
    We fit this transformer directly using the available train/test CSVs.
    """

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
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
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

            self.fvc_min = float(data["FVC"].min()) if "FVC" in data.columns else None
            self.fvc_max = float(data["FVC"].max()) if "FVC" in data.columns else None

            if self.standardisation:
                self.base_week_mean = data["Base_week"].mean()
                self.base_week_std = (
                    data["Base_week"].std() if data["Base_week"].std() != 0 else 1.0
                )
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )

                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = (
                    data["Base_FVC"].std() if data["Base_FVC"].std() != 0 else 1.0
                )
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )

                self.base_percent_mean = data["Percent"].mean()
                self.base_percent_std = (
                    data["Percent"].std() if data["Percent"].std() != 0 else 1.0
                )
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
                )

                self.age_mean = data["Age"].mean()
                self.age_std = data["Age"].std() if data["Age"].std() != 0 else 1.0
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)

                self.weeks_mean = data["Weeks"].mean()
                self.weeks_std = (
                    data["Weeks"].std() if data["Weeks"].std() != 0 else 1.0
                )
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                self.base_week_min = float(data["Base_week"].min())
                self.base_week_max = float(data["Base_week"].max())
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )

                self.base_fvc_min = float(data["Base_FVC"].min())
                self.base_fvc_max = float(data["Base_FVC"].max())
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )

                self.base_percent_min = float(data["Percent"].min())
                self.base_percent_max = float(data["Percent"].max())
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )

                self.age_min = float(data["Age"].min())
                self.age_max = float(data["Age"].max())
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)

                self.weeks_min = float(data["Weeks"].min())
                self.weeks_max = float(data["Weeks"].max())
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )

                self.min_week_min = float(data["Min_week"].min())
                self.min_week_max = float(data["Min_week"].max())
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
                )

        return data




## === cell 6
train_base = (
    train_raw.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent"]]
    .rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
)

train = train_raw.merge(train_base, on="Patient", how="left")
train["Base_week"] = train["Weeks"] - train["Min_week"]

train["Weight"] = 1.0

X_prediction = X_prediction_raw.copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

raw_test2 = raw_test.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
X_prediction = X_prediction.merge(raw_test2, how="left", on="Patient")
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]

print("train:", train.shape, "X_prediction:", X_prediction.shape)



## === cell 7
dp_fit_df = pd.concat(
    [
        train[
            [
                "FVC",
                "Weeks",
                "Percent",
                "Age",
                "Sex",
                "SmokingStatus",
                "Min_week",
                "Base_FVC",
                "Base_week",
            ]
        ],
        X_prediction[
            [
                "Weeks",
                "Percent",
                "Age",
                "Sex",
                "SmokingStatus",
                "Min_week",
                "Base_FVC",
                "Base_week",
            ]
        ].assign(FVC=train["FVC"].median()),
    ],
    axis=0,
    ignore_index=True,
)
data_prep = data_preparation(bool_normalization=True, bool_standard=False)
_ = data_prep(dp_fit_df)  # triggers fitting

train_t = data_prep(
    train[
        [
            "Patient",
            "FVC",
            "Weeks",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Min_week",
            "Base_FVC",
            "Base_week",
            "Weight",
        ]
    ]
)
train_t["Patient"] = train["Patient"].values
train_t["Weight"] = train["Weight"].values

X_prediction_t = data_prep(
    X_prediction[
        [
            "Patient",
            "Weeks",
            "Patient_Week",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Min_week",
            "Base_FVC",
            "Base_week",
        ]
    ]
)
X_prediction_t["Patient"] = X_prediction["Patient"].values
X_prediction_t["Patient_Week"] = X_prediction["Patient_Week"].values

fvc_min, fvc_max = data_prep.fvc_min, data_prep.fvc_max
train_t["FVC_n"] = (train["FVC"].values - fvc_min) / (fvc_max - fvc_min)

print("FVC scale:", fvc_min, fvc_max)
print("Transformed columns:", train_t.columns.tolist())




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1821584144.py in <cell line: 0>()
      3 dp_fit_df = pd.concat(
      4     [
----> 5         train[
      6             [
      7                 "FVC",

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['Percent'] not in index"

## === cell 8
def eval_score(y_true, y_pred):
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
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return -keras.backend.mean(metric)




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

for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if col not in train_t.columns:
        train_t[col] = 0
    if col not in X_prediction_t.columns:
        X_prediction_t[col] = 0


def create_model(lambda_loss):
    model_input = keras.Input(shape=(len(SELECTED_COLUMNS),))
    x = L.Dense(500, activation="selu", name="dense_to_freeze1")(model_input)
    x = L.Dense(100, activation="selu", name="dense_to_freeze2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="selu", name="p2")(x)
    FVC = L.Lambda(lambda t: t[0] + tf.cumsum(t[1], axis=1), name="FVC")([p1, p2])

    model = keras.Model(inputs=model_input, outputs=[FVC])
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
        ),
        loss=mloss(lambda_loss),
        metrics=[score],
    )
    return model


model = create_model(LAMBDA_LOSS)
model.summary()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4020166707.py in <cell line: 0>()
     15 # Ensure any missing one-hot columns exist (safety for small data / categories).
     16 for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
---> 17     if col not in train_t.columns:
     18         train_t[col] = 0
     19     if col not in X_prediction_t.columns:

NameError: name 'train_t' is not defined

## === cell 10
patients = train_t["Patient"].unique()
NFOLD = 8
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)

pe = np.zeros((X_prediction_t.shape[0], 3), dtype=np.float32)
pred = np.zeros((train_t.shape[0], 3), dtype=np.float32)

X_test_mat = X_prediction_t[SELECTED_COLUMNS].astype("float32").values
y_train_all = train_t[["FVC_n"]].astype("float32").values

for fold, (tr_idx, va_idx) in enumerate(kf.split(patients), start=0):
    print(f"FOLD {fold}")
    tr_pat = set(patients[tr_idx])
    va_pat = set(patients[va_idx])

    tr_mask = train_t["Patient"].isin(tr_pat).values
    va_mask = train_t["Patient"].isin(va_pat).values

    X_tr = train_t.loc[tr_mask, SELECTED_COLUMNS].astype("float32").values
    y_tr = train_t.loc[tr_mask, ["FVC_n"]].astype("float32").values
    w_tr = train_t.loc[tr_mask, "Weight"].astype("float32").values

    X_va = train_t.loc[va_mask, SELECTED_COLUMNS].astype("float32").values
    y_va = train_t.loc[va_mask, ["FVC_n"]].astype("float32").values

    model = create_model(LAMBDA_LOSS)
    _ = model.fit(
        x=X_tr,
        y=y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        sample_weight=w_tr,
        verbose=0,
    )

    tr_eval = model.evaluate(X_tr, y_tr, verbose=0, batch_size=BATCH_SIZE)
    va_eval = model.evaluate(X_va, y_va, verbose=0, batch_size=BATCH_SIZE)
    print("train eval:", tr_eval)
    print("val eval:", va_eval)

    pred[train_t.index[va_mask]] += (
        model.predict(X_va, batch_size=BATCH_SIZE, verbose=0) / NFOLD
    )
    pe += model.predict(X_test_mat, batch_size=BATCH_SIZE, verbose=0) / NFOLD



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2542001468.py in <cell line: 0>()
      1 # Deterministic, patient-level folds (original used random subsets each fold, which is unstable and can crash score expectations).
----> 2 patients = train_t["Patient"].unique()
      3 NFOLD = 8
      4 kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)
      5 

NameError: name 'train_t' is not defined

## === cell 11
scale = data_prep.fvc_max - data_prep.fvc_min
pe_ml = pe * scale + data_prep.fvc_min
pred_ml = pred * scale + data_prep.fvc_min

sigma_opt = mean_absolute_error(train_raw[["FVC"]].values, pred_ml[:, 1])
unc = pred_ml[:, 2] - pred_ml[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt(MAE):", sigma_opt, "sigma_mean:", sigma_mean)

X_prediction_out = X_prediction.copy()
X_prediction_out["FVC1"] = 0.996 * pe_ml[:, 1]
X_prediction_out["Confidence1"] = pe_ml[:, 2] - pe_ml[:, 0]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/160253800.py in <cell line: 0>()
      1 # Inverse scale predictions back to ml.
----> 2 scale = data_prep.fvc_max - data_prep.fvc_min
      3 pe_ml = pe * scale + data_prep.fvc_min
      4 pred_ml = pred * scale + data_prep.fvc_min
      5 

NameError: name 'data_prep' is not defined

## === cell 12
subm = X_prediction_out.copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

mask = ~subm["FVC1"].isnull()
subm.loc[mask, "FVC"] = subm.loc[mask, "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[mask, "Confidence"] = subm.loc[mask, "Confidence1"]



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/863422252.py in <cell line: 0>()
----> 1 subm = X_prediction_out.copy()
      2 subm["FVC"] = 3020.0
      3 subm["Confidence"] = 100.0
      4 
      5 mask = ~subm["FVC1"].isnull()

NameError: name 'X_prediction_out' is not defined

## === cell 13
otest = raw_test.copy()
for i in range(len(otest)):
    key = otest.Patient.iloc[i] + "_" + str(int(otest.Weeks.iloc[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC.iloc[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1

subm["Confidence"] = subm["Confidence"].astype(float).clip(lower=0.1)

submission = subm[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3635073275.py in <cell line: 0>()
      3 for i in range(len(otest)):
      4     key = otest.Patient.iloc[i] + "_" + str(int(otest.Weeks.iloc[i]))
----> 5     subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC.iloc[i])
      6     subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1
      7 

NameError: name 'subm' is not defined
