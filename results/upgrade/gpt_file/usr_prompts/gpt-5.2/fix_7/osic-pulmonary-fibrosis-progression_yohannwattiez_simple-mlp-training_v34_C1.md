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

-7.253935904752643

# 6. Current score

-11.09437

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -9.71374) has done: 'I fix the TensorFlow import crash by setting a safe protobuf implementation before importing TF (this resolves the `MessageFactory.GetPrototype` error in many Kaggle images). I also correct all broken input paths by pointing to the provided `/kaggle/input/osic-pulmonary-fibrosis-progression/*.csv` files and remove the dependency on the missing external pickle (`data-preparation-for-osic/data_prep`) by fitting the existing `data_preparation` class directly on the training-derived features. Finally, I make minimal Keras-API compatibility fixes (`lr`→`learning_rate`, remove deprecated `decay`) and fix a few variable/shape issues so training, prediction, and submission writing run end-to-end and produce a valid `submission.csv` with the required columns.'
- What this solution (achieved -9.71374) has done: 'I fix the immediate crash before any training starts by making the TensorFlow import robust to the known protobuf incompatibility (the `MessageFactory.GetPrototype` error) by forcing the pure-Python protobuf implementation *before* TensorFlow is imported. I also add a safe fallback that uninstalls/reloads conflicting `google.protobuf` state if the first import attempt fails, while keeping the rest of the pipeline unchanged. After that, the script run end-to-end, train the same model as before, and write a valid `submission.csv` with the required columns. No model architecture, training loop, loss, or feature logic is changed (score impact should be neutral aside from enabling execution).'
- What this solution (achieved -9.71374) has done: 'I fix the immediate TensorFlow import crash by setting the protobuf environment variables before *any* TensorFlow-related import and by adding a robust fallback that switches to the pure-Python protobuf implementation if the first import fails. Then I keep your model/training logic unchanged, but correct the feature engineering mismatch where `Base_percent` is referenced even though the column is named `Percent` (this can silently hurt training stability and score). Finally, I make the submission confidence always valid for the metric by ensuring it is positive and clipped to at least 70 at output time (score-improving but aligned with the competition’s clipping rule), while preserving your existing prediction structure and overrides for baseline test rows.'
- What this solution (achieved -11.09437) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the legacy pure-Python protobuf backend *and* (if needed) sanitizing protobuf-related environment variables **before** importing TF, with a robust fallback reload. Then I correct the sign in your `score()` metric to match the competition’s Laplace log-likelihood (higher-is-better); this doesn’t change the training core, but it fixes the loss’ direction so optimization improves the true Kaggle metric instead of pushing the wrong way. Finally, I keep your model/training loop intact and only ensure the output confidence is always valid (positive and effectively clipped) and the submission is written correctly as `submission.csv`.'
- What this solution (achieved -11.09437) has done: 'I fix the TensorFlow/protobuf crash by switching to a TF import strategy that works in Kaggle without changing any model/training logic (use the default C++ protobuf and, if needed, sanitize conflicting protobuf environment variables and retry). Then I fix the incorrect “baseline row override” in the submission: currently it sets `Confidence=0.1`, but Kaggle clips confidence to at least 70 in the metric, so this only hurts score; I set it to 70.0 while still keeping the exact baseline FVC override. Finally, I ensure the submission is aligned to `sample_submission.csv`’s `Patient_Week` ordering and contains only the required columns, so it’s always a valid submission CSV.'
- What this solution (achieved -11.09437) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the most reliable workaround for the `MessageFactory.GetPrototype` error in Kaggle images. I also make the TF import fallback more robust by clearing any already-imported protobuf modules before retrying, without changing your model/training logic. Everything else (feature engineering, model architecture, training loop, loss, and submission formatting) be kept the same so the run completes end-to-end and writes a valid `submission.csv`. This should restore execution and allow the model to train/predict normally, which is necessary to move the score upward toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import random
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception as e1:
    import sys

    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            sys.modules.pop(m, None)

    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

    import tensorflow as tf

import pickle  # kept to preserve original imports, though unused
from tensorflow import keras as K
from tensorflow.keras import layers as L

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)




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
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_raw = pd.read_csv(train_path)
raw_test = pd.read_csv(test_path)
X_prediction_raw = pd.read_csv(sample_sub_path)

print(train_raw.shape, raw_test.shape, X_prediction_raw.shape)
print("train columns:", train_raw.columns.tolist())
print("test columns:", raw_test.columns.tolist())
print("sample_submission columns:", X_prediction_raw.columns.tolist())



## === cell 3
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.8
EPOCH = 250
BATCH_SIZE = 128



## === cell 4
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    """
    Competition metric (Laplace Log Likelihood) averaged over samples.
    Kaggle uses:
      metric = - sqrt(2)*Delta/sigma_clip - log(sqrt(2)*sigma_clip)
    Higher is better.
    """
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))

    metric = -(sq2 * delta) / sigma_clip - tf.math.log(sq2 * sigma_clip)
    return K.backend.mean(metric)


def qloss(y_true, y_pred):
    qs = PINBALL_QUANTILE
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.backend.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * (
            -score(y_true, y_pred)
        )

    return loss




## === cell 5
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

                self.base_percent_mean = data["Percent"].mean()
                self.base_percent_std = data["Percent"].std()
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
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




## === cell 6
def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Patient")["Weeks"].min().rename("Min_week")
    out = df.merge(g, on="Patient", how="left")
    out["Base_week"] = out["Weeks"] - out["Min_week"]
    out = out.rename(columns={"FVC": "Base_FVC"})
    return out


train_feat = build_feature_frame(train_raw)
test_feat = build_feature_frame(raw_test)

X_prediction = X_prediction_raw.copy()
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

train_feat["Base_week"] = train_feat["Weeks"] - train_feat["Min_week"]

print("train_feat columns:", train_feat.columns.tolist())
print("X_prediction columns:", X_prediction.columns.tolist())



## === cell 7
data_prep = data_preparation(bool_normalization=True, bool_standard=False)

train_proc = data_prep(
    train_feat[
        [
            "Weeks",
            "Percent",
            "Age",
            "Sex",
            "Min_week",
            "Base_FVC",
            "Base_week",
            "SmokingStatus",
        ]
    ]
)
X_prediction_proc = data_prep(
    X_prediction[
        [
            "Weeks",
            "Percent",
            "Age",
            "Sex",
            "Min_week",
            "Base_FVC",
            "Base_week",
            "SmokingStatus",
        ]
    ]
)

train_proc["FVC"] = train_raw["FVC"].values.astype(np.float32)

print("Processed train columns:", train_proc.columns.tolist())



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

missing_train = [c for c in SELECTED_COLUMNS if c not in train_proc.columns]
missing_test = [c for c in SELECTED_COLUMNS if c not in X_prediction_proc.columns]
if missing_train or missing_test:
    raise ValueError(
        f"Missing columns. train missing={missing_train}, test missing={missing_test}"
    )

train = train_proc.reset_index(drop=True)
X_prediction_final = X_prediction_proc.reset_index(drop=True)




## === cell 9
def create_model(lambda_loss):
    model_input = K.Input(shape=(len(SELECTED_COLUMNS),))
    x = L.Dense(500, activation="selu", name="dense_to_freeze1")(model_input)
    x = L.Dense(100, activation="selu", name="dense_to_freeze2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="selu", name="p2")(x)
    FVC = L.Lambda(lambda t: t[0] + tf.cumsum(t[1], axis=1), name="FVC")([p1, p2])

    model = K.Model(inputs=model_input, outputs=[FVC])

    model.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
        ),
        loss=mloss(lambda_loss),
        metrics=[score],
    )
    return model


model = create_model(LAMBDA_LOSS)
model.summary()



## === cell 10
NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)

pe = np.zeros((X_prediction_final.shape[0], 3), dtype=np.float32)
pred = np.zeros((train.shape[0], 3), dtype=np.float32)



## === cell 11
cnt = 0
EPOCHS = 800

X_tr_full = train[SELECTED_COLUMNS].astype(np.float32).values
y_full = train[["FVC"]].astype(np.float32).values  # keep 2D for y_true[:,0]
X_test_full = X_prediction_final[SELECTED_COLUMNS].astype(np.float32).values

for tr_idx, val_idx in kf.split(X_tr_full):
    cnt += 1
    print(f"FOLD {cnt}")
    net = create_model(LAMBDA_LOSS)

    net.fit(
        X_tr_full[tr_idx],
        y_full[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_tr_full[val_idx], y_full[val_idx]),
        verbose=0,
    )

    print(
        "train",
        net.evaluate(
            X_tr_full[tr_idx], y_full[tr_idx], verbose=0, batch_size=BATCH_SIZE
        ),
    )
    print(
        "val",
        net.evaluate(
            X_tr_full[val_idx], y_full[val_idx], verbose=0, batch_size=BATCH_SIZE
        ),
    )

    print("predict val...")
    pred[val_idx] = net.predict(X_tr_full[val_idx], batch_size=BATCH_SIZE, verbose=0)

    print("predict test...")
    pe += net.predict(X_test_full, batch_size=BATCH_SIZE, verbose=0) / NFOLD

    net.save(f"model_{cnt}.keras")



## === cell 12
sigma_opt = mean_absolute_error(train[["FVC"]].values, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt:", sigma_opt, "sigma_mean:", sigma_mean)



## === cell 13
X_prediction_out = X_prediction.copy()
X_prediction_out["FVC1"] = 0.996 * pe[:, 1]
X_prediction_out["Confidence1"] = pe[:, 2] - pe[:, 0]

subm = X_prediction_out.copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]

conf_model = subm["Confidence1"].astype(float)
conf_model = np.abs(conf_model)
conf_model = np.maximum(conf_model, 70.0)

sigma_opt_clipped = float(max(sigma_opt, 70.0))

if sigma_mean < 70:
    subm["Confidence"] = sigma_opt_clipped
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = conf_model.loc[~subm.FVC1.isnull()]

otest = raw_test.copy()
for i in range(len(otest)):
    key = otest.Patient.iloc[i] + "_" + str(int(otest.Weeks.iloc[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC.iloc[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 70.0

subm_out = subm[["Patient_Week", "FVC", "Confidence"]].copy()
subm_out["FVC"] = subm_out["FVC"].astype(float)
subm_out["Confidence"] = subm_out["Confidence"].astype(float)

subm_out = (
    X_prediction_raw[["Patient_Week"]]
    .merge(subm_out, on="Patient_Week", how="left")
    .fillna({"FVC": 3020.0, "Confidence": 100.0})
)

print(subm_out.head())
print(subm_out.shape)



## === cell 14
subm_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm_out.shape)
print("Saved to:", os.path.abspath("submission.csv"))
