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

3.9

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

-7.012473122175673

# 6. Current score

-8.76189

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76189) has done: 'I fix the import-time crash by setting an environment flag that forces the pure-Python protobuf implementation, which avoids the `MessageFactory.GetPrototype` AttributeError seen in Kaggle TF/Keras stacks. Then I correct all broken input paths to point at the provided `/kaggle/input/osic-pulmonary-fibrosis-progression/...` files and remove dependencies on missing external pickles/datasets by fitting the existing `data_preparation` encoder directly on the loaded train data. Finally, I update the optimizer argument from deprecated `lr` to `learning_rate`, add a deterministic, patient-level KFold split (replacing the missing `list_patient_score` artifact), and ensure a valid `submission.csv` with the exact required columns is written.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by moving the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment setting to the very top of the notebook (before any TF/Keras import) and adding a safe fallback if TF still fails to import. Then I fix the `ValueError: None values not supported` during `model.fit()` by ensuring all model input columns are numeric and contain no missing values (imputing from train medians and re-aligning columns). Finally, I keep the modeling/training logic identical, but make the fold-weighted test ensembling sum to 1.0 (it currently sums to 1.0 but we enforce/normalize defensively) and write a valid `submission.csv` with required columns.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before any TensorFlow/Keras import and also disabling C++ proto via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION`, which is the reliable Kaggle workaround for the `MessageFactory.GetPrototype` error. Then I fix the `None values not supported` during `model.fit()` by ensuring the target `FVC` column is strictly numeric, non-missing, and float32 (the features are already handled, but `FVC` itself can still carry NaNs/None after merges/casts). Finally, I keep the exact model/training logic the same and only add defensive alignment checks so the pipeline runs end-to-end and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before any TensorFlow import and (if still needed) retrying with a safe env setting that avoids the `MessageFactory.GetPrototype` issue. Then I eliminate the `None values not supported` fit-time error by ensuring the training target `y_all` is a 2D float32 array with no missing values and by making the custom loss/metric consume `y_true[:, 0]` consistently (so Keras doesn’t pass unexpected shapes that can propagate `None`). Finally, I keep the model architecture and training loop intact, but ensure prediction arrays have the expected shape (N,3) and the submission file is always written with the required columns and `.csv` suffix.'
- What this solution (achieved -8.76189) has done: 'I fix the import-time protobuf crash by forcing the pure-Python protobuf implementation before TensorFlow is imported and by retrying the import in a clean way that actually re-runs TensorFlow’s initialization with the env vars applied. Then I fix the `None values not supported` error in `model.fit()` by ensuring `sample_weight` is always a pure numeric `float32` NumPy array aligned to `X_trn` (and by avoiding any pandas object/None leakage). Finally, I keep the model, loss, and training loop intact, and only add minimal defensive casting/shape checks so training completes and a valid `submission.csv` is always written.'
- What this solution (achieved -8.76189) has done: 'We fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before the Python process imports protobuf/TensorFlow at all* and by handling the failure with a clean error message (this is required to even reach training). Then we fix the `None values not supported` error in `model.fit()` by ensuring `sample_weight` is passed as a dense numeric NumPy array and by removing any possibility of pandas/object dtype leaking into `fit()` (the current stack trace points to a `None` inside a tensor conversion). Finally, we keep the model, loss, folds, epochs, and feature set unchanged, but make the target `y` shape consistent with the custom loss expectations (`y_true[:,0]`) and add a final defensive NaN/inf check right before training so the run always produces a valid `submission.csv`.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by making the protobuf env override happen before any protobuf/TensorFlow import and by adding a safe fallback that uses `tf.keras` even if standalone `keras` is present. Then I eliminate the `None values not supported` error during `model.fit()` by ensuring the model target matches the model output shape (3 quantiles) instead of a single-column `FVC`, which is what triggers Keras’ internal broadcasting/path that can surface `None` conversions. Finally, I keep your model architecture, loss/metric, folds, epochs, and feature pipeline unchanged, and ensure a correct `submission.csv` is always written with the required columns.'
- What this solution (achieved -8.76189) has done: 'I fix the import-time protobuf crash by ensuring the environment variables are set before TensorFlow (and protobuf) are imported, and by using the stable `tf.keras` path directly. Then I fix the `None values not supported` error during `model.fit()` by making the custom loss/metric handle both `(N, 1)` and `(N, 3)` target shapes safely and by enforcing fully numeric float32 arrays for `x`, `y`, and `sample_weight` right before training. Finally, I keep your model architecture and training loop the same, but make the loss/metric sign consistent with the competition (maximize Laplace log-likelihood), which should move the score upward toward the provided target.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf environment flags before any TensorFlow-related import and by retrying the import in a clean way that works in Kaggle’s TF stack. Then I fix the `None values not supported` during `model.fit()` by ensuring `sample_weight` is never passed when it’s not needed (all-ones), because Keras can still choke on weight tensors in some builds even when arrays look numeric; this is score-neutral and only unblocks training. Finally, I keep your model/loss/training loop intact, but add a tiny defensive cast/finite check right before `fit()` and ensure the written `submission.csv` always matches the required columns and row count.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras as K
    from tensorflow.keras import layers as L
except Exception as e:
    raise RuntimeError(
        "TensorFlow failed to import (likely protobuf incompatibility). "
        "Environment flags were set to force pure-Python protobuf, but import still failed."
    ) from e

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

print("TF version:", tf.__version__)
print("Keras module:", K)




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

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
raw_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(train.shape, raw_test.shape, sample_sub.shape)
print(train.columns.tolist())



## === cell 3
ID = "Patient_Week"
PINBALL_QUANTILE = [0.255, 0.50, 0.745]
LAMBDA_LOSS = 0.585
EPOCH = [54, 55, 20, 60, 23]
BATCH_SIZE = 128
NFOLD = 5



## === cell 4
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def _get_y_column(y_true):
    y_true = tf.convert_to_tensor(y_true)
    if y_true.shape.rank == 1:
        y = tf.reshape(y_true, (-1,))
    else:
        y = y_true[:, 0]
    return y


def score(y_true, y_pred):
    y = _get_y_column(y_true)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))

    metric = -(sq2 * delta) / sigma_clip - tf.math.log(sq2 * sigma_clip)
    return K.backend.mean(metric)


def qloss(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true)
    if y_true.shape.rank == 1:
        y = tf.reshape(y_true, (-1, 1))
    else:
        y = y_true[:, 0:1]
    qs = PINBALL_QUANTILE
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.backend.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) - (1 - _lambda) * score(y_true, y_pred)

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




## === cell 6
min_week = train.groupby("Patient")["Weeks"].min().rename("Min_week")
base_fvc = (
    train.loc[train.groupby("Patient")["Weeks"].idxmin(), ["Patient", "FVC"]]
    .set_index("Patient")["FVC"]
    .rename("Base_FVC")
)

train = train.merge(min_week, on="Patient", how="left")
train = train.merge(base_fvc, on="Patient", how="left")
train["Base_week"] = train["Weeks"] - train["Min_week"]

train["SmokingStatus"] = train["SmokingStatus"].fillna("Unknown")
raw_test["SmokingStatus"] = raw_test["SmokingStatus"].fillna("Unknown")

train.head()



## === cell 7
X_prediction = sample_sub.copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

raw_test_ren = raw_test.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
X_prediction = X_prediction.merge(
    raw_test_ren[
        ["Patient", "Min_week", "Base_FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    ],
    how="left",
    on="Patient",
)
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]

X_prediction.head()



## === cell 8
data_prep = data_preparation(bool_normalization=True, bool_standard=False)

train = data_prep(train)
X_prediction = data_prep(X_prediction)

for col in ["_Currently smokes", "_Ex-smoker", "_Never smoked"]:
    if col not in train.columns:
        train[col] = 0
    if col not in X_prediction.columns:
        X_prediction[col] = 0



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

missing_train = [c for c in SELECTED_COLUMNS if c not in train.columns]
missing_test = [c for c in SELECTED_COLUMNS if c not in X_prediction.columns]
print("Missing in train:", missing_train)
print("Missing in X_prediction:", missing_test)

for c in SELECTED_COLUMNS:
    train[c] = pd.to_numeric(train[c], errors="coerce")
    X_prediction[c] = pd.to_numeric(X_prediction[c], errors="coerce")

medians = train[SELECTED_COLUMNS].median(numeric_only=True)
train[SELECTED_COLUMNS] = train[SELECTED_COLUMNS].fillna(medians)
X_prediction[SELECTED_COLUMNS] = X_prediction[SELECTED_COLUMNS].fillna(medians)

train = train.copy()
X_prediction = X_prediction.copy()
train[SELECTED_COLUMNS] = train[SELECTED_COLUMNS].astype(np.float32)
X_prediction[SELECTED_COLUMNS] = X_prediction[SELECTED_COLUMNS].astype(np.float32)

train["FVC"] = pd.to_numeric(train["FVC"], errors="coerce")
fvc_median = float(train["FVC"].median())
train["FVC"] = train["FVC"].fillna(fvc_median).astype(np.float32)

print("NaNs in train features:", int(train[SELECTED_COLUMNS].isna().sum().sum()))
print("NaNs in pred features:", int(X_prediction[SELECTED_COLUMNS].isna().sum().sum()))
print("NaNs in train FVC:", int(train[["FVC"]].isna().sum().sum()))




## === cell 10
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



## === cell 11
patients = train["Patient"].unique()
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)

patient_to_fold = {}
for fold, (_, val_idx) in enumerate(kf.split(patients)):
    for p in patients[val_idx]:
        patient_to_fold[p] = fold

train["Fold"] = train["Patient"].map(patient_to_fold).astype(int)

train["Weight"] = 1.0
train["Weight"] = (
    pd.to_numeric(train["Weight"], errors="coerce").fillna(1.0).astype(np.float32)
)

KFOLD_confidence = [0.05, 0.15, 0.2, 0.25, 0.35]
KFOLD_confidence = (
    np.array(KFOLD_confidence, dtype=np.float32) / np.sum(KFOLD_confidence)
).tolist()
print("KFOLD_confidence sum:", sum(KFOLD_confidence))



## === cell 12
pe = np.zeros((X_prediction.shape[0], 3), dtype=np.float32)
pred = np.zeros((train.shape[0], 3), dtype=np.float32)

X_pred_np = X_prediction[SELECTED_COLUMNS].to_numpy(dtype=np.float32)

y_scalar = train[["FVC"]].to_numpy(dtype=np.float32).reshape(-1, 1)
y_all = np.repeat(y_scalar, 3, axis=1).astype(np.float32)

X_all = train[SELECTED_COLUMNS].to_numpy(dtype=np.float32)
folds_arr = train["Fold"].to_numpy(dtype=np.int32)

weights_arr = np.asarray(train["Weight"].to_numpy(), dtype=np.float32)
weights_arr[~np.isfinite(weights_arr)] = 1.0

X_all = np.ascontiguousarray(X_all, dtype=np.float32)
y_all = np.ascontiguousarray(y_all, dtype=np.float32)
X_pred_np = np.ascontiguousarray(X_pred_np, dtype=np.float32)
weights_arr = np.ascontiguousarray(weights_arr, dtype=np.float32)

use_sample_weight = not np.allclose(weights_arr, 1.0)

for i in range(NFOLD):
    print(f"FOLD {i}")
    model = create_model(LAMBDA_LOSS)

    trn_mask = folds_arr != i
    val_mask = folds_arr == i

    X_trn = np.ascontiguousarray(X_all[trn_mask], dtype=np.float32)
    y_trn = np.ascontiguousarray(y_all[trn_mask], dtype=np.float32)
    X_val = np.ascontiguousarray(X_all[val_mask], dtype=np.float32)
    y_val = np.ascontiguousarray(y_all[val_mask], dtype=np.float32)

    X_trn = X_trn.astype(np.float32, copy=False)
    y_trn = y_trn.astype(np.float32, copy=False)
    X_val = X_val.astype(np.float32, copy=False)
    y_val = y_val.astype(np.float32, copy=False)

    if not np.isfinite(X_trn).all() or not np.isfinite(y_trn).all():
        raise ValueError("Non-finite values detected in training fold inputs/targets.")
    if not np.isfinite(X_val).all() or not np.isfinite(y_val).all():
        raise ValueError(
            "Non-finite values detected in validation fold inputs/targets."
        )

    fit_kwargs = dict(
        x=X_trn,
        y=y_trn,
        validation_data=(X_val, y_val),
        epochs=EPOCH[i],
        verbose=0,
        batch_size=BATCH_SIZE,
    )

    if use_sample_weight:
        w_trn = np.ascontiguousarray(weights_arr[trn_mask], dtype=np.float32)
        if (
            (w_trn.ndim != 1)
            or (len(w_trn) != len(X_trn))
            or (not np.isfinite(w_trn).all())
        ):
            raise ValueError("Invalid sample_weight detected.")
        fit_kwargs["sample_weight"] = w_trn

    history = model.fit(**fit_kwargs)

    print("train", model.evaluate(X_trn, y_trn, verbose=0, batch_size=BATCH_SIZE))
    print("val", model.evaluate(X_val, y_val, verbose=0, batch_size=BATCH_SIZE))

    val_pred = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    test_pred = model.predict(X_pred_np, batch_size=BATCH_SIZE, verbose=0)

    if isinstance(val_pred, (list, tuple)):
        val_pred = val_pred[0]
    if isinstance(test_pred, (list, tuple)):
        test_pred = test_pred[0]

    pred[val_mask] = val_pred.astype(np.float32)
    pe += test_pred.astype(np.float32) * KFOLD_confidence[i]

    model.save(f"model_{i}.keras")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1517118502.py in <cell line: 0>()
     67         fit_kwargs["sample_weight"] = w_trn
     68 
---> 69     history = model.fit(**fit_kwargs)
     70 
     71     print("train", model.evaluate(X_trn, y_trn, verbose=0, batch_size=BATCH_SIZE))

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

## === cell 13
sigma_opt = mean_absolute_error(train[["FVC"]], pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt:", sigma_opt, "sigma_mean:", sigma_mean)

X_prediction["FVC1"] = 0.996 * pe[:, 1]
X_prediction["Confidence1"] = pe[:, 2] - pe[:, 0]

subm = X_prediction[["Patient_Week"]].copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

mask = ~X_prediction["FVC1"].isnull()
subm.loc[mask, "FVC"] = X_prediction.loc[mask, "FVC1"].astype(float).values

if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[mask, "Confidence"] = (
        X_prediction.loc[mask, "Confidence1"].astype(float).values
    )

otest = raw_test.copy()
for i in range(len(otest)):
    key = otest.Patient.iloc[i] + "_" + str(int(otest.Weeks.iloc[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC.iloc[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1

subm["Confidence"] = subm["Confidence"].clip(lower=1e-3)

subm = subm[["Patient_Week", "FVC", "Confidence"]]
subm.to_csv("submission.csv", index=False)

print(subm.head())
print("Wrote submission.csv with shape:", subm.shape)
print("Submission columns:", subm.columns.tolist())
print("Expected rows (sample_submission):", sample_sub.shape[0])
assert subm.shape[0] == sample_sub.shape[0], "Row count mismatch vs sample_submission."
assert list(subm.columns) == [
    "Patient_Week",
    "FVC",
    "Confidence",
], "Submission columns mismatch."
