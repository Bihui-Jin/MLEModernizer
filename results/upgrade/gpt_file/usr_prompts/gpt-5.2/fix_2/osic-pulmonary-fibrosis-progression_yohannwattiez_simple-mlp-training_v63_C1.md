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

-7.022315791905101

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pickle
import time

from tensorflow import keras as K
from tensorflow.keras import layers as L

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error




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
TRAIN_PATH = f"{BASE_PATH}/train.csv"
TEST_PATH = f"{BASE_PATH}/test.csv"
SAMPLE_SUB_PATH = f"{BASE_PATH}/sample_submission.csv"

train_raw = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)



## === cell 3
ID = "Patient_Week"
PINBALL_QUANTILE = [0.27, 0.50, 0.73]
LAMBDA_LOSS = 0.585
EPOCH = [54, 55, 20, 60, 23]
BATCH_SIZE = 128
NFOLD = 5



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
    FVC = L.Lambda(lambda t: t[0] + tf.cumsum(t[1], axis=1), name="FVC")([p1, p2])

    model = K.Model(inputs=model_input, outputs=[FVC])

    opt = K.optimizers.Adam(
        learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
    )
    model.compile(optimizer=opt, loss=mloss(lambda_loss), metrics=[score])
    return model




## === cell 6
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
                self.base_week_mean, self.base_week_std = (
                    data["Base_week"].mean(),
                    data["Base_week"].std(),
                )
                self.base_fvc_mean, self.base_fvc_std = (
                    data["Base_FVC"].mean(),
                    data["Base_FVC"].std(),
                )
                self.base_percent_mean, self.base_percent_std = (
                    data["Percent"].mean(),
                    data["Percent"].std(),
                )
                self.age_mean, self.age_std = data["Age"].mean(), data["Age"].std()
                self.weeks_mean, self.weeks_std = (
                    data["Weeks"].mean(),
                    data["Weeks"].std(),
                )

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
                self.base_week_min, self.base_week_max = (
                    data["Base_week"].min(),
                    data["Base_week"].max(),
                )
                self.base_fvc_min, self.base_fvc_max = (
                    data["Base_FVC"].min(),
                    data["Base_FVC"].max(),
                )
                self.base_percent_min, self.base_percent_max = (
                    data["Percent"].min(),
                    data["Percent"].max(),
                )
                self.age_min, self.age_max = data["Age"].min(), data["Age"].max()
                self.weeks_min, self.weeks_max = (
                    data["Weeks"].min(),
                    data["Weeks"].max(),
                )
                self.min_week_min, self.min_week_max = (
                    data["Min_week"].min(),
                    data["Min_week"].max(),
                )

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

        return data




## === cell 7
train = train_raw.copy()
train["Min_week"] = train.groupby("Patient")["Weeks"].transform("min")

base = (
    train.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC"]]
    .rename(columns={"Weeks": "Min_week_base", "FVC": "Base_FVC_from_first"})
)

base0 = train[train["Weeks"] == 0][["Patient", "Weeks", "FVC"]].rename(
    columns={"Weeks": "Base_week_raw", "FVC": "Base_FVC"}
)
train = train.merge(
    base0[["Patient", "Base_week_raw", "Base_FVC"]], on="Patient", how="left"
)
train["Base_week_raw"] = train["Base_week_raw"].fillna(
    train.merge(base, on="Patient", how="left")["Min_week_base"]
)
train["Base_FVC"] = train["Base_FVC"].fillna(
    train.merge(base, on="Patient", how="left")["Base_FVC_from_first"]
)

train["Base_week"] = train["Weeks"] - train["Min_week"]

train["Weight"] = 1.0



## === cell 8
X_prediction = sample_sub.copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)

X_prediction = X_prediction.merge(
    raw_test, on="Patient", how="left", suffixes=("", "_base")
)

X_prediction["Min_week"] = X_prediction["Weeks_base"]
X_prediction["Base_FVC"] = X_prediction["FVC"]
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]

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



## === cell 9
data_prep = data_preparation(bool_normalization=True, bool_standard=False)

train_prep_input = train[
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
].copy()
test_prep_input = X_prediction[
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
].copy()

train_trans = data_prep(train_prep_input)
X_prediction_trans = data_prep(test_prep_input)

for col in SELECTED_COLUMNS:
    if col not in train_trans.columns:
        train_trans[col] = 0
    if col not in X_prediction_trans.columns:
        X_prediction_trans[col] = 0

train_feat = train_trans[SELECTED_COLUMNS].astype(np.float32)
test_feat = X_prediction_trans[SELECTED_COLUMNS].astype(np.float32)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/497682197.py in <cell line: 0>()
     14     ]
     15 ].copy()
---> 16 test_prep_input = X_prediction[
     17     [
     18         "Weeks",

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

KeyError: "['Base_week'] not in index"

## === cell 10
patients = train["Patient"].drop_duplicates().values
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)
patient_to_fold = {}
for fold, (_, val_idx) in enumerate(kf.split(patients)):
    for p in patients[val_idx]:
        patient_to_fold[p] = fold
list_patient_KFOLD = [(p, patient_to_fold[p]) for p in patients]

KFOLD_confidence = [0.1, 0.1, 0.2, 0.3, 0.3]



## === cell 11
pe = np.zeros((test_feat.shape[0], 3), dtype=np.float32)
pred_oof = np.zeros((train_feat.shape[0], 3), dtype=np.float32)

y_train = train[["FVC"]].values.astype(np.float32)

for i in range(NFOLD):
    print(f"FOLD {i}")
    model = create_model(LAMBDA_LOSS)

    val_patients = [j[0] for j in list_patient_KFOLD if j[1] == i]
    is_val = train["Patient"].isin(val_patients).values
    is_tr = ~is_val

    history = model.fit(
        x=train_feat.loc[is_tr],
        y=y_train[is_tr],
        validation_data=(train_feat.loc[is_val], y_train[is_val]),
        epochs=EPOCH[i],
        sample_weight=train.loc[is_tr, "Weight"].values,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    tr_eval = model.evaluate(
        train_feat.loc[is_tr], y_train[is_tr], verbose=0, batch_size=BATCH_SIZE
    )
    va_eval = model.evaluate(
        train_feat.loc[is_val], y_train[is_val], verbose=0, batch_size=BATCH_SIZE
    )
    print("train", tr_eval)
    print("val", va_eval)

    pred_oof[train.index[is_val]] = model.predict(
        train_feat.loc[is_val], batch_size=BATCH_SIZE, verbose=0
    )
    pe += (
        model.predict(test_feat, batch_size=BATCH_SIZE, verbose=0) * KFOLD_confidence[i]
    )

    model.save(f"model_{i}.keras")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3212709530.py in <cell line: 0>()
      1 # Train per fold and predict
----> 2 pe = np.zeros((test_feat.shape[0], 3), dtype=np.float32)
      3 pred_oof = np.zeros((train_feat.shape[0], 3), dtype=np.float32)
      4 
      5 y_train = train[["FVC"]].values.astype(np.float32)

NameError: name 'test_feat' is not defined

## === cell 12
sigma_opt = mean_absolute_error(train[["FVC"]], pred_oof[:, 1])
unc = pred_oof[:, 2] - pred_oof[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt", sigma_opt, "sigma_mean", sigma_mean)

X_prediction["FVC1"] = 0.996 * pe[:, 1]
X_prediction["Confidence1"] = pe[:, 2] - pe[:, 0]

subm = X_prediction[["Patient_Week"]].copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

subm.loc[~X_prediction["FVC1"].isnull(), "FVC"] = X_prediction.loc[
    ~X_prediction["FVC1"].isnull(), "FVC1"
].values
if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[~X_prediction["FVC1"].isnull(), "Confidence"] = X_prediction.loc[
        ~X_prediction["FVC1"].isnull(), "Confidence1"
    ].values

for i in range(len(raw_test)):
    pw = raw_test.Patient[i] + "_" + str(raw_test.Weeks[i])
    subm.loc[subm["Patient_Week"] == pw, "FVC"] = float(raw_test.FVC[i])
    subm.loc[subm["Patient_Week"] == pw, "Confidence"] = 0.1

subm = sample_sub[["Patient_Week"]].merge(subm, on="Patient_Week", how="left")
subm["FVC"] = subm["FVC"].fillna(3020.0).astype(float)
subm["Confidence"] = subm["Confidence"].fillna(100.0).astype(float)

subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print(subm.head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3687072006.py in <cell line: 0>()
      1 # Calibration/post-processing consistent with original script
----> 2 sigma_opt = mean_absolute_error(train[["FVC"]], pred_oof[:, 1])
      3 unc = pred_oof[:, 2] - pred_oof[:, 0]
      4 sigma_mean = float(np.mean(unc))
      5 print("sigma_opt", sigma_opt, "sigma_mean", sigma_mean)

NameError: name 'pred_oof' is not defined
