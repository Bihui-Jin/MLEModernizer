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

-6.989142889208512

# 6. Current score

-8.76189

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -12.90565) has done: 'I fix the two blockers preventing the notebook from running: (1) the TensorFlow import crash caused by an incompatible protobuf runtime, and (2) the use of missing Kaggle datasets/paths and pickle artifacts that don’t exist in your environment. To keep the core model/loss intact, I keep your MLP architecture and custom loss exactly the same, but rebuild the missing `data_prep` object by fitting it on the available `train.csv`, and rebuild the missing patient fold assignment/weights with a deterministic KFold split by patient (instead of loading `list_patient_score`). I also update the Adam optimizer arguments to the current Keras API (`learning_rate` instead of `lr`, and remove deprecated `decay`) so model compilation works. Finally, I ensure the pipeline always writes a valid `submission.csv` with the exact required columns and that rows align to `sample_submission.csv`.'
- What this solution (achieved -8.76189) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure‑python protobuf setting and importing TensorFlow normally (this error prevents any run). Then I fix the NaNs that break MAE calibration by ensuring `Base_FVC`, `Percent`, `Age`, and related merged fields are correctly created for train/test (your current merge/rename builds NaNs, especially in `Base_FVC` for test). Finally, I make prediction storage shape-consistent (Keras returns `(n,3)` but the current assignment assumes a different structure), so `pred`, `pe`, and the `FVC1/Confidence1` columns are always created and the submission writes correctly; these are correctness fixes and should also improve score substantially versus falling back to constants.'

# 9. Code solution

## === cell 0
import os
import random
import time
import pickle

import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

print("TF version:", tf.__version__)




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

train_raw = pd.read_csv(f"{DATA_DIR}/train.csv")
raw_test = pd.read_csv(f"{DATA_DIR}/test.csv")
X_prediction = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(train_raw.shape, raw_test.shape, X_prediction.shape)



## === cell 3
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.8
EPOCH = [25, 200, 50, 50, 23]
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
        return ["{}_{}".format(name, categories[j]) for j in range(len(categories))]


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
                    data["Min_week"], self.base_week_max, self.base_week_min
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

                self.base_week_min = data["Min_week"].min()
                self.base_week_max = data["Min_week"].max()
                data["Min_week"] = normalization(
                    data["Min_week"], self.base_week_max, self.base_week_min
                )

        return data




## === cell 6
X_prediction = X_prediction.copy()
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")[0]
X_prediction["Weeks"] = (
    X_prediction["Patient_Week"].str.extract(r".*_(.*)")[0].astype(int)
)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

raw_test_base = raw_test.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"}).copy()

X_prediction = X_prediction.merge(raw_test_base, how="left", on="Patient")[
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

X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]



## === cell 7
train = train_raw.copy()
train_base = (
    train.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent"]]
    .rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC", "Percent": "Base_percent"})
)

train = train.merge(
    train_base[["Patient", "Min_week", "Base_FVC", "Base_percent"]],
    on="Patient",
    how="left",
)
train["Base_week"] = train["Weeks"] - train["Min_week"]

for col in [
    "Percent",
    "Age",
    "Sex",
    "SmokingStatus",
    "Min_week",
    "Base_FVC",
    "Base_week",
]:
    if col not in train.columns:
        raise ValueError(f"Missing expected column in train: {col}")

data_prep = data_preparation(bool_normalization=True, bool_standard=False)

_ = data_prep(
    train[
        [
            "Patient",
            "Min_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Base_week",
        ]
    ].copy()
)

train_prep = data_prep(
    train[
        [
            "Patient",
            "Min_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Base_week",
        ]
    ].copy()
)
train_prep["FVC"] = train["FVC"].values

X_prediction_prep = data_prep(X_prediction)

num_cols_train = train_prep.select_dtypes(include=[np.number]).columns
num_cols_pred = X_prediction_prep.select_dtypes(include=[np.number]).columns
train_prep[num_cols_train] = train_prep[num_cols_train].replace(
    [np.inf, -np.inf], np.nan
)
X_prediction_prep[num_cols_pred] = X_prediction_prep[num_cols_pred].replace(
    [np.inf, -np.inf], np.nan
)

train_prep[num_cols_train] = train_prep[num_cols_train].fillna(
    train_prep[num_cols_train].median(numeric_only=True)
)
X_prediction_prep[num_cols_pred] = X_prediction_prep[num_cols_pred].fillna(
    train_prep[num_cols_train].median(numeric_only=True)
)

train = train_prep
X_prediction = X_prediction_prep

print("Prepared shapes:", train.shape, X_prediction.shape)
print(
    "NaNs train:",
    int(train.isna().sum().sum()),
    "NaNs pred:",
    int(X_prediction.isna().sum().sum()),
)



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

for c in SELECTED_COLUMNS:
    if c not in train.columns:
        train[c] = 0
    if c not in X_prediction.columns:
        X_prediction[c] = 0

train[SELECTED_COLUMNS] = train[SELECTED_COLUMNS].astype(np.float32)
X_prediction[SELECTED_COLUMNS] = X_prediction[SELECTED_COLUMNS].astype(np.float32)




## === cell 9
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
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
        ),
        loss=mloss(lambda_loss),
        metrics=[score],
    )
    return model


model = create_model(LAMBDA_LOSS)
model.summary()



## === cell 10
patients = np.array(sorted(train["Patient"].unique()))
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)
patient_to_fold = {}
for fold, (_, val_idx) in enumerate(kf.split(patients)):
    for p in patients[val_idx]:
        patient_to_fold[p] = fold

list_patient_KFOLD = [[p, patient_to_fold[p]] for p in patients]

train["Weight"] = 1.0



## === cell 11
pe = np.zeros((X_prediction.shape[0], 3), dtype=np.float32)
pred = np.zeros((train.shape[0], 3), dtype=np.float32)

for i in range(NFOLD):
    print(f"FOLD {i}")
    model = create_model(LAMBDA_LOSS)
    list_patients = [j[0] for j in list_patient_KFOLD if j[1] == i]

    trn_mask = ~train.Patient.isin(list_patients)
    val_mask = train.Patient.isin(list_patients)

    history = model.fit(
        x=train.loc[trn_mask, SELECTED_COLUMNS],
        y=train.loc[trn_mask, ["FVC"]],
        validation_data=(
            train.loc[val_mask, SELECTED_COLUMNS],
            train.loc[val_mask, ["FVC"]],
        ),
        epochs=EPOCH[i],
        sample_weight=train.loc[trn_mask, "Weight"],
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    print(
        "train",
        model.evaluate(
            train.loc[trn_mask, SELECTED_COLUMNS],
            train.loc[trn_mask, ["FVC"]],
            verbose=0,
            batch_size=BATCH_SIZE,
        ),
    )
    print(
        "val",
        model.evaluate(
            train.loc[val_mask, SELECTED_COLUMNS],
            train.loc[val_mask, ["FVC"]],
            verbose=0,
            batch_size=BATCH_SIZE,
        ),
    )

    val_pred = model.predict(
        train.loc[val_mask, SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0
    )
    test_pred = model.predict(
        X_prediction[SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0
    )

    if isinstance(val_pred, (list, tuple)):
        val_pred = val_pred[0]
    if isinstance(test_pred, (list, tuple)):
        test_pred = test_pred[0]

    pred[train.loc[val_mask].index.values] = val_pred.astype(np.float32)
    pe += test_pred.astype(np.float32) / NFOLD

    model.save(f"model_{i}.keras")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3779893265.py in <cell line: 0>()
     12     val_mask = train.Patient.isin(list_patients)
     13 
---> 14     history = model.fit(
     15         x=train.loc[trn_mask, SELECTED_COLUMNS],
     16         y=train.loc[trn_mask, ["FVC"]],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1874103941.py in score(y_true, y_pred)
      7 
      8     sigma_clip = tf.maximum(sigma, C1)
----> 9     delta = tf.abs(y_true[:, 0] - fvc_pred)
     10     delta = tf.minimum(delta, C2)
     11     sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))

TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int64 of argument 'x'.

## === cell 12
sigma_opt = mean_absolute_error(train["FVC"].values, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt:", sigma_opt, "sigma_mean:", sigma_mean)

X_prediction["FVC1"] = 0.996 * pe[:, 1]
X_prediction["Confidence1"] = pe[:, 2] - pe[:, 0]



## === cell 13
subm = X_prediction.copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

subm.loc[~subm["FVC1"].isnull(), "FVC"] = subm.loc[~subm["FVC1"].isnull(), "FVC1"]
if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[~subm["FVC1"].isnull(), "Confidence"] = subm.loc[
        ~subm["FVC1"].isnull(), "Confidence1"
    ]

otest = pd.read_csv(f"{DATA_DIR}/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1

sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
subm = sample[["Patient_Week"]].merge(
    subm[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)

subm["FVC"] = subm["FVC"].fillna(3020.0).astype(float)
subm["Confidence"] = subm["Confidence"].fillna(100.0).astype(float)

subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
print(subm.head())
