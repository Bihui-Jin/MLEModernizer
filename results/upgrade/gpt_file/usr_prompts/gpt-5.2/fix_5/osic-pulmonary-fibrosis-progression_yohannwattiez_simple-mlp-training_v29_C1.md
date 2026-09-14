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

-7.468159738011274

# 6. Current score

-8.76189

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.76189) has done: 'I fix the environment/import crash by forcing TensorFlow to use the Python implementation of protobuf (this resolves the `MessageFactory` error in many Kaggle images), then correct the broken input paths so train/test/sample_submission are read from the provided dataset. I remove the missing external pickle dependency (`data-preparation-for-osic/data_prep`) by fitting the existing `data_preparation` class directly on the loaded train data, preserving the original feature pipeline and selected columns. I update the Keras Adam optimizer arguments (`lr`→`learning_rate`) to match the installed Keras version, and fix a couple of column-name typos (`Base_percent` vs `Percent`) so preprocessing runs. Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -8.76189) has done: 'I fix two execution blockers: the protobuf/TensorFlow import crash by safely falling back to `tensorflow-cpu`/standard TF import without triggering the `MessageFactory.GetPrototype` path, and the Keras 3 compile error by passing `metrics=[score]` (same metric function, just in the required container type). These changes are score-neutral but allow the full training/inference pipeline to run end-to-end and write a valid `submission.csv`. Since your current score is below target (more negative), I also make a minimal, metric-aligned calibration fix: clip predicted `Confidence` to be at least 70 everywhere (matching the evaluation’s sigma clipping), which typically improves Laplace log likelihood without changing the model architecture/training loop. All paths and core model/training logic remain unchanged.'
- What this solution (achieved -8.76189) has done: 'I first fix the TensorFlow import crash by switching protobuf to the pure-Python implementation (the current `"cpp"` setting triggers the `_message` ImportError in this environment), which unblocks all downstream cells. Next, I restore missing imports (`KFold`, `mean_absolute_error`) and make `seed_all()` resilient so it doesn’t fail if TF isn’t imported yet. Finally, I keep the modeling/training logic unchanged, but ensure the submission is always created end-to-end with the exact required columns and a `.csv` suffix, including the metric-aligned `Confidence >= 70` clipping that is already present.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

print("Python:", sys.version)
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
    try:
        tf.random.set_seed(seed)
    except Exception as e:
        print("Warning: could not set TF seed:", repr(e))


seed_all(20)



## === cell 2
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
raw_test = pd.read_csv(f"{DATA_DIR}/test.csv")
X_prediction = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(train.shape, raw_test.shape, X_prediction.shape)
train.head()



## === cell 3
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.75
BATCH_SIZE = 128

NFOLD = 5
EPOCHS = 800



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
def create_model(lambda_loss):
    model_input = K.Input(shape=(9,))
    x = L.Dense(500, activation="relu", name="dense_to_freeze1")(model_input)
    x = L.Dense(100, activation="relu", name="dense_to_freeze2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
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



## === cell 6
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

raw_test_ = raw_test.copy()
raw_test_ = raw_test_.rename(columns={"Weeks": "Base_week", "FVC": "Base_FVC"})

X_prediction = (
    X_prediction.merge(
        raw_test_[
            [
                "Patient",
                "Base_week",
                "Base_FVC",
                "Percent",
                "Age",
                "Sex",
                "SmokingStatus",
            ]
        ],
        how="left",
        on="Patient",
    )
    .loc[
        :,
        [
            "Patient",
            "Base_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ],
    ]
    .reset_index(drop=True)
)

X_prediction.head()



## === cell 7
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

        if "Sex" in data.columns:
            data["Sex"] = data["Sex"].fillna("Unknown")
        if "SmokingStatus" in data.columns:
            data["SmokingStatus"] = data["SmokingStatus"].fillna("Unknown")

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

        return data




## === cell 8
train_ = train.copy()

base = train_.loc[
    train_.groupby("Patient")["Weeks"].apply(lambda s: (s.abs().idxmin()))
].copy()
base = base.rename(
    columns={"Weeks": "Base_week", "FVC": "Base_FVC", "Percent": "Base_percent"}
)
base = base[["Patient", "Base_week", "Base_FVC", "Base_percent"]]

train_ = train_.merge(base, on="Patient", how="left")

train_["Base_week"] = train_["Base_week"]
train_["Base_FVC"] = train_["Base_FVC"]
train_ = train_.rename(columns={"Base_percent": "Base_percent"})

train_["Patient_Week"] = (
    train_["Patient"].astype(str) + "_" + train_["Weeks"].astype(int).astype(str)
)

data_prep = data_preparation(bool_normalization=True, bool_standard=False)

train_p = data_prep(
    train_[
        [
            "Patient",
            "Base_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ]
    ]
)
X_pred_p = data_prep(X_prediction)

y_train = train[["FVC"]].values.astype("float32")

print("train_p shape:", train_p.shape, "X_pred_p shape:", X_pred_p.shape)



## === cell 9
SELECTED_COLUMNS = [
    "Base_week",
    "Base_FVC",
    "Percent",
    "Age",
    "Sex",
    "Weeks",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]

missing_train = [c for c in SELECTED_COLUMNS if c not in train_p.columns]
missing_test = [c for c in SELECTED_COLUMNS if c not in X_pred_p.columns]
if missing_train or missing_test:
    raise ValueError(
        f"Missing columns. train missing={missing_train}, test missing={missing_test}"
    )


def make_numeric_no_nan(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


train_p = make_numeric_no_nan(train_p, SELECTED_COLUMNS)
X_pred_p = make_numeric_no_nan(X_pred_p, SELECTED_COLUMNS)

train_p[SELECTED_COLUMNS] = train_p[SELECTED_COLUMNS].astype("float32")
X_pred_p[SELECTED_COLUMNS] = X_pred_p[SELECTED_COLUMNS].astype("float32")

print("Any NaNs in train inputs:", train_p[SELECTED_COLUMNS].isna().any().any())
print("Any NaNs in test inputs:", X_pred_p[SELECTED_COLUMNS].isna().any().any())



## === cell 10
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=20)

pe = np.zeros((X_pred_p.shape[0], 3), dtype=np.float32)
pred = np.zeros((train_p.shape[0], 3), dtype=np.float32)

cnt = 0
for tr_idx, val_idx in kf.split(train_p):
    cnt += 1
    print(f"FOLD {cnt}/{NFOLD}")
    net = create_model(LAMBDA_LOSS)

    net.fit(
        train_p.loc[tr_idx, SELECTED_COLUMNS].values,
        y_train[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(
            train_p.loc[val_idx, SELECTED_COLUMNS].values,
            y_train[val_idx],
        ),
        verbose=0,
    )

    tr_eval = net.evaluate(
        train_p.loc[tr_idx, SELECTED_COLUMNS].values,
        y_train[tr_idx],
        verbose=0,
        batch_size=BATCH_SIZE,
    )
    va_eval = net.evaluate(
        train_p.loc[val_idx, SELECTED_COLUMNS].values,
        y_train[val_idx],
        verbose=0,
        batch_size=BATCH_SIZE,
    )
    print("train eval (loss, score):", tr_eval)
    print("val   eval (loss, score):", va_eval)

    pred[val_idx] = net.predict(
        train_p.loc[val_idx, SELECTED_COLUMNS].values, batch_size=BATCH_SIZE, verbose=0
    )
    pe += (
        net.predict(X_pred_p[SELECTED_COLUMNS].values, batch_size=BATCH_SIZE, verbose=0)
        / NFOLD
    )



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1521950089.py in <cell line: 0>()
     10     net = create_model(LAMBDA_LOSS)
     11 
---> 12     net.fit(
     13         train_p.loc[tr_idx, SELECTED_COLUMNS].values,
     14         y_train[tr_idx],

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
sigma_opt = mean_absolute_error(train[["FVC"]].values, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = float(np.mean(unc))
print("sigma_opt:", sigma_opt, "sigma_mean:", sigma_mean)



## === cell 12
X_prediction_out = X_prediction.copy()
X_prediction_out["FVC1"] = 0.996 * pe[:, 1]
X_prediction_out["Confidence1"] = pe[:, 2] - pe[:, 0]

subm = X_prediction_out.copy()
subm["FVC"] = 3020.0
subm["Confidence"] = 100.0

mask = ~subm["FVC1"].isnull()
subm.loc[mask, "FVC"] = subm.loc[mask, "FVC1"]

if sigma_mean < 70:
    subm["Confidence"] = float(sigma_opt)
else:
    subm.loc[mask, "Confidence"] = subm.loc[mask, "Confidence1"]

subm["Confidence"] = subm["Confidence"].clip(lower=70.0)



## === cell 13
otest = pd.read_csv(f"{DATA_DIR}/test.csv")
for i in range(len(otest)):
    key = otest.Patient[i] + "_" + str(int(otest.Weeks[i]))
    subm.loc[subm["Patient_Week"] == key, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == key, "Confidence"] = 0.1

subm["Confidence"] = subm["Confidence"].clip(lower=70.0)



## === cell 14
submission_path = "submission.csv"
subm[["Patient_Week", "FVC", "Confidence"]].to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(subm.head())
print("Submission shape:", subm[["Patient_Week", "FVC", "Confidence"]].shape)
print(
    "Any nulls:", subm[["Patient_Week", "FVC", "Confidence"]].isnull().any().to_dict()
)
