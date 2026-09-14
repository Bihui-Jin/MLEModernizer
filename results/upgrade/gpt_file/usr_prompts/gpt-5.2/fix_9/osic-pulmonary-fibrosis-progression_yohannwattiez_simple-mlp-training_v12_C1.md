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

-6.9561572857819725

# 6. Current score

-8.82183

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.82186) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TF (this addresses the `MessageFactory.GetPrototype` error). I also remove dependencies on the missing `/kaggle/input/prep-data/*` files by training directly from the provided competition `train.csv` and fitting the `data_preparation` encoder in-notebook, then applying it consistently to both train and test. To prevent the one-hot smoking columns from going missing at inference (causing the KeyError), I explicitly add any absent expected dummy columns with zeros after transformation. Finally, I ensure the submission is written with the exact required columns and a `.csv` suffix.'
- What this solution (achieved -8.82186) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime *before* importing TensorFlow (the current `cpp` default triggers the `_message` ImportError in this environment). Because TensorFlow currently fails to import, all downstream variables (`pd`, `random`, `tf`, `raw_train`, etc.) are undefined; once TF imports correctly, the rest of the pipeline can run with minimal changes. I preserve your existing model, loss, feature logic, and training loop, only adding small guards to ensure the one-hot columns exist and the submission is written correctly. Finally, I ensure the script writes `submission.csv` with the exact required columns and shape.'
- What this solution (achieved -8.82183) has done: 'I fix the TensorFlow/protobuf crash by pinning `protobuf` to the pure-Python implementation early and additionally forcing TensorFlow to use the legacy protobuf API via an env var that avoids the `MessageFactory.GetPrototype` issue in this Kaggle image. Then I make the data-prep one-hot column naming consistent and guaranteed (including “Unknown”), so train/test have identical feature columns without silent misalignment. Finally, I keep your model/training loop intact, but ensure predictions are numerically safe (positive confidence) and the submission is produced exactly matching `sample_submission.csv` with a `.csv` suffix.'
- What this solution (achieved -8.82183) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime/API (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation and using the legacy protobuf API before importing TensorFlow. I keep the model, loss, features, and training loop identical, only making the environment/compat changes needed so the notebook runs end-to-end. I also add a small safety fallback so if TensorFlow still fails to import for any reason, the script still produce a valid `submission.csv` using a simple baseline (score likely be worse, but it guarantees a CSV is written). No other score-affecting changes are introduced beyond restoring the intended TensorFlow pipeline.'
- What this solution (achieved -8.82183) has done: 'I fix the TensorFlow import crash by setting the correct protobuf environment variables *before* importing TensorFlow (the current `PROTOCOL_BUFFERS_PYTHON_LEGACY_API` env var name is wrong, so the legacy API isn’t enabled and TF still crashes). I keep your model, loss, features, and training loop intact; the only behavioral change is restoring the intended TF path so the learned model can run (this should improve score from the fallback baseline toward the target). I also make the TF-availability check robust by attempting the import again after setting env vars, and I keep the existing fallback submission writer as a safety net. Finally, I ensure a valid `submission.csv` is always written with the required columns and correct row alignment to `sample_submission.csv`.'
- What this solution (achieved -8.82183) has done: 'I fix the TensorFlow/protobuf crash by setting the correct legacy protobuf environment variable **before** importing TensorFlow (the current code sets a non-existent env var, so TF still hits `MessageFactory.GetPrototype`). I also make the TF import block robust by retrying after setting env vars and only falling back to the baseline writer if TF truly cannot import. These changes are execution-unblocking and restore your intended training/inference path, which should improve score from the fallback-level behavior toward your target without changing the model, loss, features, or training loop. Finally, I keep the submission-writing logic unchanged except to guarantee a valid `submission.csv` is produced in all cases.'
- What this solution (achieved -8.82183) has done: 'I fix the TensorFlow/protobuf import crash by setting the correct legacy protobuf environment variable (`PROTOCOL_BUFFERS_PYTHON_LEGACY_API=1`) before any TensorFlow-related import and by making the TF import retry logic stop cleanly without leaving the notebook in a broken state. Then I keep your existing data prep, model, loss, and training loop unchanged, only ensuring train/test feature columns are consistently present and ordered (already mostly done) so inference can’t misalign. Finally, I guarantee that a valid `submission.csv` is always written (either from the TF model path or the existing fallback), with exact required columns and row alignment to `sample_submission.csv`. This should restore the intended TF training/inference path, which is the minimal change likely to improve score from -8.82 toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C_EXTENSIONS"] = "1"

os.environ["PROTOCOL_BUFFERS_PYTHON_LEGACY_API"] = "1"

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



## === cell 1
import random
import numpy as np
import pandas as pd

TF_AVAILABLE = True
tf = None
K = None
L = None

try:
    import tensorflow as tf  # noqa: E402
    from tensorflow import keras as K  # noqa: E402
    from tensorflow.keras import layers as L  # noqa: E402
except Exception as e1:
    try:
        os.environ["PROTOCOL_BUFFERS_PYTHON_LEGACY_API"] = "1"
        import tensorflow as tf  # noqa: E402
        from tensorflow import keras as K  # noqa: E402
        from tensorflow.keras import layers as L  # noqa: E402
    except Exception as e2:
        TF_AVAILABLE = False
        print(
            "WARNING: TensorFlow failed to import; will use fallback submission writer."
        )
        print("TensorFlow import error:", repr(e2))

import pickle  # preserved from original code (not required for external files)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def seed_all(seed=20):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


seed_all(20)



## === cell 3
RAW_TEST_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv"
RAW_TRAIN_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv"
SAMPLE_SUB_PATH = (
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

raw_test = pd.read_csv(RAW_TEST_PATH)
raw_train = pd.read_csv(RAW_TRAIN_PATH)
X_prediction = pd.read_csv(SAMPLE_SUB_PATH)



## === cell 4
ID = "Patient_Week"
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.8

SELECTED_COLUMNS = [
    "Weeks",
    "Base_week",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]



## === cell 5
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        if "sparse_output" in kwargs:
            try:
                super().__init__(**kwargs)
            except TypeError:
                kwargs["sparse"] = kwargs.pop("sparse_output")
                super().__init__(**kwargs)
        else:
            super().__init__(**kwargs)
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


class data_preparation:
    def __init__(self):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder(sparse_output=True, handle_unknown="ignore")

    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)

        data["Sex"] = data["Sex"].fillna("Unknown")
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
        except (NotFittedError, AttributeError):
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
        return data




## === cell 6
base = (
    raw_train.loc[raw_train["Weeks"] == 0, ["Patient", "Weeks", "FVC", "Percent"]]
    .rename(
        columns={"Weeks": "Base_week", "FVC": "Base_FVC", "Percent": "Base_percent"}
    )
    .drop_duplicates("Patient")
)

train = raw_train.merge(base, on="Patient", how="left")

missing_base = train["Base_week"].isna()
if missing_base.any():
    earliest = (
        raw_train.sort_values(["Patient", "Weeks"])
        .groupby("Patient", as_index=False)
        .first()[["Patient", "Weeks", "FVC", "Percent"]]
        .rename(
            columns={"Weeks": "Base_week", "FVC": "Base_FVC", "Percent": "Base_percent"}
        )
    )
    train = train.drop(columns=["Base_week", "Base_FVC", "Base_percent"]).merge(
        earliest, on="Patient", how="left"
    )

train = train[
    [
        "Patient",
        "Weeks",
        "FVC",
        "Base_week",
        "Base_FVC",
        "Base_percent",
        "Age",
        "Sex",
        "SmokingStatus",
    ]
].reset_index(drop=True)

data_prep = data_preparation()
train = data_prep(train)

for c in ["_Currently smokes", "_Ex-smoker", "_Never smoked", "_Unknown"]:
    if c not in train.columns:
        train[c] = 0



## === cell 7
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

rename_cols = {
    "Weeks_y": "Base_week",
    "Weeks_x": "Weeks",
    "Percent": "Base_percent",
    "FVC": "Base_FVC",
}

X_prediction = (
    X_prediction.merge(raw_test, how="left", left_on="Patient", right_on="Patient")
    .rename(columns=rename_cols)[
        [
            "Patient",
            "Base_week",
            "Base_FVC",
            "Base_percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ]
    ]
    .reset_index(drop=True)
)

X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)

for c in ["_Currently smokes", "_Ex-smoker", "_Never smoked", "_Unknown"]:
    if c not in X_prediction.columns:
        X_prediction[c] = 0

train = train.copy()
X_prediction = X_prediction.copy()



## === cell 8
if not TF_AVAILABLE:
    sample = pd.read_csv(SAMPLE_SUB_PATH)[["Patient_Week"]]
    pred = X_prediction[["Patient_Week", "Base_FVC"]].copy()
    pred["FVC"] = pred["Base_FVC"].astype(np.float32)
    pred["Confidence"] = np.float32(200.0)

    sub = sample.merge(
        pred[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
    )
    sub["FVC"] = sub["FVC"].fillna(sub["FVC"].median()).astype(np.float32)
    sub["Confidence"] = sub["Confidence"].fillna(np.float32(200.0)).astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", sub.shape)
    print(sub.head())



## === cell 9
if TF_AVAILABLE:
    C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")

    def score(y_true, y_pred):
        y_true = tf.dtypes.cast(y_true, tf.float32)
        y_pred = tf.dtypes.cast(y_pred, tf.float32)
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
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(
                y_true, y_pred
            )

        return loss




## === cell 10
if TF_AVAILABLE:

    def create_model():
        model_input = K.Input(shape=(9,))
        x = L.Dense(500, activation="relu")(model_input)
        x = L.Dense(200, activation="relu")(x)
        x = L.Dense(100, activation="relu")(x)
        FVC = L.Dense(3, activation="relu")(x)

        model = K.Model(inputs=model_input, outputs=[FVC])
        model.compile(optimizer=K.optimizers.RMSprop(), loss=mloss(LAMBDA_LOSS))
        return model

    model = create_model()
    model.summary()



## === cell 11
if TF_AVAILABLE:
    for c in SELECTED_COLUMNS:
        if c not in train.columns:
            train[c] = 0
        if c not in X_prediction.columns:
            X_prediction[c] = 0

    X_train = train[SELECTED_COLUMNS].astype(np.float32)
    y_train = train[["FVC"]].astype(np.float32)

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=200,
        verbose=0,
    )



## === cell 12
if TF_AVAILABLE:
    y_prediction = model.predict(
        X_prediction[SELECTED_COLUMNS].astype(np.float32), verbose=0
    )

    conf = (y_prediction[:, 2] - y_prediction[:, 0]).astype(np.float32)
    conf = np.maximum(conf, 70.0)

    sub = pd.DataFrame(
        data={
            "Patient_Week": X_prediction["Patient_Week"].values,
            "FVC": y_prediction[:, 1].astype(np.float32),
            "Confidence": conf,
        }
    )

    sub = pd.read_csv(SAMPLE_SUB_PATH)[["Patient_Week"]].merge(
        sub, on="Patient_Week", how="left"
    )
    sub["FVC"] = sub["FVC"].fillna(sub["FVC"].median()).astype(np.float32)
    sub["Confidence"] = sub["Confidence"].fillna(70.0).astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", sub.shape)
    print(sub.head())



## === cell 13
if TF_AVAILABLE:
    import matplotlib.pyplot as plt

    plt.plot(history.history.get("loss", []))
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
