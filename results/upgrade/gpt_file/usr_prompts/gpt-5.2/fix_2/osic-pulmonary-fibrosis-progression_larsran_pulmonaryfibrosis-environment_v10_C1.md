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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0
tqdm==4.67.1
wandb==0.21.0

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

-6.8539

# 6. Current score

-9.32628

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -9.32628) has done: 'I remove the W&B and kaggle_secrets dependency (it’s failing due to unauthenticated secrets access) and make the imports robust to the Kaggle environment (including the protobuf/wandb crash). I also provide a minimal in-notebook `pfutils` fallback implementation (same semantics: tabular features + dense network predicting FVC and sigma) so the notebook runs even if `pfutils` isn’t available. I fix a few logic/runtime issues that currently prevent training/inference (cell numbering, Keras import usage, generator bug where Gaussian std becomes boolean, and fold index boundaries). Finally, I ensure a correctly formatted `submission.csv` is always written using `sample_submission.csv` for the required Patient_Week rows.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tqdm import tqdm

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

WANDB = False
TEST = True
DATA_GENERATOR = True
TRAIN_ON_BACKWARD_WEEKS = True
PSEUDO_TEST_PATIENTS = 0

BASE_INPUT = "../input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_INPUT, "sample_submission.csv")


def _fallback_pfutils():
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Sex"] = (out["Sex"].astype(str).str.lower() == "male").astype("float32")
        sm = out["SmokingStatus"].astype(str)
        out["Currently smokes"] = (sm == "Currently smokes").astype("float32")
        out["Ex-smoker"] = (sm == "Ex-smoker").astype("float32")
        out["Never smoked"] = (sm == "Never smoked").astype("float32")
        for c in ["Weeks", "FVC", "Percent", "Age"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
        return out

    def get_test_data(test_csv_path, input_normalization=True):
        test = pd.read_csv(test_csv_path)
        sample = pd.read_csv(SAMPLE_SUB_CSV)

        pw = sample["Patient_Week"].astype(str).str.split("_", expand=True)
        patient = pw[0].values
        week = pw[1].astype(int).values.astype("float32")

        base = test.groupby("Patient").first().reset_index()
        base = _preprocess(base)

        base_map = base.set_index("Patient")
        feat_rows = []
        for p, w in zip(patient, week):
            r = base_map.loc[p]
            weekdiff = w - float(r["Weeks"])
            feat_rows.append(
                [
                    w,
                    float(r["FVC"]),
                    float(r["Percent"]),
                    float(r["Age"]),
                    float(r["Sex"]),
                    float(r["Currently smokes"]),
                    float(r["Ex-smoker"]),
                    float(r["Never smoked"]),
                    float(weekdiff),
                ]
            )
        cols = [
            "Weeks",
            "FVC",
            "Percent",
            "Age",
            "Sex",
            "Currently smokes",
            "Ex-smoker",
            "Never smoked",
            "Weekdiff_target",
        ]
        test_data = pd.DataFrame(feat_rows, columns=cols)

        if input_normalization:
            test_data["Weeks"] = test_data["Weeks"] / 100.0
            test_data["FVC"] = test_data["FVC"] / 5000.0
            test_data["Percent"] = test_data["Percent"] / 100.0
            test_data["Age"] = test_data["Age"] / 100.0
            test_data["Weekdiff_target"] = test_data["Weekdiff_target"] / 100.0

        return test_data, sample

    def get_train_data(
        train_csv_path,
        pseudo_test_patients,
        input_normalization=True,
        train_on_backward_weeks=True,
    ):
        df = pd.read_csv(train_csv_path)
        df = _preprocess(df)

        base = (
            df.sort_values(["Patient", "Weeks"])
            .groupby("Patient")
            .first()
            .reset_index()
        )
        base = base[
            [
                "Patient",
                "Weeks",
                "FVC",
                "Percent",
                "Age",
                "Sex",
                "Currently smokes",
                "Ex-smoker",
                "Never smoked",
            ]
        ].rename(columns={"Weeks": "BaseWeeks", "FVC": "BaseFVC"})
        merged = df.merge(base, on="Patient", how="left")

        merged["Weekdiff_target"] = (merged["Weeks"] - merged["BaseWeeks"]).astype(
            "float32"
        )
        if not train_on_backward_weeks:
            merged = merged[merged["Weekdiff_target"] >= 0].copy()

        train = pd.DataFrame(
            {
                "Patient": merged["Patient"].values,
                "Weeks": merged["Weeks"].astype("float32").values,
                "FVC": merged["BaseFVC"].astype("float32").values,
                "Percent": (
                    merged["Percent_y"].astype("float32").values
                    if "Percent_y" in merged.columns
                    else merged["Percent"].astype("float32").values
                ),
                "Age": (
                    merged["Age_y"].astype("float32").values
                    if "Age_y" in merged.columns
                    else merged["Age"].astype("float32").values
                ),
                "Sex": (
                    merged["Sex_y"].astype("float32").values
                    if "Sex_y" in merged.columns
                    else merged["Sex"].astype("float32").values
                ),
                "Currently smokes": (
                    merged["Currently smokes_y"].astype("float32").values
                    if "Currently smokes_y" in merged.columns
                    else merged["Currently smokes"].astype("float32").values
                ),
                "Ex-smoker": (
                    merged["Ex-smoker_y"].astype("float32").values
                    if "Ex-smoker_y" in merged.columns
                    else merged["Ex-smoker"].astype("float32").values
                ),
                "Never smoked": (
                    merged["Never smoked_y"].astype("float32").values
                    if "Never smoked_y" in merged.columns
                    else merged["Never smoked"].astype("float32").values
                ),
                "Weekdiff_target": merged["Weekdiff_target"].astype("float32").values,
            }
        )

        y = pd.DataFrame(
            {
                0: merged["FVC"].astype("float32").values,  # target FVC
                1: np.full(len(merged), 200.0, dtype="float32"),  # initial sigma proxy
                2: merged["BaseFVC"]
                .astype("float32")
                .values,  # baseline fvc (used by generator noise code)
            }
        )

        if input_normalization:
            train["Weeks"] = train["Weeks"] / 100.0
            train["FVC"] = train["FVC"] / 5000.0
            train["Percent"] = train["Percent"] / 100.0
            train["Age"] = train["Age"] / 100.0
            train["Weekdiff_target"] = train["Weekdiff_target"] / 100.0

            y[0] = y[0] / 5000.0
            y[2] = y[2] / 5000.0

        data = {"input_features": train.drop(columns=["Patient"])}
        labels = y
        return train, data, labels

    def get_pseudo_test_data(
        train_csv_path, pseudo_test_patients, input_normalization=True
    ):
        raise NotImplementedError("Pseudo test not implemented in fallback.")

    def get_fold_indices(folds, train_df):
        pats = train_df["Patient"].astype(str).values
        unique_pats = pd.unique(pats)
        rng = np.random.RandomState(SEED)
        rng.shuffle(unique_pats)
        fold_ids = np.array_split(unique_pats, folds)

        fold_pos = [0]
        idx = np.arange(len(train_df))
        order = []
        for f in range(folds):
            sel = np.isin(pats, fold_ids[f])
            order.append(idx[sel])
            fold_pos.append(fold_pos[-1] + int(sel.sum()))
        return fold_pos

    def build_model(config):
        n = int(config["NUMBER_FEATURES"])
        hidden = config["HIDDEN_LAYERS"]
        act = config["ACTIVATION_FUNCTION"]
        dr = float(config["DROP_OUT_RATE"])

        inp = keras.Input(shape=(n,), name="input")
        x = inp
        for i, h in enumerate(hidden):
            x = layers.Dense(h, activation=act, name=f"dense_{i}")(x)
            if dr and (
                ("DROP_OUT_LAYERS" not in config)
                or (i in config.get("DROP_OUT_LAYERS", []))
            ):
                x = layers.Dropout(dr, name=f"dropout_{i}")(x)

        out = layers.Dense(2, activation=None, name="out")(x)

        model = keras.Model(inp, out)

        def laplace_nll(y_true, y_pred):
            fvc_true = y_true[:, 0]
            fvc_pred = y_pred[:, 0]
            sigma = tf.math.abs(y_pred[:, 1]) + 1e-3  # keep positive
            if bool(config.get("OUTPUT_NORMALIZATION", True)):
                fvc_true_ml = fvc_true * 5000.0
                fvc_pred_ml = fvc_pred * 5000.0
            else:
                fvc_true_ml = fvc_true
                fvc_pred_ml = fvc_pred

            sigma_ml = sigma
            sigma_clip = tf.maximum(sigma_ml, 70.0)
            delta = tf.minimum(tf.abs(fvc_true_ml - fvc_pred_ml), 1000.0)
            sq2 = tf.sqrt(2.0)
            metric = (sq2 * delta) / sigma_clip + tf.math.log(sq2 * sigma_clip)
            return tf.reduce_mean(metric)

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=float(config["MAX_LEARNING_RATE"])
            ),
            loss=laplace_nll,
        )
        return model

    def get_cosine_annealing_lr_callback(*args, **kwargs):
        return None

    return (
        get_test_data,
        get_train_data,
        get_pseudo_test_data,
        build_model,
        get_cosine_annealing_lr_callback,
        get_fold_indices,
    )


try:
    from pfutils import (
        get_test_data,
        get_train_data,
        get_pseudo_test_data,
        build_model,
        get_cosine_annealing_lr_callback,
        get_fold_indices,
    )
except Exception as e:
    (
        get_test_data,
        get_train_data,
        get_pseudo_test_data,
        build_model,
        get_cosine_annealing_lr_callback,
        get_fold_indices,
    ) = _fallback_pfutils()

print(
    "Setup complete. Using pfutils:",
    "external" if "pfutils" in sys.modules else "fallback",
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if TEST:
    PSEUDO_TEST_PATIENTS = 0



## === cell 2
WANDB = False



## === cell 3
FOLDS = 10
BATCH_SIZE = 128
NUMBER_FEATURES = 9
HIDDEN_LAYERS = [64, 64]
PREDICT_SLOPE = False

USE_GAUSSIAN_ON_FVC = True
VALUE_GAUSSIAN_NOISE_ON_FVC = 70
GAUSSIAN_NOISE_CORRELATED = False

ACTIVATION_FUNCTION = "swish"

DROP_OUT_RATE = 0
DROP_OUT_LAYERS = []

EPOCHS = 100
STEPS_PER_EPOCH = 100

L2_REGULARIZATION = False
REGULARIZATION_CONSTANT = 0.005

INPUT_NORMALIZATION = True
OUTPUT_NORMALIZATION = True

MAX_LEARNING_RATE = 5e-4
COSINE_CYCLES = 10

MODEL_NAME = "BaselineSubmission"

config = dict(
    NUMBER_FEATURES=NUMBER_FEATURES,
    L2_REGULARIZATION=L2_REGULARIZATION,
    INPUT_NORMALIZATION=INPUT_NORMALIZATION,
    ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
    DROP_OUT_RATE=DROP_OUT_RATE,
    OUTPUT_NORMALIZATION=OUTPUT_NORMALIZATION,
    EPOCHS=EPOCHS,
    STEPS_PER_EPOCH=STEPS_PER_EPOCH,
    MAX_LEARNING_RATE=MAX_LEARNING_RATE,
    COSINE_CYCLES=COSINE_CYCLES,
    MODEL_NAME=MODEL_NAME,
    USE_GAUSSIAN_ON_FVC=USE_GAUSSIAN_ON_FVC,
    VALUE_GAUSSIAN_NOISE_ON_FVC=VALUE_GAUSSIAN_NOISE_ON_FVC,
    PREDICT_SLOPE=PREDICT_SLOPE,
    HIDDEN_LAYERS=HIDDEN_LAYERS,
    REGULARIZATION_CONSTANT=REGULARIZATION_CONSTANT,
    DROP_OUT_LAYERS=DROP_OUT_LAYERS,
    BATCH_SIZE=BATCH_SIZE,
    GAUSSIAN_NOISE_CORRELATED=GAUSSIAN_NOISE_CORRELATED,
)



## === cell 4
if TEST:
    test_data, submission = get_test_data(TEST_CSV, INPUT_NORMALIZATION)

train, data, labels = get_train_data(
    TRAIN_CSV, PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION, TRAIN_ON_BACKWARD_WEEKS
)

if PSEUDO_TEST_PATIENTS > 0:
    test_data, test_check = get_pseudo_test_data(
        TRAIN_CSV, PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION
    )

print("Train rows:", len(train), "Test rows:", len(test_data))
print("Test submission rows:", len(submission))



## === cell 5
model = build_model(config)
model.summary()




## === cell 6
def _simple_fold_positions(n, folds):
    sizes = [n // folds] * folds
    for i in range(n % folds):
        sizes[i] += 1
    pos = [0]
    for s in sizes:
        pos.append(pos[-1] + s)
    return pos


try:
    fold_pos = get_fold_indices(FOLDS, train)
    if len(fold_pos) != FOLDS + 1 or fold_pos[0] != 0 or fold_pos[-1] != len(train):
        fold_pos = _simple_fold_positions(len(train), FOLDS)
except Exception:
    fold_pos = _simple_fold_positions(len(train), FOLDS)

print("fold_pos:", fold_pos)




## === cell 7
class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        list_IDs,
        config,
        validation=False,
        number_of_labels=3,
        batch_size=128,
        shuffle=True,
    ):
        self.number_features = int(config["NUMBER_FEATURES"])
        self.use_gaussian = bool(config["USE_GAUSSIAN_ON_FVC"])
        self.gauss_std = (
            float(config["VALUE_GAUSSIAN_NOISE_ON_FVC"])
            if (self.use_gaussian and (not validation))
            else 0.0
        )
        self.list_IDs = list_IDs
        self.batch_size = int(config["BATCH_SIZE"])
        self.shuffle = shuffle
        self.on_epoch_end()
        self.label_size = number_of_labels
        self.normalized = bool(config["INPUT_NORMALIZATION"])
        self.correlated = bool(config["GAUSSIAN_NOISE_CORRELATED"])

        self.data = np.load("./train_data.npy", allow_pickle=True)
        self.lab = np.load("./train_labels.npy", allow_pickle=True)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.number_features), dtype="float32")
        y = np.empty((self.batch_size, self.label_size), dtype="float32")

        for i, ID in enumerate(list_IDs_temp):
            X[i] = np.asarray(self.data[ID], dtype="float32")
            y[i] = np.asarray(self.lab[ID], dtype="float32")

        if self.gauss_std > 0:
            gauss_X = np.random.normal(0, self.gauss_std, size=self.batch_size).astype(
                "float32"
            )
            gauss_y = (
                gauss_X
                if self.correlated
                else np.random.normal(0, self.gauss_std, size=self.batch_size).astype(
                    "float32"
                )
            )

            if self.normalized:
                gauss_X = gauss_X / 5000.0
                gauss_y = gauss_y / 5000.0

            X[:, 1] += gauss_X
            y[:, 2] += gauss_X
            y[:, 0] += gauss_y

        return X, y




## === cell 8
if DATA_GENERATOR:
    train_data = train[
        [
            "Weeks",
            "FVC",
            "Percent",
            "Age",
            "Sex",
            "Currently smokes",
            "Ex-smoker",
            "Never smoked",
            "Weekdiff_target",
        ]
    ]
    train_labels = labels
    np.save("train_data.npy", train_data.to_numpy(dtype="float32"))
    np.save("train_labels.npy", train_labels.to_numpy(dtype="float32"))
    print("Saved generator arrays:", train_data.shape, train_labels.shape)



## === cell 9
predictions = []

for fold in range(FOLDS):
    if DATA_GENERATOR:
        train_ID = list(range(fold_pos[0], fold_pos[fold])) + list(
            range(fold_pos[fold + 1], len(train))
        )
        val_ID = list(range(fold_pos[fold], fold_pos[fold + 1]))
        training_generator = DataGenerator(train_ID, config)
        validation_generator = DataGenerator(val_ID, config, validation=True)
    else:
        x_train = pd.concat(
            [
                data["input_features"][: fold_pos[fold]],
                data["input_features"][fold_pos[fold + 1] :],
            ],
            axis=0,
        )
        y_train = pd.concat(
            [labels[: fold_pos[fold]], labels[fold_pos[fold + 1] :]], axis=0
        )
        x_val = data["input_features"][fold_pos[fold] : fold_pos[fold + 1]]
        y_val = labels[fold_pos[fold] : fold_pos[fold + 1]]

    model = build_model(config)

    sv = tf.keras.callbacks.ModelCheckpoint(
        f"fold-{fold}.weights.h5",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    callbacks = [sv]

    print(fold + 1, "of", FOLDS)

    if DATA_GENERATOR:
        history = model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            verbose=0,
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            verbose=0,
            callbacks=callbacks,
        )

    if TEST or PSEUDO_TEST_PATIENTS > 0:
        model.load_weights(f"fold-{fold}.weights.h5")
        preds = model.predict(test_data, batch_size=256, verbose=0)
        predictions.append(preds)



## === cell 10
if TEST:
    predictions = np.array(predictions)  # (folds, n, 2)

    if PREDICT_SLOPE:
        predictions_mean = np.mean(predictions, axis=0)
        for i in range(len(test_data)):
            submission.loc[i, "FVC"] = float(
                test_data.loc[i, "FVC"]
                + predictions_mean[i, 0] * test_data.loc[i, "Weekdiff_target"]
            )
            submission.loc[i, "Confidence"] = float(
                abs(predictions_mean[i, 1] * test_data.loc[i, "Weekdiff_target"])
            )
    else:
        preds = np.abs(predictions)
        preds[:, :, 1] = np.power(preds[:, :, 1], 2)
        preds = np.mean(preds, axis=0)
        preds[:, 1] = np.power(preds[:, 1], 0.5)

        if OUTPUT_NORMALIZATION:
            fvc_out = preds[:, 0] * 5000.0
        else:
            fvc_out = preds[:, 0]

        for i in range(len(test_data)):
            submission.loc[i, "FVC"] = float(fvc_out[i])
            submission.loc[i, "Confidence"] = float(preds[i, 1])

    submission["FVC"] = (
        pd.to_numeric(submission["FVC"], errors="coerce").fillna(0).astype("float32")
    )
    submission["Confidence"] = (
        pd.to_numeric(submission["Confidence"], errors="coerce")
        .fillna(200)
        .astype("float32")
    )
    submission["Confidence"] = np.maximum(submission["Confidence"].values, 70.0).astype(
        "float32"
    )

    submission = submission[["Patient_Week", "FVC", "Confidence"]]
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission.shape)



## === cell 11
backup = predictions



## === cell 12
import matplotlib.pyplot as plt
from scipy.stats import gmean

if PSEUDO_TEST_PATIENTS > 0:
    result = []
    for i in range(-20, 20):
        predictions = np.abs(backup)
        if i == 0:
            predictions[:, :, 1] = gmean(predictions[:, :, 1], axis=0)
            predictions = np.mean(predictions, axis=0)
        else:
            predictions[:, :, 1] = np.power(predictions[:, :, 1], i)
            predictions = np.mean(predictions, axis=0)
            predictions[:, 1] = np.power(predictions[:, 1], 1 / i)

        FVC_true = test_check["TargetFVC"].values
        FVC_pred = predictions[:, 0]
        sigma = predictions[:, 1]

        sigma_clip = np.maximum(np.abs(sigma), 70)
        delta = np.abs(FVC_true - FVC_pred)
        delta = np.minimum(delta, 1000)

        sq2 = np.sqrt(2)
        loss = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
        result.append(np.mean(loss))
    plt.plot(np.arange(-20, 20), result)
    plt.show()
