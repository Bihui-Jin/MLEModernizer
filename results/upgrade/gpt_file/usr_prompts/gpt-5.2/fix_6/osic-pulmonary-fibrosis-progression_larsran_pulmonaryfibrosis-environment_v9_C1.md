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

-6.8481

# 6. Current score

-24.65893

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.65902) has done: 'I fix the protobuf crash by pinning `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (this avoids the `MessageFactory.GetPrototype` incompatibility in this environment). Then I fix the train/test feature mismatch that caused `(None, 9)` vs `(None, 10)` by ensuring `Weeks_target` is never accidentally included and by explicitly selecting `FEATURE_COLS` after joins with proper suffix handling. Finally, I make the data generator robust to `np.save`/`np.load` producing plain numeric arrays (so it doesn’t create object arrays and shape drift), ensuring fold predictions are produced and `submission.csv` is always written in the correct format.'
- What this solution (achieved -24.65893) has done: 'I fix the immediate protobuf crash by setting the environment variable before importing TensorFlow/Keras and by forcing the pure-Python protobuf backend early, which avoids the `MessageFactory.GetPrototype` incompatibility. Then I make the training labels and loss consistent (your loss expects `[FVC_true, sigma_true?]` but you were passing a sliced view that didn’t match intent), without changing the model architecture or training loop. Finally, I adjust the prediction post-processing to use a safer ensemble reduction for `sigma` (mean instead of RMS) and keep confidence clipping aligned with the metric, which should materially improve score toward the target while preserving the core approach. The script always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

WANDB = False

TEST = True
DATA_GENERATOR = True
TRAIN_ON_BACKWARD_WEEKS = False
PSEUDO_TEST_PATIENTS = 0

DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using DATA_DIR:", DATA_DIR)
print(
    "Train CSV exists:",
    os.path.exists(TRAIN_CSV),
    "Test CSV exists:",
    os.path.exists(TEST_CSV),
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex"] = (df["Sex"] == "Male").astype("float32")

    for col, val in [
        ("Currently smokes", "Currently smokes"),
        ("Ex-smoker", "Ex-smoker"),
        ("Never smoked", "Never smoked"),
    ]:
        df[col] = (df["SmokingStatus"] == val).astype("float32")

    if "Weekdiff_target" not in df.columns:
        df["Weekdiff_target"] = 0.0

    num_cols = [
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
    for c in num_cols:
        df[c] = df[c].astype("float32")
    return df


FEATURE_COLS = [
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


def get_train_data(
    train_csv_path,
    pseudo_test_patients=0,
    input_normalization=True,
    train_on_backward_weeks=False,
):
    train = pd.read_csv(train_csv_path)
    train = _encode_features(train)

    if not train_on_backward_weeks:
        pass

    X = train[FEATURE_COLS].copy()

    if input_normalization:
        X["Weeks"] = X["Weeks"] / 100.0
        X["FVC"] = X["FVC"] / 5000.0
        X["Percent"] = X["Percent"] / 100.0
        X["Age"] = X["Age"] / 100.0
        X["Weekdiff_target"] = X["Weekdiff_target"] / 100.0

    data = {"input_features": X}

    y = np.zeros((len(train), 3), dtype="float32")
    y[:, 0] = train["FVC"].values
    y[:, 1] = 0.0
    y[:, 2] = train["FVC"].values

    labels = pd.DataFrame(y, columns=["y0", "y1", "y2"])
    return train, data, labels


def get_test_data(test_csv_path, input_normalization=True):
    test = pd.read_csv(test_csv_path)
    test = _encode_features(test)

    sub = pd.read_csv(SAMPLE_SUB_CSV)
    pw = sub["Patient_Week"].str.split("_", expand=True)
    sub_pat = pw[0].values
    sub_week = pw[1].astype(int).values

    base = test.set_index("Patient")

    expanded = pd.DataFrame({"Patient": sub_pat, "Weeks_target": sub_week})
    expanded = expanded.join(base, on="Patient", how="left")

    expanded["Weeks_target"] = expanded["Weeks_target"].astype("float32")
    expanded["Weeks"] = expanded["Weeks"].astype("float32")

    expanded["Weekdiff_target"] = expanded["Weeks_target"] - expanded["Weeks"]

    expanded["Weeks"] = expanded["Weeks_target"]
    expanded = expanded.drop(columns=["Weeks_target"])

    X = expanded.loc[:, FEATURE_COLS].copy()

    if input_normalization:
        X["Weeks"] = X["Weeks"] / 100.0
        X["FVC"] = X["FVC"] / 5000.0
        X["Percent"] = X["Percent"] / 100.0
        X["Age"] = X["Age"] / 100.0
        X["Weekdiff_target"] = X["Weekdiff_target"] / 100.0

    return X.reset_index(drop=True), sub.copy()


def get_pseudo_test_data(
    train_csv_path, pseudo_test_patients, input_normalization=True
):
    train = pd.read_csv(train_csv_path)
    pats = train["Patient"].unique()
    pseudo_pats = pats[:pseudo_test_patients]
    test_like = train[train["Patient"].isin(pseudo_pats)].copy()
    test_like = _encode_features(test_like)
    test_like["TargetFVC"] = test_like["FVC"]
    test_data = test_like[FEATURE_COLS].copy()
    if input_normalization:
        test_data["Weeks"] = test_data["Weeks"] / 100.0
        test_data["FVC"] = test_data["FVC"] / 5000.0
        test_data["Percent"] = test_data["Percent"] / 100.0
        test_data["Age"] = test_data["Age"] / 100.0
        test_data["Weekdiff_target"] = test_data["Weekdiff_target"] / 100.0
    return test_data.reset_index(drop=True), test_like.reset_index(drop=True)


def get_fold_indices(folds, train_df):
    n = len(train_df)
    fold_sizes = [n // folds] * folds
    for i in range(n % folds):
        fold_sizes[i] += 1
    pos = [0]
    for fs in fold_sizes:
        pos.append(pos[-1] + fs)
    return pos


def build_model(config):
    n_features = int(config["NUMBER_FEATURES"])
    hidden_layers = list(config["HIDDEN_LAYERS"])
    act = config["ACTIVATION_FUNCTION"]
    dropout_rate = float(config["DROP_OUT_RATE"])
    dropout_layers = set(config["DROP_OUT_LAYERS"])
    l2_reg = bool(config["L2_REGULARIZATION"])
    reg_c = float(config["REGULARIZATION_CONSTANT"])

    inputs = keras.Input(shape=(n_features,), name="input_features")
    x = inputs

    for i, units in enumerate(hidden_layers):
        if l2_reg:
            x = keras.layers.Dense(
                units, activation=act, kernel_regularizer=keras.regularizers.l2(reg_c)
            )(x)
        else:
            x = keras.layers.Dense(units, activation=act)(x)
        if dropout_rate > 0 and i in dropout_layers:
            x = keras.layers.Dropout(dropout_rate)(x)

    pred = keras.layers.Dense(1, name="pred")(x)
    sigma = keras.layers.Dense(1, activation="softplus", name="sigma")(x)
    out = keras.layers.Concatenate(name="out")([pred, sigma])

    model = keras.Model(inputs=inputs, outputs=out)

    def laplace_nll(y_true_full, y_pred):
        fvc_true = y_true_full[:, 0]
        fvc_pred = y_pred[:, 0]
        sigma_pred = y_pred[:, 1]
        sigma_clip = tf.maximum(sigma_pred, 70.0)
        delta = tf.minimum(tf.abs(fvc_true - fvc_pred), 1000.0)
        sq2 = tf.sqrt(2.0)
        return tf.reduce_mean(
            (sq2 * delta) / sigma_clip + tf.math.log(sq2 * sigma_clip)
        )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(config["MAX_LEARNING_RATE"])
        ),
        loss=laplace_nll,
    )
    return model




## === cell 2
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



## === cell 3
if TEST:
    test_data, submission = get_test_data(TEST_CSV, INPUT_NORMALIZATION)

train, data, labels = get_train_data(
    TRAIN_CSV, PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION, TRAIN_ON_BACKWARD_WEEKS
)

if PSEUDO_TEST_PATIENTS > 0:
    test_data, test_check = get_pseudo_test_data(
        TRAIN_CSV, PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION
    )

print("train shape:", train.shape, "labels shape:", labels.shape)
if TEST:
    print("test_data shape:", test_data.shape, "submission shape:", submission.shape)
    print("test_data columns:", list(test_data.columns))
    assert (
        test_data.shape[1] == NUMBER_FEATURES
    ), f"Expected {NUMBER_FEATURES} features, got {test_data.shape[1]}"



## === cell 4
model = build_model(config)
model.summary()



## === cell 5
fold_pos = get_fold_indices(FOLDS, train)
print("fold positions:", fold_pos, "n_train:", len(train))




## === cell 6
class DataGenerator(keras.utils.Sequence):
    def __init__(
        self, list_IDs, config, validation=False, number_of_labels=3, shuffle=True
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
        self.label_size = int(number_of_labels)
        self.normalized = bool(config["INPUT_NORMALIZATION"])
        self.correlated = bool(config["GAUSSIAN_NOISE_CORRELATED"])

        self.data_arr = np.load("./train_data.npy").astype("float32")
        self.lab_arr = np.load("./train_labels.npy").astype("float32")

        if self.data_arr.ndim != 2 or self.data_arr.shape[1] != self.number_features:
            raise ValueError(
                f"train_data.npy has shape {self.data_arr.shape}, expected (*, {self.number_features}). "
                "This usually indicates FEATURE_COLS mismatch."
            )
        if self.lab_arr.ndim != 2 or self.lab_arr.shape[1] != self.label_size:
            raise ValueError(
                f"train_labels.npy has shape {self.lab_arr.shape}, expected (*, {self.label_size})."
            )

        self.on_epoch_end()

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
            X[i] = self.data_arr[ID]
            y[i] = self.lab_arr[ID]

        if self.gauss_std > 0:
            gauss_X = np.random.normal(0, self.gauss_std, size=self.batch_size).astype(
                "float32"
            )
            if self.correlated:
                gauss_y = gauss_X
            else:
                gauss_y = np.random.normal(
                    0, self.gauss_std, size=self.batch_size
                ).astype("float32")

            if self.normalized:
                gauss_X = gauss_X / 5000.0

            X[:, 1] += gauss_X
            y[:, 2] += gauss_X * (5000.0 if self.normalized else 1.0)
            y[:, 0] += gauss_y

        return X, y




## === cell 7
if DATA_GENERATOR:
    train_data = train[FEATURE_COLS]
    train_labels = labels
    np.save("train_data.npy", train_data.to_numpy(dtype="float32"))
    np.save("train_labels.npy", train_labels.to_numpy(dtype="float32"))
    print(
        "Saved train_data.npy:",
        train_data.shape,
        "train_labels.npy:",
        train_labels.shape,
    )



## === cell 8
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
        model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            verbose=0,
            callbacks=callbacks,
        )
    else:
        model.fit(
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
        preds_fold = model.predict(
            test_data.to_numpy(dtype="float32"), batch_size=256, verbose=0
        )
        predictions.append(preds_fold)

print(
    "Generated fold predictions:",
    len(predictions),
    "each of shape:",
    predictions[0].shape if predictions else None,
)



## === cell 9
if TEST:
    if len(predictions) == 0:
        raise RuntimeError(
            "No fold predictions were generated; cannot write submission."
        )

    preds = np.stack(predictions, axis=0)  # (folds, n, 2)

    if PREDICT_SLOPE:
        preds = np.mean(preds, axis=0)
        fvc_pred = (
            test_data["FVC"].to_numpy()
            + preds[:, 0] * test_data["Weekdiff_target"].to_numpy()
        )
        conf_pred = np.abs(preds[:, 1] * test_data["Weekdiff_target"].to_numpy())
    else:
        fvc_pred = np.mean(preds[:, :, 0], axis=0)
        sigma_pred = np.maximum(preds[:, :, 1], 0.0)

        conf_pred = np.mean(sigma_pred, axis=0)

    fvc_pred = np.clip(fvc_pred, 0.0, 10000.0)
    conf_pred = np.clip(conf_pred, 70.0, 3000.0)

    submission = submission.copy()
    submission["FVC"] = fvc_pred.astype(np.float32)
    submission["Confidence"] = conf_pred.astype(np.float32)

    submission = submission[["Patient_Week", "FVC", "Confidence"]]
    submission.to_csv("submission.csv", index=False)

    print("Wrote submission.csv with shape:", submission.shape)
    print(submission.head())



## === cell 10
backup = predictions



## === cell 11
import matplotlib.pyplot as plt
from scipy.stats import gmean

if PSEUDO_TEST_PATIENTS > 0:
    result = []
    backup_arr = np.stack(backup, axis=0)
    for i in range(-20, 20):
        preds = backup_arr.copy()
        preds[:, :, 1] = np.maximum(preds[:, :, 1], 0.0)

        if i == 0:
            preds[:, :, 1] = gmean(np.maximum(preds[:, :, 1], 1e-6), axis=0)
            preds = np.mean(preds, axis=0)
        else:
            preds[:, :, 1] = np.power(np.maximum(preds[:, :, 1], 1e-6), i)
            preds = np.mean(preds, axis=0)
            preds[:, 1] = np.power(np.maximum(preds[:, 1], 1e-6), 1 / i)

        FVC_true = test_check["TargetFVC"].values
        FVC_pred = preds[:, 0]
        sigma = preds[:, 1]

        sigma_clip = np.maximum(np.abs(sigma), 70)
        delta = np.abs(FVC_true - FVC_pred)
        delta = np.minimum(delta, 1000)

        sq2 = np.sqrt(2)
        loss = (delta / sigma_clip) * sq2 + np.log(sigma_clip * sq2)
        result.append(np.mean(loss))

    plt.plot(np.arange(-20, 20), result)
    plt.show()
