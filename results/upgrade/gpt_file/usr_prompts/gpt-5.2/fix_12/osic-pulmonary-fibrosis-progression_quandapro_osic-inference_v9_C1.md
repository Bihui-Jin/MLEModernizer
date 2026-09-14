# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    BatchNormalization,
    Activation,
    MaxPool3D,
    Add,
    GlobalAveragePooling3D,
    Concatenate,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
MIN_MAX = {
    "Weeks": (-5.0, 133.0),
    "FVC": (827.0, 6399.0),
    "Percent": (28.877577, 153.145378),
    "Age": (49.0, 88.0),
    "typical_fvc": (827.0, 6399.0),
}



## === cell 2
IMG_SIZE = 128
NUM_OF_SCANS = 12
BATCH_SIZE = 4  # as provided
EPOCHS = 3

INPUT_ROOT = "../input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(INPUT_ROOT, "train.csv")
TEST_CSV = os.path.join(INPUT_ROOT, "test.csv")
SAMPLE_CSV = os.path.join(INPUT_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT, "train")
TEST_IMG_DIR = os.path.join(INPUT_ROOT, "test")

TRAIN_DF = pd.read_csv(TRAIN_CSV)
TEST_DF = pd.read_csv(TEST_CSV)
SAMPLE_SUBMISSION = pd.read_csv(SAMPLE_CSV)

training_features = [
    "Weeks",
    "min_week",
    "min_week_FVC",
    "typical_fvc",
    "Age",
    "Male",
    "Female",
    "Never smoked",
    "Currently smokes",
    "Ex-smoker",
]
num_of_features = len(training_features)

print(
    "train/test/sample shapes:", TRAIN_DF.shape, TEST_DF.shape, SAMPLE_SUBMISSION.shape
)




## === cell 3
def create_typical_fvc(df):
    df = df.copy()
    df["typical_fvc"] = df["FVC"] / df["Percent"] * 100.0
    return df


def normalize(df):
    df = df.copy()
    for feature, min_max in MIN_MAX.items():
        df[feature] = (df[feature] - min_max[0]) / (min_max[1] - min_max[0])
    return df


def add_onehots_and_minweek(df):
    df = df.copy()
    df = create_typical_fvc(df)

    df["Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype("uint8")
    df["Currently smokes"] = (df["SmokingStatus"] == "Currently smokes").astype("uint8")
    df["Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype("uint8")

    df["Male"] = (df["Sex"] == "Male").astype("uint8")
    df["Female"] = (df["Sex"] == "Female").astype("uint8")

    g = df.groupby("Patient", sort=False)
    min_week = g["Weeks"].min().rename("min_week")

    df2 = df.merge(min_week, on="Patient", how="left")
    df2["_is_min_week"] = (df2["Weeks"] == df2["min_week"]).astype("uint8")
    minweek_rows = df2[df2["_is_min_week"] == 1].drop_duplicates(
        "Patient", keep="first"
    )
    min_week_FVC = minweek_rows.set_index("Patient")["FVC"].rename("min_week_FVC")
    df2 = df2.merge(min_week_FVC, on="Patient", how="left")
    df2.drop(columns=["_is_min_week"], inplace=True)

    df2 = normalize(df2)
    return df2


TRAIN_DF_PROC = add_onehots_and_minweek(TRAIN_DF)
TEST_DF_PROC = add_onehots_and_minweek(TEST_DF)

print("Processed shapes:", TRAIN_DF_PROC.shape, TEST_DF_PROC.shape)




## === cell 4
def build_placeholder_volumes(patients, num_scans=NUM_OF_SCANS, img_size=IMG_SIZE):
    zero_vol = np.zeros((num_scans, img_size, img_size), dtype=np.uint8)
    return {pid: zero_vol for pid in patients}


train_patients = TRAIN_DF_PROC["Patient"].unique().tolist()
test_patients = TEST_DF_PROC["Patient"].unique().tolist()

train_volumes = build_placeholder_volumes(train_patients)
test_volumes = build_placeholder_volumes(test_patients)

print("Using placeholder volumes:", len(train_volumes), len(test_volumes))




## === cell 5
def make_train_tfdata(df_proc, batch_size=BATCH_SIZE, shuffle=True):
    tabs = df_proc[training_features].to_numpy(dtype=np.float32, copy=False)
    y = df_proc["FVC"].to_numpy(dtype=np.float32, copy=False)
    y3 = np.stack([y, y, y], axis=1).astype(np.float32, copy=False)

    img0 = np.zeros((NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((tabs, y3))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(df_proc), seed=SEED, reshuffle_each_iteration=True
        )

    def _map(tab, y3_):
        return (img0, tab), y3_

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_test_tfdata(sample_sub, test_df_proc, batch_size=BATCH_SIZE):
    sample = sample_sub.reset_index(drop=True)

    pw = sample["Patient_Week"].astype(str)
    split = pw.str.split("_", n=1, expand=True)
    patients = split[0].to_numpy()
    weeks_raw = split[1].astype(np.float32).to_numpy()

    cols = training_features[1:]
    tab_per_patient = test_df_proc.drop_duplicates("Patient", keep="first")[
        ["Patient"] + cols
    ].reset_index(drop=True)
    patient_to_row = {p: i for i, p in enumerate(tab_per_patient["Patient"].to_numpy())}
    tab_matrix = tab_per_patient[cols].to_numpy(dtype=np.float32, copy=False)
    tab_row_idx = np.fromiter(
        (patient_to_row[p] for p in patients),
        dtype=np.int32,
        count=len(patients),
    )

    w0 = MIN_MAX["Weeks"][0]
    wden = MIN_MAX["Weeks"][1] - MIN_MAX["Weeks"][0]
    weeks_norm = (weeks_raw - w0) / wden

    tabulars = np.empty((len(sample), num_of_features), dtype=np.float32)
    tabulars[:, 0] = weeks_norm
    tabulars[:, 1:] = tab_matrix[tab_row_idx]

    img0 = np.zeros((NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices(tabulars)

    def _map(tab):
        return (img0, tab)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 6
def swish(x):
    return x * K.sigmoid(x)


def conv_block(x, num_of_filters):
    x = Conv3D(
        num_of_filters,
        kernel_size=(3, 1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(
        num_of_filters,
        kernel_size=(1, 3, 3),
        padding="same",
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(
        num_of_filters,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def residual_block(x, num_of_filters):
    x1 = conv_block(x, num_of_filters)
    x2 = conv_block(x1, num_of_filters)
    return Add()([x1, x2])


def build_3d_resnet(input_tensor):
    c1 = Conv3D(64, kernel_size=(5, 7, 7), strides=(1, 2, 2), padding="same")(
        input_tensor
    )
    b1 = BatchNormalization()(c1)
    a1 = Activation("relu")(b1)
    p1 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(a1)

    r1 = residual_block(p1, 64)
    r1 = residual_block(r1, 64)
    p1 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(r1)

    r2 = residual_block(p1, 128)
    r2 = residual_block(r2, 128)
    p2 = MaxPool3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(r2)

    r3 = residual_block(p2, 256)
    r3 = residual_block(r3, 256)

    return r3


def build_model():
    input_img = Input(shape=(NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1))
    x = build_3d_resnet(input_img)
    input_latent = GlobalAveragePooling3D()(x)

    input_tabular = Input(shape=(num_of_features,))
    x = Concatenate()([input_latent, input_tabular])
    x = Dense(100)(x)
    x = Dense(100)(x)
    out = Dense(3)(x)

    model = Model(inputs=[input_img, input_tabular], outputs=out, name="tabular_model")
    return model




## === cell 7
model = build_model()

model.compile(optimizer=Adam(1e-3), loss="mse", jit_compile=True)

train_ds = make_train_tfdata(TRAIN_DF_PROC, batch_size=BATCH_SIZE, shuffle=True)
steps_per_epoch = int(np.ceil(len(TRAIN_DF_PROC) / BATCH_SIZE))

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)

gc.collect()



## === cell 8
test_ds = make_test_tfdata(SAMPLE_SUBMISSION, TEST_DF_PROC, batch_size=BATCH_SIZE)

predictions = model.predict(
    test_ds,
    verbose=1,
    steps=int(np.ceil(len(SAMPLE_SUBMISSION) / BATCH_SIZE)),
)

if predictions.shape[0] != len(SAMPLE_SUBMISSION):
    raise RuntimeError(
        f"Prediction rows {predictions.shape[0]} != sample rows {len(SAMPLE_SUBMISSION)}"
    )




## === cell 9
def denormalize(y):
    return y * (MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]) + MIN_MAX["FVC"][0]


predictions = denormalize(predictions.astype(np.float32))
FVC = predictions[:, 1]
Confidence = predictions[:, 2] - predictions[:, 0]
Confidence = np.maximum(np.abs(Confidence), 70.0)

SAMPLE_SUBMISSION["FVC"] = np.round(FVC).astype(int)
SAMPLE_SUBMISSION["Confidence"] = np.round(Confidence).astype(int)

SAMPLE_SUBMISSION = SAMPLE_SUBMISSION[["Patient_Week", "FVC", "Confidence"]]
SAMPLE_SUBMISSION.to_csv("submission.csv", index=False)

print(SAMPLE_SUBMISSION.head())
print("Wrote submission.csv with shape:", SAMPLE_SUBMISSION.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
