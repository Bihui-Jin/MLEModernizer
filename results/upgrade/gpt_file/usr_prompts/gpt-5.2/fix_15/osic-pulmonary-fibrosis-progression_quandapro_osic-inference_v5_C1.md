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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import gc
import random
import numpy as np
import pandas as pd

import cv2

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

dicom = None
DICOM_AVAILABLE = False
try:
    import pydicom as dicom  # may crash due to protobuf mismatch

    DICOM_AVAILABLE = True
    print("INFO: DICOM_AVAILABLE=True (using pydicom).")
except Exception as e:
    dicom = None
    DICOM_AVAILABLE = False
    print("INFO: DICOM_AVAILABLE=False (no-CT fallback). Reason:", repr(e))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




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
NUM_OF_SCANS = 10
BATCH_SIZE = 8  # keep memory safe; does not change core logic
EPOCHS = 8  # keep identical to provided script
N_MODELS = 3  # keep identical to provided script

DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"

TRAIN_DF = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
TEST_DF = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
SAMPLE_SUBMISSION = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

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

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")




## === cell 3
def create_typical_fvc(df):
    df = df.copy()
    df["typical_fvc"] = df["FVC"] / df["Percent"] * 100.0
    return df


def normalize(df):
    df = df.copy()
    for feature, min_max in MIN_MAX.items():
        if feature in df.columns:
            df[feature] = (df[feature] - min_max[0]) / (min_max[1] - min_max[0])
    return df


def add_onehot(df):
    df = df.copy()
    df["Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype("uint8")
    df["Currently smokes"] = (df["SmokingStatus"] == "Currently smokes").astype("uint8")
    df["Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype("uint8")
    df["Male"] = (df["Sex"] == "Male").astype("uint8")
    df["Female"] = (df["Sex"] == "Female").astype("uint8")
    return df


TRAIN_DF = create_typical_fvc(TRAIN_DF)
TEST_DF = create_typical_fvc(TEST_DF)

TRAIN_DF = add_onehot(TRAIN_DF)
TEST_DF = add_onehot(TEST_DF)

for df in (TRAIN_DF, TEST_DF):
    g = df.groupby("Patient", sort=False)
    df["min_week"] = g["Weeks"].transform("min").astype(np.float32)
    idx = g["Weeks"].idxmin()
    min_fvc_map = df.loc[idx, ["Patient", "FVC"]].set_index("Patient")["FVC"]
    df["min_week_FVC"] = df["Patient"].map(min_fvc_map).astype(np.float32)

TRAIN_DF = normalize(TRAIN_DF)
TEST_DF = normalize(TEST_DF)

TRAIN_DF["FVC_norm"] = (TRAIN_DF["FVC"] - MIN_MAX["FVC"][0]) / (
    MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]
)
TRAIN_DF["y0"] = TRAIN_DF["FVC_norm"] - 0.05
TRAIN_DF["y1"] = TRAIN_DF["FVC_norm"]
TRAIN_DF["y2"] = TRAIN_DF["FVC_norm"] + 0.05

patients = TRAIN_DF["Patient"].unique()
train_pat, val_pat = train_test_split(patients, test_size=0.2, random_state=SEED)

train_idx = TRAIN_DF.index[TRAIN_DF["Patient"].isin(train_pat)].to_numpy()
val_idx = TRAIN_DF.index[TRAIN_DF["Patient"].isin(val_pat)].to_numpy()




## === cell 4
def get_pixels_hu(scan):
    image = scan.pixel_array
    image = image.astype(np.int16)

    slope = getattr(scan, "RescaleSlope", 1)
    intercept = getattr(scan, "RescaleIntercept", 0)
    window_center = -200
    window_width = 2000
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)

    image_min = window_center - window_width // 2
    image_max = window_center + window_width // 2
    image = np.clip(image, image_min, image_max).astype(np.float64)
    image = (image - image_min) / (image_max - image_min) * 255.0
    return image.astype(np.uint8)


def _list_dicom_files(image_folder):
    try:
        out = []
        with os.scandir(image_folder) as it:
            for e in it:
                if e.is_file():
                    name = e.name
                    if name.lower().endswith(".dcm"):
                        out.append(name)
        return out
    except Exception:
        return []


CACHE_DIR = os.path.join(".", "_ct_cache_npy")
os.makedirs(CACHE_DIR, exist_ok=True)


def _try_int_stem(filename: str):
    stem = os.path.splitext(os.path.basename(filename))[0]
    try:
        return int(stem)
    except Exception:
        return None


def _select_middle_files(files, num_scans):
    stems = [_try_int_stem(f) for f in files]
    if all(s is not None for s in stems):
        order = np.argsort(np.asarray(stems, dtype=np.int64), kind="mergesort")
        files = [files[i] for i in order]
    else:
        files = sorted(files)

    n = len(files)
    if n >= num_scans:
        mid = n // 2
        half = num_scans // 2
        start = max(0, mid - half)
        end = start + num_scans
        if end > n:
            end = n
            start = n - num_scans
        files = files[start:end]
    return files


def load_patient_volume(patient_id, folder, img_size=IMG_SIZE, num_scans=NUM_OF_SCANS):
    image_folder = os.path.join(folder, patient_id)
    if not DICOM_AVAILABLE:
        return np.zeros((num_scans, img_size, img_size), dtype="float32")

    cache_path = os.path.join(CACHE_DIR, f"{patient_id}_{img_size}_{num_scans}.npy")
    try:
        if os.path.exists(cache_path):
            arr = np.load(cache_path, mmap_mode="r")
            return np.asarray(arr, dtype="float32")
    except Exception:
        pass

    try:
        files = _list_dicom_files(image_folder)
        if not files:
            return np.zeros((num_scans, img_size, img_size), dtype="float32")

        files = _select_middle_files(files, num_scans)
        fps = [os.path.join(image_folder, f) for f in files]

        imgs = []
        for fp in fps:
            try:
                s = dicom.dcmread(
                    fp, force=True, stop_before_pixels=False, specific_tags=None
                )
                px = get_pixels_hu(s)
                imgs.append(
                    cv2.resize(px, (img_size, img_size), interpolation=cv2.INTER_AREA)
                )
            except Exception:
                return np.zeros((num_scans, img_size, img_size), dtype="float32")

        if len(imgs) < num_scans:
            if len(imgs) == 0:
                imgs = [np.zeros((img_size, img_size), dtype=np.uint8)]
            while len(imgs) < num_scans:
                imgs.append(imgs[-1])
            imgs = imgs[:num_scans]

        vol = np.asarray(imgs, dtype="float32")
        try:
            np.save(cache_path, vol)
        except Exception:
            pass
        return vol
    except Exception:
        return np.zeros((num_scans, img_size, img_size), dtype="float32")


all_patients = sorted(
    set(TRAIN_DF["Patient"].unique()).union(set(TEST_DF["Patient"].unique()))
)
pid_to_index = {pid: i for i, pid in enumerate(all_patients)}
index_to_pid = np.array(all_patients)

_PID_FOLDER = {}
for pid in all_patients:
    folder = (
        TRAIN_IMG_DIR
        if os.path.isdir(os.path.join(TRAIN_IMG_DIR, pid))
        else TEST_IMG_DIR
    )
    _PID_FOLDER[pid] = folder


def _build_one_patient(i_pid):
    i, pid = i_pid
    v = load_patient_volume(pid, _PID_FOLDER[pid])  # float32 [0..255]
    v = (v / 255.0).astype(np.float32, copy=False)
    return i, v


def build_all_volumes_array():
    vols = np.empty(
        (len(all_patients), NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32
    )
    if not DICOM_AVAILABLE:
        vols.fill(0.0)
        return vols

    for i, pid in enumerate(all_patients):
        v = load_patient_volume(pid, _PID_FOLDER[pid])
        vols[i, ..., 0] = (v / 255.0).astype(np.float32, copy=False)
    return vols


ALL_VOLUMES = build_all_volumes_array()
print("Patients (train+test):", len(all_patients), "DICOM_AVAILABLE:", DICOM_AVAILABLE)
gc.collect()




## === cell 5
TRAIN_PATIENTS_ARR = TRAIN_DF["Patient"].to_numpy()
TRAIN_PID_IDX_ARR = np.fromiter(
    (pid_to_index[p] for p in TRAIN_PATIENTS_ARR),
    dtype=np.int32,
    count=len(TRAIN_PATIENTS_ARR),
)

TRAIN_TAB_ARR = TRAIN_DF[training_features].astype("float32").to_numpy(copy=True)
TRAIN_Y_ARR = TRAIN_DF[["y0", "y1", "y2"]].astype("float32").to_numpy(copy=True)

TRAIN_VOL_ARR = ALL_VOLUMES[TRAIN_PID_IDX_ARR]  # shape [n_rows, S, H, W, 1]


def make_tf_dataset(indices, training: bool):
    indices = np.asarray(indices, dtype=np.int32)
    vol = TRAIN_VOL_ARR[indices]
    tab = TRAIN_TAB_ARR[indices]
    y = TRAIN_Y_ARR[indices]

    ds = tf.data.Dataset.from_tensor_slices(((vol, tab), y))
    if training:
        ds = ds.shuffle(
            buffer_size=len(indices), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 6
def swish(x):
    return x * K.sigmoid(x)


def conv_block(x, num_of_filters, num_of_layers=3):
    for i in range(num_of_layers):
        x = Conv3D(
            num_of_filters,
            kernel_size=(3, 3, 3),
            padding="same",
            kernel_initializer="he_uniform",
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = MaxPooling3D()(x)
    return x


def build_model():
    input_img = Input(shape=(NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1))
    x = conv_block(input_img, 16, 4)
    x = conv_block(x, 32, 4)
    x = conv_block(x, 64, 4)
    input_latent = GlobalAveragePooling3D()(x)

    input_tabular = Input(shape=(num_of_features,))
    x = Concatenate()([input_latent, input_tabular])
    x = Dense(100)(x)
    x = Dense(100)(x)
    out = Dense(3)(x)

    model = tf.keras.Model(
        inputs=[input_img, input_tabular], outputs=out, name="tabular_model"
    )
    return model


class InMemoryBestWeights(tf.keras.callbacks.Callback):
    def __init__(self, monitor="val_loss", mode="min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (
            (current < self.best) if self.mode == "min" else (current > self.best)
        )
        if improved:
            self.best = float(current)
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_one_model(seed_offset=0):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    model = build_model()
    model.compile(optimizer=Adam(1e-3), loss="mse")

    train_ds = make_tf_dataset(train_idx, training=True)
    val_ds = make_tf_dataset(val_idx, training=False)

    best_cb = InMemoryBestWeights(monitor="val_loss", mode="min")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[best_cb],
    )
    return model




## === cell 7
models = [train_one_model(i) for i in range(N_MODELS)]




## === cell 8
sub = SAMPLE_SUBMISSION.copy()
sub[["Patient", "Weeks_str"]] = sub["Patient_Week"].str.split("_", expand=True)
sub["Weeks"] = sub["Weeks_str"].astype("int32")
sub = sub.drop(columns=["Weeks_str"])

static_cols = [c for c in TEST_DF.columns if c in training_features and c != "Weeks"]
test_static = TEST_DF[["Patient"] + static_cols].drop_duplicates("Patient")
test_merge = sub.merge(test_static, on="Patient", how="left")

test_merge["Weeks"] = (test_merge["Weeks"].astype("float32") - MIN_MAX["Weeks"][0]) / (
    MIN_MAX["Weeks"][1] - MIN_MAX["Weeks"][0]
)

for col in training_features:
    if col not in test_merge.columns:
        test_merge[col] = 0.0

test_merge = test_merge.reset_index(drop=True)
TEST_PATIENTS_ARR = test_merge["Patient"].to_numpy()
TEST_PID_IDX_ARR = np.fromiter(
    (pid_to_index[p] for p in TEST_PATIENTS_ARR),
    dtype=np.int32,
    count=len(TEST_PATIENTS_ARR),
)
TEST_TAB_ARR = test_merge[training_features].astype("float32").to_numpy(copy=False)

TEST_VOL_ARR = ALL_VOLUMES[TEST_PID_IDX_ARR]


def make_test_tf_dataset():
    vol = TEST_VOL_ARR
    tab = TEST_TAB_ARR
    ds = tf.data.Dataset.from_tensor_slices((vol, tab))
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_tf_dataset()

predictions = np.zeros((len(test_merge), 3), dtype="float32")
for model in models:
    predictions += model.predict(test_ds, verbose=1)
predictions = predictions / len(models)




## === cell 9
def denormalize_fvc(y):
    return y * (MIN_MAX["FVC"][1] - MIN_MAX["FVC"][0]) + MIN_MAX["FVC"][0]


pred_denorm = denormalize_fvc(predictions)

FVC = pred_denorm[:, 1]
Confidence = pred_denorm[:, 2] - pred_denorm[:, 0]
Confidence = np.maximum(Confidence, 70.0)

out = SAMPLE_SUBMISSION.copy()
out["FVC"] = np.round(FVC).astype(int)
out["Confidence"] = np.round(Confidence).astype(int)
out.to_csv("submission.csv", index=False)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
print("Columns:", list(out.columns))
