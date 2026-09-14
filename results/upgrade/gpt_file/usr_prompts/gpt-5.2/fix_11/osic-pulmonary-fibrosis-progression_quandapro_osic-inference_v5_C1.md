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
        files = [f for f in os.listdir(image_folder) if f.lower().endswith(".dcm")]
    except Exception:
        return []
    return files


def load_patient_volume(patient_id, folder, img_size=IMG_SIZE, num_scans=NUM_OF_SCANS):
    image_folder = os.path.join(folder, patient_id)
    if not DICOM_AVAILABLE:
        return np.zeros((num_scans, img_size, img_size), dtype="float32")

    try:
        files = _list_dicom_files(image_folder)
        if not files:
            return np.zeros((num_scans, img_size, img_size), dtype="float32")

        meta = []
        for f in files:
            fp = os.path.join(image_folder, f)
            try:
                ds = dicom.dcmread(
                    fp,
                    stop_before_pixels=True,
                    force=True,
                    specific_tags=["InstanceNumber"],
                )
                inst = getattr(ds, "InstanceNumber", None)
                if inst is None:
                    try:
                        inst = int(os.path.splitext(f)[0])
                    except Exception:
                        inst = 0
                meta.append((inst, fp))
            except Exception:
                continue

        if not meta:
            return np.zeros((num_scans, img_size, img_size), dtype="float32")

        meta.sort(key=lambda t: t[0])
        fps = [fp for _, fp in meta]

        if len(fps) >= num_scans:
            mid = len(fps) // 2
            half = num_scans // 2
            fps = fps[mid - half : mid + half]
        else:
            pass

        imgs = []
        for fp in fps:
            try:
                s = dicom.dcmread(fp, force=True)
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

        return np.asarray(imgs, dtype="float32")
    except Exception:
        return np.zeros((num_scans, img_size, img_size), dtype="float32")


all_patients = sorted(
    set(TRAIN_DF["Patient"].unique()).union(set(TEST_DF["Patient"].unique()))
)
pid_to_index = {pid: i for i, pid in enumerate(all_patients)}
index_to_pid = np.array(all_patients)

_VOLUME_CACHE = {}  # pid -> np.ndarray
_PID_FOLDER = {}
for pid in all_patients:
    folder = (
        TRAIN_IMG_DIR
        if os.path.isdir(os.path.join(TRAIN_IMG_DIR, pid))
        else TEST_IMG_DIR
    )
    _PID_FOLDER[pid] = folder


def get_cached_volume_by_pid(pid: str) -> np.ndarray:
    v = _VOLUME_CACHE.get(pid)
    if v is not None:
        return v
    vol = load_patient_volume(pid, _PID_FOLDER[pid])  # float32 [0..255]
    vol = (vol / 255.0).astype(np.float32, copy=False)
    vol = vol[..., None]  # add channel
    _VOLUME_CACHE[pid] = vol
    return vol


def _prime_volume_cache(pids):
    for pid in sorted(set(pids)):
        _ = get_cached_volume_by_pid(pid)
    gc.collect()


print("Patients (train+test):", len(all_patients), "DICOM_AVAILABLE:", DICOM_AVAILABLE)




## === cell 5
TRAIN_PATIENTS_ARR = TRAIN_DF["Patient"].to_numpy()
TRAIN_PID_IDX_ARR = np.fromiter(
    (pid_to_index[p] for p in TRAIN_PATIENTS_ARR),
    dtype=np.int32,
    count=len(TRAIN_PATIENTS_ARR),
)

TRAIN_TAB_ARR = TRAIN_DF[training_features].astype("float32").to_numpy(copy=True)
TRAIN_Y_ARR = TRAIN_DF[["y0", "y1", "y2"]].astype("float32").to_numpy(copy=True)


def make_tf_dataset(indices, training: bool):
    indices = np.asarray(indices, dtype=np.int32)

    def py_get_volume(pid_idx_np):
        pid = index_to_pid[int(pid_idx_np)]
        vol = get_cached_volume_by_pid(pid)
        return vol

    def map_fn(i):
        tab = tf.gather(TRAIN_TAB_ARR, i)
        pid_idx = tf.gather(TRAIN_PID_IDX_ARR, i)
        vol = tf.numpy_function(py_get_volume, [pid_idx], Tout=tf.float32)
        vol.set_shape((NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1))
        y = tf.gather(TRAIN_Y_ARR, i)
        return (vol, tab), y

    ds = tf.data.Dataset.from_tensor_slices(indices)
    if training:
        ds = ds.shuffle(
            buffer_size=len(indices), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
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


def train_one_model(seed_offset=0):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    model = build_model()
    model.compile(optimizer=Adam(1e-3), loss="mse")

    needed_pids = np.concatenate(
        [
            TRAIN_DF.loc[train_idx, "Patient"].values,
            TRAIN_DF.loc[val_idx, "Patient"].values,
        ]
    )
    _prime_volume_cache(needed_pids)

    train_ds = make_tf_dataset(train_idx, training=True)
    val_ds = make_tf_dataset(val_idx, training=False)

    ckpt_path = f"best_weights_{seed_offset}.weights.h5"
    ckpt_cb = ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=0,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[ckpt_cb],
    )
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
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


def make_test_tf_dataset():
    idx = np.arange(len(test_merge), dtype=np.int32)

    def py_get_volume(pid_idx_np):
        pid = index_to_pid[int(pid_idx_np)]
        vol = get_cached_volume_by_pid(pid)
        return vol

    def map_fn(i):
        tab = tf.gather(TEST_TAB_ARR, i)
        pid_idx = tf.gather(TEST_PID_IDX_ARR, i)
        vol = tf.numpy_function(py_get_volume, [pid_idx], Tout=tf.float32)
        vol.set_shape((NUM_OF_SCANS, IMG_SIZE, IMG_SIZE, 1))
        return (vol, tab)

    ds = tf.data.Dataset.from_tensor_slices(idx)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


_prime_volume_cache(TEST_PATIENTS_ARR)
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
