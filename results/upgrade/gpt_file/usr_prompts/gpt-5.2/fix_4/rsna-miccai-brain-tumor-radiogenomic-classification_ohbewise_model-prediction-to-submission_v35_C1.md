# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

nibabel==5.3.2
protobuf==6.33.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf

print("Python:", sys.version)
print("TF:", tf.__version__)

try:
    import pydicom  # type: ignore

    _HAS_PYDICOM = True
    print("pydicom: available")
except Exception as e:
    _HAS_PYDICOM = False
    print("pydicom: NOT available, using fallback DICOM pixel parser.", repr(e))




## === cell 1
data_dir = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
labels_path = os.path.join(data_dir, "train_labels.csv")
sample_path = os.path.join(data_dir, "sample_submission.csv")

out_dir = "/kaggle/working"
os.makedirs(out_dir, exist_ok=True)
submission_path = os.path.join(out_dir, "submission.csv")

print("Exists train_dir:", os.path.exists(train_dir))
print("Exists test_dir:", os.path.exists(test_dir))
print("Exists train_labels:", os.path.exists(labels_path))
print("Exists sample_submission:", os.path.exists(sample_path))




## === cell 2
TARGET_SHAPE = (128, 128, 64)  # (H, W, D)
SCAN_TYPE = "FLAIR"  # preserve original behavior


def _sorted_dicom_files(folder):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".dcm")
    ]
    files.sort()
    return files


def _read_dicom_pixel_array(fp: str) -> np.ndarray:
    """
    Fix: replace tfio.image.decode_dicom_image (crashes due to libtensorflow_io.so ABI mismatch).
    Prefer pydicom for correctness; otherwise use a minimal fallback that works for typical RSNA DICOM.
    Returns uint16 or int16 2D array (H,W).
    """
    if _HAS_PYDICOM:
        ds = pydicom.dcmread(fp, force=True)
        arr = ds.pixel_array  # numpy array
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr.astype(np.float32) * slope + intercept
        return arr.astype(np.float32)

    with open(fp, "rb") as f:
        data = f.read()

    tag = b"\xe0\x7f\x10\x00"
    idx = data.find(tag)
    if idx == -1:
        raise ValueError("PixelData tag not found in DICOM")

    vr = data[idx + 4 : idx + 6]
    long_vr = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"}
    if vr in long_vr:
        length = int.from_bytes(data[idx + 8 : idx + 12], "little", signed=False)
        start = idx + 12
    else:
        length = int.from_bytes(data[idx + 6 : idx + 8], "little", signed=False)
        start = idx + 8

    pixel_bytes = data[start : start + length]

    n = len(pixel_bytes)
    if n % 2 != 0:
        raise ValueError("Unexpected PixelData length")
    vals = np.frombuffer(pixel_bytes, dtype=np.uint16)
    side = int(np.sqrt(vals.size))
    if side * side == vals.size:
        arr = vals.reshape(side, side).astype(np.float32)
        return arr
    return vals.astype(np.float32)


def load_dicom_series_to_volume(series_dir):
    """Load a DICOM series directory into a float32 volume of shape (H, W, D)."""
    files = _sorted_dicom_files(series_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No .dcm files found in {series_dir}")

    slices = []
    for fp in files:
        img2d = _read_dicom_pixel_array(fp)  # (H,W) float32
        if img2d.ndim != 2:
            img2d = np.squeeze(img2d)
        slices.append(img2d.astype(np.float32))

    vol = np.stack(slices, axis=-1).astype(np.float32)
    return tf.convert_to_tensor(vol, dtype=tf.float32)


def znorm(volume):
    mean = tf.reduce_mean(volume)
    std = tf.math.reduce_std(volume)
    std = tf.where(std > 0, std, tf.constant(1.0, dtype=volume.dtype))
    return (volume - mean) / std


def rescale_to_minus1_plus1(volume):
    vmin = tf.reduce_min(volume)
    vmax = tf.reduce_max(volume)
    denom = tf.where(
        (vmax - vmin) > 0, (vmax - vmin), tf.constant(1.0, dtype=volume.dtype)
    )
    x = (volume - vmin) / denom  # [0,1]
    return x * 2.0 - 1.0


def crop_or_pad_3d(volume, target_shape=TARGET_SHAPE):
    """Center crop or pad a (H,W,D) volume to target_shape."""
    th, tw, td = target_shape

    def _crop_center(x, target, axis):
        size = tf.shape(x)[axis]
        start = tf.maximum((size - target) // 2, 0)
        begin = [0, 0, 0]
        begin[axis] = start
        size_out = [-1, -1, -1]
        size_out[axis] = tf.minimum(target, size)
        return tf.slice(x, begin, size_out)

    vol = volume
    vol = _crop_center(vol, th, 0)
    vol = _crop_center(vol, tw, 1)
    vol = _crop_center(vol, td, 2)

    shp2 = tf.shape(vol)
    ph = tf.maximum(th - shp2[0], 0)
    pw = tf.maximum(tw - shp2[1], 0)
    pd = tf.maximum(td - shp2[2], 0)

    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left
    pad_front = pd // 2
    pad_back = pd - pad_front

    vol = tf.pad(
        vol,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [pad_front, pad_back]],
        mode="CONSTANT",
        constant_values=0.0,
    )
    vol.set_shape([th, tw, td])
    return vol


def add_channel(volume):
    """(H,W,D) -> (H,W,D,1)"""
    return tf.expand_dims(volume, axis=-1)


def process_series_to_tensor(series_dir):
    """Return (H,W,D,1) float32 ready for model."""
    vol = load_dicom_series_to_volume(series_dir)  # (H,W,D)
    vol = znorm(vol)
    vol = crop_or_pad_3d(vol, TARGET_SHAPE)
    vol = rescale_to_minus1_plus1(vol)
    vol = add_channel(vol)  # (H,W,D,1)
    vol = tf.ensure_shape(vol, (*TARGET_SHAPE, 1))
    return vol


def process_series_numpy(series_dir):
    """Numpy wrapper (used by tf.data via py_function)."""
    t = process_series_to_tensor(series_dir)
    return t.numpy().astype(np.float32)




## === cell 3
labels_df = pd.read_csv(labels_path)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

bad_cases = {"00109", "00123", "00709"}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)


def series_exists(case_id, scan_type=SCAN_TYPE):
    return os.path.isdir(os.path.join(train_dir, case_id, scan_type))


labels_df = labels_df[labels_df["BraTS21ID"].map(series_exists)].reset_index(drop=True)

print("Train cases after filtering:", len(labels_df))
print(labels_df.head())

SEED = 42
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(labels_df))
val_frac = 0.15
n_val = max(1, int(len(labels_df) * val_frac))
val_idx = perm[:n_val]
train_idx = perm[n_val:]

train_df = labels_df.iloc[train_idx].reset_index(drop=True)
val_df = labels_df.iloc[val_idx].reset_index(drop=True)

print("Train/Val sizes:", len(train_df), len(val_df))
print("Val positive rate:", float(val_df["MGMT_value"].mean()))




## === cell 4
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 2  # keep small for memory; no early stopping introduced
EPOCHS = 3  # modest to finish within time; ensures end-to-end run


def make_dataset(df, training: bool):
    ids = df["BraTS21ID"].astype(str).tolist()
    labels = df["MGMT_value"].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((ids, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 256), seed=SEED, reshuffle_each_iteration=True
        )

    def _load(case_id, y):
        case_id_str = case_id.numpy().decode("utf-8")
        series_dir = os.path.join(train_dir, case_id_str, SCAN_TYPE)
        x = process_series_numpy(series_dir)  # (H,W,D,1)
        return x, np.float32(y)

    def _tf_load(case_id, y):
        x, y2 = tf.py_function(_load, inp=[case_id, y], Tout=[tf.float32, tf.float32])
        x.set_shape((*TARGET_SHAPE, 1))
        y2.set_shape(())
        return x, y2

    ds = ds.map(_tf_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df, training=False)




## === cell 5
def build_model(input_shape=(*TARGET_SHAPE, 1)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv3D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


tf.keras.utils.set_random_seed(SEED)

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
)




## === cell 6
sub_df = pd.read_csv(sample_path)
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sub_df["BraTS21ID"].tolist()

preds = []
failed = 0

for patient in test_ids:
    series_dir = os.path.join(test_dir, patient, SCAN_TYPE)
    try:
        x = process_series_to_tensor(series_dir)  # (H,W,D,1)
        x = tf.expand_dims(x, axis=0)  # (1,H,W,D,1)
        p = float(model.predict(x, verbose=0).reshape(-1)[0])
        p = float(np.clip(p, 0.0, 1.0))
    except Exception as e:
        failed += 1
        p = 0.5
        print(f"Warning: failed patient {patient} ({type(e).__name__}: {e}); using 0.5")
    preds.append(p)

sub_df["MGMT_value"] = preds
sub_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print("Rows:", len(sub_df), " Failed:", failed)
print(sub_df.head())
print("Submission columns:", list(sub_df.columns))
