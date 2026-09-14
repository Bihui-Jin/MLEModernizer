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

# 5. Target score

-1.0

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the immediate runtime crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6 by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. Then I remove the unsupported `workers/use_multiprocessing/max_queue_size` arguments from `model.fit()` and `model.predict()` (these are not accepted by the TF 2.18 Keras trainer), keeping the same training epochs and model unchanged. Finally, I ensure the pipeline completes end-to-end and writes a valid `submission.csv` with the required columns and ID formatting. These changes are execution-stability fixes and should be score-neutral aside from eliminating failures.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import re
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

print("TF:", tf.__version__)

SEED = 42
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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
_DCM_LIST_CACHE = {}  # series_dir -> list[str]


def _list_dcm_files(series_dir: str):
    files = _DCM_LIST_CACHE.get(series_dir)
    if files is not None:
        return files
    files = []
    with os.scandir(series_dir) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".dcm"):
                files.append(e.path)
    if len(files) == 0:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")
    files.sort()
    _DCM_LIST_CACHE[series_dir] = files
    return files


_TAG_ROWS = b"\x28\x00\x10\x00"
_TAG_COLS = b"\x28\x00\x11\x00"
_TAG_BITS_ALLOC = b"\x28\x00\x00\x01"
_TAG_PIXEL_REPR = b"\x28\x00\x03\x01"
_TAG_PIXEL_DATA = b"\xE0\x7F\x10\x00"


def _read_dicom_uncompressed_pixel_array(fp: str) -> np.ndarray:
    """
    Minimal DICOM reader for RSNA MRI series used in this competition.
    Assumptions (holds for this dataset):
      - Explicit VR Little Endian
      - Uncompressed PixelData (7FE0,0010)
      - MONOCHROME2, BitsAllocated 16, SamplesPerPixel 1
    Returns float32 image [H, W]. Raises if required tags are missing.
    """
    prefix_sizes = (512 * 1024, 2 * 1024 * 1024)
    data = b""
    with open(fp, "rb") as f:
        for ps in prefix_sizes:
            f.seek(0)
            data = f.read(ps)
            if (
                data.find(_TAG_ROWS) >= 0
                and data.find(_TAG_COLS) >= 0
                and data.find(_TAG_PIXEL_DATA) >= 0
            ):
                break

        def read_us_after_explicit_vr(pos: int) -> int:
            vr = data[pos + 4 : pos + 6]
            if len(vr) != 2:
                raise ValueError("Unexpected EOF while reading VR")
            length = int.from_bytes(data[pos + 6 : pos + 8], "little", signed=False)
            val_pos = pos + 8
            if length != 2:
                val = int.from_bytes(
                    data[val_pos : val_pos + length], "little", signed=False
                )
                return int(val)
            return int.from_bytes(data[val_pos : val_pos + 2], "little", signed=False)

        pos_r = data.find(_TAG_ROWS)
        pos_c = data.find(_TAG_COLS)
        if pos_r < 0 or pos_c < 0:
            raise ValueError("Missing Rows/Columns tags")
        rows = read_us_after_explicit_vr(pos_r)
        cols = read_us_after_explicit_vr(pos_c)

        pos_ba = data.find(_TAG_BITS_ALLOC)
        bits_alloc = 16 if pos_ba < 0 else read_us_after_explicit_vr(pos_ba)

        pos_pr = data.find(_TAG_PIXEL_REPR)
        pixel_repr = 0 if pos_pr < 0 else read_us_after_explicit_vr(pos_pr)

        if bits_alloc != 16:
            raise ValueError(f"Unsupported BitsAllocated={bits_alloc}")

        pos_pd = data.find(_TAG_PIXEL_DATA)
        if pos_pd < 0:
            raise ValueError("Missing PixelData tag")

        vr = data[pos_pd + 4 : pos_pd + 6]
        if vr not in (b"OW", b"OB"):
            raise ValueError(f"Unexpected PixelData VR={vr!r}")

        length = int.from_bytes(data[pos_pd + 8 : pos_pd + 12], "little", signed=False)
        val_pos = pos_pd + 12

        needed = rows * cols * 2
        dtype = np.int16 if pixel_repr == 1 else np.uint16

        if val_pos + needed <= len(data):
            pix = data[val_pos : val_pos + needed]
        else:
            f.seek(val_pos)
            pix = f.read(needed)
            if len(pix) < needed:
                raise ValueError("PixelData shorter than expected")

    arr = (
        np.frombuffer(pix, dtype=dtype)
        .reshape(rows, cols)
        .astype(np.float32, copy=False)
    )
    return arr


def load_dicom_series_to_volume(series_dir: str) -> np.ndarray:
    """
    Read all *.dcm in a folder, decode with a minimal DICOM parser, stack into [H, W, D].
    Converts to float32.
    """
    files = _list_dcm_files(series_dir)

    slices = []
    shapes = []
    for fp in files:
        try:
            arr = _read_dicom_uncompressed_pixel_array(fp)
            if arr.ndim != 2:
                continue
            slices.append(arr)
            shapes.append(arr.shape)
        except Exception:
            continue

    if len(slices) == 0:
        raise RuntimeError(f"All DICOM files failed to decode in: {series_dir}")

    shapes_arr = np.array(shapes, dtype=np.int32)
    uniq, counts = np.unique(shapes_arr, axis=0, return_counts=True)
    target_shape = tuple(uniq[np.argmax(counts)].tolist())

    slices = [s for s in slices if s.shape == target_shape]
    if len(slices) == 0:
        raise RuntimeError(f"No consistent-slice shapes found in: {series_dir}")

    vol = np.stack(slices, axis=-1).astype(np.float32)  # [H, W, D]
    return vol


def zscore_normalize(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    v = volume.astype(np.float32, copy=False)
    m = float(v.mean())
    s = float(v.std())
    return (v - m) / (s + eps)


def center_crop_or_pad_3d(
    volume: np.ndarray, target_shape=(128, 128, 64)
) -> np.ndarray:
    """Center crop/pad a [H, W, D] volume to target_shape."""
    h, w, d = volume.shape
    th, tw, td = target_shape

    out = np.zeros((th, tw, td), dtype=volume.dtype)

    sh0 = max((h - th) // 2, 0)
    sw0 = max((w - tw) // 2, 0)
    sd0 = max((d - td) // 2, 0)
    sh1 = sh0 + min(th, h)
    sw1 = sw0 + min(tw, w)
    sd1 = sd0 + min(td, d)

    dh0 = max((th - h) // 2, 0)
    dw0 = max((tw - w) // 2, 0)
    dd0 = max((td - d) // 2, 0)
    dh1 = dh0 + (sh1 - sh0)
    dw1 = dw0 + (sw1 - sw0)
    dd1 = dd0 + (sd1 - sd0)

    out[dh0:dh1, dw0:dw1, dd0:dd1] = volume[sh0:sh1, sw0:sw1, sd0:sd1]
    return out


def add_batch_channel(volume_3d: np.ndarray) -> tf.Tensor:
    """Convert [H, W, D] -> [1, H, W, D, 1] for 3D CNN input."""
    x = tf.convert_to_tensor(volume_3d, dtype=tf.float32)
    x = tf.expand_dims(x, axis=-1)  # channel
    x = tf.expand_dims(x, axis=0)  # batch
    return x


def process_dicom_folder(series_dir: str, target_shape=(128, 128, 64)) -> tf.Tensor:
    vol = load_dicom_series_to_volume(series_dir)
    vol = zscore_normalize(vol)
    vol = center_crop_or_pad_3d(vol, target_shape=target_shape)
    return add_batch_channel(vol)




## === cell 2
data_dir = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
labels_path = os.path.join(data_dir, "train_labels.csv")
sample_path = os.path.join(data_dir, "sample_submission.csv")

labels_df = pd.read_csv(labels_path)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df = pd.read_csv(sample_path)
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sub_df["BraTS21ID"].tolist()

print("Train labels:", labels_df.shape, "Test rows:", sub_df.shape)

scan_type = "T1wCE"

bad_ids = set(["00109", "00123", "00709"])
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)
print("Train labels after excluding bad IDs:", labels_df.shape)



## === cell 3
TARGET_SHAPE = (128, 128, 64)


def build_3d_cnn(input_shape=(128, 128, 64, 1)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, name="input_1")
    x = tf.keras.layers.Conv3D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="MGMT_value")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


model = build_3d_cnn(input_shape=(*TARGET_SHAPE, 1))
model.summary()



## === cell 4
_VOLUME_CACHE = {}  # key: (split, pid, scan_type, target_shape) -> np.ndarray [H,W,D,1]


def _get_cached_volume(split_key: str, pid: str) -> np.ndarray:
    key = (split_key, pid, scan_type, TARGET_SHAPE)
    v = _VOLUME_CACHE.get(key, None)
    if v is not None:
        return v
    series_dir = os.path.join(
        train_dir if split_key == "train" else test_dir, pid, scan_type
    )
    try:
        x = process_dicom_folder(series_dir, target_shape=TARGET_SHAPE)  # [1,H,W,D,1]
        v = x.numpy()[0].astype(np.float32, copy=False)  # [H,W,D,1]
    except Exception:
        v = np.zeros((*TARGET_SHAPE, 1), dtype=np.float32)
    _VOLUME_CACHE[key] = v
    return v


class DicomSequence(tf.keras.utils.Sequence):
    def __init__(self, ids, y=None, batch_size=1, shuffle=False, split_key="train"):
        self.ids = list(ids)
        self.y = None if y is None else np.array(y, dtype=np.float32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.split_key = str(split_key)  # "train" or "test"
        self._epoch = 0
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(SEED + self._epoch)
            idx = np.arange(len(self.ids))
            rng.shuffle(idx)
            self.ids = [self.ids[i] for i in idx]
            if self.y is not None:
                self.y = self.y[idx]
        self._epoch += 1

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        bs = len(batch_ids)
        X = np.empty((bs, *TARGET_SHAPE, 1), dtype=np.float32)
        Y = None if self.y is None else np.empty((bs, 1), dtype=np.float32)

        base = index * self.batch_size
        for i, pid in enumerate(batch_ids):
            X[i] = _get_cached_volume(self.split_key, pid)
            if Y is not None:
                Y[i, 0] = float(self.y[base + i])
        return (X, Y) if Y is not None else X


from sklearn.model_selection import train_test_split

train_ids = labels_df["BraTS21ID"].tolist()
train_y = labels_df["MGMT_value"].astype(np.float32).values

tr_ids, va_ids, tr_y, va_y = train_test_split(
    train_ids, train_y, test_size=0.15, random_state=SEED, stratify=train_y
)

train_seq = DicomSequence(tr_ids, tr_y, batch_size=1, shuffle=True, split_key="train")
val_seq = DicomSequence(va_ids, va_y, batch_size=1, shuffle=False, split_key="train")

print("Train/Val sizes:", len(tr_ids), len(va_ids))

for pid in va_ids:
    _get_cached_volume("train", pid)



## === cell 5
history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=2,
    verbose=1,
)



## === cell 6
test_seq = DicomSequence(
    test_ids, y=None, batch_size=1, shuffle=False, split_key="test"
)

pred = model.predict(
    test_seq,
    verbose=1,
)

pred = np.squeeze(pred).astype(np.float32)
pred = np.clip(pred, 0.0, 1.0)

sub_df["MGMT_value"] = pred

out_path = "/kaggle/working/submission.csv"
sub_df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub_df.head())
print("Rows in submission:", len(sub_df))
print("Prediction stats:", float(pred.min()), float(pred.mean()), float(pred.max()))
