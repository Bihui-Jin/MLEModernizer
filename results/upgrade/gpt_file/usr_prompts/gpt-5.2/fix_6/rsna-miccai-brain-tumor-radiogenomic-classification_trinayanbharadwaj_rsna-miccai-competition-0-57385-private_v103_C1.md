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

No external packages required in the script and installed.

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
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from skimage.transform import resize  # noqa: F401

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

IMG_PX_SIZE = 150

BAD_TRAIN_IDS = {109, 123, 709}


def _is_case_dir_entry(ent: os.DirEntry) -> bool:
    return ent.is_dir() and ent.name.isdigit()


_case_dirs_cache = {}


def _list_case_dirs(path_dir):
    if path_dir in _case_dirs_cache:
        return _case_dirs_cache[path_dir]
    out = sorted([ent.path for ent in os.scandir(path_dir) if _is_case_dir_entry(ent)])
    _case_dirs_cache[path_dir] = out
    return out


def _case_id_from_path(case_path):
    return int(os.path.basename(case_path))


_t2w_dir_cache = {}


def _t2w_dir(case_path):
    if case_path in _t2w_dir_cache:
        return _t2w_dir_cache[case_path]
    cand = os.path.join(case_path, "T2w")
    if os.path.isdir(cand):
        _t2w_dir_cache[case_path] = cand
        return cand
    for f in os.scandir(case_path):
        if f.is_dir() and f.name.lower() == "t2w":
            _t2w_dir_cache[case_path] = f.path
            return f.path
    _t2w_dir_cache[case_path] = None
    return None


_read_dicom_cache = {}


def _read_dicom_pixels(path):
    """
    Minimal DICOM reader. Returns float32 2D array (H,W). If unsupported/invalid, raises.
    """
    if path in _read_dicom_cache:
        return _read_dicom_cache[path]

    import struct

    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 140 or data[128:132] != b"DICM":
        raise ValueError("Not a DICOM file with DICM header.")

    def _read_tag(offset):
        g, e = struct.unpack_from("<HH", data, offset)
        return g, e

    def _read_vr_len(offset):
        vr = data[offset + 4 : offset + 6].decode("ascii", errors="ignore")
        if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
            length = struct.unpack_from("<I", data, offset + 8)[0]
            value_offset = offset + 12
            header_len = 12
        else:
            length = struct.unpack_from("<H", data, offset + 6)[0]
            value_offset = offset + 8
            header_len = 8
        return vr, length, value_offset, header_len

    def _parse_numeric(vr, raw):
        if vr in ("US",):
            if len(raw) < 2:
                return None
            return int(struct.unpack_from("<H", raw, 0)[0])
        if vr in ("SS",):
            if len(raw) < 2:
                return None
            return int(struct.unpack_from("<h", raw, 0)[0])
        if vr in ("UL",):
            if len(raw) < 4:
                return None
            return int(struct.unpack_from("<I", raw, 0)[0])
        if vr in ("SL",):
            if len(raw) < 4:
                return None
            return int(struct.unpack_from("<i", raw, 0)[0])
        if vr in ("DS", "IS"):
            s = raw.decode("ascii", errors="ignore").strip().strip("\x00")
            if s == "":
                return None
            try:
                if vr == "IS":
                    return int(float(s))
                return float(s)
            except Exception:
                return None
        return None

    rows = cols = bits_alloc = pix_repr = None
    slope = 1.0
    intercept = 0.0
    pixel_bytes = None

    off = 132
    n = len(data)

    max_elems = 200000
    cnt = 0

    while off + 8 <= n and cnt < max_elems:
        cnt += 1
        g, e = _read_tag(off)
        vr, length, value_offset, header_len = _read_vr_len(off)

        if length == 0xFFFFFFFF:
            off += header_len
            continue

        value_end = value_offset + length
        if value_end > n:
            break

        raw = data[value_offset:value_end]

        if (g, e) == (0x0028, 0x0010):  # Rows
            v = _parse_numeric(vr, raw)
            if v is not None:
                rows = v
        elif (g, e) == (0x0028, 0x0011):  # Columns
            v = _parse_numeric(vr, raw)
            if v is not None:
                cols = v
        elif (g, e) == (0x0028, 0x0100):  # BitsAllocated
            v = _parse_numeric(vr, raw)
            if v is not None:
                bits_alloc = v
        elif (g, e) == (0x0028, 0x0103):  # PixelRepresentation
            v = _parse_numeric(vr, raw)
            if v is not None:
                pix_repr = v
        elif (g, e) == (0x0028, 0x1053):  # RescaleSlope
            v = _parse_numeric(vr, raw)
            if v is not None:
                slope = float(v)
        elif (g, e) == (0x0028, 0x1052):  # RescaleIntercept
            v = _parse_numeric(vr, raw)
            if v is not None:
                intercept = float(v)
        elif (g, e) == (0x7FE0, 0x0010):  # PixelData
            pixel_bytes = raw
            if (
                rows is not None
                and cols is not None
                and bits_alloc is not None
                and pix_repr is not None
            ):
                break

        off = value_end

    if pixel_bytes is None or rows is None or cols is None or bits_alloc is None:
        raise ValueError("Missing required DICOM tags/pixel data.")

    if bits_alloc == 16:
        dtype = np.int16 if int(pix_repr or 0) == 1 else np.uint16
    elif bits_alloc == 8:
        dtype = np.int8 if int(pix_repr or 0) == 1 else np.uint8
    else:
        raise ValueError(f"Unsupported BitsAllocated={bits_alloc}")

    arr = np.frombuffer(pixel_bytes, dtype=dtype)
    if arr.size < rows * cols:
        raise ValueError("Pixel data smaller than expected.")
    arr = arr[: rows * cols].reshape((rows, cols)).astype(np.float32, copy=False)

    if slope != 1.0 or intercept != 0.0:
        arr = arr * float(slope) + float(intercept)

    _read_dicom_cache[path] = arr
    return arr


def _resize_to_150(arr2d: np.ndarray) -> np.ndarray:
    x = tf.convert_to_tensor(arr2d, dtype=tf.float32)
    x = tf.expand_dims(x, axis=-1)  # H,W,1
    x = tf.image.resize(
        x, (IMG_PX_SIZE, IMG_PX_SIZE), method="bilinear", antialias=True
    )
    x = tf.squeeze(x, axis=-1)
    return x.numpy().astype(np.float32, copy=False)


_case_slices_cache = {}


def _load_case_t2_slices(case_path, max_slices=6):
    """
    Load up to `max_slices` informative slices from the T2w series of a case.
    Returns a list of (150,150,3) float32 images in [0,1].
    """
    cache_key = (case_path, max_slices, IMG_PX_SIZE)
    if cache_key in _case_slices_cache:
        return _case_slices_cache[cache_key]

    t2dir = _t2w_dir(case_path)
    if t2dir is None:
        _case_slices_cache[cache_key] = []
        return []

    img_paths = sorted(
        [
            f.path
            for f in os.scandir(t2dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    out = []
    for p in img_paths:
        try:
            arr = _read_dicom_pixels(p)
        except Exception:
            continue

        if float(arr.sum()) <= 100000:
            continue

        arr_rs = _resize_to_150(arr)

        mx = float(np.max(arr_rs))
        if mx <= 0:
            continue
        arr_rs = arr_rs / mx

        stacked = np.stack([arr_rs, arr_rs, arr_rs], axis=-1).astype(
            np.float32, copy=False
        )

        if float(stacked.sum()) <= 1000:
            continue

        out.append(stacked)
        if len(out) >= max_slices:
            break

    _case_slices_cache[cache_key] = out
    return out




## === cell 1
def load_images_as_six_arrays_from_case_paths(case_paths, max_slices=6):
    """
    Ensure image arrays align 1:1 with the provided case_paths ordering.
    Returns six arrays (N,150,150,3).

    Speed: preallocate output arrays (instead of Python lists of arrays) to reduce overhead.
    """
    n = len(case_paths)
    out = [
        np.empty((n, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
        for _ in range(max_slices)
    ]
    pad = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)

    for i, case_path in enumerate(case_paths):
        slices = _load_case_t2_slices(case_path, max_slices=max_slices)
        if len(slices) == 0:
            slices = [pad] * max_slices
        elif len(slices) < max_slices:
            slices = slices + [slices[-1]] * (max_slices - len(slices))

        for k in range(max_slices):
            out[k][i] = slices[k]

    for i in range(max_slices):
        mx = float(np.max(out[i]))
        if mx > 0:
            out[i] = out[i] / mx

    print("Number of T2 images loaded are ", ", ".join(str(len(a)) for a in out))
    return tuple(out)


def load_images_as_six_arrays(path_dir, max_slices=6):
    case_paths = _list_case_dirs(path_dir)
    return load_images_as_six_arrays_from_case_paths(case_paths, max_slices=max_slices)




## === cell 2
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int)

labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_TRAIN_IDS)].reset_index(
    drop=True
)

train_case_paths_all = _list_case_dirs(TRAIN_DIR)
train_ids_all = np.array(
    [_case_id_from_path(p) for p in train_case_paths_all], dtype=int
)

label_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))
keep_mask = np.array([cid in label_map for cid in train_ids_all], dtype=bool)

train_case_paths = [p for p, k in zip(train_case_paths_all, keep_mask) if k]
train_ids = np.array([cid for cid, k in zip(train_ids_all, keep_mask) if k], dtype=int)
y = np.array([label_map[cid] for cid in train_ids], dtype=np.float32)

print(f"Train cases after filtering: {len(train_ids)}")
print("y mean:", float(y.mean()), "y std:", float(y.std()))




## === cell 3
def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 4
train_arrays = load_images_as_six_arrays_from_case_paths(train_case_paths, max_slices=6)

X = train_arrays[2]
print("X shape:", X.shape, "y shape:", y.shape)

if len(X) != len(y):
    raise ValueError(
        f"Mismatch X ({len(X)}) vs y ({len(y)}). Check filtering/alignment."
    )

idx = np.arange(len(X))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
split = int(0.85 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, y_tr = X[tr_idx], y[tr_idx]
X_va, y_va = X[va_idx], y[va_idx]

model_T2 = build_model(input_shape=X.shape[1:])
history = model_T2.fit(
    X_tr, y_tr, validation_data=(X_va, y_va), epochs=3, batch_size=16, verbose=2
)

model_T2 = build_model(input_shape=X.shape[1:])
model_T2.fit(X, y, epochs=3, batch_size=16, verbose=2)




## === cell 5
test_arrays = load_images_as_six_arrays(TEST_DIR, max_slices=6)
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = test_arrays

test_stack = np.concatenate(
    [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6], axis=0
)
preds_stack = model_T2.predict(test_stack, verbose=0).reshape(-1)

n = len(pixels_1)
preds_1 = preds_stack[0 * n : 1 * n]
preds_2 = preds_stack[1 * n : 2 * n]
preds_3 = preds_stack[2 * n : 3 * n]
preds_4 = preds_stack[3 * n : 4 * n]
preds_5 = preds_stack[4 * n : 5 * n]
preds_6 = preds_stack[5 * n : 6 * n]

prediction = (preds_1 + preds_2 + preds_3 + preds_4 + preds_5 + preds_6) / 6.0
prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)

print(
    "Test prediction stats:",
    float(prediction.min()),
    float(prediction.mean()),
    float(prediction.max()),
)




## === cell 6
def create_sub(path_test, preds):
    case_paths = _list_case_dirs(path_test)
    cases = [_case_id_from_path(p) for p in case_paths]
    if len(cases) != len(preds):
        raise ValueError(f"Mismatch cases ({len(cases)}) vs preds ({len(preds)})")
    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": preds})
    return df


sub_df = create_sub(TEST_DIR, prediction)

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(int)
sub_df = sample_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.dtypes)
print(sub_df.head())
