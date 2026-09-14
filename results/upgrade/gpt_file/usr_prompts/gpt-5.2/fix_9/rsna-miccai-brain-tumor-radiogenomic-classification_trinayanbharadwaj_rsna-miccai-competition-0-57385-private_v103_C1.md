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

# 5. Target score

-1.0

# 6. Current score

0.54765

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.41176) has done: 'I remove the failing `SimpleITK` import (it triggers a protobuf `MessageFactory.GetPrototype` crash in this environment) and replace DICOM reading with a small, robust `pydicom`-based loader. Then I fix the train/label misalignment that caused `IndexError` by ensuring images are loaded only for the same filtered `train_case_paths` used to build `y`. Finally, I make inference and submission generation run end-to-end by keeping `model_T2` defined and writing `submission.csv` with the exact required columns and ordering from `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by extremely expensive Python-level DICOM parsing and per-slice `tf.image.resize(...).numpy()` calls repeated thousands of times, plus building `test_stack` to run one big predict pass. The core model/training logic is kept identical, but data loading is made much faster by (1) replacing the slow custom DICOM parser with `tensorflow.io.decode_dicom_image` (same pixel values/semantics, just implemented in optimized C++), (2) resizing/normalization staying the same but executed through TF without round-tripping to NumPy per slice, and (3) parallelizing case loading with a thread pool while preserving deterministic ordering. For prediction, we avoid concatenating all slices into one huge array (same outputs) by predicting per-slice array and averaging, which reduces peak memory and copies. All paths, slice-selection criteria, model architecture, epochs, batch size, and averaging logic remain unchanged.'
- What this solution (achieved 0.54765) has done: 'You’re hitting the same protobuf `MessageFactory.GetPrototype` crash, but now it’s coming indirectly from `tf.io.decode_dicom_image` in this Kaggle environment. I keep the model/training logic identical and only swap the DICOM reading backend to a robust pure-`pydicom` loader (no SimpleITK, no TF DICOM decoder), while keeping the same slice filtering, resizing to 150, normalization, and 6-slice averaging semantics. I also make the DICOM ordering deterministic by sorting via `InstanceNumber` (fallback to filename) to avoid accidental slice-order noise that can hurt AUC. Finally, I ensure the pipeline completes and writes `submission.csv` with the exact required columns and order from `sample_submission.csv`.'
- What this solution (achieved 0.54765) has done: 'I fix the runtime crash caused by a protobuf incompatibility that’s being triggered indirectly during DICOM reading by switching the DICOM backend to a pure-NumPy decoder using `tf.io.decode_dicom_image`, and I guard it with a safe fallback to `pydicom` if needed. This keeps the same slice selection, resizing to 150, normalization, 6-slice padding, and 6-pred averaging logic intact, but avoids the environment-specific `MessageFactory.GetPrototype` failure. I also ensure the script still writes a valid `submission.csv` with the exact required columns and sample ordering. These changes are execution-stability focused and should be score-neutral to mildly positive due to more consistent, deterministic slice ordering and fewer failed reads.'

# 9. Code solution

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


try:
    import pydicom  # fallback only
except Exception:
    pydicom = None


_read_dicom_cache = {}


def _read_dicom_pixels(path):
    if path in _read_dicom_cache:
        return _read_dicom_cache[path]

    try:
        b = tf.io.read_file(path)
        img = tf.io.decode_dicom_image(
            b,
            dtype=tf.uint16,
            color_dim=False,
            on_error="skip",
            scale="auto",
        )
        img = tf.squeeze(img)  # -> H,W
        arr = img.numpy().astype(np.float32, copy=False)
        _read_dicom_cache[path] = arr
        return arr
    except Exception:
        if pydicom is None:
            raise

    ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
    arr = ds.pixel_array.astype(np.float32, copy=False)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept

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


def _sorted_dicom_paths(t2dir: str):
    paths = [
        f.path
        for f in os.scandir(t2dir)
        if f.is_file() and f.name.lower().endswith(".dcm")
    ]
    if not paths:
        return []

    inst_nums = []
    for p in paths:
        try:
            if pydicom is not None:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                inst = getattr(ds, "InstanceNumber", None)
                inst_nums.append(inst if inst is not None else -1)
            else:
                inst_nums.append(-1)
        except Exception:
            inst_nums.append(-1)

    if sum([1 for x in inst_nums if x != -1]) < max(3, len(paths) // 3):
        return sorted(paths)
    return [p for _, p in sorted(zip(inst_nums, paths), key=lambda t: (t[0], t[1]))]


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

    img_paths = _sorted_dicom_paths(t2dir)

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_images_as_six_arrays_from_case_paths(case_paths, max_slices=6):
    """
    Ensure image arrays align 1:1 with the provided case_paths ordering.
    Returns six arrays (N,150,150,3).
    """
    from concurrent.futures import ThreadPoolExecutor

    n = len(case_paths)
    out = [
        np.empty((n, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
        for _ in range(max_slices)
    ]
    pad = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)

    def _load_one(i_case_path):
        i, case_path = i_case_path
        slices = _load_case_t2_slices(case_path, max_slices=max_slices)
        if len(slices) == 0:
            slices = [pad] * max_slices
        elif len(slices) < max_slices:
            slices = slices + [slices[-1]] * (max_slices - len(slices))
        return i, slices

    max_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, slices in ex.map(_load_one, enumerate(case_paths), chunksize=8):
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

preds_1 = model_T2.predict(pixels_1, verbose=0).reshape(-1)
preds_2 = model_T2.predict(pixels_2, verbose=0).reshape(-1)
preds_3 = model_T2.predict(pixels_3, verbose=0).reshape(-1)
preds_4 = model_T2.predict(pixels_4, verbose=0).reshape(-1)
preds_5 = model_T2.predict(pixels_5, verbose=0).reshape(-1)
preds_6 = model_T2.predict(pixels_6, verbose=0).reshape(-1)

prediction = (preds_1 + preds_2 + preds_3 + preds_4 + preds_5 + preds_6) / 6.0
prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)

print(
    "Test prediction stats:",
    float(prediction.min()),
    float(prediction.mean()),
    float(prediction.max()),
)


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
