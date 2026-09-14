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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.54353) has done: 'I remove the nonessential imports that trigger the protobuf/pydicom `MessageFactory` error, and I load DICOMs via `pydicom.dcmread` (from the `pydicom` package directly) to avoid that crash. Because the referenced pretrained `.h5` files are not available in your input paths, I replace those loads with a minimal TensorFlow CNN that preserves the same “predict 6 slices then average” pipeline, and train it on the provided training set (excluding the known-bad IDs) so the notebook can run end-to-end. I also fix `load_test_T2W_images` to actually return NumPy arrays (not Python lists) and to normalize safely, and fix `create_sub` so it produces one prediction per test case instead of repeatedly overwriting a single vector. Finally, I ensure the submission matches `sample_submission.csv` order and writes `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
tf.config.threading.set_inter_op_parallelism_threads(max(1, os.cpu_count() // 2))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing labels: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)
labels_df.head(), sample_sub.head()




## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 6
T2_FOLDER_NAME = "T2w"

BAD_TRAIN_IDS = {"00109", "00123", "00709"}

try:
    import tensorflow_io as tfio  # type: ignore
except Exception:
    tfio = None


def _safe_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mx = float(np.max(x))
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def _read_dicom_pixel_array_tfio(dcm_path: str):
    if tfio is None:
        return None
    try:
        b = tf.io.read_file(dcm_path)
        img = tfio.image.decode_dicom_image(
            b,
            dtype=tf.uint16,
            color_dim=False,
            on_error="skip",
        )
        img = tf.squeeze(img)
        arr = img.numpy()
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None


def _resize_to(arr2d: np.ndarray, img_size: int) -> np.ndarray:
    t = tf.convert_to_tensor(arr2d, dtype=tf.float32)
    t = t[None, ..., None]  # (1,H,W,1)
    t = tf.image.resize(t, (img_size, img_size), method="bilinear", antialias=True)
    out = tf.squeeze(t, axis=(0, 3)).numpy()
    return out.astype(np.float32, copy=False)


from functools import lru_cache


@lru_cache(maxsize=None)
def _sorted_dcms_numeric_cached(dir_path: str):
    files = []
    try:
        with os.scandir(dir_path) as it:
            for f in it:
                if f.is_file() and f.name.lower().endswith(".dcm"):
                    name = f.name
                    try:
                        n = int(name.split("-")[1].split(".")[0])
                    except Exception:
                        n = 1_000_000_000
                    files.append((n, f.path))
    except FileNotFoundError:
        return tuple()
    files.sort(key=lambda x: x[0])
    return tuple(p for _, p in files)


def _sorted_dcms_numeric(dir_path: str):
    return list(_sorted_dcms_numeric_cached(dir_path))


def load_case_t2_slices(
    case_path: str, n_slices: int = N_SLICES, img_size: int = IMG_PX_SIZE
):
    """
    Returns: np.ndarray shape (n_slices, img_size, img_size, 3), dtype float32
    If not enough informative slices, pads with zeros.
    """
    t2_path = os.path.join(case_path, T2_FOLDER_NAME)
    if not os.path.isdir(t2_path):
        candidates = []
        try:
            with os.scandir(case_path) as it:
                for d in it:
                    if d.is_dir() and "t2" in d.name.lower():
                        candidates.append(d.path)
        except FileNotFoundError:
            candidates = []
        t2_path = candidates[0] if candidates else None
    if not t2_path or not os.path.isdir(t2_path):
        return np.zeros((n_slices, img_size, img_size, 3), dtype=np.float32)

    dcm_files = _sorted_dcms_numeric(t2_path)
    if len(dcm_files) == 0:
        return np.zeros((n_slices, img_size, img_size, 3), dtype=np.float32)

    chosen = []
    for fp in dcm_files:
        arr = _read_dicom_pixel_array_tfio(fp)
        if arr is None:
            continue

        s = float(np.sum(arr))
        if s <= 100000:
            continue

        img = _resize_to(arr, img_size)
        stacked = np.stack([img, img, img], axis=-1).astype(np.float32, copy=False)
        stacked = _safe_float01(stacked)

        if float(np.sum(stacked)) <= 2000:
            continue

        chosen.append(stacked)
        if len(chosen) >= n_slices:
            break

    if len(chosen) < n_slices:
        pad_count = n_slices - len(chosen)
        if pad_count:
            chosen.extend(
                [np.zeros((img_size, img_size, 3), dtype=np.float32)] * pad_count
            )

    return np.stack(chosen, axis=0).astype(np.float32, copy=False)




## === cell 3
def load_dataset_from_dir(base_dir: str, ids: list, labels: pd.Series = None):
    """
    Builds a per-slice dataset:
      X: (num_cases*n_slices, 150, 150, 3)
      y: (num_cases*n_slices,) if labels is provided
    Also returns: case_id list (one per case).
    """
    n_cases = len(ids)
    X = np.zeros((n_cases * N_SLICES, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    y = None
    if labels is not None:
        y = np.zeros((n_cases * N_SLICES,), dtype=np.float32)

    case_ids_out = []
    out_i = 0
    for cid in ids:
        case_path = os.path.join(base_dir, cid)
        slices = load_case_t2_slices(case_path, n_slices=N_SLICES, img_size=IMG_PX_SIZE)
        X[out_i : out_i + N_SLICES] = slices
        case_ids_out.append(cid)
        if y is not None:
            y_val = float(labels.loc[cid])
            y[out_i : out_i + N_SLICES] = y_val
        out_i += N_SLICES

    if labels is not None:
        return X, y, case_ids_out
    return X, case_ids_out




## === cell 4
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_TRAIN_IDS)].reset_index(
    drop=True
)

train_ids_all = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
labels_map = labels_df.set_index("BraTS21ID")["MGMT_value"]
labels_id_set = set(labels_df["BraTS21ID"].values)
train_ids = [i for i in train_ids_all if i in labels_id_set]

from sklearn.model_selection import train_test_split

stratify_y = labels_map.loc[train_ids].values

tr_ids, va_ids = train_test_split(
    train_ids,
    test_size=0.15,
    random_state=SEED,
    stratify=stratify_y,
)

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def _load_case_job(args):
    cid, base_dir = args
    case_path = os.path.join(base_dir, cid)
    slices = load_case_t2_slices(case_path, n_slices=N_SLICES, img_size=IMG_PX_SIZE)
    return cid, slices


def load_dataset_from_dir_parallel(base_dir: str, ids: list, labels: pd.Series = None):
    n_cases = len(ids)
    X = np.zeros((n_cases * N_SLICES, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    y = None
    if labels is not None:
        y = np.zeros((n_cases * N_SLICES,), dtype=np.float32)

    max_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    ctx = mp.get_context("spawn")

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        for cid, slices in ex.map(
            _load_case_job, [(cid, base_dir) for cid in ids], chunksize=4
        ):
            results[cid] = slices

    case_ids_out = []
    out_i = 0
    for cid in ids:
        slices = results[cid]
        X[out_i : out_i + N_SLICES] = slices
        case_ids_out.append(cid)
        if y is not None:
            y_val = float(labels.loc[cid])
            y[out_i : out_i + N_SLICES] = y_val
        out_i += N_SLICES

    if labels is not None:
        return X, y, case_ids_out
    return X, case_ids_out


X_tr, y_tr, _ = load_dataset_from_dir_parallel(TRAIN_DIR, tr_ids, labels=labels_map)
X_va, y_va, _ = load_dataset_from_dir_parallel(TRAIN_DIR, va_ids, labels=labels_map)

X_tr.shape, y_tr.shape, X_va.shape, y_va.shape




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
BrokenProcessPool                         Traceback (most recent call last)
/tmp/ipykernel_11/4015142008.py in <cell line: 0>()
     68 
     69 
---> 70 X_tr, y_tr, _ = load_dataset_from_dir_parallel(TRAIN_DIR, tr_ids, labels=labels_map)
     71 X_va, y_va, _ = load_dataset_from_dir_parallel(TRAIN_DIR, va_ids, labels=labels_map)
     72 

/tmp/ipykernel_11/4015142008.py in load_dataset_from_dir_parallel(base_dir, ids, labels)
     47     results = {}
     48     with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
---> 49         for cid, slices in ex.map(
     50             _load_case_job, [(cid, base_dir) for cid in ids], chunksize=4
     51         ):

/usr/lib/python3.11/concurrent/futures/process.py in map(self, fn, timeout, chunksize, *iterables)
    835             raise ValueError("chunksize must be >= 1.")
    836 
--> 837         results = super().map(partial(_process_chunk, fn),
    838                               _get_chunks(*iterables, chunksize=chunksize),
    839                               timeout=timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in map(self, fn, timeout, chunksize, *iterables)
    606             end_time = timeout + time.monotonic()
    607 
--> 608         fs = [self.submit(fn, *args) for args in zip(*iterables)]
    609 
    610         # Yield must be hidden in closure so that the futures are submitted

/usr/lib/python3.11/concurrent/futures/_base.py in <listcomp>(.0)
    606             end_time = timeout + time.monotonic()
    607 
--> 608         fs = [self.submit(fn, *args) for args in zip(*iterables)]
    609 
    610         # Yield must be hidden in closure so that the futures are submitted

/usr/lib/python3.11/concurrent/futures/process.py in submit(self, fn, *args, **kwargs)
    789         with self._shutdown_lock:
    790             if self._broken:
--> 791                 raise BrokenProcessPool(self._broken)
    792             if self._shutdown_thread:
    793                 raise RuntimeError('cannot schedule new futures after shutdown')

BrokenProcessPool: A child process terminated abruptly, the process pool is not usable anymore

## === cell 5
def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = build_model()
model_T2.summary()




## === cell 6
def train_one_model(seed_offset: int):
    tf.random.set_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    m = build_model()

    ds_tr = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .shuffle(
            buffer_size=len(y_tr),
            seed=SEED + seed_offset,
            reshuffle_each_iteration=True,
        )
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_va = (
        tf.data.Dataset.from_tensor_slices((X_va, y_va))
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    m.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=3,
        verbose=2,
    )
    return m


model_T2 = train_one_model(0)
model_T2_2 = train_one_model(1)
model_T2_3 = train_one_model(2)
model_T2_4 = train_one_model(3)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3052788255.py in <cell line: 0>()
     30 
     31 
---> 32 model_T2 = train_one_model(0)
     33 model_T2_2 = train_one_model(1)
     34 model_T2_3 = train_one_model(2)

/tmp/ipykernel_11/3052788255.py in train_one_model(seed_offset)
      6 
      7     ds_tr = (
----> 8         tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
      9         .shuffle(
     10             buffer_size=len(y_tr),

NameError: name 'X_tr' is not defined

## === cell 7
def load_test_T2W_images(path_test):
    """
    Collect 6 slices per case:
    - return numpy arrays
    - safe normalization inside load_case_t2_slices
    """
    case_ids = sorted([d.name for d in os.scandir(path_test) if d.is_dir()])
    n_cases = len(case_ids)

    max_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    ctx = mp.get_context("spawn")

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        for cid, slices in ex.map(
            _load_case_job, [(cid, path_test) for cid in case_ids], chunksize=4
        ):
            results[cid] = slices

    X_all = np.zeros((n_cases, N_SLICES, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    for i, cid in enumerate(case_ids):
        X_all[i] = results[cid]

    arrays = tuple(X_all[:, s].copy() for s in range(N_SLICES))

    print("Number of T2 images loaded are", ", ".join(str(a.shape[0]) for a in arrays))
    return arrays




## === cell 8
test = TEST_DIR
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
BrokenProcessPool                         Traceback (most recent call last)
/tmp/ipykernel_11/68258430.py in <cell line: 0>()
      1 test = TEST_DIR
----> 2 pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
      3 
      4 

/tmp/ipykernel_11/2909675558.py in load_test_T2W_images(path_test)
     14     results = {}
     15     with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
---> 16         for cid, slices in ex.map(
     17             _load_case_job, [(cid, path_test) for cid in case_ids], chunksize=4
     18         ):

/usr/lib/python3.11/concurrent/futures/process.py in map(self, fn, timeout, chunksize, *iterables)
    835             raise ValueError("chunksize must be >= 1.")
    836 
--> 837         results = super().map(partial(_process_chunk, fn),
    838                               _get_chunks(*iterables, chunksize=chunksize),
    839                               timeout=timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in map(self, fn, timeout, chunksize, *iterables)
    606             end_time = timeout + time.monotonic()
    607 
--> 608         fs = [self.submit(fn, *args) for args in zip(*iterables)]
    609 
    610         # Yield must be hidden in closure so that the futures are submitted

/usr/lib/python3.11/concurrent/futures/_base.py in <listcomp>(.0)
    606             end_time = timeout + time.monotonic()
    607 
--> 608         fs = [self.submit(fn, *args) for args in zip(*iterables)]
    609 
    610         # Yield must be hidden in closure so that the futures are submitted

/usr/lib/python3.11/concurrent/futures/process.py in submit(self, fn, *args, **kwargs)
    789         with self._shutdown_lock:
    790             if self._broken:
--> 791                 raise BrokenProcessPool(self._broken)
    792             if self._shutdown_thread:
    793                 raise RuntimeError('cannot schedule new futures after shutdown')

BrokenProcessPool: A child process terminated abruptly, the process pool is not usable anymore

## === cell 9
def _pred_col(model, x):
    p = model.predict(x, verbose=0, batch_size=128).reshape(-1)
    return p.astype(np.float32, copy=False)


prediction_1 = _pred_col(model_T2, pixels_1)
prediction_2 = _pred_col(model_T2, pixels_2)
prediction_3 = _pred_col(model_T2, pixels_3)
prediction_4 = _pred_col(model_T2, pixels_4)
prediction_5 = _pred_col(model_T2, pixels_5)
prediction_6 = _pred_col(model_T2, pixels_6)

prediction_101 = _pred_col(model_T2_2, pixels_1)
prediction_102 = _pred_col(model_T2_2, pixels_2)
prediction_103 = _pred_col(model_T2_2, pixels_3)
prediction_104 = _pred_col(model_T2_2, pixels_4)
prediction_105 = _pred_col(model_T2_2, pixels_5)
prediction_106 = _pred_col(model_T2_2, pixels_6)

prediction_201 = _pred_col(model_T2_3, pixels_1)
prediction_202 = _pred_col(model_T2_3, pixels_2)
prediction_203 = _pred_col(model_T2_3, pixels_3)
prediction_204 = _pred_col(model_T2_3, pixels_4)
prediction_205 = _pred_col(model_T2_3, pixels_5)
prediction_206 = _pred_col(model_T2_3, pixels_6)

prediction_301 = _pred_col(model_T2_4, pixels_1)
prediction_302 = _pred_col(model_T2_4, pixels_2)
prediction_303 = _pred_col(model_T2_4, pixels_3)
prediction_304 = _pred_col(model_T2_4, pixels_4)
prediction_305 = _pred_col(model_T2_4, pixels_5)
prediction_306 = _pred_col(model_T2_4, pixels_6)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3203264389.py in <cell line: 0>()
      4 
      5 
----> 6 prediction_1 = _pred_col(model_T2, pixels_1)
      7 prediction_2 = _pred_col(model_T2, pixels_2)
      8 prediction_3 = _pred_col(model_T2, pixels_3)

NameError: name 'pixels_1' is not defined

## === cell 10
def create_sub(
    cases,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
):
    """
    Compute one averaged prediction per case.
    """
    pred = (
        p1
        + p2
        + p3
        + p4
        + p5
        + p6
        + p101
        + p102
        + p103
        + p104
        + p105
        + p106
        + p201
        + p202
        + p203
        + p204
        + p205
        + p206
        + p301
        + p302
        + p303
        + p304
        + p305
        + p306
    ) / 24.0

    pred = np.clip(pred.astype(np.float32, copy=False), 0.0, 1.0)
    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": pred})
    return df


test_case_paths = sorted([f.path for f in os.scandir(TEST_DIR) if f.is_dir()])
test_cases = [os.path.basename(p) for p in test_case_paths]

sub_df = create_sub(
    test_cases,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
)

sub_df.head(), sub_df.shape




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1375844622.py in <cell line: 0>()
     66 sub_df = create_sub(
     67     test_cases,
---> 68     prediction_1,
     69     prediction_2,
     70     prediction_3,

NameError: name 'prediction_1' is not defined

## === cell 11
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

sub_df.to_csv("submission.csv", index=False)
sub_df.head()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/163311345.py in <cell line: 0>()
----> 1 sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
      2 sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
      3 
      4 sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
      5 sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

NameError: name 'sub_df' is not defined

## === cell 12
assert os.path.isfile("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["BraTS21ID", "MGMT_value"]
assert chk["MGMT_value"].between(0, 1).all()
chk.shape, chk.describe(include="all")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/821736153.py in <cell line: 0>()
----> 1 assert os.path.isfile("submission.csv")
      2 chk = pd.read_csv("submission.csv")
      3 assert list(chk.columns) == ["BraTS21ID", "MGMT_value"]
      4 assert chk["MGMT_value"].between(0, 1).all()
      5 chk.shape, chk.describe(include="all")

AssertionError:
