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

0.55882

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The timeout is dominated by Python-level DICOM I/O and per-slice TensorFlow ops (decode → resize → normalize) executed thousands of times sequentially for train+test. I keep the exact slice-selection logic and the exact model/training loop, but make the DICOM loading pipeline much faster by (1) caching decoded+preprocessed slices on disk in `/kaggle/working` so repeated runs don’t redo expensive decoding, (2) vectorizing/compiling the resize/normalize with `tf.function` to reduce eager overhead, and (3) parallelizing subject feature extraction with a thread pool (safe here because the workload is I/O-heavy and TensorFlow decoding releases the GIL). These changes are provably equivalent in outputs (same files, same order, same thresholds, same preprocessing math) aside from negligible float round-off, and do not alter the model or training semantics.'
- What this solution (achieved 0.55882) has done: 'The crash happens before any of your cells run because TensorFlow’s DICOM decoder triggers a protobuf incompatibility in this Kaggle image (`MessageFactory.GetPrototype`). To keep your core model and slice-selection logic intact while restoring end-to-end execution, I replace the TF-based DICOM read with a pure‑Python DICOM loader using `pydicom` (available in this competition environment) and keep the same resize/normalize path. I also pin `TF_USE_LEGACY_KERAS=1` before importing TensorFlow to avoid Keras/TensorFlow package mismatch issues on Kaggle. These changes are strictly I/O/compatibility fixes; training, architecture, and submission formatting stay the same, and it write `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import hashlib
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
tf.config.threading.set_inter_op_parallelism_threads(max(1, os.cpu_count() // 2))
tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Missing LABELS_CSV: {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing SAMPLE_SUB_CSV: {SAMPLE_SUB_CSV}"




## === cell 2
@tf.function(reduce_retracing=True)
def _resize_and_normalize_tf(img2d, size: int):
    x = tf.cast(img2d, tf.float32)
    x = tf.expand_dims(x, axis=-1)  # (H,W,1)
    x = tf.image.resize(x, (size, size), method="bilinear")
    x = tf.squeeze(x, axis=-1)  # (size,size)

    mn = tf.reduce_min(x)
    mx = tf.reduce_max(x)
    x = (x - mn) / (mx - mn + tf.constant(1e-6, tf.float32))
    return x  # float32 (size,size)


def resize2d(img2d: np.ndarray, size: int) -> np.ndarray:
    return _resize_and_normalize_tf(tf.convert_to_tensor(img2d), size).numpy()


def normalize_to_0_1(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    img = img.astype(np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))
    return (img - mn) / (mx - mn + eps)




## === cell 3
CACHE_DIR = "/kaggle/working/dcm_cache_v1"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key_for_modality(modality_dir: str, img_px_size: int, n_slices: int) -> str:
    s = f"{modality_dir}|{img_px_size}|{n_slices}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _cache_path_for_modality(modality_dir: str, img_px_size: int, n_slices: int) -> str:
    return os.path.join(
        CACHE_DIR, f"{_cache_key_for_modality(modality_dir, img_px_size, n_slices)}.npy"
    )


def _read_dicom_pixel_array_py(dcm_path: str) -> np.ndarray:
    import pydicom

    ds = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
    arr = ds.pixel_array  # typically uint16
    if arr is None:
        return None
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float32, copy=False)


def load_case_slices_from_modality(
    modality_dir: str, img_px_size: int = 150, n_slices: int = 6
) -> np.ndarray:
    cache_path = _cache_path_for_modality(modality_dir, img_px_size, n_slices)
    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        return arr.astype(np.float32, copy=False)

    if not os.path.isdir(modality_dir):
        arr = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
        np.save(cache_path, arr)
        return arr

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(modality_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    selected = []
    count = 0

    for fp in dcm_files:
        if count >= n_slices:
            break
        try:
            px = _read_dicom_pixel_array_py(fp)
        except Exception:
            continue

        if px is None:
            continue
        if float(np.sum(px)) <= 100000:
            continue

        px_norm = _resize_and_normalize_tf(
            tf.convert_to_tensor(px), img_px_size
        ).numpy()

        stacked = np.stack([px_norm, px_norm, px_norm], axis=-1)  # (H,W,3)
        if float(np.sum(stacked)) <= 2000:
            continue

        selected.append(stacked.astype(np.float32))
        count += 1

    if len(selected) == 0:
        arr = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    else:
        while len(selected) < n_slices:
            selected.append(np.zeros((img_px_size, img_px_size, 3), dtype=np.float32))
        arr = np.stack(selected[:n_slices], axis=0).astype(np.float32)

    np.save(cache_path, arr)
    return arr




## === cell 4
MODALITIES = ["T2w", "T1wCE"]
IMG_PX_SIZE = 150
N_SLICES = 6


def get_subject_dirs(root_dir: str):
    out = []
    for f in os.scandir(root_dir):
        if not f.is_dir():
            continue
        base = os.path.basename(f.path)
        if base.isdigit():
            out.append(f.path)
    return sorted(out)


def load_subject_features(subject_dir: str) -> np.ndarray:
    slices = []
    for m in MODALITIES:
        slices.append(
            load_case_slices_from_modality(
                os.path.join(subject_dir, m), IMG_PX_SIZE, N_SLICES
            )
        )
    return np.concatenate(slices, axis=0)


def subject_id_from_dir(subject_dir: str) -> int:
    base = os.path.basename(subject_dir)
    if not base.isdigit():
        raise ValueError(f"Non-numeric subject dir basename: {base} from {subject_dir}")
    return int(base)




## === cell 5
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int)

bad_ids = {109, 123, 709}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

train_subject_dirs = get_subject_dirs(TRAIN_DIR)
train_ids_available = {subject_id_from_dir(d) for d in train_subject_dirs}
labels_df = labels_df[labels_df["BraTS21ID"].isin(train_ids_available)].reset_index(
    drop=True
)

MAX_TRAIN_SUBJECTS = 160  # keep as-is
if len(labels_df) > MAX_TRAIN_SUBJECTS:
    labels_df = (
        labels_df.sort_values("BraTS21ID")
        .head(MAX_TRAIN_SUBJECTS)
        .reset_index(drop=True)
    )

print(labels_df.shape)
print(labels_df.head().to_string(index=False))




## === cell 6
from concurrent.futures import ThreadPoolExecutor

X_list = [None] * len(labels_df)
y = labels_df["MGMT_value"].astype(np.float32).to_numpy(copy=True)

id_to_dir = {subject_id_from_dir(d): d for d in train_subject_dirs}
brats_ids = labels_df["BraTS21ID"].astype(int).to_list()


def _load_one_train(i: int):
    brats_id = int(brats_ids[i])
    subj_dir = id_to_dir.get(brats_id)
    if subj_dir is None:
        return i, None
    feats = load_subject_features(subj_dir)  # (2*6,150,150,3)
    return i, feats


max_workers = min(8, max(1, (os.cpu_count() or 4)))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feats in ex.map(_load_one_train, range(len(labels_df))):
        X_list[i] = feats

keep_idx = [i for i, v in enumerate(X_list) if v is not None]
X = np.stack([X_list[i] for i in keep_idx], axis=0).astype(np.float32)
y = y[keep_idx]

print(X.shape, y.shape, float(y.mean()))




## === cell 7
S = X.shape[1]
H = X.shape[2]
W = X.shape[3]
C = X.shape[4]

inp = keras.Input(shape=(S, H, W, C), name="slices")

x = layers.TimeDistributed(layers.Conv2D(16, 3, padding="same", activation="relu"))(inp)
x = layers.TimeDistributed(layers.MaxPool2D())(x)
x = layers.TimeDistributed(layers.Conv2D(32, 3, padding="same", activation="relu"))(x)
x = layers.TimeDistributed(layers.MaxPool2D())(x)
x = layers.TimeDistributed(layers.Conv2D(64, 3, padding="same", activation="relu"))(x)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)  # (N, S, 64)

x = layers.GlobalAveragePooling1D()(x)  # (N, 64)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1, activation="sigmoid", name="MGMT_value")(x)

model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")

model.summary()




## === cell 8
from sklearn.model_selection import train_test_split

strat = (y > 0.5).astype(int)
unique, counts = np.unique(strat, return_counts=True)
use_stratify = (len(unique) == 2) and np.all(counts >= 2)

X_tr, X_va, y_tr, y_va = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=strat if use_stratify else None,
)

history = model.fit(
    X_tr, y_tr, validation_data=(X_va, y_va), epochs=5, batch_size=4, verbose=2
)




## === cell 9
test_subject_dirs = get_subject_dirs(TEST_DIR)
test_ids = [subject_id_from_dir(d) for d in test_subject_dirs]

X_test_list = [None] * len(test_subject_dirs)


def _load_one_test(i: int):
    return i, load_subject_features(test_subject_dirs[i])


max_workers = min(8, max(1, (os.cpu_count() or 4)))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feats in ex.map(_load_one_test, range(len(test_subject_dirs))):
        X_test_list[i] = feats

X_test = np.stack(X_test_list, axis=0).astype(np.float32)

pred = model.predict(X_test, batch_size=4, verbose=0).reshape(-1).astype(np.float32)

print(len(test_ids), X_test.shape, pred.shape, float(pred.min()), float(pred.max()))




## === cell 10
sample = pd.read_csv(SAMPLE_SUB_CSV)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

pred_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": pred})
sub_df = sample[["BraTS21ID"]].merge(pred_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = (
    sub_df["MGMT_value"].astype(np.float32).fillna(0.5).clip(0.0, 1.0)
)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].map(lambda x: f"{int(x):05d}")

print(sub_df.head().to_string(index=False))
print(sub_df.shape)

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape={sub_df.shape} and columns={list(sub_df.columns)}")
print(sub_df.head(10).to_string(index=False))
