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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.6) has done: 'I fix the import/runtime crash caused by incompatible optional packages (pympler/protobuf) by removing unused imports and relying only on Kaggle’s standard libraries. Since the referenced pretrained model file is not available in your input paths, I replace that load with a small TensorFlow/Keras CNN trained on the same extracted T2w slices (same overall “predict from images with a CNN” core idea) so the notebook can run end-to-end and output valid probabilities. I also fix multiple logic bugs in the data pipeline (missing `resize` import usage, list/ndarray normalization, incorrect slice predictions using pixels_1/pixels_2 for later predictions, and submission alignment per-case). Finally, I ensure the submission uses the exact required columns and ID formatting and writes `submission.csv` successfully.'
- What this solution (achieved 0.62941) has done: 'I fix the runtime crash happening at import time by avoiding the `pydicom` dependency that’s triggering a protobuf incompatibility in this environment, and instead read the DICOM slices using SimpleITK (commonly available on Kaggle for this competition). The rest of the pipeline stays the same: extract T2w slices, resize/normalize, train the same small Keras CNN, and average per-slice predictions into per-case probabilities. I also add a small safety check to ensure test-case ordering aligns with `sample_submission.csv` and that missing predictions are filled with 0.5, producing a valid `submission.csv`. These changes should restore end-to-end execution and keep (or slightly improve) the score without changing the modeling approach.'
- What this solution (achieved 0.65059) has done: 'The import-time crash comes from a protobuf/SimpleITK incompatibility; to make the pipeline run end-to-end we remove the SimpleITK dependency and read DICOMs with `pydicom` (which is available in the RSNA Kaggle environment). The rest of the solution stays the same: load a small fixed number of T2w slices per case, resize/normalize, train the same small Keras CNN, and average per-slice predictions into per-case probabilities. I also add a tiny robustness fix so slice ordering is numeric (not lexicographic) and ensure the submission aligns exactly to `sample_submission.csv` with proper zero-padded IDs. These changes are runtime-critical and should be score-neutral to slightly positive without changing the core modeling approach.'
- What this solution (achieved 0.5) has done: 'The crash happens before any training because importing `pydicom` triggers a protobuf `MessageFactory.GetPrototype` incompatibility in this Kaggle environment. To keep the same core approach (read T2w DICOM slices → resize/normalize → small Keras CNN → average slice predictions per case), I replace the DICOM reader with a pure-standard-library fallback that reads pixel data from the DICOM file directly (no `pydicom`/SimpleITK). I also add a small safety check to ensure we always return exactly `N_SLICES` slices per case and that submission IDs/predictions stay aligned to `sample_submission.csv`. These changes are runtime-critical and should be score-neutral to slightly positive by restoring correct image loading deterministically.'

# 9. Code solution

## === cell 0
import os
import re
import struct
import numpy as np
import pandas as pd

from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
tf.random.set_seed(0)



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

bad_cases = set(["00109", "00123", "00709"])

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)
labels_df.head()



## === cell 2
_num_re = re.compile(r"(\d+)")


def _numeric_key_from_filename(path):
    base = os.path.basename(path)
    m = _num_re.findall(base)
    return int(m[-1]) if m else 0


def _read_dcm_pixel_array(path):
    with open(path, "rb") as f:
        data = f.read()

    tag = b"\xE0\x7F\x10\x00"  # (7FE0,0010) little endian
    i = data.find(tag)
    if i == -1:
        raise ValueError("PixelData tag not found")

    vr = data[i + 4 : i + 6]
    if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
        length = struct.unpack("<I", data[i + 8 : i + 12])[0]
        start = i + 12
    else:
        length = struct.unpack("<H", data[i + 6 : i + 8])[0]
        start = i + 8

    pixel_bytes = data[start : start + length]
    if len(pixel_bytes) == 0:
        raise ValueError("Empty PixelData")

    if len(pixel_bytes) % 2 == 0:
        arr16 = np.frombuffer(pixel_bytes, dtype="<u2")
        if arr16.size > 0 and (arr16.max(initial=0) > 255 or arr16.size >= 512 * 512):
            arr = arr16.astype(np.float32)
        else:
            arr = np.frombuffer(pixel_bytes, dtype=np.uint8).astype(np.float32)
    else:
        arr = np.frombuffer(pixel_bytes, dtype=np.uint8).astype(np.float32)

    n = int(np.sqrt(arr.size))
    if n * n == arr.size:
        arr = arr.reshape(n, n)
    else:
        for s in (512, 256, 384):
            if arr.size >= s * s:
                arr = arr[: s * s].reshape(s, s)
                break
        else:
            s = int(np.floor(np.sqrt(arr.size)))
            arr = arr[: s * s].reshape(s, s)
    return arr


def _load_case_t2_slices(case_dir, img_px_size=150, n_slices=8):
    t2_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2_dir):
        return None

    img_paths = [
        f.path
        for f in os.scandir(t2_dir)
        if f.is_file() and f.name.lower().endswith(".dcm")
    ]
    if len(img_paths) == 0:
        return None

    img_paths = sorted(img_paths, key=_numeric_key_from_filename)

    collected = []
    for p in img_paths:
        try:
            arr = _read_dcm_pixel_array(p)
        except Exception:
            continue

        if float(arr.sum()) > 100000:
            img = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked = np.stack((img,) * 3, axis=-1)
            mx = float(np.max(stacked))
            if mx <= 0:
                continue
            stacked_norm = stacked / mx
            if float(stacked_norm.sum()) > 2500:
                collected.append(stacked_norm)
                if len(collected) >= n_slices:
                    break

    if len(collected) == 0:
        mid = img_paths[len(img_paths) // 2]
        try:
            arr = _read_dcm_pixel_array(mid)
            img = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked = np.stack((img,) * 3, axis=-1)
            mx = float(np.max(stacked))
            if mx > 0:
                collected = [stacked / mx]
        except Exception:
            pass

    if len(collected) == 0:
        collected = [np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)]

    while len(collected) < n_slices:
        collected.append(collected[-1].copy())

    return np.stack(collected[:n_slices], axis=0).astype(np.float32)


def load_dataset_t2(dir_path, ids, img_px_size=150, n_slices=8):
    xs = []
    valid_ids = []
    for sid in ids:
        case_dir = os.path.join(dir_path, sid)
        arr = _load_case_t2_slices(case_dir, img_px_size=img_px_size, n_slices=n_slices)
        if arr is None:
            continue
        xs.append(arr)
        valid_ids.append(sid)
    if len(xs) == 0:
        return (
            np.empty((0, n_slices, img_px_size, img_px_size, 3), dtype=np.float32),
            [],
        )
    return np.stack(xs, axis=0).astype(np.float32), valid_ids




## === cell 3
IMG_PX_SIZE = 150
N_SLICES = 8

train_ids = labels_df["BraTS21ID"].tolist()
X_cases, valid_train_ids = load_dataset_t2(
    TRAIN_DIR, train_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)

valid_mask = labels_df["BraTS21ID"].isin(valid_train_ids)
y_cases = labels_df.loc[valid_mask, "MGMT_value"].astype(np.float32).values

X_train = X_cases.reshape(-1, IMG_PX_SIZE, IMG_PX_SIZE, 3).astype(np.float32)
y_train = np.repeat(y_cases, N_SLICES).astype(np.float32)

X_train.shape, y_train.shape, len(valid_train_ids)



## === cell 4
model = keras.Sequential(
    [
        layers.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

BATCH_SIZE = 32
EPOCHS = 2

_ = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
)



## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sample_sub["BraTS21ID"].tolist()

X_test_cases, valid_test_ids = load_dataset_t2(
    TEST_DIR, test_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
X_test = X_test_cases.reshape(-1, IMG_PX_SIZE, IMG_PX_SIZE, 3).astype(np.float32)

X_test.shape, len(valid_test_ids)



## === cell 6
slice_preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).reshape(-1)

n_case = len(valid_test_ids)
expected = n_case * N_SLICES
if slice_preds.size != expected:
    if slice_preds.size < expected:
        pad = np.full((expected - slice_preds.size,), 0.5, dtype=slice_preds.dtype)
        slice_preds = np.concatenate([slice_preds, pad], axis=0)
    else:
        slice_preds = slice_preds[:expected]

case_preds = slice_preds.reshape(-1, N_SLICES).mean(axis=1)

pred_map = {sid: float(p) for sid, p in zip(valid_test_ids, case_preds)}
sub_df = sample_sub.copy()
sub_df["MGMT_value"] = sub_df["BraTS21ID"].map(pred_map)
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0).astype(float)

sub_df.head(), sub_df["MGMT_value"].describe()



## === cell 7
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head(3).to_string(index=False))
