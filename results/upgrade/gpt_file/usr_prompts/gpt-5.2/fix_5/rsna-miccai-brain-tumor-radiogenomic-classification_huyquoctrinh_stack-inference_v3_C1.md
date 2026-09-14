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

0.61059

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61059) has done: 'The timeout is dominated by Python-side DICOM discovery/reading: repeated `glob+sorted+lambda` for every patient plus slow `pydicom.dcmread` calls, all done serially. I keep the exact slice-selection logic (middle 50%, step=3 if >=10) and the same model/training, but speed up data loading by (1) using `os.scandir` + numeric parsing instead of `glob+sorted(key=...)`, (2) reading DICOMs with `stop_before_pixels=True` then decoding pixels once, (3) parallelizing slice decoding with a deterministic thread pool (I/O-bound), and (4) reducing pandas per-row overhead by iterating over pre-extracted numpy arrays. These changes are equivalent in semantics (same files selected and same pixel preprocessing) but drastically reduce wall time.'
- What this solution (achieved 0.61059) has done: 'I fix the runtime crash happening before any training by addressing the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known protobuf/TensorFlow compatibility issue in Kaggle images; the minimal robust fix is to force the Python protobuf implementation via an environment variable set before importing TensorFlow. I also remove a redundant double `pydicom.dcmread` call in `load_dicom` that wastes time and can increase failure risk, while keeping the exact same image normalization/resizing semantics. Finally, I keep the model/training logic unchanged and ensure the submission is always written as `submission.csv` with the required columns and ID formatting.'
- What this solution (achieved 0.61059) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf implementation is forced early enough and by adding a safe fallback that restarts the kernel process if TensorFlow was imported before the env var took effect. I also add a small guard so the pipeline doesn’t crash if no DICOM slices are read (it then write a valid 0.5 baseline submission), while keeping the same data selection, model, and training logic when data loads correctly. Finally, I keep the exact submission formatting and ensure `submission.csv` is always produced in the working directory.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import random
import warnings

import numpy as np
import pandas as pd
import cv2
import pydicom
from tqdm.notebook import tqdm

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    msg = repr(e)
    if ("GetPrototype" in msg) or ("protobuf" in msg.lower()):
        if os.environ.get("__TF_PROTOBUF_RESTARTED__", "0") != "1":
            os.environ["__TF_PROTOBUF_RESTARTED__"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)
    raise

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_LABELS_PATH)
test_df = pd.read_csv(SAMPLE_SUB_PATH)

train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)

print(train_df.shape, test_df.shape)
print(train_df.head())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _extract_image_index(filename: str) -> int:
    base = os.path.splitext(filename)[0]
    try:
        return int(base.split("-")[-1])
    except Exception:
        return -1


def load_dicom(path, size=224):
    """
    Robust DICOM reader:
    - uses pydicom.dcmread
    - converts to uint8 image
    - resizes to (size, size)
    Returns None if the slice cannot be read/decoded.
    """
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array.astype(np.float32)

        mx = float(np.max(data))
        if mx > 0:
            data = data / mx
        data = (data * 255.0).clip(0, 255).astype(np.uint8)

        return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        return None


def get_all_image_paths(brats21id, image_type, folder="train"):
    assert image_type in TYPES

    patient_path = os.path.join(
        DATA_ROOT, folder, str(int(brats21id)).zfill(5), image_type
    )

    try:
        entries = []
        with os.scandir(patient_path) as it:
            for e in it:
                if e.is_file():
                    idx = _extract_image_index(e.name)
                    if idx >= 0:
                        entries.append((idx, e.path))
    except FileNotFoundError:
        return np.array([], dtype=object)

    if not entries:
        return np.array([], dtype=object)

    entries.sort(key=lambda t: t[0])
    paths = [p for _, p in entries]

    num_images = len(paths)
    start = int(num_images * 0.25)
    end = int(num_images * 0.75)
    interval = 3 if num_images >= 10 else 1

    return np.array(paths[start:end:interval], dtype=object)


from concurrent.futures import ThreadPoolExecutor


def get_all_images(brats21id, image_type, folder="train", size=224, max_workers=8):
    paths = get_all_image_paths(brats21id, image_type, folder)
    if len(paths) == 0:
        return []

    def _load(p):
        return load_dicom(p, size=size)

    imgs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for im in ex.map(_load, list(paths)):
            if im is not None:
                imgs.append(im)
    return imgs




## === cell 2
IMAGE_SIZE = 128


def get_all_data_for_train(image_type):
    X, y, train_ids = [], [], []

    ids = train_df["BraTS21ID"].to_numpy()
    labels = train_df["MGMT_value"].to_numpy(dtype=np.float32)

    for brats_id, label in tqdm(list(zip(ids, labels)), total=len(ids)):
        brats_id = int(brats_id)

        images = get_all_images(brats_id, image_type, folder="train", size=IMAGE_SIZE)
        if len(images) == 0:
            continue

        X.extend(images)
        y.extend([float(label)] * len(images))
        train_ids.extend([brats_id] * len(images))

    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y, dtype=np.float32)
    train_ids = np.asarray(train_ids, dtype=np.int32)
    return X, y, train_ids


def get_all_data_for_test(image_type):
    X, test_ids = [], []

    ids = test_df["BraTS21ID"].to_numpy()
    for brats_id in tqdm(ids, total=len(ids)):
        brats_id = int(brats_id)

        images = get_all_images(brats_id, image_type, folder="test", size=IMAGE_SIZE)
        if len(images) == 0:
            continue

        X.extend(images)
        test_ids.extend([brats_id] * len(images))

    X = np.asarray(X, dtype=np.uint8)
    test_ids = np.asarray(test_ids, dtype=np.int32)
    return X, test_ids




## === cell 3
X, y, trainidt = get_all_data_for_train("T1wCE")
X_test, testidt = get_all_data_for_test("T1wCE")

print("Train slices:", X.shape, y.shape, trainidt.shape)
print("Test slices:", X_test.shape, testidt.shape)

if X.size == 0 or X_test.size == 0:
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    submission = sample.copy()
    submission["MGMT_value"] = 0.5
    submission["BraTS21ID"] = submission["BraTS21ID"].apply(
        lambda x: str(int(x)).zfill(5)
    )
    submission.to_csv("submission.csv", index=False)
    print(
        "No slices loaded; wrote baseline submission.csv with shape:", submission.shape
    )
    raise SystemExit(0)

X = (X.astype(np.float32) / 255.0)[..., None]
X_test = (X_test.astype(np.float32) / 255.0)[..., None]

assert X.ndim == 4 and X.shape[-1] == 1
assert X_test.ndim == 4 and X_test.shape[-1] == 1



## === cell 4
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(X_tr.shape, X_va.shape, y_tr.mean(), y_va.mean())




## === cell 5
def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_model(X.shape[1:])
model.summary()



## === cell 6
BATCH_SIZE = 64
EPOCHS = 3

history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 7
y_pred_test = model.predict(X_test, batch_size=128, verbose=0).reshape(-1)

pred_df = pd.DataFrame(
    {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred_test.astype(float)}
)
pred_df = pred_df.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

sample = pd.read_csv(SAMPLE_SUB_PATH)
sample_ids = sample["BraTS21ID"].astype(int).values

pred_df = pred_df.set_index("BraTS21ID").reindex(sample_ids)
pred_df["MGMT_value"] = pred_df["MGMT_value"].astype(float).fillna(0.5)

submission = pd.DataFrame(
    {"BraTS21ID": sample_ids, "MGMT_value": pred_df["MGMT_value"].values}
)
submission["BraTS21ID"] = submission["BraTS21ID"].apply(lambda x: str(int(x)).zfill(5))

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
