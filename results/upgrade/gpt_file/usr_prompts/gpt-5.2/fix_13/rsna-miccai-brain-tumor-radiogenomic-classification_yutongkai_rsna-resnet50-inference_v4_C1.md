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

3.10

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

0.57647

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.57412) has done: 'I fix the test ID parsing bug that caused `BraTS21ID` to include the literal string `"test"` by correctly listing only numeric subject folders. Then I make TensorFlow/Keras imports robust to the known protobuf error in this environment by falling back to `tf.keras` (or exiting cleanly if no backend is available), which unblocks training and inference. Finally, I keep the same model/training logic but ensure the pipeline runs end-to-end and always writes a valid `submission.csv` with the required columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.57412) has done: 'I fix the TensorFlow import crash caused by the protobuf incompatibility (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before importing it, and by providing a robust fallback to `tf.keras` once it imports. This is a runtime-only fix that preserves the exact same model/training logic and should restore end-to-end execution. I also make the Keras fallback safer by not attempting standalone `keras` if TensorFlow fails for protobuf reasons (since it typically depends on the same stack here). Finally, I keep the existing submission alignment logic and ensure `submission.csv` is always written with the correct columns.'
- What this solution (achieved 0.57412) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf environment variables are set before any TensorFlow-related import happens, and by hard-disabling C++ protobuf in a way that works in Kaggle’s Python 3.10 environment. I also make the import path more robust by trying `tensorflow-cpu`-style TensorFlow import normally after setting env vars, without changing your model/training logic. Finally, I keep your existing data pipeline and submission alignment unchanged, so the produced `submission.csv` remains valid while unblocking end-to-end execution (and thus preserving/allowing your previous score level).'
- What this solution (achieved 0.57412) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf env vars before *any* import that might indirectly touch TensorFlow/protobuf, and by adding a safe fallback to write a valid baseline submission if TensorFlow still cannot import in this environment. I also ensure IDs are consistently treated as 5-digit strings when building the final submission (matching the competition format) without changing your core data/model logic. The model architecture, training loop, slice aggregation, and submission alignment remain the same; these changes are runtime/stability only. This should restore end-to-end execution and let you reproduce/improve upon the prior 0.57412 score instead of failing before training.'
- What this solution (achieved 0.57412) has done: 'I fix the TensorFlow/protobuf import crash that’s currently stopping training by setting the protobuf env vars *and* forcing the pure-Python protobuf module before any TensorFlow import, plus adding a safe fallback that still produces a valid `submission.csv` if TF cannot load. I also make the `tqdm` import robust (notebook vs console) to avoid runtime errors in non-notebook Kaggle runs. These changes are runtime/stability-only and preserve your existing data pipeline, model architecture, training loop, and submission aggregation, so the score should recover to (or slightly improve from) your previous 0.57412 rather than failing.'
- What this solution (achieved 0.57412) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow-related import* and by removing the explicit `google.protobuf.message_factory` import that triggers the `MessageFactory.GetPrototype` AttributeError in this environment. This is a runtime-only change that preserves your exact model, training loop, and inference logic, so it should restore end-to-end execution and recover your prior scoring behavior instead of falling back to the 0.5 baseline. I also keep the test ID parsing and submission alignment unchanged, ensuring `submission.csv` is always written with the required columns and correct 5-digit IDs.'
- What this solution (achieved 0.57647) has done: 'The timeout is dominated by DICOM decoding and repeated per-slice disk I/O across thousands of images; the model training itself is comparatively small. I keep the exact same slice selection and CNN/training loop, but make data loading faster by (1) avoiding `pydicom.dcmread(...).pixel_array` overhead via direct pixel-data decoding (same result), (2) resizing/normalizing in-place with fewer temporary arrays, and (3) eliminating per-patient `np.stack` and repeated allocations by writing each slice directly into the preallocated `X` array. I also reduce Python overhead in the threaded loader by mapping directly to a lean loader function and by using `executor.map` on the already-selected path list. These changes are computation/order equivalent and preserve labels, slice selection, and training semantics.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import random
import numpy as np
import pandas as pd

import pydicom
import cv2

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

try:
    import pylibjpeg  # noqa: F401
except Exception:
    pass



## === cell 1
TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # out of 255
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_df = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))

_test_glob = glob.glob(os.path.join(DATA_ROOT, "test", "*"))
test_ids = sorted(
    [
        os.path.basename(p)
        for p in _test_glob
        if os.path.isdir(p) and os.path.basename(p).isdigit()
    ]
)
test_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": 0.5})

train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)


def load_dicom(path, size=224):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)

        arr = None
        try:
            if getattr(ds, "PixelData", None) is not None:
                rows = int(ds.Rows)
                cols = int(ds.Columns)
                bits = int(getattr(ds, "BitsAllocated", 16))
                spp = int(getattr(ds, "SamplesPerPixel", 1))
                if spp == 1 and bits in (8, 16):
                    dtype = np.uint8 if bits == 8 else np.uint16
                    arr = np.frombuffer(ds.PixelData, dtype=dtype, count=rows * cols)
                    if arr.size == rows * cols:
                        arr = arr.reshape((rows, cols))
        except Exception:
            arr = None

        if arr is None:
            arr = ds.pixel_array
            if arr.ndim > 2:
                arr = arr[..., 0]

        arr = arr.astype(np.float32, copy=False)
        mx = float(arr.max()) if arr.size else 0.0
        if mx > 0:
            arr = arr * (255.0 / mx)
        else:
            arr = arr * 0.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8, copy=False)

        return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        return np.zeros((size, size), dtype=np.uint8)




## === cell 2
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import repeat

IMAGE_SIZE = 224


@lru_cache(maxsize=None)
def _list_sorted_dicom_paths(patient_path, image_type):
    d = os.path.join(patient_path, image_type)
    try:
        entries = []
        with os.scandir(d) as it:
            for e in it:
                if e.is_file():
                    name = e.name
                    stem = os.path.splitext(name)[0]
                    try:
                        idx = int(stem.split("-")[-1])
                    except Exception:
                        continue
                    entries.append((idx, e.path))
        entries.sort(key=lambda t: t[0])
        return [p for _, p in entries]
    except FileNotFoundError:
        return []


@lru_cache(maxsize=None)
def _get_all_image_paths_cached(brats21id_int, image_type, folder):
    assert image_type in TYPES
    patient_path = os.path.join(DATA_ROOT, folder, str(int(brats21id_int)).zfill(5))
    paths = _list_sorted_dicom_paths(patient_path, image_type)

    num_images = len(paths)
    if num_images == 0:
        return ()

    if num_images > 10:
        start = int(num_images * 0.25)
        end = int(num_images * 0.75)
    else:
        start = 0
        end = num_images

    interval = 1
    return tuple(paths[start:end:interval])


def get_all_image_paths(brats21id, image_type, folder="train"):
    return np.array(
        _get_all_image_paths_cached(int(brats21id), image_type, folder), dtype=object
    )


def get_all_images(brats21id, image_type, folder="train", size=224, executor=None):
    paths = _get_all_image_paths_cached(int(brats21id), image_type, folder)
    if not paths:
        return []
    if executor is None:
        return [load_dicom(p, size) for p in paths]
    return list(executor.map(load_dicom, paths, repeat(size), chunksize=16))


def _count_slices(df, image_type, folder):
    ids = df["BraTS21ID"].astype(int).to_numpy()
    counts = np.empty(ids.shape[0], dtype=np.int32)
    total = 0
    for i, bid in enumerate(ids):
        c = len(_get_all_image_paths_cached(int(bid), image_type, folder))
        counts[i] = c
        total += c
    return ids, counts, total


def get_all_data_for_train(image_type):
    ids, counts, total = _count_slices(train_df, image_type, "train")
    labels = train_df["MGMT_value"].to_numpy(dtype=np.float32)

    X = np.empty((total, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    y = np.empty((total,), dtype=np.float32)
    train_ids = np.empty((total,), dtype=np.int32)

    max_workers = min(16, max(4, (os.cpu_count() or 2)))
    pos = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in tqdm(range(len(ids))):
            bid = int(ids[i])
            n = int(counts[i])
            if n == 0:
                continue
            label = float(labels[i])

            imgs = get_all_images(bid, image_type, "train", IMAGE_SIZE, executor=ex)
            end = pos + n
            for j, im in enumerate(imgs):
                X[pos + j] = im
            y[pos:end] = label
            train_ids[pos:end] = bid
            pos = end

    if pos != total:
        X = X[:pos]
        y = y[:pos]
        train_ids = train_ids[:pos]
    return X, y, train_ids


def get_all_data_for_test(image_type):
    ids, counts, total = _count_slices(test_df, image_type, "test")
    X = np.empty((total, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    test_ids_arr = np.empty((total,), dtype=np.int32)

    max_workers = min(16, max(4, (os.cpu_count() or 2)))
    pos = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in tqdm(range(len(ids))):
            bid = int(ids[i])
            n = int(counts[i])
            if n == 0:
                continue

            imgs = get_all_images(bid, image_type, "test", IMAGE_SIZE, executor=ex)
            end = pos + n
            for j, im in enumerate(imgs):
                X[pos + j] = im
            test_ids_arr[pos:end] = bid
            pos = end

    if pos != total:
        X = X[:pos]
        test_ids_arr = test_ids_arr[:pos]
    return X, test_ids_arr




## === cell 3
X_train_raw, y_train, trainidt = get_all_data_for_train("T1wCE")
X_test_raw, testidt = get_all_data_for_test("T1wCE")

X_train = X_train_raw.astype(np.float32)
X_train *= 1.0 / 255.0
X_train = X_train[..., None]

X_test = X_test_raw.astype(np.float32)
X_test *= 1.0 / 255.0
X_test = X_test[..., None]

y_train = y_train.astype(np.float32)

print("Train slices:", X_train.shape, "Train labels:", y_train.shape)
print("Test slices:", X_test.shape, "Test ids:", testidt.shape)



## === cell 4
seed = 42
random.seed(seed)
np.random.seed(seed)

TF_AVAILABLE = True
tf_import_error = None

try:
    import tensorflow as tf  # noqa: F401

    tf.random.set_seed(seed)
    keras = tf.keras
    layers = tf.keras.layers
    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            min(8, os.cpu_count() or 2)
        )
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass
except Exception as e:
    TF_AVAILABLE = False
    tf_import_error = repr(e)
    print(
        "WARNING: TensorFlow failed to import; will use sklearn fallback and still write submission. Error:",
        tf_import_error,
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
y_pred = None

if TF_AVAILABLE:

    def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)):
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)  # probability for ROC-AUC
        model = keras.Model(inputs, outputs)
        return model

    model_best = build_model()

    model_best.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    history = model_best.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=3,
        batch_size=32,
        verbose=1,
    )

    y_pred = model_best.predict(X_test, batch_size=64, verbose=1).reshape(-1)
else:
    from sklearn.linear_model import LogisticRegression

    Xtr = X_train.reshape((X_train.shape[0], -1)).astype(np.float32)
    Xte = X_test.reshape((X_test.shape[0], -1)).astype(np.float32)

    lr = LogisticRegression(
        solver="liblinear",
        random_state=seed,
        max_iter=200,
    )
    lr.fit(Xtr, y_train.astype(int))
    y_pred = lr.predict_proba(Xte)[:, 1].astype(np.float64)



## === cell 6
sample = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
sample["BraTS21ID_int"] = sample["BraTS21ID"].astype(int)

if y_pred is not None:
    pred_df = pd.DataFrame(
        {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred.astype(float)}
    )
    pred_df = pred_df.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()
    pred_df_map = pred_df.set_index("BraTS21ID")["MGMT_value"]
    sample["MGMT_value"] = sample["BraTS21ID_int"].map(pred_df_map)
else:
    sample["MGMT_value"] = 0.5

sample["MGMT_value"] = sample["MGMT_value"].fillna(0.5).clip(0.0, 1.0)
sample["BraTS21ID"] = sample["BraTS21ID_int"].map(lambda x: str(int(x)).zfill(5))

submission = sample[["BraTS21ID", "MGMT_value"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("TF_AVAILABLE:", TF_AVAILABLE)
if tf_import_error is not None:
    print("TensorFlow import error was:", tf_import_error)
