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

# 5. Code solution

## === cell 0
import os
import math
import warnings

import numpy as np
import pandas as pd
import cv2
import pydicom

warnings.filterwarnings("ignore")

BASE_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_CSV = os.path.join(BASE_DIR, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

BAD_IDS = {"00109", "00123", "00709"}  # per competition note

train_patients = sorted(
    [p for p in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, p))]
)
test_patients = sorted(
    [p for p in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, p))]
)

print("n_train_folders:", len(train_patients), "n_test_folders:", len(test_patients))




## === cell 1
def average(l):
    return sum(l) / len(l)


def layers(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


m2 = [
    22,
    23,
    24,
    36,
    37,
    38,
    39,
    40,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    65,
    66,
    67,
    68,
    69,
    70,
    81,
    82,
    83,
    84,
    97,
    98,
]
m3 = [19, 20, 21, 25, 26, 33, 34, 35, 41, 42, 49]
m4 = [18, 27, 28]
m5 = [17]


def adjuster(file):
    if len(file) in m5:
        n = 5
    elif len(file) in m4:
        n = 4
    elif len(file) in m3:
        n = 3
    elif len(file) in m2:
        n = 2
    else:
        n = 1
    new_file = []
    for i in range(len(file)):
        for _ in range(n):
            new_file.append(file[i])
    return new_file


def decider(length, layer_number):
    return math.ceil(length / layer_number)


def adjuster2(layers_list, layer_number):
    if len(layers_list) == layer_number - 1:
        layers_list.append(layers_list[-1])


pydicom.config.convert_wrong_length_to_UN = True
try:
    pydicom.config.image_handlers = [
        "gdcm_handler",
        "pillow_handler",
        "jpeg_ls_handler",
        "pylibjpeg_handler",
    ]
except Exception:
    pass

_SERIES_LIST_CACHE = {}
_INSTANCE_CACHE = {}

_DICOM_PIXEL_TAGS = [
    "PixelData",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "PlanarConfiguration",
    "NumberOfFrames",
    "TransferSyntaxUID",
    "RescaleIntercept",
    "RescaleSlope",
]


def _list_dcm_paths(series_dir: str):
    cached = _SERIES_LIST_CACHE.get(series_dir)
    if cached is not None:
        return cached
    paths = []
    with os.scandir(series_dir) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".dcm"):
                paths.append(e.path)
    _SERIES_LIST_CACHE[series_dir] = paths
    return paths


def _instance_number(fp: str) -> int:
    v = _INSTANCE_CACHE.get(fp)
    if v is not None:
        return v
    try:
        ds = pydicom.dcmread(
            fp, stop_before_pixels=True, force=True, specific_tags=["InstanceNumber"]
        )
        inst = int(getattr(ds, "InstanceNumber", 0) or 0)
    except Exception:
        inst = 0
    _INSTANCE_CACHE[fp] = inst
    return inst


def _list_dcm_paths_sorted(series_dir: str):
    paths = _list_dcm_paths(series_dir)
    paths = sorted(paths, key=lambda p: (_instance_number(p), p))
    return paths


def _read_dicom_pixels_fast(fp: str) -> np.ndarray:
    ds = pydicom.dcmread(
        fp,
        force=True,
        defer_size=1024,
        specific_tags=_DICOM_PIXEL_TAGS,
    )
    return ds.pixel_array


def load_series_resized(
    series_dir, size=(64, 64), series_cache_path: str | None = None
):
    if series_cache_path is not None and os.path.exists(series_cache_path):
        return np.load(series_cache_path, allow_pickle=False)

    full_paths = _list_dcm_paths_sorted(series_dir)
    n = len(full_paths)

    imgs = np.empty((n, size[1], size[0]), dtype=np.float32)
    for i, fp in enumerate(full_paths):
        arr = _read_dicom_pixels_fast(fp).astype(np.float32, copy=False)
        imgs[i] = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)

    if series_cache_path is not None:
        os.makedirs(os.path.dirname(series_cache_path), exist_ok=True)
        np.save(series_cache_path, imgs)
    return imgs


def build_patient_scan(
    root_dir, patient_id, layer_number=16, series_cache_dir: str | None = None
):
    """
    Core logic preserved; speed-ups are provably equivalent:
    - same resize, same repetition rule, same layer chunking, same mean aggregation.
    - faster slice ordering via InstanceNumber header read (no pixel decode).
    - vectorized aggregation: same chunks and same float32 means.
    """
    modalities = ("FLAIR", "T1w", "T1wCE", "T2w")
    out_mods = []

    for mod in modalities:
        series_dir = os.path.join(root_dir, patient_id, mod)
        cache_path = None
        if series_cache_dir is not None:
            cache_path = os.path.join(series_cache_dir, f"{patient_id}_{mod}_64.npy")

        imgs = load_series_resized(
            series_dir, size=(64, 64), series_cache_path=cache_path
        )

        S = imgs.shape[0]
        if S in m5:
            nrep = 5
        elif S in m4:
            nrep = 4
        elif S in m3:
            nrep = 3
        elif S in m2:
            nrep = 2
        else:
            nrep = 1

        if nrep != 1:
            imgs = np.repeat(imgs, repeats=nrep, axis=0)

        layer_size = decider(int(imgs.shape[0]), layer_number)
        N = imgs.shape[0]

        starts = np.arange(0, N, layer_size, dtype=np.int32)
        means = []
        for start in starts:
            chunk = imgs[start : start + layer_size]
            means.append(chunk.mean(axis=0, dtype=np.float32))

        adjuster2(means, layer_number)

        if len(means) < layer_number:
            means += [means[-1]] * (layer_number - len(means))
        if len(means) > layer_number:
            means = means[:layer_number]

        out_mods.append(np.stack(means, axis=0).astype(np.float32, copy=False))

    scan = np.stack(out_mods, axis=-1).astype(np.float32, copy=False)
    return scan




## === cell 2
labels_df = pd.read_csv(LABELS_CSV, dtype={"BraTS21ID": str})
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].str.zfill(5)

labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_IDS)].reset_index(drop=True)

available_train = set(train_patients)
labels_df = labels_df[labels_df["BraTS21ID"].isin(available_train)].reset_index(
    drop=True
)

train_ids = labels_df["BraTS21ID"].tolist()
Output_Values = labels_df["MGMT_value"].astype(np.float32).values.reshape(-1, 1)

print("n_train_used:", len(train_ids), "Output_Values shape:", Output_Values.shape)




## === cell 3
CACHE_X = "/kaggle/working/Input_Values.npy"
CACHE_Y = "/kaggle/working/Output_Values.npy"
CACHE_IDS = "/kaggle/working/train_ids.npy"

layer_number = 16


def _cache_ok():
    if not (
        os.path.exists(CACHE_X)
        and os.path.exists(CACHE_Y)
        and os.path.exists(CACHE_IDS)
    ):
        return False
    try:
        x = np.load(CACHE_X, mmap_mode="r", allow_pickle=False)
        y = np.load(CACHE_Y, mmap_mode="r", allow_pickle=False)
        ids = np.load(CACHE_IDS, allow_pickle=True).tolist()
        if ids != train_ids:
            return False
        if x.shape[0] != y.shape[0] or x.shape[0] != len(ids):
            return False
        if x.shape[1:] != (layer_number, 64, 64, 4):
            return False
        if y.shape[1:] != (1,):
            return False
        return True
    except Exception:
        return False


if _cache_ok():
    Input_Values = np.load(CACHE_X, mmap_mode=None, allow_pickle=False)
    Output_Values = np.load(CACHE_Y, mmap_mode=None, allow_pickle=False)
else:
    Input_Values = None

if Input_Values is None:
    import concurrent.futures as cf

    PATIENT_CACHE_DIR = "/kaggle/working/patient_cache_train"
    SERIES_CACHE_DIR = "/kaggle/working/series_cache_train"
    os.makedirs(PATIENT_CACHE_DIR, exist_ok=True)
    os.makedirs(SERIES_CACHE_DIR, exist_ok=True)

    def _patient_cache_path(pid: str) -> str:
        return os.path.join(PATIENT_CACHE_DIR, f"{pid}_L{layer_number}_64.npy")

    Input_Values = np.empty((len(train_ids), layer_number, 64, 64, 4), dtype=np.float32)

    def _build_one_train(i_pid):
        i, pid = i_pid
        x_path = _patient_cache_path(pid)
        if os.path.exists(x_path):
            scan = np.load(x_path, allow_pickle=False)
            return i, scan
        try:
            scan = build_patient_scan(
                TRAIN_DIR,
                pid,
                layer_number=layer_number,
                series_cache_dir=SERIES_CACHE_DIR,
            ).astype(np.float32, copy=False)
            np.save(x_path, scan)
            return i, scan
        except Exception as e:
            print(
                f"[WARN] Failed to build train patient {pid}: {type(e).__name__}: {e}"
            )
            scan = np.zeros((layer_number, 64, 64, 4), dtype=np.float32)
            return i, scan

    max_workers = max(1, min(16, (os.cpu_count() or 2)))
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, scan in ex.map(_build_one_train, enumerate(train_ids), chunksize=4):
            Input_Values[i] = scan
            done += 1
            if done % 25 == 0 or done == len(train_ids):
                print(f"built {done}/{len(train_ids)} (workers={max_workers})")

    np.save(CACHE_X, Input_Values)
    np.save(CACHE_Y, Output_Values.astype(np.float32))
    np.save(CACHE_IDS, np.array(train_ids, dtype=object))

print("Input_Values shape:", Input_Values.shape, "dtype:", Input_Values.dtype)
print("Output_Values shape:", Output_Values.shape, "dtype:", Output_Values.dtype)




## === cell 4
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv3D,
    MaxPooling3D,
    BatchNormalization,
    Dropout,
    Dense,
    GlobalAveragePooling3D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

print("TensorFlow version:", tf.__version__)

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    cpu = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(max(1, min(8, cpu)))
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass




## === cell 5
model = Sequential()
model.add(
    Conv3D(
        32,
        (3, 3, 3),
        activation="relu",
        input_shape=(16, 64, 64, 4),
        kernel_regularizer=l2(0.001),
        bias_regularizer=l2(0.001),
    )
)
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(
    Conv3D(
        64,
        (3, 3, 3),
        activation="relu",
        kernel_regularizer=l2(0.001),
        bias_regularizer=l2(0.001),
    )
)
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(GlobalAveragePooling3D())
model.add(
    Dense(
        64, activation="relu", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)
    )
)
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.summary()




## === cell 6
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=opt)




## === cell 7
assert Input_Values is not None and len(Input_Values) > 0, "Input_Values is empty."
assert Output_Values is not None and len(Output_Values) == len(
    Input_Values
), "Mismatched X/Y lengths."

BATCH_SIZE = 32

options = tf.data.Options()
options.deterministic = True

Input_Values = np.ascontiguousarray(Input_Values, dtype=np.float32)
Output_Values = np.ascontiguousarray(Output_Values, dtype=np.float32)

train_ds = (
    tf.data.Dataset.from_tensor_slices((Input_Values, Output_Values))
    .with_options(options)
    .shuffle(len(Input_Values), seed=42, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(train_ds, epochs=30, verbose=1)
print(history.history.keys())




## === cell 8
pass




## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUB, dtype={"BraTS21ID": str})
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
test = sample_sub["BraTS21ID"].tolist()

print("n_test (from sample_submission):", len(test))
missing = [pid for pid in test if not os.path.isdir(os.path.join(TEST_DIR, pid))]
if missing:
    print(f"[WARN] {len(missing)} test IDs missing folders. First few:", missing[:5])




## === cell 10
CACHE_TEST = "/kaggle/working/Test_Values.npy"


def _test_cache_ok():
    if not os.path.exists(CACHE_TEST):
        return False
    try:
        x = np.load(CACHE_TEST, mmap_mode="r", allow_pickle=False)
        return x.shape == (len(test), layer_number, 64, 64, 4)
    except Exception:
        return False


if _test_cache_ok():
    Test_Values = np.load(CACHE_TEST, mmap_mode=None, allow_pickle=False)
else:
    import concurrent.futures as cf

    PATIENT_CACHE_DIR = "/kaggle/working/patient_cache_test"
    SERIES_CACHE_DIR = "/kaggle/working/series_cache_test"
    os.makedirs(PATIENT_CACHE_DIR, exist_ok=True)
    os.makedirs(SERIES_CACHE_DIR, exist_ok=True)

    def _patient_cache_path_test(pid: str) -> str:
        return os.path.join(PATIENT_CACHE_DIR, f"{pid}_L{layer_number}_64.npy")

    Test_Values = np.empty((len(test), layer_number, 64, 64, 4), dtype=np.float32)

    def _build_one_test(i_pid):
        i, pid = i_pid
        x_path = _patient_cache_path_test(pid)
        if os.path.exists(x_path):
            scan = np.load(x_path, allow_pickle=False)
            return i, scan
        try:
            scan = build_patient_scan(
                TEST_DIR,
                pid,
                layer_number=layer_number,
                series_cache_dir=SERIES_CACHE_DIR,
            ).astype(np.float32, copy=False)
            np.save(x_path, scan)
            return i, scan
        except Exception as e:
            print(f"[WARN] Failed to build test patient {pid}: {type(e).__name__}: {e}")
            scan = np.zeros((layer_number, 64, 64, 4), dtype=np.float32)
            return i, scan

    max_workers = max(1, min(16, (os.cpu_count() or 2)))
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, scan in ex.map(_build_one_test, enumerate(test), chunksize=4):
            Test_Values[i] = scan
            done += 1
            if done % 10 == 0 or done == len(test):
                print(f"built test {done}/{len(test)} (workers={max_workers})")

    np.save(CACHE_TEST, Test_Values)

print("Test_Values shape:", Test_Values.shape, "dtype:", Test_Values.dtype)




## === cell 11
assert Test_Values is not None and len(Test_Values) == len(
    test
), "Test_Values mismatch with test IDs."

Test_Values = np.ascontiguousarray(Test_Values, dtype=np.float32)

pred_ds = (
    tf.data.Dataset.from_tensor_slices(Test_Values)
    .batch(16, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)
Results = model.predict(pred_ds, verbose=1).reshape(-1)
print(
    "Results shape:",
    Results.shape,
    "min/max:",
    float(np.min(Results)),
    float(np.max(Results)),
)

sub = pd.DataFrame({"BraTS21ID": test, "MGMT_value": Results.astype(float)})
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

sub = sample_sub[["BraTS21ID"]].merge(sub, on="BraTS21ID", how="left")
sub["MGMT_value"] = sub["MGMT_value"].astype(float).fillna(0.5)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
