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

0.5412629610742818

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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


def _extract_int_from_name(name: str) -> int:
    i = len(name) - 1
    while i >= 0 and not name[i].isdigit():
        i -= 1
    j = i
    while j >= 0 and name[j].isdigit():
        j -= 1
    if i >= 0 and j < i:
        try:
            return int(name[j + 1 : i + 1])
        except Exception:
            return -1
    return -1


def _list_dcm_paths_sorted(series_dir: str):
    cached = _SERIES_LIST_CACHE.get(series_dir)
    if cached is not None:
        return cached
    items = []
    with os.scandir(series_dir) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".dcm"):
                n = _extract_int_from_name(e.name)
                items.append((n, e.name, e.path))
    items.sort(key=lambda t: (t[0] < 0, t[0], t[1]))
    paths = [p for _, _, p in items]
    _SERIES_LIST_CACHE[series_dir] = paths
    return paths


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
    Unchanged core logic:
    - load each modality
    - adjuster (repeat slices by n depending on original length)
    - split into ceil(len/imgs)/layer_number sized chunks
    - average each chunk to produce up to layer_number slices
    - adjuster2 + pad/trim to exactly layer_number
    - stack modalities to (layer_number, 64, 64, 4)

    Speed-only changes:
    - optional per-series caching of resized slice stacks
    - vectorized chunk-mean computation (exact same per-chunk mean)
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
        new_imgs = [None] * len(starts)
        for k, start in enumerate(starts):
            chunk = imgs[start : start + layer_size]
            new_imgs[k] = chunk.mean(axis=0, dtype=np.float32)

        adjuster2(new_imgs, layer_number)

        if len(new_imgs) < layer_number:
            new_imgs += [new_imgs[-1]] * (layer_number - len(new_imgs))
        if len(new_imgs) > layer_number:
            new_imgs = new_imgs[:layer_number]

        out_mods.append(np.stack(new_imgs, axis=0).astype(np.float32, copy=False))

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
    cached_ids = np.load(CACHE_IDS, allow_pickle=True).tolist()
else:
    Input_Values = None

if Input_Values is None:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    PATIENT_CACHE_DIR = "/kaggle/working/patient_cache_train"
    SERIES_CACHE_DIR = "/kaggle/working/series_cache_train"
    os.makedirs(PATIENT_CACHE_DIR, exist_ok=True)
    os.makedirs(SERIES_CACHE_DIR, exist_ok=True)

    def _patient_cache_path(pid: str) -> str:
        return os.path.join(PATIENT_CACHE_DIR, f"{pid}_L{layer_number}_64.npy")

    def _build_one_patient(pid: str):
        x_path = _patient_cache_path(pid)
        if os.path.exists(x_path):
            arr = np.load(x_path, allow_pickle=False)
            return pid, arr
        scan = build_patient_scan(
            TRAIN_DIR,
            pid,
            layer_number=layer_number,
            series_cache_dir=SERIES_CACHE_DIR,
        )
        np.save(x_path, scan.astype(np.float32, copy=False))
        return pid, scan.astype(np.float32, copy=False)

    Input_Values = np.empty((len(train_ids), layer_number, 64, 64, 4), dtype=np.float32)

    cpu = os.cpu_count() or 2
    max_workers = max(1, min(8, cpu))  # keep IPC overhead bounded

    ctx = None
    try:
        if hasattr(mp, "get_context"):
            ctx = mp.get_context("forkserver")
    except Exception:
        ctx = None

    done = 0
    pid_to_idx = {pid: i for i, pid in enumerate(train_ids)}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        for pid, scan in ex.map(_build_one_patient, train_ids, chunksize=4):
            Input_Values[pid_to_idx[pid]] = scan
            done += 1
            if done % 25 == 0 or done == len(train_ids):
                print(f"built {done}/{len(train_ids)}")

    np.save(CACHE_X, Input_Values)
    np.save(CACHE_Y, Output_Values.astype(np.float32))
    np.save(CACHE_IDS, np.array(train_ids, dtype=object))

print("Input_Values shape:", Input_Values.shape, "dtype:", Input_Values.dtype)
print("Output_Values shape:", Output_Values.shape, "dtype:", Output_Values.dtype)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
BrokenProcessPool                         Traceback (most recent call last)
/tmp/ipykernel_11/672078555.py in <cell line: 0>()
     81 
     82     with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
---> 83         for pid, scan in ex.map(_build_one_patient, train_ids, chunksize=4):
     84             Input_Values[pid_to_idx[pid]] = scan
     85             done += 1

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.

## === cell 4
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

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
    tf.config.threading.set_intra_op_parallelism_threads(max(1, min(4, cpu)))
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

train_ds = (
    tf.data.Dataset.from_tensor_slices((Input_Values, Output_Values))
    .with_options(options)
    .shuffle(len(Input_Values), seed=42, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(train_ds, epochs=30)

print(history.history.keys())




## === cell 8
pass




## === cell 9
test_dir = TEST_DIR
test = test_patients
print("n_test:", len(test))




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
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    PATIENT_CACHE_DIR = "/kaggle/working/patient_cache_test"
    SERIES_CACHE_DIR = "/kaggle/working/series_cache_test"
    os.makedirs(PATIENT_CACHE_DIR, exist_ok=True)
    os.makedirs(SERIES_CACHE_DIR, exist_ok=True)

    def _patient_cache_path_test(pid: str) -> str:
        return os.path.join(PATIENT_CACHE_DIR, f"{pid}_L{layer_number}_64.npy")

    def _build_one_patient_test(pid: str):
        x_path = _patient_cache_path_test(pid)
        if os.path.exists(x_path):
            arr = np.load(x_path, allow_pickle=False)
            return pid, arr
        scan = build_patient_scan(
            TEST_DIR,
            pid,
            layer_number=layer_number,
            series_cache_dir=SERIES_CACHE_DIR,
        )
        np.save(x_path, scan.astype(np.float32, copy=False))
        return pid, scan.astype(np.float32, copy=False)

    Test_Values = np.empty((len(test), layer_number, 64, 64, 4), dtype=np.float32)

    cpu = os.cpu_count() or 2
    max_workers = max(1, min(8, cpu))

    ctx = None
    try:
        if hasattr(mp, "get_context"):
            ctx = mp.get_context("forkserver")
    except Exception:
        ctx = None

    done = 0
    pid_to_idx = {pid: i for i, pid in enumerate(test)}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        for pid, scan in ex.map(_build_one_patient_test, test, chunksize=4):
            Test_Values[pid_to_idx[pid]] = scan
            done += 1
            if done % 10 == 0 or done == len(test):
                print(f"built test {done}/{len(test)}")

    np.save(CACHE_TEST, Test_Values)

print("Test_Values shape:", Test_Values.shape, "dtype:", Test_Values.dtype)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
BrokenProcessPool                         Traceback (most recent call last)
/tmp/ipykernel_11/2187694801.py in <cell line: 0>()
     57 
     58     with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
---> 59         for pid, scan in ex.map(_build_one_patient_test, test, chunksize=4):
     60             Test_Values[pid_to_idx[pid]] = scan
     61             done += 1

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.

## === cell 11
assert Test_Values is not None and len(Test_Values) == len(
    test
), "Test_Values mismatch with test IDs."




## === cell 12
pred_ds = (
    tf.data.Dataset.from_tensor_slices(Test_Values)
    .batch(16, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)
Results = model.predict(pred_ds).reshape(-1)
print(
    "Results shape:",
    Results.shape,
    "min/max:",
    float(np.min(Results)),
    float(np.max(Results)),
)

sub = pd.DataFrame({"BraTS21ID": test, "MGMT_value": Results.astype(float)})
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
