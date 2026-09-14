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

0.43176

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.37647) has done: 'Most of the timeout is spent in DICOM I/O and decoding: the script reads thousands of slices per subject across 4 sequences for all train + test, and does it again every run. To keep the core model and training loop identical, the main speedup is to (1) aggressively avoid unnecessary DICOM header parsing by sorting with a fast filename-derived key when possible, (2) reduce per-slice overhead by using a single-pass slice selection and caching series directory listings, and (3) persist the extracted 4-channel volumes to disk as `.npy` so repeated runs (and test-time after training) don’t redo DICOM decoding work. These changes preserve the exact semantics of “evenly spaced slices, per-slice min-max normalization, resize, 4 sequences, same model/epochs/batch size” while cutting wall time dominated by repeated file parsing and pixel decoding.'
- What this solution (achieved 0.40118) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf incompatibility) by cleanly falling back to the existing sklearn pipeline when TF fails, instead of letting the notebook error out mid-run. I also ensure `Input_Values`/`Output_Values` are consistently aligned after filtering missing volumes so training doesn’t silently mismatch features and labels. Finally, I make submission generation robust by deriving the test ID list from `sample_submission.csv` (to guarantee exact row/order match) and writing a valid `submission.csv`. These changes preserve the model/training semantics you already have (CNN if TF works, otherwise the same LogisticRegression features) and should run end-to-end reliably in the Kaggle environment.'
- What this solution (achieved 0.47412) has done: 'I fix the TensorFlow/protobuf crash by making the TF import block robust to *any* exception (including the `MessageFactory.GetPrototype` AttributeError) and cleanly falling back to the existing sklearn pipeline so the notebook always runs end-to-end. I also add a small guard to ensure `Input_Values` and `Output_Values` stay aligned and non-empty after volume-building (to prevent silent shape/label mismatches). Finally, I keep the submission generation identical but make sure IDs are consistently zero-padded strings and predictions are written to `./submission.csv` with the exact required columns/order; these changes are score-neutral except that they prevent TF crashes that can otherwise stop a run or force an unintended path.'
- What this solution (achieved 0.37294) has done: 'I fix the TensorFlow/protobuf crash by making the TF import guard actually catch and disable TF before Keras is used, ensuring the script always completes and writes `submission.csv`. I also add a small, score-improving-but-still-minimal calibration step for the sklearn fallback: use a stratified train/validation split and Platt scaling (`CalibratedClassifierCV` with `sigmoid`) so predicted probabilities are better aligned with ROC-AUC, without changing the core feature extraction or base LogisticRegression model. Finally, I make path selection robust (use the existing `../input/...` but fall back to `/kaggle/input/...` if needed) and keep the submission IDs/order exactly matching `sample_submission.csv`.'
- What this solution (achieved 0.41412) has done: 'I fix the TensorFlow/protobuf crash so it’s correctly caught and the notebook cleanly falls back to the sklearn pipeline (right now the error is shown as an “ERROR in cell 3” even though the intent is to continue). I also make the TF probe more robust by importing TF in a separate guarded function and explicitly disabling TF usage on any exception, preventing downstream cells from trying to use Keras when TF is broken. To nudge ROC-AUC upward without changing the feature extraction or base model, I keep your LogisticRegression+Platt scaling setup but add `class_weight="balanced"` (a minimal, common calibration-friendly improvement for imbalanced medical labels). Finally, I keep submission generation identical but add a hard check that prediction length matches the sample submission length before writing `submission.csv`.'
- What this solution (achieved 0.37059) has done: 'I fix the TensorFlow/protobuf crash so it’s fully contained and the script reliably falls back to the sklearn pipeline (right now the run stops at the TF import error). I do this by moving TF probing into a separate subprocess: if TF can’t import cleanly, we set `use_tf=False` without raising, preserving your existing “TF CNN else sklearn LR” core logic. I also keep your volume caching/feature extraction/training semantics intact, only making the TF availability check robust and ensuring a valid `submission.csv` is always written. This should run end-to-end consistently in the Kaggle environment and (since TF is broken here) produce the same kind of sklearn-based score as before, but without execution failure risk.'
- What this solution (achieved 0.35294) has done: 'I fix the TensorFlow/protobuf crash so it’s fully contained: the current cell 3 still errors out because a TF-related `AttributeError` escapes before `use_tf` is set to False. To keep your core “TF CNN if available else sklearn LR” logic unchanged, I wrap both the subprocess probe and the in-process TF import in broad exception handlers and explicitly force `use_tf=False` on any failure so downstream cells don’t execute Keras code. This is primarily a correctness/stability fix (score-neutral) and ensure the script always runs end-to-end and writes a valid `submission.csv`. I also keep submission ID order exactly matching `sample_submission.csv` and keep all I/O paths unchanged.'
- What this solution (achieved 0.43176) has done: 'I fix the TensorFlow/protobuf crash that currently stops execution by ensuring TF is never imported in-process when the subprocess probe fails (the error is being raised outside your try/except path). This keeps your core “TF CNN if available else sklearn LR” logic intact, but makes the TF gate actually work so the notebook reliably falls back to the sklearn pipeline and always writes `./submission.csv`. I also make the TF probe more conservative by defaulting to `use_tf=False` unless the probe conclusively succeeds, preventing any accidental Keras usage after a failed probe. No feature extraction, model architecture, training loop, or submission formatting is changed beyond this stability fix.'

# 9. Code solution

## === cell 0
import os
import math
import re
import hashlib
import numpy as np
import pandas as pd
import pydicom
import cv2

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
try:
    cv2.setNumThreads(0)
except Exception:
    pass

RNG_SEED = 42
np.random.seed(RNG_SEED)

BASE_DIR = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
if not os.path.isdir(BASE_DIR):
    alt = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
    if os.path.isdir(alt):
        BASE_DIR = alt

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

patients = sorted(os.listdir(TRAIN_DIR)) if os.path.isdir(TRAIN_DIR) else []
test_dirs = sorted(os.listdir(TEST_DIR)) if os.path.isdir(TEST_DIR) else []


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


_imgnum_re = re.compile(r"image-(\d+)\.dcm$", re.IGNORECASE)
_series_listing_cache = {}


def _dcmread_header(path):
    return pydicom.dcmread(path, force=True, stop_before_pixels=True)


def _dcmread_pixels(path):
    return pydicom.dcmread(path, force=True)


def _slice_sort_key_from_header(ds):
    if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            pass
    if hasattr(ds, "InstanceNumber"):
        try:
            return int(ds.InstanceNumber)
        except Exception:
            pass
    return 0.0


def _list_dcm_paths_with_fast_key(series_dir):
    cached = _series_listing_cache.get(series_dir)
    if cached is not None:
        return cached

    out = []
    try:
        with os.scandir(series_dir) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if not name.lower().endswith(".dcm"):
                    continue
                m = _imgnum_re.search(name)
                if m:
                    out.append((e.path, int(m.group(1))))
                else:
                    out.append((e.path, None))
    except FileNotFoundError:
        out = []

    _series_listing_cache[series_dir] = out
    return out


def load_series_mid_slices(series_dir, layer_number=16, size=64):
    """
    Same semantics:
    - Sort slices by spatial/instance order (use filename-derived InstanceNumber when possible;
      otherwise fall back to DICOM header key).
    - Select layer_number evenly spaced indices.
    - For each selected slice: read pixels, min-max normalize per-slice, resize to (size,size).
    Returns: (layer_number, size, size) float32.
    """
    items = _list_dcm_paths_with_fast_key(series_dir)
    n = len(items)
    if n == 0:
        return np.zeros((layer_number, size, size), dtype=np.float32)

    need_hdr = [i for i, (_, k) in enumerate(items) if k is None]
    if need_hdr:
        items = list(items)
        for i in need_hdr:
            fp, _ = items[i]
            try:
                ds = _dcmread_header(fp)
                items[i] = (fp, _slice_sort_key_from_header(ds))
            except Exception:
                items[i] = (fp, 0.0)

    items.sort(key=lambda x: x[1])

    idxs = np.linspace(0, n - 1, num=layer_number, dtype=np.int32)

    out = np.zeros((layer_number, size, size), dtype=np.float32)
    for i, idx in enumerate(idxs.tolist()):
        fp = items[int(idx)][0]
        try:
            ds = _dcmread_pixels(fp)
            img = ds.pixel_array.astype(np.float32, copy=False)
            mn = float(img.min())
            mx = float(img.max())
            if mx > mn:
                img = (img - mn) / (mx - mn)
            else:
                img = img * 0.0
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            out[i] = img
        except Exception:
            out[i] = 0.0
    return out


def build_4ch_volume(subject_dir, layer_number=16, size=64):
    """
    Returns a (layer_number, size, size, 4) float32 tensor: [FLAIR, T1w, T1wCE, T2w]
    """
    flair = load_series_mid_slices(
        os.path.join(subject_dir, "FLAIR"), layer_number, size
    )
    t1w = load_series_mid_slices(os.path.join(subject_dir, "T1w"), layer_number, size)
    t1wce = load_series_mid_slices(
        os.path.join(subject_dir, "T1wCE"), layer_number, size
    )
    t2w = load_series_mid_slices(os.path.join(subject_dir, "T2w"), layer_number, size)
    vol = np.stack([flair, t1w, t1wce, t2w], axis=-1).astype(np.float32, copy=False)
    return vol


def _cache_key_for_subject(subj_dir, layer_number, size):
    key = f"{subj_dir}|ln={layer_number}|sz={size}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


CACHE_DIR = "./_vol_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def build_4ch_volume_cached(subject_dir, layer_number=16, size=64):
    ck = _cache_key_for_subject(subject_dir, layer_number, size)
    path = os.path.join(CACHE_DIR, f"{ck}.npy")
    if os.path.exists(path):
        try:
            arr = np.load(path, mmap_mode=None)
            if arr.shape == (layer_number, size, size, 4):
                return arr.astype(np.float32, copy=False)
        except Exception:
            pass
    vol = build_4ch_volume(subject_dir, layer_number=layer_number, size=size)
    try:
        np.save(path, vol)
    except Exception:
        pass
    return vol


print("BASE_DIR:", BASE_DIR)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)



## === cell 1
df = pd.read_csv(LABELS_PATH)

bad_ids = {"00109", "00123", "00709"}
df["BraTS21ID"] = df["BraTS21ID"].astype(str).str.zfill(5)
df_train = df[~df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

y = df_train["MGMT_value"].astype(np.float32).to_numpy().reshape(-1, 1)
train_ids = df_train["BraTS21ID"].tolist()

print("Train rows after exclusion:", len(train_ids), "Labels shape:", y.shape)



## === cell 2
layer_number = 16
size = 64

from concurrent.futures import ThreadPoolExecutor


def _build_one_train(idx_pid):
    idx, pid = idx_pid
    subj_dir = os.path.join(TRAIN_DIR, pid)
    if not os.path.isdir(subj_dir):
        return idx, pid, None
    vol = build_4ch_volume_cached(subj_dir, layer_number=layer_number, size=size)
    return idx, pid, vol


max_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
vols = [None] * len(train_ids)
kept = [False] * len(train_ids)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for j, (idx, pid, vol) in enumerate(
        ex.map(_build_one_train, enumerate(train_ids)), start=1
    ):
        if vol is not None:
            vols[idx] = vol
            kept[idx] = True
        if j % 50 == 0:
            print(f"Built train volumes: {j}/{len(train_ids)}")

keep_mask = np.asarray(kept, dtype=bool)
kept_ids = [pid for pid, k in zip(train_ids, keep_mask) if k]

if keep_mask.sum() == 0:
    raise RuntimeError(
        "No training volumes were built successfully; cannot train a model."
    )

Input_Values = np.stack([v for v in vols if v is not None], axis=0)
Output_Values = (
    df_train.loc[keep_mask, "MGMT_value"].astype(np.float32).to_numpy().reshape(-1, 1)
)

if Input_Values.shape[0] != Output_Values.shape[0]:
    raise RuntimeError(
        f"Train feature/label mismatch: X={Input_Values.shape[0]} vs y={Output_Values.shape[0]}"
    )

print("Kept volumes:", Input_Values.shape[0], "Kept labels:", Output_Values.shape[0])
print("Input_Values shape:", Input_Values.shape)
print("Output_Values shape:", Output_Values.shape)



## === cell 3
import subprocess, sys

use_tf = False
tf_import_error = None

tf = None
keras = None
Conv3D = MaxPooling3D = BatchNormalization = Dropout = Dense = (
    GlobalAveragePooling3D
) = LeakyReLU = None
Sequential = None
l2 = None

probe_code = r"""
import sys
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Dropout, Dense, GlobalAveragePooling3D, LeakyReLU
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.regularizers import l2
    print("OK", tf.__version__)
except Exception as e:
    print("FAIL", repr(e))
    sys.exit(1)
"""

try:
    res = subprocess.run(
        [sys.executable, "-c", probe_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    if res.returncode == 0:
        use_tf = True
    else:
        use_tf = False
        tf_import_error = (res.stdout or "").strip()[-500:]
except BaseException as e:
    use_tf = False
    tf_import_error = repr(e)

if use_tf:
    try:
        import tensorflow as _tf
        from tensorflow import keras as _keras
        from tensorflow.keras.layers import (
            Conv3D as _Conv3D,
            MaxPooling3D as _MaxPooling3D,
            BatchNormalization as _BatchNormalization,
            Dropout as _Dropout,
            Dense as _Dense,
            GlobalAveragePooling3D as _GlobalAveragePooling3D,
            LeakyReLU as _LeakyReLU,
        )
        from tensorflow.keras.models import Sequential as _Sequential
        from tensorflow.keras.regularizers import l2 as _l2

        tf = _tf
        keras = _keras
        Conv3D, MaxPooling3D, BatchNormalization, Dropout, Dense = (
            _Conv3D,
            _MaxPooling3D,
            _BatchNormalization,
            _Dropout,
            _Dense,
        )
        GlobalAveragePooling3D, LeakyReLU = _GlobalAveragePooling3D, _LeakyReLU
        Sequential, l2 = _Sequential, _l2

        try:
            tf.random.set_seed(RNG_SEED)
        except Exception:
            pass

        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
        except Exception:
            pass

    except BaseException as e:
        use_tf = False
        tf_import_error = repr(e)

print("TensorFlow available:", use_tf)
if not use_tf:
    print("TensorFlow import/probe error:", tf_import_error)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
if use_tf:
    model = Sequential()
    model.add(
        Conv3D(
            32,
            (3, 3, 3),
            activation="relu",
            input_shape=(16, 64, 64, 4),
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
        )
    )
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling3D(pool_size=(2, 3, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(
        Conv3D(
            64,
            (3, 3, 3),
            activation="relu",
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
        )
    )
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(GlobalAveragePooling3D())
    model.add(
        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()



## === cell 5
if use_tf:
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=opt)



## === cell 6
if use_tf:
    history = model.fit(
        Input_Values, Output_Values, epochs=45, batch_size=32, shuffle=True, verbose=2
    )



## === cell 7
if not use_tf:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV

    X = Input_Values.reshape(Input_Values.shape[0], -1, 4)
    feat_mean = X.mean(axis=1)
    feat_std = X.std(axis=1)
    X_feat = np.concatenate([feat_mean, feat_std], axis=1).astype(np.float32)
    y_bin = Output_Values.ravel().astype(int)

    base_lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000, random_state=RNG_SEED, class_weight="balanced"
                ),
            ),
        ]
    )

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_feat, y_bin, test_size=0.2, random_state=RNG_SEED, stratify=y_bin
    )
    base_lr.fit(X_tr, y_tr)

    clf = CalibratedClassifierCV(base_lr, method="sigmoid", cv="prefit")
    clf.fit(X_cal, y_cal)

    print("Trained calibrated sklearn LogisticRegression on features:", X_feat.shape)



## === cell 8
test_dir = TEST_DIR

sub_template = pd.read_csv(SAMPLE_SUB_PATH)
sub_template["BraTS21ID"] = sub_template["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sub_template["BraTS21ID"].tolist()

print("Submission template rows:", len(test_ids))
print("Example test id:", test_ids[0] if test_ids else None)



## === cell 9
from concurrent.futures import ThreadPoolExecutor


def _build_one_test(idx_pid):
    idx, pid = idx_pid
    subj_dir = os.path.join(test_dir, pid)
    vol = build_4ch_volume_cached(subj_dir, layer_number=layer_number, size=size)
    return idx, vol


max_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
test_vols = [None] * len(test_ids)
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for j, (idx, vol) in enumerate(
        ex.map(_build_one_test, enumerate(test_ids)), start=1
    ):
        test_vols[idx] = vol
        if j % 20 == 0:
            print(f"Built test volumes: {j}/{len(test_ids)}")

Test_Values = np.stack(test_vols, axis=0)
print("Test_Values shape:", Test_Values.shape)



## === cell 10
if use_tf:
    Results = model.predict(Test_Values, batch_size=8, verbose=0).reshape(-1)
else:
    Xt = Test_Values.reshape(Test_Values.shape[0], -1, 4)
    feat_mean_t = Xt.mean(axis=1)
    feat_std_t = Xt.std(axis=1)
    X_feat_t = np.concatenate([feat_mean_t, feat_std_t], axis=1).astype(np.float32)
    Results = clf.predict_proba(X_feat_t)[:, 1].reshape(-1)

Results = np.clip(Results, 0.0, 1.0)
print(
    "Preds shape:",
    Results.shape,
    "min/max:",
    float(Results.min()),
    float(Results.max()),
)

if len(Results) != len(sub_template):
    raise RuntimeError(
        f"Prediction length {len(Results)} != submission template length {len(sub_template)}"
    )

sub = sub_template.copy()
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
sub["MGMT_value"] = Results.astype(np.float32)

sub = sub[["BraTS21ID", "MGMT_value"]]
sub.to_csv("./submission.csv", index=False)

print(sub.head())
print("Wrote ./submission.csv with shape:", sub.shape)
print("Submission columns:", list(sub.columns))
