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

0.51882

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51882) has done: 'I fix two execution blockers: (1) the `pydicom` import crash caused by an incompatible protobuf version in the Kaggle image by forcing the pure-Python protobuf backend before importing `pydicom`, and (2) the submission ordering KeyError caused by mismatched zero-padding between your `BraTS21ID` strings and the sample submission IDs. The model/training/inference core logic stays the same; changes are only to stabilize imports and ensure IDs are consistently 5-digit strings everywhere. Finally, I make the submission writing robust by merging on `BraTS21ID` rather than `.loc[...]` indexing, guaranteeing the exact required row order and a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the current import-time crash by removing the fragile `pydicom` dependency entirely and replacing DICOM reading with a safe, standard-library-only approach that uses TensorFlow’s built-in `tf.io.decode_dicom_image` (this avoids the protobuf `MessageFactory.GetPrototype` issue). I keep the same downstream preprocessing and model/training logic (same slices-per-subject, resize/stack/normalize, same CNN, same training loop). I also make the DICOM loader robust to varying slice ordering by sorting with `InstanceNumber` when available, while still falling back to filename sorting; this is a minimal correctness improvement that can nudge AUC upward without changing the core approach. Submission formatting and ordering logic remain as-is to ensure a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash by forcing the pure-Python protobuf implementation before TensorFlow is imported, because TensorFlow’s DICOM decoder path can still trigger the same protobuf `MessageFactory.GetPrototype` issue otherwise. I also make the optional `skimage` import safer by not letting it crash the run if it indirectly pulls problematic deps, keeping the existing tf.image.resize fallback. The rest of the pipeline (DICOM decoding via `tf.io.decode_dicom_image`, slice selection, CNN, training loop, subject-level mean aggregation, and submission merge-on-ID ordering) remain unchanged so behavior/score stays consistent while the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'We fix the current import-time crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf backend *and* proactively removing any already-imported `google.protobuf` modules before TensorFlow is imported, which is a common cause of this exact issue in Kaggle images. This is an execution-stability fix and should be score-neutral (it only changes the import order/initialization). I also add a safe fallback that, if DICOM decoding still fails at runtime for any reason, returns zero-images so the pipeline completes and always writes a valid `submission.csv`. Core data loading logic (T2w, slice filtering, resize/stack/normalize), model architecture, training loop, and subject-mean aggregation remain unchanged.'
- What this solution (achieved 0.51882) has done: 'We fix the import-time protobuf crash that happens when importing TensorFlow by setting the pure-Python protobuf environment variables *and* forcing a clean interpreter state for any preloaded `google.protobuf` modules before TensorFlow is first imported. Then we add a safe fallback DICOM decoding path: try TensorFlow’s `tf.io.decode_dicom_image` when available, otherwise use `pydicom` only if it can be imported (so the pipeline still runs even if TF’s DICOM path is broken in this image). These changes are execution-stability focused and keep your data selection, preprocessing, model, training loop, and submission ordering logic the same; score impact should be neutral to slightly positive (more slices decoded successfully instead of silently returning None/zeros). Finally, we keep the merge-on-ID submission creation to guarantee a valid `submission.csv` with correct ordering and columns.'
- What this solution (achieved 0.51882) has done: 'I fix the TensorFlow import-time crash caused by the protobuf backend not being applied early enough by moving the protobuf environment setup to the very top (before any TensorFlow-related imports can occur) and by also setting `TF_CPP_MIN_LOG_LEVEL` for cleaner logs. To keep your core pipeline identical, I won’t change the model, slice selection, training loop, or aggregation logic; the only functional change is making DICOM decoding work reliably by ensuring TensorFlow can import. I also add a tiny safety fallback so that if TensorFlow’s DICOM decoder isn’t available, the code can still run via `pydicom` when possible (without changing the normal path). The submission writing remains the same merge-on-ID approach to guarantee correct ordering and a valid `submission.csv`.'
- What this solution (achieved 0.51882) has done: 'I fix the current execution blocker (`MessageFactory` has no `GetPrototype`) by ensuring the pure-Python protobuf backend is applied *before any protobuf/TensorFlow import* and by forcing a clean reload of protobuf modules (this is the known root cause in this Kaggle image). I also make the `pydicom` import strictly lazy (only attempted inside the fallback reader) so it can’t crash the run at import time, while keeping the default DICOM path via `tf.io.decode_dicom_image` unchanged. These changes are execution-stability focused and should be score-neutral, preserving your existing model, training loop, slice selection, and aggregation logic. The script still write a correctly ordered `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import importlib

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]
if "google" in sys.modules:
    try:
        importlib.reload(sys.modules["google"])
    except Exception:
        pass

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    from skimage.transform import resize as sk_resize
except Exception as e:
    sk_resize = None
    print(
        "WARNING: skimage.transform.resize import failed; will fallback to tf.image.resize. Error:",
        repr(e),
    )

print("TensorFlow:", tf.__version__)



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

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

BAD_CASES = {109, 123, 709}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

print("Train labels:", labels_df.shape, "Sample submission:", sample_sub.shape)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)




## === cell 2
def _safe_resize(img2d: np.ndarray, out_hw: int) -> np.ndarray:
    """Resize a 2D image to (out_hw, out_hw) float32."""
    img2d = img2d.astype(np.float32)
    if sk_resize is not None:
        return sk_resize(
            img2d, (out_hw, out_hw), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
    x = tf.convert_to_tensor(img2d[None, ..., None], dtype=tf.float32)
    x = tf.image.resize(x, (out_hw, out_hw), method="bilinear")
    return x[0, ..., 0].numpy().astype(np.float32)


_HAS_TF_DICOM = hasattr(tf.io, "decode_dicom_image")


def _read_dicom_pixel_array(dcm_path: str):
    """
    Robust DICOM read. Prefer tf.io.decode_dicom_image when available, else fall back to pydicom if possible.
    Returns float32 2D array or None on failure.
    """
    if _HAS_TF_DICOM:
        try:
            raw = tf.io.read_file(dcm_path)
            decoded = tf.io.decode_dicom_image(
                raw,
                dtype=tf.uint16,
                color_dim=False,
                scale="auto",
            )
            arr = decoded.numpy()
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                return None
            return arr.astype(np.float32)
        except Exception:
            pass

    try:
        import pydicom  # type: ignore

        ds = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
        arr = ds.pixel_array
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None
        return arr.astype(np.float32)
    except Exception:
        return None


def _dicom_instance_number(dcm_path: str):
    """
    Try to parse InstanceNumber to sort slices in anatomical order.
    Uses only stdlib; if unavailable/parse fails, return None.
    """
    try:
        with open(dcm_path, "rb") as f:
            data = f.read(2_000_000)  # bounded read; usually enough for header tags
        tag = b"\x20\x00\x13\x00"
        i = data.find(tag)
        if i == -1:
            return None
        if i + 10 >= len(data):
            return None
        vr = data[i + 4 : i + 6]
        if vr != b"IS":
            return None
        length = int.from_bytes(data[i + 6 : i + 8], "little")
        val_bytes = data[i + 8 : i + 8 + length]
        val = val_bytes.decode(errors="ignore").strip().strip("\x00")
        return int(val)
    except Exception:
        return None


def load_subject_t2w_stack(
    subject_path: str, img_px_size: int = 150, n_slices: int = 6
) -> np.ndarray:
    """
    Core logic preserved: pick up to 6 informative T2w slices,
    resize to IMG_PX_SIZE, stack to 3 channels, normalize.
    Returns: (n_slices, img_px_size, img_px_size, 3)
    """
    t2w_dir = os.path.join(subject_path, "T2w")
    if not os.path.isdir(t2w_dir):
        return np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)

    dcm_files = [
        os.path.join(t2w_dir, f)
        for f in os.listdir(t2w_dir)
        if f.lower().endswith(".dcm")
    ]
    if len(dcm_files) == 0:
        return np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)

    inst = []
    any_inst = False
    for p in dcm_files:
        v = _dicom_instance_number(p)
        inst.append(v)
        if v is not None:
            any_inst = True
    if any_inst:
        dcm_files = [
            p
            for _, p in sorted(
                zip(inst, dcm_files),
                key=lambda x: (x[0] is None, x[0] if x[0] is not None else 10**9),
            )
        ]
    else:
        dcm_files = sorted(dcm_files)

    selected = []
    for p in dcm_files:
        arr = _read_dicom_pixel_array(p)
        if arr is None:
            continue

        if arr.sum() <= 100000:
            continue

        img = _safe_resize(arr, img_px_size)
        stacked = np.stack((img,) * 3, axis=-1)

        mx = np.max(stacked)
        if mx > 0:
            stacked = stacked / mx

        if stacked.sum() > 2000:
            selected.append(stacked.astype(np.float32))
            if len(selected) >= n_slices:
                break

    out = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    if len(selected) > 0:
        out[: len(selected)] = np.stack(selected, axis=0)

    mx = np.max(out)
    if mx > 0:
        out = out / mx

    return out




## === cell 3
def load_dataset_from_dir(
    base_dir: str, ids: list[int], img_px_size: int = 150, n_slices: int = 6
):
    """
    Load stacks for a list of subject IDs.
    Returns X shape: (len(ids)*n_slices, H, W, 3) and a mapping subject_index for aggregation.
    """
    X_list = []
    subj_index = []
    for idx, brats_id in enumerate(ids):
        subj_folder = os.path.join(base_dir, f"{brats_id:05d}")
        stack = load_subject_t2w_stack(
            subj_folder, img_px_size=img_px_size, n_slices=n_slices
        )
        X_list.append(stack)
        subj_index.extend([idx] * n_slices)
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    subj_index = np.array(subj_index, dtype=np.int32)
    return X, subj_index


all_train_ids = labels_df["BraTS21ID"].astype(int).tolist()
all_y = labels_df["MGMT_value"].astype(np.float32).values

from sklearn.model_selection import train_test_split

train_ids, val_ids, y_train, y_val = train_test_split(
    all_train_ids, all_y, test_size=0.15, random_state=SEED, stratify=all_y
)

IMG_PX_SIZE = 150
N_SLICES = 6

print("Loading training images...")
X_train, train_subj_idx = load_dataset_from_dir(
    TRAIN_DIR, train_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
print("Loading validation images...")
X_val, val_subj_idx = load_dataset_from_dir(
    TRAIN_DIR, val_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)

y_train_rep = np.repeat(y_train, N_SLICES).astype(np.float32)
y_val_rep = np.repeat(y_val, N_SLICES).astype(np.float32)

print("X_train:", X_train.shape, "y_train:", y_train_rep.shape)
print("X_val:", X_val.shape, "y_val:", y_val_rep.shape)




## === cell 4
def build_model(input_shape=(150, 150, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_model((IMG_PX_SIZE, IMG_PX_SIZE, 3))
model.summary()

history = model.fit(
    X_train,
    y_train_rep,
    validation_data=(X_val, y_val_rep),
    epochs=2,
    batch_size=32,
    verbose=2,
)




## === cell 5
def predict_subject_probs(model, X: np.ndarray, subj_index: np.ndarray) -> np.ndarray:
    """
    Predict per-slice probabilities then aggregate per subject by mean.
    Returns probs per subject ordered by subject index (0..n_subjects-1).
    """
    p = model.predict(X, batch_size=64, verbose=0).reshape(-1).astype(np.float32)
    n_subjects = int(subj_index.max()) + 1 if subj_index.size else 0
    out = np.zeros(n_subjects, dtype=np.float32)
    for i in range(n_subjects):
        out[i] = float(p[subj_index == i].mean()) if np.any(subj_index == i) else 0.5
    return out


from sklearn.metrics import roc_auc_score

val_probs = predict_subject_probs(model, X_val, val_subj_idx)
val_auc = roc_auc_score(y_val, val_probs)
print("Validation subject-level AUC:", val_auc)



## === cell 6
test_ids_int = sample_sub["BraTS21ID"].astype(int).tolist()
test_ids_str = [f"{i:05d}" for i in test_ids_int]

print("Loading test images...")
X_test, test_subj_idx = load_dataset_from_dir(
    TEST_DIR, test_ids_int, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)

test_probs = predict_subject_probs(model, X_test, test_subj_idx)
print(
    "Test probs:",
    test_probs.shape,
    "min/max:",
    float(test_probs.min()),
    float(test_probs.max()),
)

sub_pred = pd.DataFrame(
    {"BraTS21ID": test_ids_str, "MGMT_value": test_probs.astype(float)}
)

sample_order = sample_sub.copy()
sample_order["BraTS21ID"] = (
    sample_order["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}")
)

sub_df = sample_order[["BraTS21ID"]].merge(sub_pred, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

assert sub_df.shape[0] == sample_sub.shape[0]
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape", sub_df.shape)
print(sub_df.head())
