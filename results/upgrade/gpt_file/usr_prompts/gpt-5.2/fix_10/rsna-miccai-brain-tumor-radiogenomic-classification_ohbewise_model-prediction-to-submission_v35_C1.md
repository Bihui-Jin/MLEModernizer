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

nibabel==5.3.2
protobuf==6.33.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

- What this solution (achieved 0.5) has done: 'I fix the protobuf/tensorflow-io import crash by pinning protobuf’s Python implementation before importing TensorFlow/TFIO, which resolves the `MessageFactory` error in Kaggle’s environment. Then I remove the missing external model dependency and replace it with a minimal 3D CNN trained on the provided `train/` DICOM series using the same preprocessing pipeline already in your code (z-norm → center crop/pad → rescale → channel/batch). I also exclude the known-bad training cases `[00109, 00123, 00709]` and ensure the submission is written to `/kaggle/working/submission.csv` with correct columns and zero-padded IDs. The overall approach remains: load a single sequence type (FLAIR), preprocess to a fixed 3D tensor, train, and predict probabilities for ROC-AUC evaluation.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by Python-level DICOM decoding (pydicom per-slice) and repeated directory scans, both in the tf.data pipeline and again during test inference. I keep the exact same preprocessing steps and model/training loop, but speed them up by (1) caching the preprocessed volumes to disk (once per case) so subsequent epochs and inference reuse them, (2) switching tf.data mapping to deterministic parallel map with prefetch, and (3) reducing filesystem overhead by using `os.scandir` and avoiding repeated sorting work. These changes are correctness-preserving because cached tensors store the exact output of `process_series_to_tensor` (same math, just reused), and determinism is maintained via fixed seeds and deterministic dataset options.'
- What this solution (achieved 0.55412) has done: 'I fix two runtime blockers with minimal impact on the modeling logic: (1) the protobuf/TensorFlow import crash by switching to the safe `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting *before* importing TensorFlow and forcing the pure-Python protobuf backend, and (2) the cache write race/bug where `np.save()` was creating `*.npy.tmp.npy` but the code tried to rename `*.npy.tmp`, causing `FileNotFoundError` under parallel mapping. I also make cached writes atomic and safe under parallel workers by writing to a unique temp name and then `os.replace`, and ensure cache directories exist. These are correctness/stability fixes and should keep score behavior essentially the same while allowing training/inference to run end-to-end and produce `submission.csv`.'
- What this solution (achieved 0.55412) has done: 'We fix the protobuf/TensorFlow import crash that stops the notebook in cell 1 by forcing the pure-Python protobuf runtime *before* any protobuf/TensorFlow-related imports, and by avoiding direct imports of `google.protobuf.internal.api_implementation` (which can trigger the `MessageFactory.GetPrototype` mismatch in this environment). This is a stability-only change and should not affect model behavior/score. Then we add a small safety check to guarantee the submission predictions length matches the sample submission rows (fallback fill with 0.5 if anything goes wrong) so a valid `submission.csv` is always written. All preprocessing, model architecture, training loop, and inference logic remain unchanged.'
- What this solution (achieved 0.5) has done: 'The crash happens before any training because TensorFlow pulls in protobuf symbols that are incompatible with the current pure-Python protobuf fallback in this environment, so we force the C++ protobuf backend instead (and do it before importing TensorFlow). Then we keep the rest of the pipeline unchanged (same preprocessing, same 3D CNN, same training loop), only adding a small guard to disable TF-IO import side effects and ensuring pydicom is detected safely. These changes are stability-only and should preserve your current ~0.55 AUC behavior while allowing the notebook to run end-to-end and always write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.55412) has done: 'I fix the TensorFlow import crash by switching the protobuf runtime to the pure-Python implementation (the current `"cpp"` setting is what triggers the missing `google.protobuf.pyext._message` in this environment), and I do it before importing TensorFlow. Then I keep the exact same preprocessing, caching, dataset pipeline, model, training loop, and inference logic, only making sure the notebook continues to define `tf` successfully so later cells don’t raise `NameError`. These changes are stability-only and should keep your score behavior essentially unchanged while producing a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.55882) has done: 'The current crash happens before training because TensorFlow’s protobuf dependency is incompatible with the forced pure-Python protobuf runtime in this Kaggle image, triggering the `MessageFactory.GetPrototype` AttributeError. I fix this by switching to the C++ protobuf backend (and not importing anything protobuf-related explicitly) before importing TensorFlow; this is a stability-only change and should keep model behavior/score essentially unchanged. I also add a tiny defensive fallback to set `TF_USE_LEGACY_KERAS=1` (common Kaggle TF2.18 quirk) to avoid rare Keras/protobuf interaction issues, without changing the model/training logic. Everything else (preprocessing, caching, dataset pipeline, model, training loop, and submission writing) stays the same so your score remains close to the current 0.55412 while producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf

print("Python:", sys.version)
print("TF:", tf.__version__)

try:
    import pydicom  # type: ignore

    _HAS_PYDICOM = True
    print("pydicom: available")
except Exception as e:
    _HAS_PYDICOM = False
    print("pydicom: NOT available, using fallback DICOM pixel parser.", repr(e))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
labels_path = os.path.join(data_dir, "train_labels.csv")
sample_path = os.path.join(data_dir, "sample_submission.csv")

out_dir = "/kaggle/working"
os.makedirs(out_dir, exist_ok=True)
submission_path = os.path.join(out_dir, "submission.csv")

print("Exists train_dir:", os.path.exists(train_dir))
print("Exists test_dir:", os.path.exists(test_dir))
print("Exists train_labels:", os.path.exists(labels_path))
print("Exists sample_submission:", os.path.exists(sample_path))




## === cell 2
TARGET_SHAPE = (128, 128, 64)  # (H, W, D)
SCAN_TYPE = "FLAIR"  # preserve original behavior

CACHE_ROOT = os.path.join(out_dir, "cache_preproc")
CACHE_VERSION = "v1"  # bump if preprocessing changes
CACHE_DIR = os.path.join(
    CACHE_ROOT,
    f"{CACHE_VERSION}_{SCAN_TYPE}_{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}x{TARGET_SHAPE[2]}",
)
os.makedirs(CACHE_DIR, exist_ok=True)


def _sorted_dicom_files(folder: str):
    files = []
    with os.scandir(folder) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".dcm"):
                files.append(e.path)
    files.sort()
    return files


def _read_dicom_pixel_array(fp: str) -> np.ndarray:
    """
    Prefer pydicom for correctness; otherwise use a minimal fallback that works for typical RSNA DICOM.
    Returns float32 2D array (H,W).
    """
    if _HAS_PYDICOM:
        ds = pydicom.dcmread(fp, force=True)
        arr = ds.pixel_array  # numpy array
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr.astype(np.float32) * slope + intercept
        return arr.astype(np.float32)

    with open(fp, "rb") as f:
        data = f.read()

    tag = b"\xe0\x7f\x10\x00"
    idx = data.find(tag)
    if idx == -1:
        raise ValueError("PixelData tag not found in DICOM")

    vr = data[idx + 4 : idx + 6]
    long_vr = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"}
    if vr in long_vr:
        length = int.from_bytes(data[idx + 8 : idx + 12], "little", signed=False)
        start = idx + 12
    else:
        length = int.from_bytes(data[idx + 6 : idx + 8], "little", signed=False)
        start = idx + 8

    pixel_bytes = data[start : start + length]

    n = len(pixel_bytes)
    if n % 2 != 0:
        raise ValueError("Unexpected PixelData length")
    vals = np.frombuffer(pixel_bytes, dtype=np.uint16)
    side = int(np.sqrt(vals.size))
    if side * side == vals.size:
        arr = vals.reshape(side, side).astype(np.float32)
        return arr
    return vals.astype(np.float32)


def load_dicom_series_to_volume(series_dir):
    """Load a DICOM series directory into a float32 volume of shape (H, W, D)."""
    files = _sorted_dicom_files(series_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No .dcm files found in {series_dir}")

    slices = []
    for fp in files:
        img2d = _read_dicom_pixel_array(fp)  # (H,W) float32
        if img2d.ndim != 2:
            img2d = np.squeeze(img2d)
        slices.append(img2d.astype(np.float32))

    vol = np.stack(slices, axis=-1).astype(np.float32)
    return tf.convert_to_tensor(vol, dtype=tf.float32)


def znorm(volume):
    mean = tf.reduce_mean(volume)
    std = tf.math.reduce_std(volume)
    std = tf.where(std > 0, std, tf.constant(1.0, dtype=volume.dtype))
    return (volume - mean) / std


def rescale_to_minus1_plus1(volume):
    vmin = tf.reduce_min(volume)
    vmax = tf.reduce_max(volume)
    denom = tf.where(
        (vmax - vmin) > 0, (vmax - vmin), tf.constant(1.0, dtype=volume.dtype)
    )
    x = (volume - vmin) / denom  # [0,1]
    return x * 2.0 - 1.0


def crop_or_pad_3d(volume, target_shape=TARGET_SHAPE):
    """Center crop or pad a (H,W,D) volume to target_shape."""
    th, tw, td = target_shape

    def _crop_center(x, target, axis):
        size = tf.shape(x)[axis]
        start = tf.maximum((size - target) // 2, 0)
        begin = [0, 0, 0]
        begin[axis] = start
        size_out = [-1, -1, -1]
        size_out[axis] = tf.minimum(target, size)
        return tf.slice(x, begin, size_out)

    vol = volume
    vol = _crop_center(vol, th, 0)
    vol = _crop_center(vol, tw, 1)
    vol = _crop_center(vol, td, 2)

    shp2 = tf.shape(vol)
    ph = tf.maximum(th - shp2[0], 0)
    pw = tf.maximum(tw - shp2[1], 0)
    pd = tf.maximum(td - shp2[2], 0)

    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left
    pad_front = pd // 2
    pad_back = pd - pad_front

    vol = tf.pad(
        vol,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [pad_front, pad_back]],
        mode="CONSTANT",
        constant_values=0.0,
    )
    vol.set_shape([th, tw, td])
    return vol


def add_channel(volume):
    """(H,W,D) -> (H,W,D,1)"""
    return tf.expand_dims(volume, axis=-1)


def process_series_to_tensor(series_dir):
    """Return (H,W,D,1) float32 ready for model."""
    vol = load_dicom_series_to_volume(series_dir)  # (H,W,D)
    vol = znorm(vol)
    vol = crop_or_pad_3d(vol, TARGET_SHAPE)
    vol = rescale_to_minus1_plus1(vol)
    vol = add_channel(vol)  # (H,W,D,1)
    vol = tf.ensure_shape(vol, (*TARGET_SHAPE, 1))
    return vol


def _cache_path(split: str, case_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{split}_{case_id}.npy")


def _atomic_save_npy(final_path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp_path = final_path + f".{uuid.uuid4().hex}.tmp.npy"
    np.save(tmp_path, arr)
    os.replace(tmp_path, final_path)


def process_series_numpy_cached(
    series_dir: str, split: str, case_id: str
) -> np.ndarray:
    cp = _cache_path(split, case_id)
    if os.path.exists(cp):
        arr = np.load(cp, mmap_mode="r")
        return np.asarray(arr, dtype=np.float32)

    t = process_series_to_tensor(series_dir)
    arr = t.numpy().astype(np.float32)
    _atomic_save_npy(cp, arr)
    return arr




## === cell 3
labels_df = pd.read_csv(labels_path)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

bad_cases = {"00109", "00123", "00709"}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)


def series_exists(case_id, scan_type=SCAN_TYPE):
    return os.path.isdir(os.path.join(train_dir, case_id, scan_type))


labels_df = labels_df[labels_df["BraTS21ID"].map(series_exists)].reset_index(drop=True)

print("Train cases after filtering:", len(labels_df))
print(labels_df.head())

SEED = 42
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(labels_df))
val_frac = 0.15
n_val = max(1, int(len(labels_df) * val_frac))
val_idx = perm[:n_val]
train_idx = perm[n_val:]

train_df = labels_df.iloc[train_idx].reset_index(drop=True)
val_df = labels_df.iloc[val_idx].reset_index(drop=True)

print("Train/Val sizes:", len(train_df), len(val_df))
print("Val positive rate:", float(val_df["MGMT_value"].mean()))




## === cell 4
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 2  # keep small for memory; no early stopping introduced
EPOCHS = 3  # unchanged from provided script


def make_dataset(df, training: bool, split_name: str):
    ids = df["BraTS21ID"].astype(str).tolist()
    labels = df["MGMT_value"].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((ids, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 256), seed=SEED, reshuffle_each_iteration=True
        )

    def _load(case_id, y):
        case_id_str = case_id.numpy().decode("utf-8")
        series_dir = os.path.join(train_dir, case_id_str, SCAN_TYPE)
        x = process_series_numpy_cached(
            series_dir, split=split_name, case_id=case_id_str
        )
        return x, np.float32(y)

    def _tf_load(case_id, y):
        x, y2 = tf.py_function(_load, inp=[case_id, y], Tout=[tf.float32, tf.float32])
        x.set_shape((*TARGET_SHAPE, 1))
        y2.set_shape(())
        return x, y2

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_tf_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, training=True, split_name="train")
val_ds = make_dataset(val_df, training=False, split_name="val")




## === cell 5
def build_model(input_shape=(*TARGET_SHAPE, 1)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv3D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


tf.keras.utils.set_random_seed(SEED)

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
)




## === cell 6
sub_df = pd.read_csv(sample_path)
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sub_df["BraTS21ID"].tolist()

preds = []
failed = 0

TEST_BATCH = 2  # keep small for memory; batching does not change semantics

x_batch = []


def _flush_batch(xb):
    global preds
    if not xb:
        return
    X = tf.stack(xb, axis=0)  # (B,H,W,D,1)
    P = model.predict(X, verbose=0).reshape(-1)
    for p in P:
        preds.append(float(np.clip(float(p), 0.0, 1.0)))


for patient in test_ids:
    series_dir = os.path.join(test_dir, patient, SCAN_TYPE)
    try:
        cp = _cache_path("test", patient)
        if os.path.exists(cp):
            x_np = np.load(cp, mmap_mode="r")
            x = tf.convert_to_tensor(np.asarray(x_np, dtype=np.float32))
            x = tf.ensure_shape(x, (*TARGET_SHAPE, 1))
        else:
            x = process_series_to_tensor(series_dir)  # (H,W,D,1)
            arr = x.numpy().astype(np.float32)
            _atomic_save_npy(cp, arr)

        x_batch.append(x)

        if len(x_batch) >= TEST_BATCH:
            _flush_batch(x_batch)
            x_batch.clear()

    except Exception as e:
        failed += 1
        _flush_batch(x_batch)
        x_batch.clear()

        preds.append(0.5)
        print(f"Warning: failed patient {patient} ({type(e).__name__}: {e}); using 0.5")

_flush_batch(x_batch)

if len(preds) != len(sub_df):
    print(
        f"Warning: preds length {len(preds)} != sub_df length {len(sub_df)}; fixing with 0.5 fill/truncate."
    )
    if len(preds) < len(sub_df):
        preds = preds + [0.5] * (len(sub_df) - len(preds))
    else:
        preds = preds[: len(sub_df)]

sub_df["MGMT_value"] = preds
sub_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print("Rows:", len(sub_df), " Failed:", failed)
print(sub_df.head())
print("Submission columns:", list(sub_df.columns))
