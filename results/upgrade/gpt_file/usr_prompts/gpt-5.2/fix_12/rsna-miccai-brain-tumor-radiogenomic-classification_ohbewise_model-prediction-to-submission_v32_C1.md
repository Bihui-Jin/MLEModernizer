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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the protobuf environment override that is triggering the `MessageFactory.GetPrototype` crash with TF 2.18 + protobuf 6, so imports work normally. Then I make the external SavedModel dependency optional: if the `/kaggle/input/dataset-to-model-with-tensorflow/...` model folder doesn’t exist, the script fall back to training the same simple 3D CNN on-the-fly from the provided train DICOMs (excluding the known-bad cases) and use it for test inference. This fixes the missing-model runtime error and guarantees a valid `submission.csv` is produced end-to-end in the Kaggle environment. The training/inference pipeline is kept minimal and deterministic, and outputs are clipped to valid probabilities for the AUC metric.'
- What this solution (achieved 0.5) has done: 'I fix the two runtime blockers: (1) the protobuf crash happens because the env var is removed after importing TensorFlow, so I move that `os.environ.pop(...)` to run before any TF/Keras import; (2) TensorFlow-IO’s DICOM decoder is failing to load its shared library, so I remove the tfio dependency and replace DICOM reading with a lightweight `pydicom` reader (installed in Kaggle) while keeping the same preprocessing shapes and the same 3D CNN/training loop. These changes are execution/stability fixes and keep the model/training semantics the same, so score should be similar or slightly better than the all-0.5 fallback. The script still optionally load the external SavedModel if present, otherwise it trains locally and always writes `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I fix two runtime blockers: the protobuf `MessageFactory.GetPrototype` crash (caused by unsetting the protobuf implementation env var) and the invalid reshape coming from the custom “trilinear resize” function. The protobuf fix is score-neutral but required to even import TF reliably in this environment; the resize fix preserves the same preprocessing intent (resize to 128×128×64) but makes it correct and stable by using proper 3D resizing. With these changes, the local fallback training run end-to-end and produce a valid `/kaggle/working/submission.csv`; score should move above the constant-0.5 baseline because the model actually train instead of failing. I keep the model and training loop unchanged otherwise.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf crash by ensuring the protobuf implementation env var is set before importing TensorFlow/Keras, which prevents the `MessageFactory.GetPrototype` error. Then I fix the 3D resize bug: `tf.image.resize` only supports 3D/4D tensors, so the current 5D usage fails; I replace it with a true 3D trilinear resize using `tf.keras.layers.Resizing` applied slice-wise (H/W) plus `tf.signal.resample` for depth. These are runtime/logic fixes that keep the same preprocessing intent (128×128×64) and the same 3D CNN/training loop. Finally, I keep submission writing unchanged but ensure the pipeline always runs end-to-end and outputs `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix two execution blockers while keeping your model and training loop unchanged: (1) the protobuf crash by forcing the pure-Python protobuf implementation before TensorFlow import, and (2) the missing `tf.signal.resample` API by replacing only the depth-resize step with a small linear interpolation function (still trilinear intent: resize H/W per-slice + linear along depth). These are runtime/logic fixes that preserve the same preprocessing target (128×128×64) and should let the fallback training actually run (instead of failing), which should move the AUC above the constant 0.5 baseline. I also keep submission writing identical and ensure it always outputs `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by Python-level DICOM I/O and per-slice preprocessing repeated for every case; the current code reads every DICOM file with `pydicom.dcmread()` and rebuilds Keras resize layers on every call, which is extremely slow. I keep the exact same preprocessing steps and model semantics, but speed them up by (1) reading only the Pixel Data via `pydicom.dcmread(..., stop_before_pixels=False, specific_tags=[...])`, (2) sorting slices by DICOM metadata (InstanceNumber/ImagePositionPatient) once, (3) reusing a single `keras.layers.Resizing` instance instead of recreating it per scan, and (4) using a threaded prefetch pipeline for test inference so CPU decoding overlaps with model execution. These changes are provably equivalent to the current logic (same slices, same normalization, same resize math), just with less overhead and better parallelism.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf import crash by setting the correct protobuf environment variable *before* importing TensorFlow/Keras instead of popping it, which prevents the `MessageFactory.GetPrototype` error in TF 2.18 + protobuf 6. Then I fix the depth-resize bug that’s causing the “reshape with 63 vs 62 values” error by rewriting `_resize_depth_linear` to use a shape-safe, batch/vectorized `tf.gather` interpolation (same linear-depth intent, just correct). Finally, I make inference robust when training fails by ensuring `inference_model` is always a callable model (train fallback wrapped in try/except, otherwise default to constant 0.5), so a valid `/kaggle/working/submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

print("TF:", tf.__version__)
print("Keras:", keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

DATA_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = f"{DATA_DIR}/train"
TEST_DIR = f"{DATA_DIR}/test"
LABELS_CSV = f"{DATA_DIR}/train_labels.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"

BAD_CASES = {"00109", "00123", "00709"}  # known broken cases per competition note



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pydicom
from pydicom.pixels import pixel_array as _pyd_pixel_array

_RESIZER_128 = keras.layers.Resizing(128, 128, interpolation="bilinear")

_DCM_SUFFIX = ".dcm"


def _sorted_dicom_files(series_dir: str):
    if not os.path.isdir(series_dir):
        return []
    return [
        os.path.join(series_dir, fn)
        for fn in os.listdir(series_dir)
        if fn.lower().endswith(_DCM_SUFFIX)
    ]


_ORDER_TAGS = [
    (0x0020, 0x0013),  # InstanceNumber
    (0x0020, 0x0032),  # ImagePositionPatient
    (0x0008, 0x0018),  # SOPInstanceUID (fallback stable key)
]

_PIXEL_TAGS = _ORDER_TAGS + [
    (0x7FE0, 0x0010),  # PixelData
    (0x0028, 0x0002),  # SamplesPerPixel
    (0x0028, 0x0004),  # PhotometricInterpretation
    (0x0028, 0x0010),  # Rows
    (0x0028, 0x0011),  # Columns
    (0x0028, 0x0100),  # BitsAllocated
    (0x0028, 0x0101),  # BitsStored
    (0x0028, 0x0102),  # HighBit
    (0x0028, 0x0103),  # PixelRepresentation
    (0x0028, 0x1052),  # RescaleIntercept
    (0x0028, 0x1053),  # RescaleSlope
]


def _slice_sort_key(ds) -> tuple:
    inst = getattr(ds, "InstanceNumber", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None:
        try:
            z = float(ipp[2])
        except Exception:
            z = None
    else:
        z = None
    uid = getattr(ds, "SOPInstanceUID", "")
    inst_key = int(inst) if inst is not None else 10**9
    z_key = z if z is not None else float("inf")
    return (inst_key, z_key, uid)


def _safe_get_float(ds, attr, default=None):
    v = getattr(ds, attr, default)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _decode_pixels_robust(ds) -> np.ndarray:
    """
    Some competition DICOMs can miss PhotometricInterpretation (0028,0004),
    which makes ds.pixel_array raise. Fall back to pydicom.pixels.pixel_array(raw=True),
    then apply rescale slope/intercept if present.
    """
    try:
        arr = ds.pixel_array
    except Exception:
        arr = _pyd_pixel_array(ds, raw=True)

    arr = np.asarray(arr, dtype=np.float32)

    slope = _safe_get_float(ds, "RescaleSlope", default=1.0)
    intercept = _safe_get_float(ds, "RescaleIntercept", default=0.0)
    if slope is None:
        slope = 1.0
    if intercept is None:
        intercept = 0.0
    arr = arr * np.float32(slope) + np.float32(intercept)

    return arr


def read_dicom_series(series_dir: str):
    """
    Reads a DICOM series folder into a float32 tensor [H, W, D].
    Uses pydicom to avoid TF-IO shared library issues.
    """
    fns = _sorted_dicom_files(series_dir)
    if len(fns) == 0:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    headers = []
    for fp in fns:
        ds = pydicom.dcmread(
            fp,
            force=True,
            stop_before_pixels=True,
            specific_tags=_ORDER_TAGS,
        )
        headers.append((fp, ds))

    headers.sort(key=lambda t: _slice_sort_key(t[1]))

    slices = []
    for fp, _ in headers:
        ds = pydicom.dcmread(
            fp,
            force=True,
            stop_before_pixels=False,
            specific_tags=_PIXEL_TAGS,
        )
        arr = _decode_pixels_robust(ds)  # [H, W]
        slices.append(arr)

    vol = np.stack(slices, axis=-1)  # [H, W, D]
    return tf.convert_to_tensor(vol, dtype=tf.float32)


def z_normalize(volume: tf.Tensor, eps: float = 1e-6):
    mean = tf.reduce_mean(volume)
    std = tf.math.reduce_std(volume)
    return (volume - mean) / (std + eps)


def _resize_depth_linear(vol_hw_d: tf.Tensor, target_d: int):
    """
    BUGFIX: Previous implementation could trigger invalid reshape errors in graph mode.
    This keeps the same intent (linear interpolation along depth) but does it with
    shape-safe vectorized gather on axis=2.
    Input:  [H, W, D]
    Output: [H, W, target_d]
    """
    vol_hw_d = tf.convert_to_tensor(vol_hw_d, dtype=tf.float32)
    target_d = int(target_d)

    if vol_hw_d.shape.rank != 3:
        vol_hw_d = tf.reshape(
            vol_hw_d, [tf.shape(vol_hw_d)[0], tf.shape(vol_hw_d)[1], -1]
        )

    d = tf.shape(vol_hw_d)[2]

    def _degenerate():
        return tf.zeros(
            [tf.shape(vol_hw_d)[0], tf.shape(vol_hw_d)[1], target_d], dtype=tf.float32
        )

    def _resize():
        d_f = tf.cast(d, tf.float32)
        x = tf.linspace(0.0, d_f - 1.0, target_d)  # [target_d]
        x0 = tf.cast(tf.floor(x), tf.int32)  # [target_d]
        x1 = tf.minimum(x0 + 1, d - 1)  # [target_d]
        w = x - tf.cast(x0, tf.float32)  # [target_d]

        v0 = tf.gather(vol_hw_d, x0, axis=2)  # [H, W, target_d]
        v1 = tf.gather(vol_hw_d, x1, axis=2)  # [H, W, target_d]
        w = tf.reshape(w, [1, 1, target_d])  # broadcast to [H,W,target_d]
        return (1.0 - w) * v0 + w * v1

    return tf.cond(d > 0, _resize, _degenerate)


def resize_volume_trilinear(volume: tf.Tensor, target_hw=(128, 128), target_d=64):
    """
    Trilinear-ish resize to 128x128x64:
    1) Resize H/W per-slice using keras.layers.Resizing (bilinear).
    2) Resize depth using linear interpolation.
    Input:  [H, W, D]
    Output: [Ht, Wt, Dt]
    """
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    target_d = int(target_d)

    vol = tf.convert_to_tensor(volume, dtype=tf.float32)

    if vol.shape.rank != 3:
        vol = tf.reshape(vol, [tf.shape(vol)[0], tf.shape(vol)[1], -1])

    vol_dhwc = tf.transpose(vol, [2, 0, 1])[..., tf.newaxis]  # [D,H,W,1]

    if target_h == 128 and target_w == 128:
        vol_dhwc = _RESIZER_128(vol_dhwc)
    else:
        resizer = keras.layers.Resizing(target_h, target_w, interpolation="bilinear")
        vol_dhwc = resizer(vol_dhwc)  # [D,target_h,target_w,1]

    vol_hw_d = tf.transpose(tf.squeeze(vol_dhwc, axis=-1), [1, 2, 0])  # [H,W,D]
    vol_hw_d = _resize_depth_linear(vol_hw_d, target_d=target_d)

    return vol_hw_d


def center_crop_or_pad_depth(volume: tf.Tensor, target_d: int = 64):
    """
    Ensure depth == target_d with symmetric crop/pad on the last axis.
    """
    volume = tf.convert_to_tensor(volume, dtype=tf.float32)
    target_d = int(target_d)

    if volume.shape.rank is not None and volume.shape.rank < 3:
        volume = tf.reshape(volume, [128, 128, -1])

    d = tf.shape(volume)[-1]

    def _crop():
        start = (d - target_d) // 2
        return volume[..., start : start + target_d]

    def _pad():
        pad_total = target_d - d
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return tf.pad(volume, paddings=[[0, 0], [0, 0], [pad_before, pad_after]])

    return tf.cond(d >= target_d, _crop, _pad)


def process_scan_from_dicom(series_dir: str):
    """
    Creates model input tensor with shape [1, 128, 128, 64, 1]
    from a DICOM series directory.
    """
    vol = read_dicom_series(series_dir)  # [H, W, D]
    vol = z_normalize(vol)
    vol = resize_volume_trilinear(vol, target_hw=(128, 128), target_d=64)
    vol = center_crop_or_pad_depth(vol, target_d=64)
    vol = vol[..., tf.newaxis]  # [H, W, D, 1]
    vol = vol[tf.newaxis, ...]  # [1, H, W, D, 1]
    return vol


def _model_predict_single_prob(model, case_tensor: tf.Tensor) -> float:
    """
    SavedModel via TFSMLayer may return dict or tensor; handle both robustly.
    Returns scalar probability.
    """
    out = model(case_tensor, training=False)
    if isinstance(out, dict):
        for k in (
            "outputs",
            "predictions",
            "logits",
            "output_0",
            "dense",
            "activation",
        ):
            if k in out:
                out = out[k]
                break
        else:
            out = next(iter(out.values()))
    out = tf.convert_to_tensor(out)
    out = tf.reshape(out, [-1])
    return float(out[0].numpy())




## === cell 2
def build_3d_cnn(input_shape=(128, 128, 64, 1)):
    inputs = keras.Input(shape=input_shape, dtype=tf.float32, name="input")
    x = keras.layers.Conv3D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling3D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def try_load_external_savedmodel(scan_type: str):
    candidates = [
        f"../input/dataset-to-model-with-tensorflow/models/{scan_type}/",
        f"/kaggle/input/dataset-to-model-with-tensorflow/models/{scan_type}/",
    ]
    for model_dir in candidates:
        sm_pb = os.path.join(model_dir, "saved_model.pb")
        sm_pbtxt = os.path.join(model_dir, "saved_model.pbtxt")
        if os.path.isdir(model_dir) and (
            os.path.exists(sm_pb) or os.path.exists(sm_pbtxt)
        ):
            endpoint = "serving_default"
            try:
                tfsml = keras.layers.TFSMLayer(model_dir, call_endpoint=endpoint)
            except Exception:
                loaded = tf.saved_model.load(model_dir)
                sigs = list(getattr(loaded, "signatures", {}).keys())
                if len(sigs) == 0:
                    continue
                endpoint = sigs[0]
                tfsml = keras.layers.TFSMLayer(model_dir, call_endpoint=endpoint)

            inp = keras.Input(shape=(128, 128, 64, 1), dtype=tf.float32, name="input")
            out = tfsml(inp)
            inference_model = keras.Model(inputs=inp, outputs=out)
            return inference_model, model_dir, endpoint
    return None, None, None




## === cell 3
def make_tf_dataset_from_ids(
    ids, labels_map, scan_type="T1w", batch_size=1, shuffle=False
):
    def _load_one(brats_id):
        brats_id = brats_id.numpy().decode("utf-8")
        series_dir = f"{TRAIN_DIR}/{brats_id}/{scan_type}/"
        x = process_scan_from_dicom(series_dir)  # [1,128,128,64,1]
        x = tf.squeeze(x, axis=0)  # [128,128,64,1]
        y = np.float32(labels_map[brats_id])
        return x, y

    def _tf_load_one(brats_id):
        x, y = tf.py_function(_load_one, [brats_id], Tout=(tf.float32, tf.float32))
        x.set_shape((128, 128, 64, 1))
        y.set_shape(())
        return x, y

    ds = tf.data.Dataset.from_tensor_slices(ids)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(ids), 256),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    ds = ds.map(_tf_load_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 4
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

labels_map = dict(
    zip(labels_df["BraTS21ID"].tolist(), labels_df["MGMT_value"].astype(int).tolist())
)
all_ids = labels_df["BraTS21ID"].tolist()

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(all_ids))
split = int(0.85 * len(all_ids))
train_ids = [all_ids[i] for i in perm[:split]]
val_ids = [all_ids[i] for i in perm[split:]]

print("Train/Val sizes:", len(train_ids), len(val_ids))



## === cell 5
scan_type = "T1w"

inference_model, model_dir, endpoint = try_load_external_savedmodel(scan_type)

if inference_model is not None:
    print(f"Loaded external SavedModel from: {model_dir}")
    print(f"Using SavedModel endpoint: {endpoint}")
else:
    print(
        "External SavedModel not found; training a small 3D CNN locally (minimal fallback)."
    )
    try:
        model = build_3d_cnn(input_shape=(128, 128, 64, 1))

        train_ds = make_tf_dataset_from_ids(
            train_ids, labels_map, scan_type=scan_type, batch_size=1, shuffle=True
        )
        val_ds = make_tf_dataset_from_ids(
            val_ids, labels_map, scan_type=scan_type, batch_size=1, shuffle=False
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            verbose=2,
        )
        inference_model = model
    except Exception as e:
        print("WARNING: training failed; will fall back to constant 0.5 predictions.")
        print("Training exception:", repr(e))
        inference_model = None



## === cell 6
sub = pd.read_csv(SAMPLE_SUB)
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

test_ids = sub["BraTS21ID"].tolist()


def _load_one_test(brats_id):
    brats_id = brats_id.numpy().decode("utf-8")
    series_dir = f"{TEST_DIR}/{brats_id}/{scan_type}/"
    try:
        x = process_scan_from_dicom(series_dir)  # [1,128,128,64,1]
        ok = np.int32(1)
    except Exception:
        x = tf.zeros([1, 128, 128, 64, 1], dtype=tf.float32)
        ok = np.int32(0)
    return x, ok


def _tf_load_one_test(brats_id):
    x, ok = tf.py_function(_load_one_test, [brats_id], Tout=(tf.float32, tf.int32))
    x.set_shape((1, 128, 128, 64, 1))
    ok.set_shape(())
    return x, ok


test_ds = tf.data.Dataset.from_tensor_slices(test_ids)
test_ds = test_ds.map(_tf_load_one_test, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

preds = np.empty((len(test_ids),), dtype=np.float32)
missing = 0
model_missing = 0

i = 0
for x, ok in test_ds:
    if int(ok.numpy()) == 0:
        preds[i] = 0.5
        missing += 1
    else:
        if inference_model is None:
            preds[i] = 0.5
            model_missing += 1
        else:
            p = _model_predict_single_prob(inference_model, x)
            if not np.isfinite(p):
                p = 0.5
            preds[i] = np.float32(np.clip(p, 0.0, 1.0))
    i += 1

sub["MGMT_value"] = preds.astype(float)

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)

print(f"Wrote {out_path} with {len(sub)} rows.")
print(
    f"DICOM fallback(0.5) used for {missing} cases; model-missing fallback used for {model_missing} cases."
)
print(sub.head())
print("Submission columns:", list(sub.columns))
print("MGMT_value min/max:", float(np.min(preds)), float(np.max(preds)))
