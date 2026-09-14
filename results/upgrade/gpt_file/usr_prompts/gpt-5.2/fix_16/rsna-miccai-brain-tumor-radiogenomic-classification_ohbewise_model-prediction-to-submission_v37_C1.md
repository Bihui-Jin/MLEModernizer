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

0.62

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.62) has done: 'The timeout is dominated by repeated DICOM decoding plus per-case TensorFlow preprocessing executed eagerly for every slice, both in the small training feature extraction loop and across all test patients. I keep the exact same feature definition and 1D logistic training loop, but make the pipeline faster by (1) sorting slices using DICOM metadata (avoids costly filename sorting and mis-order), (2) reading only pixel data and required tags with `pydicom.dcmread(..., stop_before_pixels=False, specific_tags=...)`, (3) moving `preprocess_volume` into a single `@tf.function`-compiled graph to eliminate Python/eager overhead, and (4) caching features on disk in `/kaggle/working/` so reruns don’t recompute. These changes are semantics-preserving: same voxels, same resize/crop/pad/normalization, same feature computation, same optimizer/steps—just less overhead and less redundant work.'
- What this solution (achieved 0.62) has done: 'I fix the runtime error caused by `tf.config.experimental.enable_op_determinism()` under this environment’s protobuf/TensorFlow combination by guarding it so the script continues deterministically where supported instead of crashing. I also make the `pydicom` import more robust: if `pydicom` (or its transitive deps) fails to import, the pipeline still complete by falling back to the label prior and producing a valid submission. These changes are score-neutral in intent (they don’t alter the model/feature logic when `pydicom` works) and primarily ensure the notebook runs end-to-end and always writes `submission.csv`. No changes are made to the feature definition, training loop, or prediction semantics.'
- What this solution (achieved 0.62) has done: 'I fix the crash happening at import time by avoiding TensorFlow’s determinism call path that triggers the protobuf `MessageFactory.GetPrototype` AttributeError in this environment, while keeping seeds and all modeling logic unchanged. I also make the pydicom import more robust for Kaggle by installing a safe fallback path that still produces a valid CSV if pydicom isn’t available. No changes are made to the feature definition, the 1D logistic training loop, or prediction semantics, so the score behavior should remain essentially the same (and should not move away from your current 0.62). Finally, I ensure the script runs end-to-end and always writes `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.62) has done: 'I fix the crash that happens before any training/inference by preventing TensorFlow from importing code paths that trigger the protobuf `MessageFactory.GetPrototype` incompatibility, while keeping the model/feature logic unchanged. Concretely, I delay importing TensorFlow until after basic setup and keep determinism disabled (as you already intended), so the notebook can run end-to-end. I also add a safe fallback to still write a valid `submission.csv` using the label prior if TensorFlow fails to import at runtime, ensuring you always get a valid submission file. These changes are intended to be score-neutral when TensorFlow imports successfully (so your ~0.62 behavior should remain).'
- What this solution (achieved 0.62) has done: 'I fix the crash happening during TensorFlow import by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow, which avoids the protobuf `MessageFactory.GetPrototype` incompatibility in this environment. I keep the rest of the pipeline (feature extraction, preprocessing, 1D logistic training loop, and submission writing) unchanged so the score behavior stays close to your current ~0.62 while restoring end-to-end execution. I also keep the existing safe fallbacks (prior-only) so a valid `submission.csv` is always produced even if TensorFlow/pydicom fails for any reason. These changes are directly tied to stability/runtime and should be score-neutral except that they allow the intended model path to run.'
- What this solution (achieved 0.62) has done: 'I fix the immediate crash happening before any training/inference by forcing protobuf to use the pure-Python implementation *and* disabling the compiled C++ protobuf in this process, which is the typical root cause of the `MessageFactory.GetPrototype` AttributeError when importing TensorFlow in this environment. This is a runtime/stability fix and does not change your feature definition, preprocessing, logistic training loop, or prediction logic, so the score should stay close to your current ~0.62 (target score is -1.0, so we should not try to move it downward). I also add a small safety fallback: if TensorFlow still can’t import, the code proceed with prior-only predictions and still write a valid `submission.csv`. Finally, I keep paths and submission formatting unchanged and ensure the script runs end-to-end.'
- What this solution (achieved 0.62) has done: 'I fix the crash that happens before training by ensuring the protobuf implementation environment variables are set before any TensorFlow/protobuf code is imported, and by fully isolating TensorFlow import so a protobuf incompatibility cannot abort the run. I keep your feature extraction, preprocessing, and 1D logistic training loop identical, only adding safe guards and clearer fallbacks so the pipeline always finishes and writes `/kaggle/working/submission.csv`. I also add a small robustness fix so `preprocess_volume` is never referenced unless TensorFlow is actually available, preventing secondary NameErrors if TF fails. These changes are intended to be score-neutral (keeping your ~0.62 behavior when TF imports successfully), while restoring end-to-end execution.'
- What this solution (achieved 0.62) has done: 'I fix the immediate TensorFlow/protobuf import crash by ensuring protobuf is forced into the pure-Python implementation *before any protobuf-related module is imported*, and by adding a safe “try TF import / fallback” wrapper so the notebook always completes. I keep your feature extraction, preprocessing, and 1D logistic training loop unchanged; the only logic change is stability-related and should keep the score near your current ~0.62 when TF loads. I also make the pydicom series read slightly more robust by explicitly stopping before pixels = False (so PixelData is actually read when requested) without changing which pixels/tags you use. Finally, I ensure submission writing always happens to `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.62) has done: 'I fix the immediate crash happening before any modeling by ensuring protobuf is forced to the pure-Python implementation *before* TensorFlow (or any protobuf-dependent module) is imported, and by adding a safe TensorFlow import wrapper that cannot abort the run. This is a runtime/stability change and does not alter your feature definition, preprocessing, logistic training loop, or submission formatting when TensorFlow imports successfully (so score behavior should remain near your current ~0.62). If TensorFlow still cannot be imported in this environment, the code fall back to prior-only predictions and still write a valid `/kaggle/working/submission.csv`. No score-targeting changes are made because your target score is -1.0 (non-sensical for AUC) and your current score is already valid.'
- What this solution (achieved 0.62) has done: 'I fix the immediate TensorFlow/protobuf crash by ensuring the protobuf environment variables are set before any protobuf/TensorFlow-related imports happen, and by wrapping TensorFlow import so it cannot abort the run. I also add a safe, score-neutral fallback path that produces a valid `submission.csv` even if TensorFlow cannot be imported (using the label prior). Finally, I make sure feature extraction only references TensorFlow objects when TF is actually available, preventing secondary NameErrors, while keeping the model/feature logic unchanged when TF works.'
- What this solution (achieved 0.62) has done: 'You’re crashing in cell 0 during TensorFlow import due to a protobuf runtime mismatch (`MessageFactory.GetPrototype`), so the code never reaches feature extraction or submission writing. I fix this by setting an additional environment variable to force the pure-Python protobuf backend *before* any TensorFlow/protobuf import, and by moving the TensorFlow import into a stricter try/except that guarantees the script continues (prior-only fallback) if TF still can’t load. I also add a tiny safety guard so feature cache save/load only happens when the feature cache dict is serializable, but I won’t change any model/feature/training logic—so your ~0.62 score behavior should remain essentially the same when TF loads successfully. The script always write `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.62) has done: 'The crash happens before any modeling because TensorFlow import triggers a protobuf incompatibility (`MessageFactory.GetPrototype`), so I prevent TensorFlow from importing any compiled protobuf implementation by setting the correct environment variables up-front (and removing the invalid one). I also make TensorFlow import fully optional: if it still fails, the code safely fall back to the prior-only prediction path and still write a valid `submission.csv`. These changes are stability-focused and keep your feature definition, preprocessing, and 1D logistic training loop unchanged when TensorFlow successfully imports, so the score behavior should remain close to your current ~0.62. Finally, I add a tiny guard so `preprocess_volume` is never referenced unless TensorFlow is actually available, avoiding secondary errors.'

# 9. Code solution

## === cell 0
import os, sys, csv
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

print("Python:", sys.version)

SEED = 42
np.random.seed(SEED)

TF_AVAILABLE = False
TF_IMPORT_ERROR = None
tf = None

try:
    import pydicom  # type: ignore

    PYDICOM_AVAILABLE = True
    PYDICOM_IMPORT_ERROR = None
except Exception as e:
    PYDICOM_AVAILABLE = False
    PYDICOM_IMPORT_ERROR = repr(e)

print("pydicom available:", PYDICOM_AVAILABLE)
if not PYDICOM_AVAILABLE:
    print("pydicom import error:", PYDICOM_IMPORT_ERROR)

try:
    import tensorflow as tf  # type: ignore

    TF_AVAILABLE = True
    tf.random.set_seed(SEED)
    print("TensorFlow:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    print("TensorFlow import failed; will use prior-only fallback.")
    print("TF import error:", TF_IMPORT_ERROR)

print(
    "Determinism enable skipped to avoid protobuf/TensorFlow crash in this environment."
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _list_dicom_files(series_dir: str):
    if not os.path.isdir(series_dir):
        return []
    return [str(p) for p in Path(series_dir).glob("*.dcm")]


def _slice_sort_key(ds):
    inst = getattr(ds, "InstanceNumber", None)
    if inst is not None:
        try:
            return (0, int(inst))
        except Exception:
            pass
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        try:
            return (1, float(ipp[2]))
        except Exception:
            pass
    return (2, 0)


def read_dicom_series(series_dir: str) -> np.ndarray:
    """Read a DICOM series directory into a 3D float32 volume shaped [H, W, Z]."""
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom not available; cannot decode DICOM series.")

    files = _list_dicom_files(series_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    dsets = []
    specific_tags = [
        "PixelData",
        "RescaleSlope",
        "RescaleIntercept",
        "InstanceNumber",
        "ImagePositionPatient",
        "BitsAllocated",
        "BitsStored",
        "PixelRepresentation",
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "Rows",
        "Columns",
    ]
    for fp in files:
        ds = pydicom.dcmread(
            fp,
            force=True,
            stop_before_pixels=False,
            specific_tags=specific_tags,
        )
        dsets.append(ds)

    dsets.sort(key=_slice_sort_key)

    slices = []
    for ds in dsets:
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        if arr.ndim != 2:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Unexpected DICOM pixel_array shape {arr.shape}")

        slices.append(arr)

    vol = np.stack(slices, axis=-1).astype(np.float32)  # [H, W, Z]
    return vol


if TF_AVAILABLE:

    @tf.function(reduce_retracing=True)
    def _preprocess_volume_tf(
        vol_hwd: "tf.Tensor", th: "tf.Tensor", tw: "tf.Tensor", tz: "tf.Tensor"
    ) -> "tf.Tensor":
        vol = tf.cast(vol_hwd, tf.float32)  # [H,W,Z]
        vol = tf.transpose(vol, [2, 0, 1])  # [Z,H,W]
        vol = tf.expand_dims(vol, axis=-1)  # [Z,H,W,1]
        vol = tf.image.resize(vol, size=[th, tw], method="bilinear", antialias=True)
        vol = tf.squeeze(vol, axis=-1)  # [Z,th,tw]

        def _center_crop(vol_, tz_):
            start = tf.maximum(0, (tf.shape(vol_)[0] - tz_) // 2)
            return vol_[start : start + tz_]

        def _center_pad(vol_, tz_):
            pad_total = tz_ - tf.shape(vol_)[0]
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            return tf.pad(vol_, [[pad_before, pad_after], [0, 0], [0, 0]])

        vol = tf.cond(
            tf.shape(vol)[0] > tz,
            lambda: _center_crop(vol, tz),
            lambda: tf.cond(
                tf.shape(vol)[0] < tz, lambda: _center_pad(vol, tz), lambda: vol
            ),
        )

        vol = tf.transpose(vol, [1, 2, 0])  # [th,tw,tz]

        mean = tf.reduce_mean(vol)
        std = tf.math.reduce_std(vol)
        vol = tf.cond(std > 0, lambda: (vol - mean) / (std + 1e-6), lambda: vol - mean)

        vol = tf.expand_dims(vol, axis=-1)  # [th,tw,tz,1]
        vol = tf.expand_dims(vol, axis=0)  # [1,th,tw,tz,1]
        return vol

    def preprocess_volume(
        volume_hwd: np.ndarray, target_hw=(128, 128), target_z=64
    ) -> "tf.Tensor":
        """
        Convert [H,W,Z] float32 volume -> model input [1,H,W,Z,1]
        Steps: resize H,W per-slice; center-crop/pad Z; z-normalize (robust to constant volumes).
        """
        vol = tf.convert_to_tensor(volume_hwd, dtype=tf.float32)
        th = tf.constant(int(target_hw[0]), dtype=tf.int32)
        tw = tf.constant(int(target_hw[1]), dtype=tf.int32)
        tz = tf.constant(int(target_z), dtype=tf.int32)
        return _preprocess_volume_tf(vol, th, tw, tz)




## === cell 2
data_dir = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
train_dir = f"{data_dir}train"
test_dir = f"{data_dir}test"

train_labels_path = f"{data_dir}train_labels.csv"
sample_path = f"{data_dir}sample_submission.csv"

train_labels = pd.read_csv(train_labels_path)
sample_sub = pd.read_csv(sample_path)

train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

test_ids = sample_sub["BraTS21ID"].tolist()
print(f"Total train labels: {len(train_labels)}")
print(f"Total test IDs (from sample_submission): {len(test_ids)}")

bad_ids = {"00109", "00123", "00709"}
train_labels = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].reset_index(
    drop=True
)
print(f"Train labels after excluding known-bad cases: {len(train_labels)}")



## === cell 3
prior = float(train_labels["MGMT_value"].mean())
prior = float(np.clip(prior, 0.0, 1.0))
print("Training label prior (mean MGMT_value):", prior)

scan_type = "T2w"
target_hw = (128, 128)
target_z = 64

FEATURE_CACHE_PATH = "/kaggle/working/feature_cache_t2w_128x128x64.npz"
_feature_cache = {}
if PYDICOM_AVAILABLE and os.path.exists(FEATURE_CACHE_PATH):
    try:
        loaded = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
        _feature_cache = loaded["cache"].item()
        if not isinstance(_feature_cache, dict):
            _feature_cache = {}
        print(
            f"Loaded feature cache with {len(_feature_cache)} entries from {FEATURE_CACHE_PATH}"
        )
    except Exception as e:
        print("Failed to load feature cache:", repr(e))
        _feature_cache = {}


def _cheap_feature_from_series(series_dir: str) -> float:
    key = os.path.abspath(series_dir)
    if key in _feature_cache:
        return float(_feature_cache[key])

    vol = read_dicom_series(series_dir)

    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available; cannot preprocess volume.")

    x = preprocess_volume(vol, target_hw=target_hw, target_z=target_z)  # [1,H,W,Z,1]
    z0 = target_z // 4
    z1 = 3 * target_z // 4
    band = x[:, :, :, z0:z1, :]
    feat = tf.reduce_mean(tf.abs(band)).numpy().item()
    if not np.isfinite(feat):
        raise ValueError("Non-finite feature")

    _feature_cache[key] = float(feat)
    return float(feat)


use_feature_model = PYDICOM_AVAILABLE and TF_AVAILABLE

if use_feature_model:
    max_train_cases = 120  # bounded to stay within 600s (kept identical)
    rows = train_labels.copy()
    if max_train_cases is not None and len(rows) > max_train_cases:
        rows = rows.sample(n=max_train_cases, random_state=SEED).reset_index(drop=True)

    feats = []
    ys = []
    ok_ids = []
    for r in rows.itertuples(index=False):
        pid = r.BraTS21ID
        y = float(r.MGMT_value)
        series_dir = f"{train_dir}/{pid}/{scan_type}/"
        try:
            f = _cheap_feature_from_series(series_dir)
            feats.append(f)
            ys.append(y)
            ok_ids.append(pid)
        except Exception:
            continue

    feats = np.asarray(feats, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    print(
        f"Feature extraction: {len(feats)} successful cases out of {len(rows)} attempted"
    )

    if len(feats) >= 20 and len(np.unique(ys)) >= 2:
        f_mean = feats.mean()
        f_std = feats.std() + 1e-6
        feats_z = (feats - f_mean) / f_std

        w = tf.Variable(0.0, dtype=tf.float32)
        b = tf.Variable(
            np.log(prior / (1.0 - prior + 1e-6) + 1e-6), dtype=tf.float32
        )  # init to prior logit
        opt = tf.keras.optimizers.Adam(learning_rate=0.05)

        x_tf = tf.constant(feats_z.reshape(-1, 1), dtype=tf.float32)
        y_tf = tf.constant(ys.reshape(-1, 1), dtype=tf.float32)

        for _ in range(200):
            with tf.GradientTape() as tape:
                logits = x_tf * w + b
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=y_tf, logits=logits)
                )
            grads = tape.gradient(loss, [w, b])
            opt.apply_gradients(zip(grads, [w, b]))

        print(
            "Trained 1D logistic model:", "w=", float(w.numpy()), "b=", float(b.numpy())
        )

        def predict_prob_from_feature(f: float) -> float:
            fz = (float(f) - float(f_mean)) / float(f_std)
            logit = fz * float(w.numpy()) + float(b.numpy())
            p = 1.0 / (1.0 + np.exp(-logit))
            return float(np.clip(p, 0.0, 1.0))

        feature_model = (predict_prob_from_feature, _cheap_feature_from_series)
    else:
        feature_model = None
else:
    feature_model = None

if PYDICOM_AVAILABLE and isinstance(_feature_cache, dict):
    try:
        np.savez_compressed(FEATURE_CACHE_PATH, cache=_feature_cache)
        print(
            f"Saved feature cache with {len(_feature_cache)} entries to {FEATURE_CACHE_PATH}"
        )
    except Exception as e:
        print("Failed to save feature cache:", repr(e))



## === cell 4
preds = {}

for patient in test_ids:
    if feature_model is None:
        p = prior
    else:
        predict_prob_from_feature, feat_fn = feature_model
        series_dir = f"{test_dir}/{patient}/{scan_type}/"
        try:
            f = feat_fn(series_dir)
            p = predict_prob_from_feature(f)
        except Exception:
            p = prior
    preds[patient] = float(p)

submission = sample_sub.copy()
submission["MGMT_value"] = submission["BraTS21ID"].map(preds).astype(float)
submission["MGMT_value"] = submission["MGMT_value"].fillna(prior).clip(0.0, 1.0)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print(submission.head())
print(f"Wrote submission to: {out_path} with {len(submission)} rows")
assert os.path.exists(out_path) and out_path.endswith(".csv")
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["BraTS21ID", "MGMT_value"]
