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

- What this solution (achieved 0.5) has done: 'I fix the environment-breaking import error by avoiding `pydicom` (which is triggering a protobuf `GetPrototype` crash) and instead loading DICOM pixel arrays via the already-available `tensorflow.io.decode_dicom_image`. I also fix the training crash by changing the AUC metric to a binary AUC computed from the positive-class probability (current 2-class softmax + sparse labels causes a shape mismatch in TF/Keras metrics). Finally, I correct test case discovery to ignore non-ID folders (e.g., a stray `test` directory name) and ensure `BraTS21ID` is written as a zero-padded string matching `sample_submission.csv`, producing a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the environment-breaking protobuf error by forcing TensorFlow’s pure-Python protobuf implementation before importing TensorFlow/Keras, which avoids the `MessageFactory.GetPrototype` crash. I also fix the training-time AUC metric bug by creating a single `tf.keras.metrics.AUC` instance in `build_model()` instead of instantiating it inside the metric function (which is what triggers the `tf.function only supports singleton tf.Variables` error). These changes keep the model architecture, loss, data loading, and training loop intact while allowing the notebook to run end-to-end. Finally, I keep the submission writing logic the same and ensure `submission.csv` is produced.'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by ensuring the protobuf environment variables are set **before any TensorFlow-related import** and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` unconditionally (some environments ignore `setdefault`). I also make the Keras imports consistent (use `tf.keras` everywhere) to avoid mixed `keras`/`tf.keras` behavior that can trigger protobuf/serialization edge issues in Kaggle. These changes are execution/stability-only and keep your data loading, model architecture, training loop, and submission formatting the same, so they should not materially change your score beyond negligible floating-point noise. Finally, I keep writing `submission.csv` with the required columns and ID zero-padding.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import and by avoiding other imports that may indirectly load compiled protobuf first. I also make the TensorFlow/Keras imports consistent (`tf.keras` everywhere) to reduce the chance of mixed-keras initialization issues. The rest of the pipeline (DICOM loading via `tf.io.decode_dicom_image`, slice selection, model architecture, training loop, and submission formatting) be kept the same so the score behavior remains essentially unchanged while the script runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow crash (`MessageFactory.GetPrototype`) by setting the protobuf env vars *unconditionally* and also forcing the pure-Python implementation via `google.protobuf.internal.api_implementation` **before** importing TensorFlow. To keep the core training/inference logic unchanged while improving stability, I also make sure we don’t import `skimage` (not guaranteed installed) by providing a small, deterministic resize fallback using TensorFlow when `skimage` is unavailable. Finally, I keep the same model/training loop and submission formatting, ensuring `submission.csv` is always written with the required columns and ID alignment.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow crash by moving all protobuf-forcing logic to the very top and avoiding importing `google.protobuf` before TensorFlow, since that can lock in the incompatible C++ implementation and trigger the `MessageFactory.GetPrototype` error. To keep your core model/training/inference logic unchanged, I won’t touch the architecture, loss, slice selection, or averaging scheme; changes are only to environment setup/import order stability. I also add a small safety fallback so that if DICOM decoding fails for a slice, the loader continues gracefully (same behavior as your current `None`-skip), and ensure the submission is always written as `submission.csv` with correct columns/ID formatting.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation at the very top *before* any TensorFlow import, and additionally setting the implementation through `google.protobuf.internal.api_implementation` to prevent the `MessageFactory.GetPrototype` failure in Kaggle. I keep your data loading via `tf.io.decode_dicom_image`, model architecture, training loop, and submission formatting the same. I also add a small safety fallback in `_safe_dcm_read` to handle occasional multi-frame DICOM tensors deterministically (take the first frame), which improves robustness without changing the core approach. The script run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I fix the immediate protobuf/TensorFlow crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf implementation *before anything else* and preventing the C++ implementation from being imported first (this is the root cause in Kaggle for this competition). I also make the TensorFlow import more robust by clearing any already-imported protobuf modules in the current kernel session before importing TensorFlow, while keeping your data loading via `tf.io.decode_dicom_image`, model, training loop, and submission logic unchanged. These changes are execution/stability-focused and should not materially change the score beyond negligible numeric differences. The script still train, predict, and write a valid `submission.csv` with the required columns and ID formatting.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation in a way that reliably takes effect before TensorFlow loads any compiled protobuf, without changing your modeling/training/prediction logic. Specifically, I move the protobuf-environment setup to the absolute top, avoid importing `google.protobuf` before importing TensorFlow, and also set `TF_PROTOBUF_IMPLEMENTATION=python` (a TF-recognized knob) to prevent the `MessageFactory.GetPrototype` failure. Everything else (DICOM loading via `tf.io.decode_dicom_image`, slice selection, model architecture, training loop, and submission formatting) is preserved so score behavior should remain essentially unchanged while the notebook runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation even more defensively *before* importing TensorFlow, and by clearing any already-imported `google.protobuf*` modules in the session (this is execution/stability-only). I also make the custom AUC metric safe by resetting its internal state each call to avoid state accumulation across batches/epochs (keeps the same “AUC on positive-class probability” semantics). Finally, I add a small guard so test-case discovery never returns an empty list (which would break stacking) and ensure the submission is always written as `submission.csv` with correct ID formatting and alignment to `sample_submission.csv`. These changes preserve your core loading logic, model architecture, training loop, and prediction averaging scheme.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation in the most reliable way *before* importing TensorFlow, and by clearing any already-imported `google.protobuf*` modules. I keep your data loading (via `tf.io.decode_dicom_image`), model architecture, training loop, and submission formatting unchanged so evaluation semantics and score behavior remain essentially the same. The only functional change is ensuring the environment setup happens early enough to prevent the crash, so the pipeline runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the immediate protobuf/TensorFlow crash by making the protobuf forcing truly “first” and removing the risky `google.protobuf` import/monkeypatch that can itself lock in the incompatible implementation before TensorFlow loads. I keep your DICOM loading via `tf.io.decode_dicom_image`, slice selection, model, training loop, and submission formatting unchanged, so behavior/score should remain essentially the same aside from negligible numeric noise. I also make the AUC metric state handling safe by using Keras’ built-in `AUC(curve="ROC")` directly on the positive-class probability without resetting internal state per batch (resetting per batch makes the metric meaningless and can hurt training feedback). The script run end-to-end and always write a valid `submission.csv` with the required columns and zero-padded IDs.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf implementation *before any TensorFlow-related import* and by ensuring no compiled protobuf modules are already loaded in-session (this is the root cause of your current failure in cell 0). I keep your data loading via `tf.io.decode_dicom_image`, slice selection, model architecture, loss, and training loop unchanged so scoring behavior stays essentially the same. I also make the AUC metric implementation safe under `tf.function` by using a proper metric object directly (same semantics: ROC AUC on the positive-class probability). Finally, I keep the submission formatting/alignment logic intact and ensure `submission.csv` is always written correctly.'
- What this solution (achieved 0.5) has done: 'I fix the crash caused by TensorFlow/protobuf incompatibility by ensuring the pure-Python protobuf implementation is locked in *before* importing TensorFlow, without importing/monkeypatching `google.protobuf` (which can accidentally load the C++ backend first). I do this by setting the environment variables at the very top and starting the script from cell 1 (Kaggle runs from the top anyway), removing the unsafe “delete google.protobuf modules” block that doesn’t reliably prevent the `MessageFactory.GetPrototype` issue. The rest of the pipeline (DICOM loading via `tf.io.decode_dicom_image`, slice selection, model, training loop, and submission formatting) is kept the same so the score behavior should remain essentially unchanged while producing a valid `submission.csv`. This should unblock end-to-end execution and keep the submission format aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["TF_PROTOBUF_IMPLEMENTATION"] = "python"

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

try:
    from skimage.transform import resize as _sk_resize

    def resize(img, out_shape, preserve_range=True, anti_aliasing=True):
        return _sk_resize(
            img,
            out_shape,
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing,
        )

except Exception:

    def resize(img, out_shape, preserve_range=True, anti_aliasing=True):
        x = tf.convert_to_tensor(img, dtype=tf.float32)
        x = x[tf.newaxis, ..., tf.newaxis]  # (1,H,W,1)
        x = tf.image.resize(
            x, out_shape, method="bilinear", antialias=bool(anti_aliasing)
        )
        x = tf.squeeze(x, axis=(0, 3))
        return x.numpy()


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

INPUT_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(INPUT_ROOT, "train")
TEST_DIR = os.path.join(INPUT_ROOT, "test")
LABELS_CSV = os.path.join(INPUT_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(INPUT_ROOT, "sample_submission.csv")

IMG_PX_SIZE = 150
SLICES_PER_CASE = 6
BAD_CASES = {"00109", "00123", "00709"}

print("Using TRAIN_DIR:", TRAIN_DIR)
print("Using TEST_DIR:", TEST_DIR)
print("Using LABELS_CSV:", LABELS_CSV)


def _safe_dcm_read(path):
    """
    Returns a 2D float32 numpy array or None.
    Uses tf.io.decode_dicom_image to avoid pydicom runtime issues.
    """
    try:
        b = tf.io.read_file(path)
        img = tf.io.decode_dicom_image(
            b,
            dtype=tf.uint16,
            color_dim=False,
            scale="preserve",
            expand_animations=True,
        )
        img = tf.squeeze(img)

        try:
            if getattr(img, "shape", None) is not None and img.shape.rank == 3:
                img = img[0]
        except Exception:
            pass

        arr = img.numpy().astype(np.float32)
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _normalize_img(img2d):
    img2d = img2d.astype(np.float32)
    mn = np.min(img2d)
    mx = np.max(img2d)
    if mx - mn < 1e-6:
        return None
    img2d = (img2d - mn) / (mx - mn)
    return img2d


def _load_case_t2_slices(
    case_dir, img_px_size=150, slices_per_case=6, sum_thr=90000.0, norm_sum_thr=3000.0
):
    """
    Core logic preserved: scan T2w DICOMs, pick slices passing thresholds,
    resize to IMG_PX_SIZE, stack to 3 channels, normalize.
    Returns (slices_per_case, img_px_size, img_px_size, 3) float32.
    If not enough slices, pads by repeating last valid (or zeros if none).
    """
    subdirs = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
    t2_dir = None
    for d in subdirs:
        if os.path.basename(d).lower() == "t2w":
            t2_dir = d
            break
    if t2_dir is None:
        if len(subdirs) > 0:
            t2_dir = subdirs[-1]
        else:
            out = np.zeros(
                (slices_per_case, img_px_size, img_px_size, 3), dtype=np.float32
            )
            return out

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(t2_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    collected = []
    for fp in dcm_files:
        arr = _safe_dcm_read(fp)
        if arr is None:
            continue
        if float(np.sum(arr)) <= sum_thr:
            continue
        arr = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        arr = _normalize_img(arr)
        if arr is None:
            continue
        stacked = np.stack((arr, arr, arr), axis=-1)  # (H,W,3)
        mx = np.max(stacked)
        if mx > 1e-6:
            stacked = stacked / mx
        if float(np.sum(stacked)) <= norm_sum_thr:
            continue
        collected.append(stacked)
        if len(collected) >= slices_per_case:
            break

    if len(collected) == 0:
        out = np.zeros((slices_per_case, img_px_size, img_px_size, 3), dtype=np.float32)
        return out

    while len(collected) < slices_per_case:
        collected.append(collected[-1])

    return np.stack(collected[:slices_per_case], axis=0).astype(np.float32)


def load_dataset_t2(train_dir, labels_df, img_px_size=150, slices_per_case=6):
    X_list, y_list = [], []
    available_cases = sorted([f.name for f in os.scandir(train_dir) if f.is_dir()])
    id_to_label = dict(
        zip(
            labels_df["BraTS21ID"].astype(str).str.zfill(5),
            labels_df["MGMT_value"].astype(int),
        )
    )

    for case_id in available_cases:
        if case_id in BAD_CASES:
            continue
        if case_id not in id_to_label:
            continue
        case_dir = os.path.join(train_dir, case_id)
        x_case = _load_case_t2_slices(
            case_dir, img_px_size=img_px_size, slices_per_case=slices_per_case
        )
        X_list.append(x_case)
        y_list.append(np.full((x_case.shape[0],), id_to_label[case_id], dtype=np.int32))

    if len(X_list) == 0:
        raise RuntimeError("No training data loaded. Check paths and labels.")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def load_test_cases_t2(test_dir, img_px_size=150, slices_per_case=6):
    case_ids = sorted(
        [
            f.name
            for f in os.scandir(test_dir)
            if f.is_dir() and f.name.isdigit() and len(f.name) == 5
        ]
    )
    if len(case_ids) == 0:
        raise RuntimeError(f"No test cases found in {test_dir}. Check dataset path.")
    X_cases = []
    for case_id in case_ids:
        case_dir = os.path.join(test_dir, case_id)
        x_case = _load_case_t2_slices(
            case_dir, img_px_size=img_px_size, slices_per_case=slices_per_case
        )
        X_cases.append(x_case)
    X_cases = np.stack(X_cases, axis=0)  # (N_cases, S, H, W, 3)
    return case_ids, X_cases




## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels["BraTS21ID"] = labels["BraTS21ID"].astype(str).str.zfill(5)
print(labels.head())
print("Labels shape:", labels.shape)

X, y = load_dataset_t2(
    TRAIN_DIR, labels, img_px_size=IMG_PX_SIZE, slices_per_case=SLICES_PER_CASE
)
print("Loaded X:", X.shape, "y:", y.shape, "pos_rate:", float(y.mean()))

idx = np.arange(len(y))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.85 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, y_tr = X[tr_idx], y[tr_idx]
X_va, y_va = X[va_idx], y[va_idx]
print("Train:", X_tr.shape, "Val:", X_va.shape)




## === cell 3
def build_model(input_shape=(150, 150, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    class AUCPos(tf.keras.metrics.AUC):
        def __init__(self, name="auc", **kwargs):
            super().__init__(name=name, curve="ROC", **kwargs)

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(y_true, tf.float32)
            y_pred_pos = y_pred[:, 1]
            return super().update_state(y_true, y_pred_pos, sample_weight=sample_weight)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[AUCPos(name="auc")],
    )
    return model


model_T2 = build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))
model_T2.summary()



## === cell 4
BATCH_SIZE = 32
EPOCHS = 3  # keep as provided

history = model_T2.fit(
    X_tr,
    y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 5
test_case_ids, X_test_cases = load_test_cases_t2(
    TEST_DIR, img_px_size=IMG_PX_SIZE, slices_per_case=SLICES_PER_CASE
)
print("Test cases:", len(test_case_ids), "X_test_cases:", X_test_cases.shape)

N = X_test_cases.shape[0]
S = X_test_cases.shape[1]
X_test_flat = X_test_cases.reshape(N * S, IMG_PX_SIZE, IMG_PX_SIZE, 3)

preds_flat = model_T2.predict(X_test_flat, batch_size=BATCH_SIZE, verbose=0)
pred_pos_flat = preds_flat[:, 1].astype(np.float32)

pred_pos_cases = pred_pos_flat.reshape(N, S).mean(axis=1)
print(
    "Pred stats:",
    float(pred_pos_cases.min()),
    float(pred_pos_cases.max()),
    float(pred_pos_cases.mean()),
)




## === cell 6
def create_sub(case_ids, pred_pos):
    df = pd.DataFrame(
        {
            "BraTS21ID": [str(x).zfill(5) for x in case_ids],
            "MGMT_value": pred_pos.astype(float),
        }
    )
    return df


sub_df = create_sub(test_case_ids, pred_pos_cases)

sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(float(sub_df["MGMT_value"].mean()))

print(sub_df.head())
print("Submission shape:", sub_df.shape)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_df.columns))
print("File size (bytes):", os.path.getsize("submission.csv"))
