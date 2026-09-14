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

0.55765

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The initial crash happens before any model code runs due to an incompatibility between `pydicom` and the protobuf runtime in this Kaggle image; I fix that by forcing the pure-Python protobuf implementation before importing `pydicom`. The invalid submission row-count error is because the image loaders silently skip cases when they can’t find enough slices, so predictions don’t align 1:1 with all test IDs; I preserve your core approach but make loaders return exactly one (or N) image(s) per case by adding per-case fallbacks (use the middle slice, or a zero image) instead of skipping. Finally, I keep your averaging logic but ensure the submission uses the same ordered IDs as `sample_submission.csv` so Kaggle row counts always match.'
- What this solution (achieved 0.47529) has done: 'I fix the protobuf/pydicom incompatibility that is crashing the notebook before any model code runs by forcing the Python protobuf implementation *and* disabling C++ descriptors via an additional env var set before importing pydicom. I also make the pydicom import robust by falling back to a minimal DICOM reader path if needed, while keeping your image selection/normalization logic unchanged. Finally, I ensure test-folder ordering matches `sample_submission.csv` IDs exactly (so predictions always align 1:1 with required rows) without changing your modeling approach or averaging logic.'
- What this solution (achieved 0.55765) has done: 'I fix the crash in the first cell by hard-pinning protobuf to the pure-Python implementation *and* blocking the C++ descriptor path before any protobuf/pydicom-related imports, which avoids the `MessageFactory.GetPrototype` error. Then I make the DICOM pixel loader robust by adding a tiny fallback reader using SimpleITK (commonly available on Kaggle for this competition) when `pydicom` cannot be used, without changing your slice-picking, resizing, normalization, or prediction-averaging logic. Finally, I keep the exact submission ordering tied to `sample_submission.csv` and ensure we always produce one prediction per required ID, so the pipeline always writes a valid `submission.csv`.'
- What this solution (achieved 0.48235) has done: 'I fix the immediate crash in the import cell by preventing the problematic `pydicom` import path that triggers the protobuf `MessageFactory.GetPrototype` error, and instead reliably using `SimpleITK` as the DICOM reader (with a safe fallback to zeros if neither works). This is a correctness/stability fix that keeps your core pipeline (slice picking, resizing, normalization, simple CNN, averaging) unchanged while making the notebook run end-to-end. I also make the reader selection explicit and deterministic so test loading never silently breaks mid-run. No score-tuning changes are introduced because the current score is already valid and the target score provided (-1.0) is not a meaningful ROC-AUC target.'
- What this solution (achieved 0.52824) has done: 'I fix the immediate runtime crash by preventing protobuf from loading its C++ implementation before TensorFlow (which is what triggers the `MessageFactory.GetPrototype` error in this environment), while keeping your “no pydicom, use SimpleITK” approach unchanged. I also make the DICOM reader slightly more robust by forcing SimpleITK to read DICOM as a single 2D slice and falling back cleanly, without changing your slice-picking/resizing/normalization logic. Since your current score is already a valid AUC and the provided target score (-1.0) is not meaningful for ROC-AUC, I not make any score-tuning changes—only stability/correctness fixes. The script still write a valid `submission.csv` with exactly the `sample_submission.csv` ordering and row count.'
- What this solution (achieved 0.56235) has done: 'I fix the crash happening before any model code runs by making the protobuf/pydicom-related environment handling more robust and, critically, by avoiding importing `keras` from the standalone `keras` package (which is what commonly triggers the `MessageFactory.GetPrototype` protobuf error in this Kaggle image). I keep your core pipeline identical (SimpleITK DICOM reading, slice selection, resizing/normalization, two small CNNs, and averaging predictions) and only change imports to use `tf.keras` consistently. I also keep the submission ordering tied exactly to `sample_submission.csv` and ensure the script always writes `submission.csv` with the correct two columns and row count. No score-tuning changes are introduced because the provided target score (-1.0) is not meaningful for ROC-AUC and your current score is already valid.'
- What this solution (achieved 0.54941) has done: 'I fix the protobuf/keras import crash by forcing the pure-Python protobuf implementation and importing TensorFlow in a way that avoids triggering the problematic protobuf descriptor path. I also make the SimpleITK DICOM read more robust by explicitly using `ImageFileReader` with `GDCMImageIO` when available, which reduces the chance of `None` images that later cause shape mismatches. These changes are stability-focused and keep your model, preprocessing, and averaging logic the same, so the score impact should be minimal while ensuring the notebook runs end-to-end. The script still write a correctly ordered `submission.csv` matching `sample_submission.csv`.'
- What this solution (achieved 0.56588) has done: 'I fix the crash in the first cell by ensuring protobuf’s pure-Python implementation is enforced *before* any TensorFlow/Keras-related imports and by importing TensorFlow via a safer path that avoids triggering the `MessageFactory.GetPrototype` protobuf descriptor issue in this Kaggle image. This is a runtime-only stability fix: it does not change your preprocessing, model architecture, or averaging logic, so score impact should be negligible (and your current 0.54941 is already in a normal range; the provided target score -1.0 isn’t meaningful for ROC-AUC). I also add a small safety fallback so that if TensorFlow still cannot import for any environment reason, the script still produce a valid `submission.csv` (using 0.5 probabilities) rather than crashing. The submission ordering/row-count logic tied to `sample_submission.csv` stays unchanged.'
- What this solution (achieved 0.53294) has done: 'I fix the crash in the first cell caused by protobuf/TensorFlow incompatibility by forcing protobuf’s pure-Python implementation earlier and importing TensorFlow only after that environment setup is guaranteed. To keep your pipeline intact, I won’t change preprocessing, slice selection, model architecture, or averaging logic; I only make the TensorFlow import path more robust and deterministic. If TensorFlow still cannot import, the code safely fall back to 0.5 predictions and still write a valid `submission.csv`. This should preserve (or slightly improve) your existing score by ensuring the real model runs rather than crashing.'
- What this solution (achieved 0.55765) has done: 'I fix the crash occurring in the very first cell (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by making the protobuf/TensorFlow import handling more defensive: enforce the pure-Python protobuf implementation *and* temporarily remove the C++ protobuf module if it is already present before importing TensorFlow. This is a stability fix that preserves your existing core pipeline (SimpleITK DICOM reading, slice selection, resizing/normalization, two small CNN stubs, averaging). I also ensure the notebook always writes a valid `submission.csv` with the exact `sample_submission.csv` ordering, even if TensorFlow still fails to import (in which case it fall back to 0.5 predictions, as your script already intends). No score-tuning changes are introduced because your current score is already valid and the provided target score (-1.0) is not meaningful for ROC-AUC.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_DISABLE_CPP_IMPLEMENTATION"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys

for mod in [
    "google._upb._message",
    "google.protobuf.pyext._message",
    "google.protobuf.internal._api_implementation",
]:
    if mod in sys.modules:
        del sys.modules[mod]

import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None

dicom = None
_PYDICOM_OK = False
dicom_import_error = "disabled to avoid protobuf MessageFactory.GetPrototype crash"

_SITK_OK = False
sitk_import_error = None
sitk = None
try:
    import SimpleITK as sitk  # noqa: F401

    _SITK_OK = True
except Exception as e:
    _SITK_OK = False
    sitk_import_error = repr(e)

_TF_OK = False
tf_import_error = None
tf = None
keras = None
layers = None
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # noqa: F401
    from tensorflow.keras import layers  # noqa: F401

    _TF_OK = True
except Exception as e:
    _TF_OK = False
    tf_import_error = repr(e)

try:
    from skimage.transform import resize as _sk_resize
except Exception:
    _sk_resize = None


def resize(img2d, out_hw):
    """Resize 2D numpy array to (H, W). Uses skimage if available; otherwise uses TF bilinear."""
    out_h, out_w = out_hw
    if _sk_resize is not None:
        return _sk_resize(
            img2d, (out_h, out_w), preserve_range=True, anti_aliasing=False
        ).astype(np.float32)
    if not _TF_OK or tf is None:
        in_h, in_w = img2d.shape[:2]
        if in_h == 0 or in_w == 0:
            return np.zeros((out_h, out_w), dtype=np.float32)
        ys = (np.linspace(0, in_h - 1, out_h)).astype(np.int32)
        xs = (np.linspace(0, in_w - 1, out_w)).astype(np.int32)
        return img2d[ys[:, None], xs[None, :]].astype(np.float32)

    x = tf.convert_to_tensor(img2d, dtype=tf.float32)[None, ..., None]  # (1,H,W,1)
    x = tf.image.resize(x, (out_h, out_w), method="bilinear")
    return x[0, ..., 0].numpy()


np.random.seed(42)
if _TF_OK and tf is not None:
    tf.random.set_seed(42)

print("pydicom import ok:", _PYDICOM_OK)
if not _PYDICOM_OK:
    print("pydicom import error (will use fallback reader):", dicom_import_error)
print("SimpleITK import ok:", _SITK_OK)
if not _SITK_OK:
    print("SimpleITK import error:", sitk_import_error)
print("TensorFlow import ok:", _TF_OK)
if not _TF_OK:
    print(
        "TensorFlow import error (will use 0.5 fallback predictions):", tf_import_error
    )




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _build_stub_model(input_size):
    if not _TF_OK or keras is None or layers is None:
        return None
    inp = keras.Input(shape=(input_size, input_size, 3))
    x = layers.Rescaling(1.0)(inp)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inp, out)
    return model


model_1 = _build_stub_model(299)  # for FLAIR stacks in this script
model_2 = _build_stub_model(150)  # for T2W stacks in this script




## === cell 2
def _safe_normalize_img(x):
    x = x.astype(np.float32)
    mx = float(np.max(x))
    if not np.isfinite(mx) or mx <= 0.0:
        return x
    return x / mx


def _read_dcm_pixel_array(p):
    if _SITK_OK and sitk is not None:
        try:
            reader = sitk.ImageFileReader()
            try:
                reader.SetImageIO("GDCMImageIO")
            except Exception:
                pass
            reader.SetFileName(p)
            img = reader.Execute()
            arr = sitk.GetArrayFromImage(img)
            if arr.ndim == 3:
                arr = arr[0]
            elif arr.ndim != 2:
                return None
            return arr
        except Exception:
            try:
                img = sitk.ReadImage(p)
                arr = sitk.GetArrayFromImage(img)
                if arr.ndim == 3:
                    arr = arr[0]
                elif arr.ndim != 2:
                    return None
                return arr
            except Exception:
                return None
    return None


def _stack_resize_norm(px2d, img_px_size):
    if px2d is None:
        img = np.zeros((img_px_size, img_px_size), dtype=np.float32)
    else:
        img = resize(px2d, (img_px_size, img_px_size))
    stacked_img = np.stack((img,) * 3, axis=-1)
    return _safe_normalize_img(stacked_img)


def _pick_best_or_middle_slice(dcm_paths, img_px_size, min_sum_thresh=0):
    """Pick first slice passing threshold; otherwise use middle slice; otherwise zeros."""
    if len(dcm_paths) == 0:
        return _stack_resize_norm(None, img_px_size)

    for p in dcm_paths:
        px = _read_dcm_pixel_array(p)
        if px is None:
            continue
        if px.sum() > min_sum_thresh:
            return _stack_resize_norm(px, img_px_size)

    mid_p = dcm_paths[len(dcm_paths) // 2]
    px = _read_dcm_pixel_array(mid_p)
    return _stack_resize_norm(px, img_px_size)


def load_test_flair_images(path_test, case_ids=None):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 299

    if case_ids is None:
        path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    else:
        path_cases = [os.path.join(path_test, cid) for cid in case_ids]

    for case_path in path_cases:
        if not os.path.isdir(case_path):
            mri_type = []
        else:
            mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])

        flair_dir = mri_type[3] if len(mri_type) >= 4 else None
        if flair_dir is None or (not os.path.isdir(flair_dir)):
            dcm_paths = []
        else:
            dcm_paths = sorted(
                [
                    f.path
                    for f in os.scandir(flair_dir)
                    if f.is_file() and f.name.lower().endswith(".dcm")
                ]
            )

        chosen_imgs = []
        for p in dcm_paths:
            px = _read_dcm_pixel_array(p)
            if px is None:
                continue
            if px.sum() > 100000:
                img3 = _stack_resize_norm(px, IMG_PX_SIZE)
                if img3.sum() > 5000:
                    chosen_imgs.append(img3)
            if len(chosen_imgs) >= 6:
                break

        while len(chosen_imgs) < 6:
            chosen_imgs.append(
                _pick_best_or_middle_slice(dcm_paths, IMG_PX_SIZE, min_sum_thresh=0)
            )

        array_1.append(chosen_imgs[0])
        array_2.append(chosen_imgs[1])
        array_3.append(chosen_imgs[2])
        array_4.append(chosen_imgs[3])
        array_5.append(chosen_imgs[4])
        array_6.append(chosen_imgs[5])

    a1, a2, a3, a4, a5, a6 = map(
        lambda lst: np.stack(lst, axis=0).astype(np.float32),
        [array_1, array_2, array_3, array_4, array_5, array_6],
    )

    def _safe_global_norm(arr):
        mx = float(np.max(arr)) if arr.size else 0.0
        return arr / mx if mx > 0 else arr

    a1, a2, a3, a4, a5, a6 = map(_safe_global_norm, [a1, a2, a3, a4, a5, a6])

    print(
        "Number of flair images loaded are ",
        len(a1),
        ",",
        len(a2),
        ",",
        len(a3),
        ",",
        len(a4),
        ",",
        len(a5),
        "and",
        len(a6),
    )
    return a1, a2, a3, a4, a5, a6


def load_test_T1W_images(path_test, case_ids=None):
    array = []
    IMG_PX_SIZE = 150

    if case_ids is None:
        path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    else:
        path_cases = [os.path.join(path_test, cid) for cid in case_ids]

    for case_path in path_cases:
        if not os.path.isdir(case_path):
            mri_type = []
        else:
            mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])

        t1_dir = mri_type[1] if len(mri_type) >= 2 else None
        if t1_dir is None or (not os.path.isdir(t1_dir)):
            dcm_paths = []
        else:
            dcm_paths = sorted(
                [
                    f.path
                    for f in os.scandir(t1_dir)
                    if f.is_file() and f.name.lower().endswith(".dcm")
                ]
            )

        img3 = _pick_best_or_middle_slice(dcm_paths, IMG_PX_SIZE, min_sum_thresh=100000)
        array.append(img3)

    arr = np.stack(array, axis=0).astype(np.float32)
    mx = float(np.max(arr)) if arr.size else 0.0
    arr = arr / mx if mx > 0 else arr
    print("Number of T1W images loaded are ", len(arr))
    return arr


def load_test_T1wCE_images(path_test, case_ids=None):
    array = []
    IMG_PX_SIZE = 150

    if case_ids is None:
        path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    else:
        path_cases = [os.path.join(path_test, cid) for cid in case_ids]

    for case_path in path_cases:
        if not os.path.isdir(case_path):
            mri_type = []
        else:
            mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])

        t1ce_dir = mri_type[2] if len(mri_type) >= 3 else None
        if t1ce_dir is None or (not os.path.isdir(t1ce_dir)):
            dcm_paths = []
        else:
            dcm_paths = sorted(
                [
                    f.path
                    for f in os.scandir(t1ce_dir)
                    if f.is_file() and f.name.lower().endswith(".dcm")
                ]
            )

        img3 = _pick_best_or_middle_slice(dcm_paths, IMG_PX_SIZE, min_sum_thresh=100000)
        array.append(img3)

    arr = np.stack(array, axis=0).astype(np.float32)
    mx = float(np.max(arr)) if arr.size else 0.0
    arr = arr / mx if mx > 0 else arr
    print("Number of T1wCE images loaded are ", len(arr))
    return arr


def load_test_T2W_images(path_test, case_ids=None):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    if case_ids is None:
        path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    else:
        path_cases = [os.path.join(path_test, cid) for cid in case_ids]

    for case_path in path_cases:
        if not os.path.isdir(case_path):
            mri_type = []
        else:
            mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])

        t2_dir = mri_type[3] if len(mri_type) >= 4 else None
        if t2_dir is None or (not os.path.isdir(t2_dir)):
            dcm_paths = []
        else:
            dcm_paths = sorted(
                [
                    f.path
                    for f in os.scandir(t2_dir)
                    if f.is_file() and f.name.lower().endswith(".dcm")
                ]
            )

        chosen_imgs = []
        for p in dcm_paths:
            px = _read_dcm_pixel_array(p)
            if px is None:
                continue
            if px.sum() > 100000:
                img3 = _stack_resize_norm(px, IMG_PX_SIZE)
                if img3.sum() > 3000:
                    chosen_imgs.append(img3)
            if len(chosen_imgs) >= 6:
                break

        while len(chosen_imgs) < 6:
            chosen_imgs.append(
                _pick_best_or_middle_slice(dcm_paths, IMG_PX_SIZE, min_sum_thresh=0)
            )

        array_1.append(chosen_imgs[0])
        array_2.append(chosen_imgs[1])
        array_3.append(chosen_imgs[2])
        array_4.append(chosen_imgs[3])
        array_5.append(chosen_imgs[4])
        array_6.append(chosen_imgs[5])

    a1, a2, a3, a4, a5, a6 = map(
        lambda lst: np.stack(lst, axis=0).astype(np.float32),
        [array_1, array_2, array_3, array_4, array_5, array_6],
    )

    def _safe_global_norm(arr):
        mx = float(np.max(arr)) if arr.size else 0.0
        return arr / mx if mx > 0 else arr

    a1, a2, a3, a4, a5, a6 = map(_safe_global_norm, [a1, a2, a3, a4, a5, a6])

    print(
        "Number of T2W images loaded are ",
        len(a1),
        ",",
        len(a2),
        ",",
        len(a3),
        ",",
        len(a4),
        ",",
        len(a5),
        "and",
        len(a6),
    )
    return a1, a2, a3, a4, a5, a6




## === cell 3
test = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
sample_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"

sample_sub = pd.read_csv(sample_path)
sample_ids = sample_sub["BraTS21ID"].astype(str).str.zfill(5).tolist()

case_dir_ids = sorted([f.name for f in os.scandir(test) if f.is_dir()])
if len(case_dir_ids) != len(sample_ids):
    print(
        f"Warning: found {len(case_dir_ids)} test folders, sample_submission has {len(sample_ids)} rows"
    )

missing_folders = [cid for cid in sample_ids if cid not in set(case_dir_ids)]
if missing_folders:
    print(
        "Warning: missing test folders for these IDs (will use zero-image fallbacks):",
        missing_folders[:10],
    )



## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_flair_images(
    test, case_ids=sample_ids
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_T2W_images(
    test, case_ids=sample_ids
)

n_cases = len(sample_ids)
if len(pixels_1) != n_cases or len(pixels_7) != n_cases:
    print(
        f"Warning: expected {n_cases} cases from sample_submission, got pixels_1={len(pixels_1)}, pixels_7={len(pixels_7)}"
    )




## === cell 5
def _predict_or_default(model, x, n_expected, input_size):
    if model is None:
        return np.full((n_expected,), 0.5, dtype=np.float32)
    if x is None or x.ndim != 4 or len(x) != n_expected or x.shape[1] != input_size:
        return np.full((n_expected,), 0.5, dtype=np.float32)
    preds = model.predict(x, verbose=0)
    if preds.ndim == 2 and preds.shape[1] >= 2:
        return preds[:, 1].astype(np.float32)
    return np.full((n_expected,), 0.5, dtype=np.float32)


prediction_1 = _predict_or_default(model_1, pixels_1, n_cases, 299)
prediction_7 = _predict_or_default(model_2, pixels_7, n_cases, 150)




## === cell 6
def create_sub(ids_order, p1, p7):
    cases = [str(x).zfill(5) for x in ids_order]

    p1 = np.asarray(p1, dtype=np.float32).reshape(-1)
    p7 = np.asarray(p7, dtype=np.float32).reshape(-1)
    if len(p1) != len(cases) or len(p7) != len(cases):
        raise ValueError(
            f"Prediction length mismatch: len(cases)={len(cases)}, len(p1)={len(p1)}, len(p7)={len(p7)}"
        )

    prediction = (p1 + p7) / 2.0
    prediction = np.clip(prediction, 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df


sub_df = create_sub(sample_ids, prediction_1, prediction_7)



## === cell 7
if sns is not None:
    try:
        sns.displot(sub_df.MGMT_value)
    except Exception:
        pass

sub_df.head()



## === cell 8
sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
print(
    "MGMT_value stats:",
    float(sub_df.MGMT_value.min()),
    float(sub_df.MGMT_value.max()),
    float(sub_df.MGMT_value.mean()),
)
