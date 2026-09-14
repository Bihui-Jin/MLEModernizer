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

0.44

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.44) has done: 'I fix the immediate runtime crash in DICOM decoding by removing the TensorFlow DICOM dependency (which is broken in this environment) and switching to a simple, robust DICOM pixel extractor using `pydicom` (available on Kaggle for this competition). Then I fix the submission ID extraction bug by listing only numeric case directories and formatting IDs as 5-digit strings to match the sample submission, which also avoids the `int('test')` error. Finally, I keep the model/training logic the same but make the test-case ordering explicitly match `sample_submission.csv` so predictions align correctly and a valid `submission.csv` is always produced.'
- What this solution (achieved 0.44) has done: 'I fix the runtime crash happening before any training by addressing an incompatibility between `pydicom` and the installed `protobuf` version (the `MessageFactory.GetPrototype` AttributeError). The safest minimal fix in Kaggle is to force protobuf to use the pure-Python implementation via an environment variable set before importing `pydicom`/TensorFlow. I also add a small fallback to `pydicom.pixels`/`pixel_array` handling so DICOM decoding is more robust without changing the model/training logic. No score-tuning changes are needed beyond making the pipeline run again, since your current score (0.44) is already well above the (odd) target of -1.0.'
- What this solution (achieved 0.44) has done: 'We fix the runtime crash caused by an incompatibility between `pydicom` and the default `protobuf` C-implementation by forcing protobuf to use the pure-Python implementation *before* any protobuf-dependent imports and by also setting the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` safeguard. Then we make DICOM decoding more robust by using pydicom’s `pixel_array` when available and falling back to `pydicom.pixel_data_handlers.util.apply_voi_lut` when needed, without changing your model/training loop or slice-selection logic. Finally, we keep the submission ordering locked to `sample_submission.csv` and ensure the output is always a valid `submission.csv`.'
- What this solution (achieved 0.44) has done: 'I fix the immediate runtime crash by setting the protobuf implementation environment variables before *any* library imports (including TensorFlow), and by adding a safe fallback that makes DICOM decoding work even when `pydicom.pixel_array` needs the VOI LUT path. I also correct a small logic bug in the DICOM reader where the VOI LUT branch was unreachable due to re-calling `ds.pixel_array` in the exception path. These changes are execution/stability fixes and should not materially change the model core logic or training semantics, while ensuring the notebook runs end-to-end and always writes a valid `submission.csv` in the required format/order.'
- What this solution (achieved 0.44) has done: 'I fix the crash in cell 0 caused by the `protobuf`/`pydicom` incompatibility by forcing the pure-Python protobuf implementation *and* disabling C++ descriptors before any protobuf-dependent imports. I also fix a logic bug in the DICOM reader where the VOI LUT fallback was unreachable due to re-calling `ds.pixel_array` inside the exception path, making pixel extraction robust again without changing the downstream model/training logic. Finally, I keep the existing sample-submission ordering and CSV writing intact so the pipeline always completes and writes a valid `submission.csv`. No score-targeting changes are needed because the target score (-1.0) is already far below the current score (0.44), so we focus on stability and correctness only.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_C_DESCRIPTORS", "1")

import random
import numpy as np
import pandas as pd

from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception as e:
    raise RuntimeError(
        "pydicom is required to read DICOMs in this environment but could not be imported."
    ) from e


def read_dicom_pixel_array(path: str):
    """
    Returns a 2D float32 numpy array (H, W) from a DICOM file.
    Robust to common DICOM variations; returns None on failure.
    Core behavior preserved (still returns pixel data as float32).
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)

        try:
            arr = ds.pixel_array
        except Exception:
            return None

        try:
            arr2 = apply_voi_lut(arr, ds)
            if arr2 is not None:
                arr = arr2
        except Exception:
            pass

        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[0]
        arr = arr.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        return arr
    except Exception:
        return None




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

train_labels = pd.read_csv(TRAIN_LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

BAD_CASES = {109, 123, 709}
train_labels = train_labels[~train_labels["BraTS21ID"].isin(BAD_CASES)].reset_index(
    drop=True
)

print("Train labels:", train_labels.shape)
print("Sample submission:", sample_sub.shape)




## === cell 2
def _get_case_dir(base_dir: str, brats_id: int) -> str:
    return os.path.join(base_dir, f"{brats_id:05d}")


def _sorted_dcm_paths(series_dir: str):
    if not os.path.isdir(series_dir):
        return []
    paths = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.lower().endswith(".dcm")
    ]
    paths.sort()
    return paths


def load_case_t2_slices(case_dir: str, img_px_size: int = 150, n_slices: int = 6):
    """
    Core logic preserved: read T2W DICOMs, filter by pixel sum thresholds, resize,
    stack to 3 channels, normalize. Returns exactly n_slices images (pads if needed).
    """
    t2_dir = os.path.join(case_dir, "T2w")
    dcm_paths = _sorted_dcm_paths(t2_dir)

    collected = []
    for p in dcm_paths:
        arr = read_dicom_pixel_array(p)
        if arr is None:
            continue

        if arr.sum() > 100000:
            resized_img = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked_img = np.stack((resized_img,) * 3, axis=-1)

            mx = np.max(stacked_img)
            if mx > 0:
                stacked_img_normalize = stacked_img / mx
            else:
                stacked_img_normalize = stacked_img

            if stacked_img_normalize.sum() > 2500:
                collected.append(stacked_img_normalize)

        if len(collected) >= n_slices:
            break

    if len(collected) == 0:
        collected = [
            np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for _ in range(n_slices)
        ]
    elif len(collected) < n_slices:
        last = collected[-1]
        collected.extend([last] * (n_slices - len(collected)))

    return np.stack(collected, axis=0).astype(np.float32)  # (n_slices, H, W, 3)




## === cell 3
def load_test_T2W_images(path_test, img_px_size: int = 150, ids_in_order=None):
    """
    Fix: allow enforcing the exact test ID order (sample_submission order) so predictions align.
    """
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []

    if ids_in_order is None:
        path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    else:
        path_cases = [os.path.join(path_test, str(i).zfill(5)) for i in ids_in_order]

    for case_path in path_cases:
        slices = load_case_t2_slices(case_path, img_px_size=img_px_size, n_slices=6)
        array_1.append(slices[0])
        array_2.append(slices[1])
        array_3.append(slices[2])
        array_4.append(slices[3])
        array_5.append(slices[4])
        array_6.append(slices[5])

    array_1 = np.asarray(array_1, dtype=np.float32)
    array_2 = np.asarray(array_2, dtype=np.float32)
    array_3 = np.asarray(array_3, dtype=np.float32)
    array_4 = np.asarray(array_4, dtype=np.float32)
    array_5 = np.asarray(array_5, dtype=np.float32)
    array_6 = np.asarray(array_6, dtype=np.float32)

    def _safe_norm(x):
        mx = np.max(x) if x.size else 1.0
        return x / mx if mx > 0 else x

    array_1 = _safe_norm(array_1)
    array_2 = _safe_norm(array_2)
    array_3 = _safe_norm(array_3)
    array_4 = _safe_norm(array_4)
    array_5 = _safe_norm(array_5)
    array_6 = _safe_norm(array_6)

    print(
        "Number of T2 images loaded are ",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        ",",
        len(array_6),
    )
    return array_1, array_2, array_3, array_4, array_5, array_6




## === cell 4
def build_model(input_shape=(150, 150, 3)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


model_T2 = build_model()



## === cell 5
IMG_PX_SIZE = 150
N_SLICES = 6

X_list = []
y_list = []

MAX_TRAIN_SUBJECTS = min(220, len(train_labels))

for brats_id, mgmt in (
    train_labels[["BraTS21ID", "MGMT_value"]]
    .iloc[:MAX_TRAIN_SUBJECTS]
    .itertuples(index=False)
):
    case_dir = _get_case_dir(TRAIN_DIR, int(brats_id))
    if not os.path.isdir(case_dir):
        continue
    slices = load_case_t2_slices(case_dir, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES)
    X_list.append(slices)
    y_list.append(np.full((N_SLICES,), float(mgmt), dtype=np.float32))

X = (
    np.concatenate(X_list, axis=0)
    if X_list
    else np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
)
y = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), dtype=np.float32)

print(
    "Training slice dataset:",
    X.shape,
    y.shape,
    "Pos rate:",
    float(y.mean()) if y.size else None,
)



## === cell 6
if X.shape[0] > 0:
    model_T2.fit(X, y, batch_size=16, epochs=2, verbose=1)
else:
    print(
        "Warning: No training data found; model will be untrained and likely produce near-constant outputs."
    )



## === cell 7
test = TEST_DIR
test_ids = sample_sub["BraTS21ID"].astype(int).tolist()

pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    test, img_px_size=IMG_PX_SIZE, ids_in_order=test_ids
)



## === cell 8
preds_1 = model_T2.predict(pixels_1, verbose=0).reshape(-1)
prediction_1 = preds_1.astype(np.float32)

preds_2 = model_T2.predict(pixels_2, verbose=0).reshape(-1)
prediction_2 = preds_2.astype(np.float32)

preds_3 = model_T2.predict(pixels_3, verbose=0).reshape(-1)
prediction_3 = preds_3.astype(np.float32)

preds_4 = model_T2.predict(pixels_4, verbose=0).reshape(-1)
prediction_4 = preds_4.astype(np.float32)

preds_5 = model_T2.predict(pixels_5, verbose=0).reshape(-1)
prediction_5 = preds_5.astype(np.float32)

preds_6 = model_T2.predict(pixels_6, verbose=0).reshape(-1)
prediction_6 = preds_6.astype(np.float32)




## === cell 9
def create_sub(ids, p1, p2, p3, p4, p5, p6):
    """
    Use the provided ordered test IDs (from sample_submission) and ensure correct formatting.
    """
    ids = [int(i) for i in ids]
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    p3 = np.asarray(p3, dtype=np.float32)
    p4 = np.asarray(p4, dtype=np.float32)
    p5 = np.asarray(p5, dtype=np.float32)
    p6 = np.asarray(p6, dtype=np.float32)

    n = len(ids)
    for arr in (p1, p2, p3, p4, p5, p6):
        if len(arr) != n:
            raise ValueError(
                f"Prediction length {len(arr)} does not match number of cases {n}"
            )

    prediction = (p1 + p2 + p3 + p4 + p5 + p6) / 6.0

    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": prediction})
    df["BraTS21ID"] = df["BraTS21ID"].map(lambda x: f"{int(x):05d}")
    return df


sub_df = create_sub(
    test_ids,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
)

sub_out = sample_sub.copy()
sub_out["BraTS21ID"] = sub_out["BraTS21ID"].map(lambda x: f"{int(x):05d}")
sub_out = sub_out[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_out["MGMT_value"] = sub_out["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

print(sub_out.head())
print("Submission shape:", sub_out.shape)

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
