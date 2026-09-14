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

- What this solution (achieved 0.5) has done: 'I fix the crash in DICOM decoding by removing the TensorFlow DICOM path that triggers the `MessageFactory.GetPrototype` protobuf error and replacing it with a lightweight pydicom-based reader that works in Kaggle’s RSNA environment. Then I fix the Keras `predict()` failure for the fallback (untrained) models by explicitly compiling them so prediction runs reliably, while keeping the same “2-class softmax → take class-1 probability” semantics. Finally, I make submission generation robust by always aligning to `sample_submission.csv`, filling any missing predictions with 0.5, and writing a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing pydicom to use the pure-Python protobuf implementation before any TensorFlow/protobuf code is imported, which is the common Kaggle-side cause of this error. I also make the DICOM reader more robust by explicitly disabling GDCM/pylibjpeg handlers (which can trigger protobuf usage) and falling back cleanly if a slice can’t be decoded. Finally, I keep the exact existing inference/ensembling logic and ensure the pipeline always writes a valid `submission.csv` aligned to `sample_submission.csv` as it already does.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    import pydicom
except Exception as e:
    raise ImportError(
        "pydicom is required to read DICOMs in this notebook environment. "
        "It is expected to be available on Kaggle."
    ) from e


def _read_dicom_pixel_array(fp: str) -> np.ndarray:
    """
    Returns a 2D numpy array (H, W) for a DICOM file using pydicom.
    Robustness: disable optional external pixel-data handlers that can trigger protobuf/C++ issues.
    """
    try:
        pydicom.config.image_handlers = []
    except Exception:
        pass

    ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
    arr = ds.pixel_array  # typically (H, W)
    if arr is None:
        return None
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _safe_norm01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + eps)


def _load_slices_for_sequence(
    case_dir: str, seq_name: str, img_px_size: int = 150, max_slices: int = 6
):
    """
    Loads up to `max_slices` informative slices from a given sequence folder.
    Returns list of (H,W,3) float32 images in [0,1]. Always returns length<=max_slices.
    """
    seq_dir = os.path.join(case_dir, seq_name)
    if not os.path.isdir(seq_dir):
        return []

    files = sorted(
        [
            f.path
            for f in os.scandir(seq_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    out = []
    for fp in files:
        try:
            arr = _read_dicom_pixel_array(fp)
        except Exception:
            continue

        if arr is None:
            continue

        if float(np.sum(arr)) <= 100000:
            continue

        img = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        if np.max(img) <= 0:
            continue

        img = _safe_norm01(img)
        stacked = np.stack([img, img, img], axis=-1)

        if float(np.sum(stacked)) <= 2000:
            continue

        out.append(stacked)
        if len(out) >= max_slices:
            break
    return out


def load_test_images_by_slices(
    path_test: str, img_px_size: int = 150, max_slices: int = 6
):
    """
    Returns 18 arrays:
      T2w:   pixels_1..pixels_6
      FLAIR: pixels_7..pixels_12
      T1w:   pixels_13..pixels_18
    Each pixels_k is a numpy array shape (N,150,150,3).
    """
    if os.path.isdir(os.path.join(path_test, "test")):
        path_test = os.path.join(path_test, "test")

    case_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    t2_lists = [[] for _ in range(max_slices)]
    fl_lists = [[] for _ in range(max_slices)]
    t1_lists = [[] for _ in range(max_slices)]

    for case_dir in case_dirs:
        t2 = _load_slices_for_sequence(case_dir, "T2w", img_px_size, max_slices)
        fl = _load_slices_for_sequence(case_dir, "FLAIR", img_px_size, max_slices)
        t1 = _load_slices_for_sequence(case_dir, "T1w", img_px_size, max_slices)

        def pad_to(seq):
            if len(seq) == 0:
                z = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                return [z for _ in range(max_slices)]
            if len(seq) < max_slices:
                seq = seq + [seq[-1]] * (max_slices - len(seq))
            return seq[:max_slices]

        t2 = pad_to(t2)
        fl = pad_to(fl)
        t1 = pad_to(t1)

        for i in range(max_slices):
            t2_lists[i].append(t2[i])
            fl_lists[i].append(fl[i])
            t1_lists[i].append(t1[i])

    t2_arrays = [np.asarray(lst, dtype=np.float32) for lst in t2_lists]
    fl_arrays = [np.asarray(lst, dtype=np.float32) for lst in fl_lists]
    t1_arrays = [np.asarray(lst, dtype=np.float32) for lst in t1_lists]

    print("Loaded test cases:", len(case_dirs))
    return (*t2_arrays, *fl_arrays, *t1_arrays)




## === cell 2
def build_fallback_model(input_shape=(150, 150, 3)):
    """
    Used only when the original pretrained .h5 files are unavailable.
    Keeps the 'keras model predicting a 2-class probability' semantics.
    """
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
    )
    return model


def safe_load_model(path: str):
    if path and os.path.exists(path):
        m = keras.models.load_model(path, compile=False)
        return m
    return None


model_paths = {
    "model_T2": "../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "model_T2_2": "../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "model_T2_3": "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b500_t1w_6k_0.63auc_imgs.h5",
    "model_T2_5": "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "model_T2_6": "../input/trained-model-for-rsnamiccai/rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
    "model_T2_7": "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5",
}

model_T2 = safe_load_model(model_paths["model_T2"]) or build_fallback_model()
model_T2_2 = safe_load_model(model_paths["model_T2_2"]) or build_fallback_model()
model_T2_3 = safe_load_model(model_paths["model_T2_3"]) or build_fallback_model()
model_T2_5 = safe_load_model(model_paths["model_T2_5"]) or build_fallback_model()
model_T2_6 = safe_load_model(model_paths["model_T2_6"]) or build_fallback_model()
model_T2_7 = safe_load_model(model_paths["model_T2_7"]) or build_fallback_model()




## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"

(
    pixels_1,
    pixels_2,
    pixels_3,
    pixels_4,
    pixels_5,
    pixels_6,
    pixels_7,
    pixels_8,
    pixels_9,
    pixels_10,
    pixels_11,
    pixels_12,
    pixels_13,
    pixels_14,
    pixels_15,
    pixels_16,
    pixels_17,
    pixels_18,
) = load_test_images_by_slices(test)




## === cell 4
def _predict_pos(model, x: np.ndarray, batch_size: int = 16) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    preds = model.predict(x, batch_size=batch_size, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] >= 2:
        return preds[:, 1].astype(np.float32)
    return preds.reshape(-1).astype(np.float32)


prediction_1 = _predict_pos(model_T2, pixels_1)
prediction_2 = _predict_pos(model_T2, pixels_2)
prediction_3 = _predict_pos(model_T2, pixels_3)
prediction_4 = _predict_pos(model_T2, pixels_4)
prediction_5 = _predict_pos(model_T2, pixels_5)
prediction_6 = _predict_pos(model_T2, pixels_6)

prediction_101 = _predict_pos(model_T2_2, pixels_1)
prediction_102 = _predict_pos(model_T2_2, pixels_2)
prediction_103 = _predict_pos(model_T2_2, pixels_3)
prediction_104 = _predict_pos(model_T2_2, pixels_4)
prediction_105 = _predict_pos(model_T2_2, pixels_5)
prediction_106 = _predict_pos(model_T2_2, pixels_6)

prediction_201 = _predict_pos(model_T2_3, pixels_13)
prediction_202 = _predict_pos(model_T2_3, pixels_14)
prediction_203 = _predict_pos(model_T2_3, pixels_15)
prediction_204 = _predict_pos(model_T2_3, pixels_16)
prediction_205 = _predict_pos(model_T2_3, pixels_17)
prediction_206 = _predict_pos(model_T2_3, pixels_18)

prediction_401 = _predict_pos(model_T2_5, pixels_1)
prediction_402 = _predict_pos(model_T2_5, pixels_2)
prediction_403 = _predict_pos(model_T2_5, pixels_3)
prediction_404 = _predict_pos(model_T2_5, pixels_4)
prediction_405 = _predict_pos(model_T2_5, pixels_5)
prediction_406 = _predict_pos(model_T2_5, pixels_6)

prediction_501 = _predict_pos(model_T2_6, pixels_1)
prediction_502 = _predict_pos(model_T2_6, pixels_2)
prediction_503 = _predict_pos(model_T2_6, pixels_3)
prediction_504 = _predict_pos(model_T2_6, pixels_4)
prediction_505 = _predict_pos(model_T2_6, pixels_5)
prediction_506 = _predict_pos(model_T2_6, pixels_6)

prediction_601 = _predict_pos(model_T2_7, pixels_7)
prediction_602 = _predict_pos(model_T2_7, pixels_8)
prediction_603 = _predict_pos(model_T2_7, pixels_9)
prediction_604 = _predict_pos(model_T2_7, pixels_10)
prediction_605 = _predict_pos(model_T2_7, pixels_11)
prediction_606 = _predict_pos(model_T2_7, pixels_12)




## === cell 5
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p401,
    p402,
    p403,
    p404,
    p405,
    p406,
    p501,
    p502,
    p503,
    p504,
    p505,
    p506,
    p601,
    p602,
    p603,
    p604,
    p605,
    p606,
):
    if os.path.isdir(os.path.join(path_test, "test")):
        path_test = os.path.join(path_test, "test")

    case_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    case_ids = [os.path.basename(p) for p in case_dirs]  # strings like '00002'

    preds = (
        p1.astype(np.float32)
        + p2.astype(np.float32)
        + p3.astype(np.float32)
        + p4.astype(np.float32)
        + p5.astype(np.float32)
        + p6.astype(np.float32)
        + p101.astype(np.float32)
        + p102.astype(np.float32)
        + p103.astype(np.float32)
        + p104.astype(np.float32)
        + p105.astype(np.float32)
        + p106.astype(np.float32)
        + p201.astype(np.float32)
        + p202.astype(np.float32)
        + p203.astype(np.float32)
        + p204.astype(np.float32)
        + p205.astype(np.float32)
        + p206.astype(np.float32)
        + p401.astype(np.float32)
        + p402.astype(np.float32)
        + p403.astype(np.float32)
        + p404.astype(np.float32)
        + p405.astype(np.float32)
        + p406.astype(np.float32)
        + p501.astype(np.float32)
        + p502.astype(np.float32)
        + p503.astype(np.float32)
        + p504.astype(np.float32)
        + p505.astype(np.float32)
        + p506.astype(np.float32)
        + p601.astype(np.float32)
        + p602.astype(np.float32)
        + p603.astype(np.float32)
        + p604.astype(np.float32)
        + p605.astype(np.float32)
        + p606.astype(np.float32)
    ) / 36.0

    preds = np.clip(preds, 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": case_ids, "MGMT_value": preds.astype(np.float32)})
    df = df.sort_values("BraTS21ID").reset_index(drop=True)
    return df


sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
    prediction_501,
    prediction_502,
    prediction_503,
    prediction_504,
    prediction_505,
    prediction_506,
    prediction_601,
    prediction_602,
    prediction_603,
    prediction_604,
    prediction_605,
    prediction_606,
)
print(sub_df.head())
print("sub_df shape:", sub_df.shape)




## === cell 6
sample_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_path)

sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

out = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
out["MGMT_value"] = out["MGMT_value"].astype(np.float32).fillna(0.5).clip(0.0, 1.0)

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
