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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the DICOM-loading crash by avoiding the protobuf `MessageFactory.GetPrototype` path (which is triggered by `pydicom.pixel_array` in this environment) and instead reading pixel data via OpenCV from each DICOM file, which is stable and keeps the same overall “2D slice → CNN prediction → patient mean” logic. I also fix the missing `K` import in the custom loss and make model loading robust: only load `.h5` files that actually exist in the Kaggle input, otherwise fall back to a safe constant prediction so a valid `submission.csv` is always produced. Finally, I correct the inference/post-processing to output probabilities (AUC metric) rather than `argmax` class indices, and ensure the submission rows align exactly with `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the DICOM-loading crash by removing reliance on `pydicom.pixel_array` (which triggers the protobuf `MessageFactory.GetPrototype` error here) and instead decoding DICOM pixel data using OpenCV from the raw `PixelData`, keeping the same “slice → CNN prediction → patient mean” core logic. I also make the notebook robust to earlier-cell failures by consolidating the path constants and data loading so later cells always have `SAMPLE_SUB_PATH`, `X_test`, and `testidt` defined. Finally, I keep the ensemble/model-loading logic intact but ensure inference always uses the correct 3-channel input and always writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the crash happening in the first cell by ensuring we never trigger the problematic protobuf/pydicom pixel decoding path during imports/reads, while keeping your existing “OpenCV-based PixelData decode” logic intact. I also make DICOM file discovery more robust (so it won’t error on unexpected filenames) and ensure the pipeline always produces a correctly aligned `submission.csv` even if some scans yield zero slices. These changes are execution/stability fixes and should be score-neutral to slightly positive (more consistent slice loading → less missing/constant-filled patients). The model ensemble and probability post-processing are preserved.'
- What this solution (achieved 0.5) has done: 'I fix the crash in the first cell by preventing `pydicom` (and its pixel handler imports) from being imported at module import time, since that triggers the protobuf `MessageFactory.GetPrototype` error in this environment. Then I make `load_dicom()` robust by doing a lazy `pydicom` import inside the function and falling back cleanly to OpenCV decoding if `pydicom` can’t be used, keeping your existing “slice → CNN → patient mean” logic unchanged. Finally, I keep the ensemble/prediction logic intact but ensure the script always writes a valid `submission.csv` aligned to `sample_submission.csv` even if no images/models are found.'
- What this solution (achieved 0.5) has done: 'I fix the crash happening at import time by removing the global `cv2`/`tqdm.notebook`/TensorFlow imports from the first cell and switching to safe, lazy imports only after the environment is initialized (this avoids the protobuf `MessageFactory.GetPrototype` path being triggered prematurely). I also make the DICOM loading fully independent of `pydicom.pixel_data_handlers` by decoding `PixelData` directly to a NumPy array (keeping your same “slice → 2D CNN → patient mean” pipeline intact). Finally, I keep your ensemble and probability post-processing unchanged, but add small robustness checks so the script always writes a valid `submission.csv` aligned to `sample_submission.csv` even if some slices/patients can’t be decoded.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # unused but kept to preserve original intent
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_LABELS_PATH)
test_df = pd.read_csv(SAMPLE_SUB_PATH)

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(int)

train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)



## === cell 1
import cv2
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

tf.random.set_seed(SEED)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def _dicom_to_uint8_from_dataset(ds):
    """
    Bugfix: do not call pydicom pixel handlers / apply_voi_lut, which can trigger the protobuf crash.
    Instead, decode PixelData directly using metadata and then min-max scale to uint8.
    """
    try:
        rows = int(getattr(ds, "Rows", 0))
        cols = int(getattr(ds, "Columns", 0))
        if rows <= 0 or cols <= 0:
            return None

        samples = int(getattr(ds, "SamplesPerPixel", 1))
        if samples != 1:
            return None

        bits = int(getattr(ds, "BitsAllocated", 16))
        photo = str(getattr(ds, "PhotometricInterpretation", "")).upper()

        raw = getattr(ds, "PixelData", None)
        if raw is None:
            return None

        pixel_repr = int(getattr(ds, "PixelRepresentation", 0))

        if bits == 8:
            dtype = np.int8 if pixel_repr == 1 else np.uint8
        else:
            dtype = np.int16 if pixel_repr == 1 else np.uint16

        arr = np.frombuffer(raw, dtype=dtype)
        if arr.size != rows * cols:
            return None

        arr = arr.reshape(rows, cols).astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        if photo == "MONOCHROME1":
            arr = np.max(arr) - arr

        mn, mx = float(np.min(arr)), float(np.max(arr))
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = arr * 0.0

        return (arr * 255.0).astype(np.uint8)
    except Exception:
        return None


def load_dicom(path, size=224):
    """
    Keep same intent: load a DICOM slice -> resize -> uint8.
    Bugfix: lazy import pydicom; no pixel_array access; robust fallback.
    """
    try:
        import pydicom  # lazy import

        ds = pydicom.dcmread(
            path, force=True, stop_before_pixels=False, specific_tags=None
        )
        img = _dicom_to_uint8_from_dataset(ds)
        if img is not None:
            return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        pass

    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((size, size), dtype=np.uint8)
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        return np.zeros((size, size), dtype=np.uint8)


def _slice_sort_key(p):
    b = os.path.basename(p)
    stem = os.path.splitext(b)[0]
    try:
        return int(stem.split("-")[-1])
    except Exception:
        return 10**9


def get_all_image_paths(brats21id, image_type, folder="train"):
    assert image_type in TYPES
    patient_path = os.path.join(DATA_ROOT, f"{folder}", str(int(brats21id)).zfill(5))
    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=_slice_sort_key,
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval])


def get_all_images(brats21id, image_type, folder="train", size=224):
    paths = get_all_image_paths(brats21id, image_type, folder)
    return [load_dicom(path, size) for path in paths]


IMAGE_SIZE = 128


def get_all_data_for_test(image_type):
    X = []
    test_ids = []
    for i in tqdm(test_df.index, total=len(test_df)):
        brats_id = int(test_df.loc[i, "BraTS21ID"])
        images = get_all_images(brats_id, image_type, "test", IMAGE_SIZE)
        if len(images) > 0:
            X += images
            test_ids += [brats_id] * len(images)
    return np.array(X), np.array(test_ids)


X_test, testidt = get_all_data_for_test("T1wCE")

if X_test.size == 0:
    X_test = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    testidt = np.array([int(test_df.loc[0, "BraTS21ID"])])



## === cell 3
X_test = tf.expand_dims(X_test, -1)
X_test_ = tf.concat([X_test, X_test, X_test], axis=-1)
X_test_.shape



## === cell 4
file_path = "../input/fork-of-rsna-miccai-2dcnn-training/best_model_inception.h5"



## === cell 5
margin = 0.6
theta = lambda t: (K.sign(t) + 1.0) / 2.0


def loss(y_true, y_pred):
    return -(
        1
        - theta(y_true - margin) * theta(y_pred - margin)
        - theta(1 - margin - y_true) * theta(1 - margin - y_pred)
    ) * (y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8))


def mish(inputs):
    x = tf.nn.softplus(inputs)
    x = tf.nn.tanh(x)
    x = tf.multiply(x, inputs)
    return x




## === cell 6
glob.glob("../input/cnndd2/*")



## === cell 7
candidate_paths = [
    "../input/cnndd2/best_model0.747.h5",
    "../input/cnndd2/best_model0.741.h5",
    "../input/cnndd2/best_model0.739.h5",
    "../input/cnndd2/best_model0.725.h5",
    "../input/cnndd2/best_model0.719.h5",
    "../input/cnndd2/best_model0.717.h5",
    "../input/cnndd2/best_model0.707.h5",
    "../input/cnndd2/best_model0.703.h5",
    file_path,  # optional single model path
]
model_path = [p for p in candidate_paths if os.path.exists(p)]

pred_list = []
for path in model_path:
    model_best = tf.keras.models.load_model(
        filepath=path,
        custom_objects={"loss": loss, "leaky_relu": tf.nn.leaky_relu, "mish": mish},
        compile=False,
    )
    try:
        y_pred = model_best.predict(X_test_, verbose=0)
    except Exception:
        y_pred = model_best.predict(X_test, verbose=0)
    pred_list.append(y_pred)

len(pred_list), model_path[:3]



## === cell 8
if len(pred_list) >= 8:
    pred_final = (
        pred_list[0] + pred_list[1] + pred_list[2] + pred_list[3]
    ) / 4 * 0.9 + (pred_list[4] + pred_list[5] + pred_list[6] + pred_list[7]) / 4 * 0.1
elif len(pred_list) > 0:
    pred_final = sum(pred_list) / len(pred_list)
else:
    pred_final = None  # handled in next cell



## === cell 9
sample = pd.read_csv(SAMPLE_SUB_PATH)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

if pred_final is None:
    result2 = sample.copy()
    result2["MGMT_value"] = 0.5
else:
    y_pred = pred_final

    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    elif y_pred.ndim == 2 and y_pred.shape[1] == 2:
        y_pred = y_pred[:, 1]
    elif y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)

    if len(y_pred) != len(testidt):
        result2 = sample.copy()
        result2["MGMT_value"] = 0.5
    else:
        result = pd.DataFrame(
            {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred.astype(float)}
        )
        result2 = result.groupby("BraTS21ID", as_index=False).mean()

        result2 = sample[["BraTS21ID"]].merge(result2, on="BraTS21ID", how="left")
        result2["MGMT_value"] = result2["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

result2.to_csv("submission.csv", index=False)
result2.head()



## === cell 10
int((result2["MGMT_value"] > 0.9).sum())
