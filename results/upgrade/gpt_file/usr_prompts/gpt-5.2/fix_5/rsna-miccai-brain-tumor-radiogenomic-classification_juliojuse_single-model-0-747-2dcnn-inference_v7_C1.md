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

0.61529

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61529) has done: 'I fix the initial crash coming from `pydicom.read_file` by switching to `pydicom.dcmread` (this avoids the protobuf-related `MessageFactory.GetPrototype` issue seen in some Kaggle images). Then I fix the inference pipeline so it actually produces `X_test` and uses the intended 3‑channel tensor (`X_test_`) for prediction (the current code mistakenly predicts on 1‑channel `X_test`). Since the referenced external `.h5` model files are not present, I keep the same “load pretrained models and ensemble” logic but make it robust: it discover any available `.h5` files in `/kaggle/input` and, if none exist, fall back to a minimal CNN trained on the extracted slices just to generate a valid submission. Finally, I correct the submission logic to output probabilities (not `argmax` class labels) aggregated per patient, matching the ROC-AUC metric and the required CSV format.'
- What this solution (achieved 0.61529) has done: 'I fix the crash happening in DICOM loading by avoiding the protobuf-triggering `pydicom.pixel_array` pathway and instead decoding the image using OpenCV from the raw PixelData bytes (this keeps the same downstream 0–255 uint8 behavior and preserves the rest of your pipeline). I also correct the cell numbering to start at 1 so the notebook/script format is valid, while keeping your model/ensemble/fallback logic unchanged. Finally, I add a small safety around the DICOM slice sorting so it doesn’t break if filenames don’t match the expected pattern, preventing silent empty loads and ensuring a submission CSV is always produced.'
- What this solution (achieved 0.61529) has done: 'I fix the DICOM loading crash by switching away from `pydicom` decoding entirely and using OpenCV to read the slice pixels, which avoids the protobuf `MessageFactory.GetPrototype` error you’re hitting. This keeps your downstream pipeline intact (same slice selection, same tensor shapes, same ensemble/fallback logic), but makes image loading robust in this Kaggle environment. I also keep `pydicom` as an optional fallback if OpenCV fails on a file, so no cases silently drop. No score-target calibration changes are needed since your target score is -1.0 and higher is better; we focus on correctness and stable end-to-end submission generation.'
- What this solution (achieved 0.61529) has done: 'I fix the remaining protobuf-triggered crash by making DICOM decoding robust without relying on `pydicom.pixel_array` (which is what hits `MessageFactory.GetPrototype` in this environment), while keeping your downstream slice selection and tensor construction unchanged. Concretely, I prefer OpenCV decoding but add a safe raw PixelData decode path via `pydicom.dcmread(..., force=True)` + manual reshape using DICOM metadata (no `pixel_array`), and only if that fails fall back to `pixel_array` as a last resort. I also keep the ensemble discovery/averaging logic intact and ensure predictions are always made on the intended 3‑channel tensor `X_test_`. These are execution-stability fixes and should be score-neutral to slightly positive by preventing empty/failed loads that would otherwise force 0.5 fill values.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    import pydicom  # noqa: F401

    _HAVE_PYDICOM = True
except Exception:
    _HAVE_PYDICOM = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # unused but kept to preserve original globals
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
train_df = pd.read_csv(f"{DATA_ROOT}/train_labels.csv")
test_df = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)


def _decode_dicom_without_pixel_array(ds):
    """
    BUGFIX: Avoid ds.pixel_array which can crash in some Kaggle images due to protobuf.
    Decode from raw PixelData using Rows/Columns and BitsAllocated when possible.
    """
    if not hasattr(ds, "PixelData"):
        return None
    if not (hasattr(ds, "Rows") and hasattr(ds, "Columns")):
        return None

    rows = int(ds.Rows)
    cols = int(ds.Columns)

    bits_alloc = int(getattr(ds, "BitsAllocated", 16))
    if bits_alloc == 8:
        dtype = np.uint8
    elif bits_alloc == 16:
        pix_repr = int(getattr(ds, "PixelRepresentation", 0))
        dtype = np.int16 if pix_repr == 1 else np.uint16
    else:
        return None

    arr = np.frombuffer(ds.PixelData, dtype=dtype)
    if arr.size < rows * cols:
        return None

    arr = arr[: rows * cols].reshape(rows, cols)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        arr = arr.astype(np.float32) * slope + intercept

    return arr


def load_dicom(path, size=224):
    """
    BUGFIX: Robust DICOM decoding without triggering protobuf errors.
    Priority:
      1) OpenCV DICOM decode (fast path)
      2) pydicom dcmread + manual PixelData decode (no ds.pixel_array)
      3) pydicom ds.pixel_array as last resort (may still fail in some envs)
    Keep the same 0..255 uint8 output and resizing behavior.
    """
    img = None

    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    if img is None and _HAVE_PYDICOM:
        try:
            import pydicom  # local import

            ds = pydicom.dcmread(path, force=True, stop_before_pixels=False)
            img = _decode_dicom_without_pixel_array(ds)
        except Exception:
            img = None

    if img is None and _HAVE_PYDICOM:
        try:
            import pydicom  # local import

            ds = pydicom.dcmread(path, force=True, stop_before_pixels=False)
            img = ds.pixel_array
        except Exception:
            img = None

    if img is None:
        raise RuntimeError(f"Failed to decode DICOM: {path}")

    img = np.asarray(img)

    if img.ndim == 3:
        img = img[..., 0]

    img = img.astype(np.float32)
    mx = float(np.max(img)) if img.size else 0.0
    if mx != 0.0 and np.isfinite(mx):
        img = img / mx
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _safe_sort_key(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    try:
        return int(stem.split("-")[-1])
    except Exception:
        return 10**18


def get_all_image_paths(brats21id, image_type, folder="train"):
    assert image_type in TYPES

    patient_path = os.path.join(
        f"{DATA_ROOT}/{folder}/",
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")), key=_safe_sort_key
    )

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval], dtype=object)


def get_all_images(brats21id, image_type, folder="train", size=225):
    paths = get_all_image_paths(brats21id, image_type, folder)
    return [load_dicom(p, size) for p in paths]


IMAGE_SIZE = 128


def get_all_data_for_train(image_type):
    X, y, train_ids = [], [], []
    for i in range(len(train_df)):
        row = train_df.loc[i]
        images = get_all_images(int(row["BraTS21ID"]), image_type, "train", IMAGE_SIZE)
        label = float(row["MGMT_value"])
        if len(images) == 0:
            continue
        X += images
        y += [label] * len(images)
        train_ids += [int(row["BraTS21ID"])] * len(images)
    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type):
    X, test_ids = [], []
    for i in range(len(test_df)):
        row = test_df.loc[i]
        images = get_all_images(int(row["BraTS21ID"]), image_type, "test", IMAGE_SIZE)
        if len(images) == 0:
            continue
        X += images
        test_ids += [int(row["BraTS21ID"])] * len(images)
    return np.array(X), np.array(test_ids)


X_test, testidt = get_all_data_for_test("T1wCE")
if len(X_test) == 0:
    raise RuntimeError("No test images were loaded. Check dataset paths/structure.")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
X_test = tf.expand_dims(X_test, -1)  # (N, H, W, 1)



## === cell 2
X_test_ = tf.concat([X_test, X_test, X_test], axis=-1)  # (N, H, W, 3)
X_test_.shape



## === cell 3
file_path = "../input/fork-of-rsna-miccai-2dcnn-training/best_model_inception.h5"



## === cell 4
from tensorflow.keras import backend as K


def loss(y_true, y_pred):
    return -(
        1
        - theta(y_true - margin) * theta(y_pred - margin)
        - theta(1 - margin - y_true) * theta(1 - margin - y_pred)
    ) * (y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8))


margin = 0.6
theta = lambda t: (K.sign(t) + 1.0) / 2.0


def mish(inputs):
    x = tf.nn.softplus(inputs)
    x = tf.nn.tanh(x)
    x = tf.multiply(x, inputs)
    return x




## === cell 5
def find_h5_models(search_roots=("../input",), max_models=8):
    h5s = []
    for root in search_roots:
        h5s.extend(glob.glob(os.path.join(root, "**", "*.h5"), recursive=True))
        h5s.extend(glob.glob(os.path.join(root, "**", "*.hdf5"), recursive=True))
    h5s_sorted = sorted(
        h5s, key=lambda p: (("best" not in os.path.basename(p).lower()), p)
    )
    return h5s_sorted[:max_models]


available_models = find_h5_models(search_roots=("../input",), max_models=8)
available_models[:10], len(available_models)



## === cell 6
pred_list = []
loaded_paths = []

custom_objects = {"loss": loss, "leaky_relu": tf.nn.leaky_relu, "mish": mish}

for path in available_models:
    try:
        model_best = tf.keras.models.load_model(
            path, custom_objects=custom_objects, compile=False
        )
        y_pred = model_best.predict(X_test_, verbose=0)
        pred_list.append(y_pred)
        loaded_paths.append(path)
    except Exception:
        continue

loaded_paths, len(pred_list)



## === cell 7
if len(pred_list) >= 1:
    pred_final = np.mean(np.stack(pred_list, axis=0), axis=0)
else:
    X_train, y_train, train_ids = get_all_data_for_train("T1wCE")
    if len(X_train) == 0:
        raise RuntimeError(
            "No training images were loaded; cannot train fallback model."
        )

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_train = tf.expand_dims(X_train, -1)
    X_train_ = tf.concat([X_train, X_train, X_train], axis=-1)

    y_train = tf.convert_to_tensor(y_train.reshape(-1, 1), dtype=tf.float32)

    inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = layers.Rescaling(1.0 / 255.0)(inp)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    fallback_model = keras.Model(inp, out)

    fallback_model.compile(
        optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy"
    )
    fallback_model.fit(X_train_, y_train, batch_size=32, epochs=2, verbose=1)

    pred_final = fallback_model.predict(X_test_, verbose=0)



## === cell 8
sample = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")

y_pred = np.asarray(pred_final).reshape(-1)
result = pd.DataFrame(
    {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred.astype(float)}
)

result2 = result.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

result2["BraTS21ID"] = result2["BraTS21ID"].astype(int)
sample_ids = sample["BraTS21ID"].astype(int)

result2 = (
    sample[["BraTS21ID"]]
    .assign(BraTS21ID=sample_ids)
    .merge(result2, on="BraTS21ID", how="left")
)

result2["MGMT_value"] = result2["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

result2["BraTS21ID"] = result2["BraTS21ID"].apply(lambda x: str(int(x)).zfill(5))

result2.to_csv("submission.csv", index=False)
result2.head()



## === cell 9
assert os.path.exists("submission.csv")
print(result2.shape)
print(result2.columns.tolist())
print(result2["MGMT_value"].describe())
print("Num > 0.9:", int((result2["MGMT_value"] > 0.9).sum()))
print("Saved to submission.csv")
