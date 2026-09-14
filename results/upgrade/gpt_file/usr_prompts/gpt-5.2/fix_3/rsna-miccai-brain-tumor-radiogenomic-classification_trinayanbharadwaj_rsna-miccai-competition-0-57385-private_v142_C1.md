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

0.56353

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.56353) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation early (a common Kaggle TF/protobuf incompatibility), which unblocks the whole pipeline. Then I fix `create_sub()` so it only collects numeric case folders from the test directory and uses the already-loaded `kept_test_ids` order, preventing the `'test'` folder parsing error and any ID/prediction misalignment. Finally, I keep the model/training and slice-averaging logic unchanged, and ensure a valid `submission.csv` with the exact required columns is always written to the working directory.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Missing LABELS_CSV: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing SAMPLE_SUB: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
sample_sub_df = pd.read_csv(SAMPLE_SUB)

print("Train labels:", labels_df.shape, labels_df.columns.tolist())
print("Sample submission:", sample_sub_df.shape, sample_sub_df.columns.tolist())



## === cell 2
BAD_CASES = {"00109", "00123", "00709"}


def _safe_dcm_pixel_array(dcm_path: str):
    """Read DICOM pixel array safely; return None on failure."""
    try:
        dcm = dicom.dcmread(dcm_path)
        arr = dcm.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def load_case_slices_t2(
    case_dir: str,
    img_px_size: int = 150,
    max_slices: int = 6,
    pixel_sum_thresh: float = 100000.0,
    norm_sum_thresh: float = 2000.0,
):
    """
    Core logic preserved:
    - iterate T2w DICOMs
    - keep slices with sufficient signal
    - resize to (img_px_size, img_px_size)
    - stack to 3 channels and normalize
    - return up to max_slices slices; if fewer found, pad by repeating last
    """
    t2_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2_dir):
        return None  # no data

    dcm_files = sorted(
        [
            os.path.join(t2_dir, f)
            for f in os.listdir(t2_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    if len(dcm_files) == 0:
        return None

    slices = []
    for fp in dcm_files:
        arr = _safe_dcm_pixel_array(fp)
        if arr is None:
            continue
        if float(arr.sum()) <= pixel_sum_thresh:
            continue

        resized_img = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        stacked = np.stack((resized_img,) * 3, axis=-1)

        mx = float(np.max(stacked))
        if mx <= 0:
            continue
        stacked_norm = stacked / mx

        if float(stacked_norm.sum()) <= norm_sum_thresh:
            continue

        slices.append(stacked_norm)
        if len(slices) >= max_slices:
            break

    if len(slices) == 0:
        mid = dcm_files[len(dcm_files) // 2]
        arr = _safe_dcm_pixel_array(mid)
        if arr is None:
            return None
        resized_img = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        stacked = np.stack((resized_img,) * 3, axis=-1)
        mx = float(np.max(stacked))
        stacked_norm = stacked / mx if mx > 0 else stacked
        slices = [stacked_norm]

    while len(slices) < max_slices:
        slices.append(slices[-1].copy())

    return np.stack(slices[:max_slices], axis=0)  # (max_slices, H, W, 3)




## === cell 3
def build_dataset_from_dir(
    root_dir: str,
    ids: list,
    y_map: dict = None,
    img_px_size: int = 150,
    max_slices: int = 6,
):
    """
    Build X as (N*max_slices, H, W, 3), and y as (N*max_slices,) repeating per-slice label.
    Preserves predicting per-slice then averaging per case.
    """
    X_list = []
    y_list = []
    kept_ids = []

    for id_str in ids:
        case_dir = os.path.join(root_dir, id_str)
        if not os.path.isdir(case_dir):
            continue

        case_slices = load_case_slices_t2(
            case_dir, img_px_size=img_px_size, max_slices=max_slices
        )
        if case_slices is None:
            continue

        X_list.append(case_slices)
        kept_ids.append(id_str)

        if y_map is not None:
            y_val = float(y_map[int(id_str)])
            y_list.append(np.full((max_slices,), y_val, dtype=np.float32))

    if len(X_list) == 0:
        raise RuntimeError(f"No cases loaded from {root_dir} for provided ids.")

    X = np.concatenate(X_list, axis=0).astype(np.float32)  # (N*max_slices, H, W, 3)
    if y_map is not None:
        y = np.concatenate(y_list, axis=0).astype(np.float32)
        return X, y, kept_ids
    return X, kept_ids


train_ids_all = sorted(
    [d.name for d in os.scandir(TRAIN_DIR) if d.is_dir() and d.name.isdigit()]
)
train_ids_all = [x for x in train_ids_all if x not in BAD_CASES]

test_ids_all = sorted(
    [d.name for d in os.scandir(TEST_DIR) if d.is_dir() and d.name.isdigit()]
)

print("Train cases (after exclusions):", len(train_ids_all))
print("Test cases:", len(test_ids_all))

y_map = dict(
    zip(
        labels_df["BraTS21ID"].astype(int).tolist(),
        labels_df["MGMT_value"].astype(float).tolist(),
    )
)




## === cell 4
def make_model(input_shape=(150, 150, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


IMG_PX_SIZE = 150
MAX_SLICES = 6

X_train, y_train, kept_train_ids = build_dataset_from_dir(
    TRAIN_DIR,
    train_ids_all,
    y_map=y_map,
    img_px_size=IMG_PX_SIZE,
    max_slices=MAX_SLICES,
)
print("X_train:", X_train.shape, "y_train:", y_train.shape)

model_T2 = make_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))

BATCH_SIZE = 32
EPOCHS = 3

history = model_T2.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
)

model_T2_2 = model_T2
model_T2_3 = model_T2
model_T2_4 = model_T2



## === cell 5
X_test_slices, kept_test_ids = build_dataset_from_dir(
    TEST_DIR, test_ids_all, y_map=None, img_px_size=IMG_PX_SIZE, max_slices=MAX_SLICES
)
print("X_test_slices:", X_test_slices.shape, "kept_test_ids:", len(kept_test_ids))

N_test = len(kept_test_ids)
X_test_5d = X_test_slices.reshape(N_test, MAX_SLICES, IMG_PX_SIZE, IMG_PX_SIZE, 3)
pixels_1 = X_test_5d[:, 0]
pixels_2 = X_test_5d[:, 1]
pixels_3 = X_test_5d[:, 2]
pixels_4 = X_test_5d[:, 3]
pixels_5 = X_test_5d[:, 4]
pixels_6 = X_test_5d[:, 5]

print("Per-slice batches:", pixels_1.shape, pixels_6.shape)



## === cell 6
preds_1 = model_T2.predict(pixels_1, verbose=0).reshape(-1)
prediction_1 = preds_1
preds_2 = model_T2.predict(pixels_2, verbose=0).reshape(-1)
prediction_2 = preds_2
preds_3 = model_T2.predict(pixels_3, verbose=0).reshape(-1)
prediction_3 = preds_3
preds_4 = model_T2.predict(pixels_4, verbose=0).reshape(-1)
prediction_4 = preds_4
preds_5 = model_T2.predict(pixels_5, verbose=0).reshape(-1)
prediction_5 = preds_5
preds_6 = model_T2.predict(pixels_6, verbose=0).reshape(-1)
prediction_6 = preds_6

preds_101 = model_T2_2.predict(pixels_1, verbose=0).reshape(-1)
prediction_101 = preds_101
preds_102 = model_T2_2.predict(pixels_2, verbose=0).reshape(-1)
prediction_102 = preds_102
preds_103 = model_T2_2.predict(pixels_3, verbose=0).reshape(-1)
prediction_103 = preds_103
preds_104 = model_T2_2.predict(pixels_4, verbose=0).reshape(-1)
prediction_104 = preds_104
preds_105 = model_T2_2.predict(pixels_5, verbose=0).reshape(-1)
prediction_105 = preds_105
preds_106 = model_T2_2.predict(pixels_6, verbose=0).reshape(-1)
prediction_106 = preds_106

preds_201 = model_T2_3.predict(pixels_1, verbose=0).reshape(-1)
prediction_201 = preds_201
preds_202 = model_T2_3.predict(pixels_2, verbose=0).reshape(-1)
prediction_202 = preds_202
preds_203 = model_T2_3.predict(pixels_3, verbose=0).reshape(-1)
prediction_203 = preds_203
preds_204 = model_T2_3.predict(pixels_4, verbose=0).reshape(-1)
prediction_204 = preds_204
preds_205 = model_T2_3.predict(pixels_5, verbose=0).reshape(-1)
prediction_205 = preds_205
preds_206 = model_T2_3.predict(pixels_6, verbose=0).reshape(-1)
prediction_206 = preds_206

preds_301 = model_T2_4.predict(pixels_1, verbose=0).reshape(-1)
prediction_301 = preds_301
preds_302 = model_T2_4.predict(pixels_2, verbose=0).reshape(-1)
prediction_302 = preds_302
preds_303 = model_T2_4.predict(pixels_3, verbose=0).reshape(-1)
prediction_303 = preds_303
preds_304 = model_T2_4.predict(pixels_4, verbose=0).reshape(-1)
prediction_304 = preds_304
preds_305 = model_T2_4.predict(pixels_5, verbose=0).reshape(-1)
prediction_305 = preds_305
preds_306 = model_T2_4.predict(pixels_6, verbose=0).reshape(-1)
prediction_306 = preds_306

print("Pred arrays length:", len(prediction_1), len(prediction_6))




## === cell 7
def create_sub_from_ids(
    ids,
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
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
):
    cases = [int(x) for x in ids]
    n = len(cases)

    preds = [
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
        p301,
        p302,
        p303,
        p304,
        p305,
        p306,
    ]
    for i, p in enumerate(preds):
        if len(p) != n:
            raise ValueError(f"Prediction vector {i} length {len(p)} != num cases {n}")

    prediction = np.mean(np.vstack([p.astype(np.float32) for p in preds]), axis=0)
    prediction = np.clip(prediction, 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction.astype(np.float32)})
    df = df.sort_values("BraTS21ID").reset_index(drop=True)
    return df


sub_df = create_sub_from_ids(
    kept_test_ids,
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
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
)

print(sub_df.head())
print(sub_df.shape)

sub_df = sample_sub_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(np.float32)

print("Final submission shape:", sub_df.shape)
print("Any NA:", sub_df["MGMT_value"].isna().any())



## === cell 8
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_df.head())
