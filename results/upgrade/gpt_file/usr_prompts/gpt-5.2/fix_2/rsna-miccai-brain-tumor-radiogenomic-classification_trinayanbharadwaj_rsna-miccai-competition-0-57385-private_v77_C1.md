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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



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

assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Missing: {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)




## === cell 2
def _safe_norm(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d.astype(np.float32)
    vmin = np.percentile(img2d, 1)
    vmax = np.percentile(img2d, 99)
    img2d = np.clip(img2d, vmin, vmax)
    denom = (vmax - vmin) if (vmax - vmin) > 1e-6 else 1.0
    img2d = (img2d - vmin) / denom
    return img2d


def _read_dicom_pixel_array(dcm_path: str) -> np.ndarray:
    dcm = dicom.dcmread(dcm_path)
    arr = dcm.pixel_array
    return arr


def _get_case_modality_dir(case_dir: str, modality: str = "T2w") -> str:
    mod_dir = os.path.join(case_dir, modality)
    if not os.path.isdir(mod_dir):
        raise FileNotFoundError(f"Missing modality folder: {mod_dir}")
    return mod_dir


def _select_evenly_spaced(items, k):
    n = len(items)
    if n == 0:
        return []
    if n >= k:
        idx = np.linspace(0, n - 1, k).round().astype(int)
        return [items[i] for i in idx]
    out = list(items)
    while len(out) < k:
        out.append(items[-1])
    return out


def load_case_t2w_slices(
    case_dir: str, img_px_size: int = 150, n_slices: int = 8
) -> np.ndarray:
    """
    Returns (n_slices, img_px_size, img_px_size, 3) float32 in [0,1].
    """
    t2_dir = _get_case_modality_dir(case_dir, "T2w")
    dcm_files = sorted(
        [
            os.path.join(t2_dir, f)
            for f in os.listdir(t2_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    chosen = _select_evenly_spaced(dcm_files, n_slices)

    out = np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    for i, p in enumerate(chosen):
        arr = _read_dicom_pixel_array(p)
        arr = _safe_norm(arr)
        arr = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        arr3 = np.stack([arr, arr, arr], axis=-1)
        out[i] = arr3
    return out


def load_dataset_slices(
    root_dir: str, ids: list, img_px_size: int = 150, n_slices: int = 8
) -> np.ndarray:
    X = np.zeros((len(ids) * n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    j = 0
    for brats_id in ids:
        case_dir = os.path.join(root_dir, brats_id)
        case_slices = load_case_t2w_slices(
            case_dir, img_px_size=img_px_size, n_slices=n_slices
        )
        X[j : j + n_slices] = case_slices
        j += n_slices
    return X




## === cell 3
BAD_CASES = {"00109", "00123", "00709"}

train_ids_all = sorted(
    [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
)
train_ids_all = [i for i in train_ids_all if i not in BAD_CASES]

train_labels_map = dict(
    zip(labels_df["BraTS21ID"], labels_df["MGMT_value"].astype(np.float32))
)
train_ids = [i for i in train_ids_all if i in train_labels_map]

MAX_TRAIN_CASES = 220  # 220*8=1760 images; typically OK on Kaggle CPU/GPU
train_ids = train_ids[: min(MAX_TRAIN_CASES, len(train_ids))]

y_cases = np.array([train_labels_map[i] for i in train_ids], dtype=np.float32)

N_SLICES = 8
y_slices = np.repeat(y_cases, N_SLICES).astype(np.float32)

IMG_PX_SIZE = 150



## === cell 4
X_train = load_dataset_slices(
    TRAIN_DIR, train_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
assert X_train.shape[0] == y_slices.shape[0]

perm = np.random.RandomState(SEED).permutation(len(X_train))
X_train = X_train[perm]
y_slices = y_slices[perm]

split = int(0.9 * len(X_train))
X_tr, X_val = X_train[:split], X_train[split:]
y_tr, y_val = y_slices[:split], y_slices[split:]

X_tr.shape, X_val.shape



## === cell 5
model_T2 = keras.Sequential(
    [
        layers.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]
)

model_T2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

y_tr_i = y_tr.astype(np.int32)
y_val_i = y_val.astype(np.int32)

history = model_T2.fit(
    X_tr, y_tr_i, validation_data=(X_val, y_val_i), epochs=3, batch_size=16, verbose=2
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1290757242.py in <cell line: 0>()
     25 y_val_i = y_val.astype(np.int32)
     26 
---> 27 history = model_T2.fit(
     28     X_tr, y_tr_i, validation_data=(X_val, y_val_i), epochs=3, batch_size=16, verbose=2
     29 )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/ops/math.py in _segment_reduce_validation(data, segment_ids)
     25         and segment_ids_shape[0] != data_shape[0]
     26     ):
---> 27         raise ValueError(
     28             "Argument `segment_ids` and `data` should have same leading "
     29             f"dimension. Got {segment_ids_shape} v.s. "

ValueError: Argument `segment_ids` and `data` should have same leading dimension. Got (32,) v.s. (16,).

## === cell 6
def load_test_T2W_images(path_test):
    """
    Original function returned 8 arrays (one slice position each) with shape (n_cases, H, W, 3).
    Fixes:
    - Ensure resize is defined (imported).
    - Return numpy arrays (not python lists) to support numpy ops and model.predict.
    - Do not rely on modality index ordering; use explicit "T2w" folder.
    """
    IMG_PX_SIZE = 150
    N_SLICES = 8

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    arrays = [
        np.zeros((len(path_cases), IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
        for _ in range(N_SLICES)
    ]

    for i, case_path in enumerate(path_cases):
        case_slices = load_case_t2w_slices(
            case_path, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
        )
        for s in range(N_SLICES):
            arrays[s][i] = case_slices[s]

    print("Number of T2w slice-batches loaded:", [a.shape[0] for a in arrays])
    return tuple(arrays)




## === cell 7
test = TEST_DIR
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7, pixels_8 = (
    load_test_T2W_images(test)
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/840297435.py in <cell line: 0>()
      1 test = TEST_DIR
      2 pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7, pixels_8 = (
----> 3     load_test_T2W_images(test)
      4 )
      5 

/tmp/ipykernel_11/3105651072.py in load_test_T2W_images(path_test)
     18 
     19     for i, case_path in enumerate(path_cases):
---> 20         case_slices = load_case_t2w_slices(
     21             case_path, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
     22         )

/tmp/ipykernel_11/2754378619.py in load_case_t2w_slices(case_dir, img_px_size, n_slices)
     43     Returns (n_slices, img_px_size, img_px_size, 3) float32 in [0,1].
     44     """
---> 45     t2_dir = _get_case_modality_dir(case_dir, "T2w")
     46     dcm_files = sorted(
     47         [

/tmp/ipykernel_11/2754378619.py in _get_case_modality_dir(case_dir, modality)
     19     mod_dir = os.path.join(case_dir, modality)
     20     if not os.path.isdir(mod_dir):
---> 21         raise FileNotFoundError(f"Missing modality folder: {mod_dir}")
     22     return mod_dir
     23 

FileNotFoundError: Missing modality folder: /kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test/test/T2w

## === cell 8
preds_1 = model_T2.predict(pixels_1, verbose=0)
prediction_1 = preds_1[:, 1]
preds_2 = model_T2.predict(pixels_2, verbose=0)
prediction_2 = preds_2[:, 1]
preds_3 = model_T2.predict(pixels_3, verbose=0)
prediction_3 = preds_3[:, 1]
preds_4 = model_T2.predict(pixels_4, verbose=0)
prediction_4 = preds_4[:, 1]
preds_5 = model_T2.predict(pixels_5, verbose=0)
prediction_5 = preds_5[:, 1]
preds_6 = model_T2.predict(pixels_6, verbose=0)
prediction_6 = preds_6[:, 1]
preds_7 = model_T2.predict(pixels_7, verbose=0)
prediction_7 = preds_7[:, 1]
preds_8 = model_T2.predict(pixels_8, verbose=0)
prediction_8 = preds_8[:, 1]




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/793983092.py in <cell line: 0>()
      1 # Predict per slice-position and take positive class probability
----> 2 preds_1 = model_T2.predict(pixels_1, verbose=0)
      3 prediction_1 = preds_1[:, 1]
      4 preds_2 = model_T2.predict(pixels_2, verbose=0)
      5 prediction_2 = preds_2[:, 1]

NameError: name 'pixels_1' is not defined

## === cell 9
def create_sub(path_test, p1, p2, p3, p4, p5, p6, p7, p8):
    """
    Fixes:
    - Compute prediction vector once (not overwritten inside the loop).
    - Ensure BraTS21ID format matches submission requirement: 5-digit string.
    - Ensure length alignment between IDs and predictions.
    """
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [os.path.basename(p) for p in path_cases]  # already zero-padded strings

    prediction = (
        p1.astype(np.float32)
        + p2.astype(np.float32)
        + p3.astype(np.float32)
        + p4.astype(np.float32)
        + p5.astype(np.float32)
        + p6.astype(np.float32)
        + p7.astype(np.float32)
        + p8.astype(np.float32)
    ) / 8.0

    assert len(cases) == len(
        prediction
    ), f"ID/pred length mismatch: {len(cases)} vs {len(prediction)}"

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df


sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_7,
    prediction_8,
)

sub_df.head(), sub_df.shape



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2903286216.py in <cell line: 0>()
     30 sub_df = create_sub(
     31     test,
---> 32     prediction_1,
     33     prediction_2,
     34     prediction_3,

NameError: name 'prediction_1' is not defined

## === cell 10
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = (
    sub_df["MGMT_value"].fillna(sub_df["MGMT_value"].mean()).clip(0.0, 1.0)
)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2711940135.py in <cell line: 0>()
      2 sample = pd.read_csv(SAMPLE_SUB)
      3 sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)
----> 4 sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
      5 
      6 sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

NameError: name 'sub_df' is not defined
