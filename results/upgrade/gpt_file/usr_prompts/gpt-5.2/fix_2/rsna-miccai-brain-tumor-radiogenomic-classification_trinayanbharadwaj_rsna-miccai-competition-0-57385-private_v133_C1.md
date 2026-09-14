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
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import pydicom as dicom
except Exception as e:
    dicom = None
    print(
        "WARNING: pydicom failed to import; DICOM reading will be unavailable.", repr(e)
    )

from skimage.transform import resize  # fixes NameError: resize not defined

import tensorflow as tf
from tensorflow import keras
from keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.exists(TRAIN_LABELS_CSV), f"Missing train labels: {TRAIN_LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing sample submission: {SAMPLE_SUB_CSV}"

train_labels = pd.read_csv(TRAIN_LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

BAD_CASES = {109, 123, 709}
train_labels = train_labels[
    ~train_labels["BraTS21ID"].astype(int).isin(BAD_CASES)
].reset_index(drop=True)

train_labels.head(), sample_sub.head()



## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 6


def _safe_dcmread(path):
    """Read DICOM safely; return pixel array or None on failure."""
    if dicom is None:
        return None
    try:
        ds = dicom.dcmread(path, force=True)
        arr = ds.pixel_array
        return arr
    except Exception:
        return None


def _get_series_dir(case_dir, series_name):
    p = os.path.join(case_dir, series_name)
    if os.path.isdir(p):
        return p
    return None


def _load_case_slices(
    case_dir, series_name, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
):
    """
    Load up to n_slices slices from a given series for one case.
    Returns: list of (H,W,3) float32 images in [0,1], length <= n_slices.
    """
    series_dir = _get_series_dir(case_dir, series_name)
    if series_dir is None:
        return []

    files = sorted(
        [
            os.path.join(series_dir, f)
            for f in os.listdir(series_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    if not files:
        return []

    out = []
    count = 0
    for fp in files:
        pix = _safe_dcmread(fp)
        if pix is None:
            continue

        if np.sum(pix) <= 100000:
            continue

        resized_img = resize(
            pix, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        mx = float(np.max(resized_img))
        if mx <= 0:
            continue

        stacked = np.stack((resized_img,) * 3, axis=-1)
        stacked_norm = stacked / mx

        if float(np.sum(stacked_norm)) <= 2000:
            continue

        out.append(stacked_norm.astype(np.float32))
        count += 1
        if count >= n_slices:
            break

    return out


def _load_dataset_slices(
    root_dir, series_name, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
):
    """
    Load slices for all cases in root_dir, padding with zeros if fewer than n_slices found.
    Returns:
      X: (num_cases, n_slices, H, W, 3) float32
      ids: list of int BraTS21ID in the same order as X
    """
    case_dirs = sorted(
        [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
    )
    ids = [int(os.path.basename(p)) for p in case_dirs]

    X = np.zeros(
        (len(case_dirs), n_slices, img_px_size, img_px_size, 3), dtype=np.float32
    )
    for i, case_dir in enumerate(case_dirs):
        imgs = _load_case_slices(
            case_dir, series_name, img_px_size=img_px_size, n_slices=n_slices
        )
        for j, im in enumerate(imgs[:n_slices]):
            X[i, j] = im
    return X, ids




## === cell 3


def load_test_T2W_images(path_test):
    X, ids = _load_dataset_slices(
        path_test, "T2w", img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
    )
    arrays = [X[:, i] for i in range(N_SLICES)]
    print("Number of T2 images loaded are ", *(a.shape[0] for a in arrays))
    return (*arrays, ids)


def load_test_flair_images(path_test):
    X, ids = _load_dataset_slices(
        path_test, "FLAIR", img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
    )
    arrays = [X[:, i] for i in range(N_SLICES)]
    print("Number of flair images loaded are ", *(a.shape[0] for a in arrays))
    return (*arrays, ids)




## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, test_ids = (
    load_test_T2W_images(TEST_DIR)
)

assert (
    len(test_ids) == pixels_1.shape[0] == sample_sub.shape[0]
), "Test size mismatch with sample submission."
pixels_1.shape, test_ids[:5]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3366442740.py in <cell line: 0>()
      1 # Load test data (T2w as originally intended by the notebook's variable names)
      2 pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, test_ids = (
----> 3     load_test_T2W_images(TEST_DIR)
      4 )
      5 

/tmp/ipykernel_11/2577307631.py in load_test_T2W_images(path_test)
      3 
      4 def load_test_T2W_images(path_test):
----> 5     X, ids = _load_dataset_slices(
      6         path_test, "T2w", img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
      7     )

/tmp/ipykernel_11/4237024517.py in _load_dataset_slices(root_dir, series_name, img_px_size, n_slices)
     91         ]
     92     )
---> 93     ids = [int(os.path.basename(p)) for p in case_dirs]
     94 
     95     X = np.zeros(

/tmp/ipykernel_11/4237024517.py in <listcomp>(.0)
     91         ]
     92     )
---> 93     ids = [int(os.path.basename(p)) for p in case_dirs]
     94 
     95     X = np.zeros(

ValueError: invalid literal for int() with base 10: 'test'

## === cell 5


def build_slice_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


X_train_all, train_ids_all = _load_dataset_slices(
    TRAIN_DIR, "T2w", img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
id_to_idx = {int(i): k for k, i in enumerate(train_ids_all)}

ordered_ids = []
y = []
rows = []
for _, r in train_labels.iterrows():
    bid = int(r["BraTS21ID"])
    if bid in id_to_idx:
        ordered_ids.append(bid)
        y.append(float(r["MGMT_value"]))
        rows.append(id_to_idx[bid])

X_train_all = X_train_all[np.array(rows, dtype=int)]
y = np.array(y, dtype=np.float32)

Xs = X_train_all.reshape(-1, IMG_PX_SIZE, IMG_PX_SIZE, 3)
ys = np.repeat(y, N_SLICES)

rng = np.random.RandomState(SEED)
idx = np.arange(len(ys))
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

model_T2_5 = build_slice_model()
model_T2_5.fit(
    Xs[tr_idx],
    ys[tr_idx],
    validation_data=(Xs[va_idx], ys[va_idx]),
    epochs=2,
    batch_size=32,
    verbose=2,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3197946107.py in <cell line: 0>()
     22 # Load training slices (T2w), using labels CSV order for alignment
     23 # Build a mapping from id -> index in loaded array.
---> 24 X_train_all, train_ids_all = _load_dataset_slices(
     25     TRAIN_DIR, "T2w", img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
     26 )

/tmp/ipykernel_11/4237024517.py in _load_dataset_slices(root_dir, series_name, img_px_size, n_slices)
     91         ]
     92     )
---> 93     ids = [int(os.path.basename(p)) for p in case_dirs]
     94 
     95     X = np.zeros(

/tmp/ipykernel_11/4237024517.py in <listcomp>(.0)
     91         ]
     92     )
---> 93     ids = [int(os.path.basename(p)) for p in case_dirs]
     94 
     95     X = np.zeros(

ValueError: invalid literal for int() with base 10: 'train'

## === cell 6

preds_401 = model_T2_5.predict(pixels_1, batch_size=32, verbose=0).reshape(-1)
prediction_401 = preds_401.astype(np.float32)

preds_402 = model_T2_5.predict(pixels_2, batch_size=32, verbose=0).reshape(-1)
prediction_402 = preds_402.astype(np.float32)

preds_403 = model_T2_5.predict(pixels_3, batch_size=32, verbose=0).reshape(-1)
prediction_403 = preds_403.astype(np.float32)

preds_404 = model_T2_5.predict(pixels_4, batch_size=32, verbose=0).reshape(-1)
prediction_404 = preds_404.astype(np.float32)

preds_405 = model_T2_5.predict(pixels_5, batch_size=32, verbose=0).reshape(-1)
prediction_405 = preds_405.astype(np.float32)

preds_406 = model_T2_5.predict(pixels_6, batch_size=32, verbose=0).reshape(-1)
prediction_406 = preds_406.astype(np.float32)

prediction_401[:3], prediction_406[:3]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3081878091.py in <cell line: 0>()
      2 # Fix original bug: prediction_406 should come from pixels_6, not pixels_5.
      3 
----> 4 preds_401 = model_T2_5.predict(pixels_1, batch_size=32, verbose=0).reshape(-1)
      5 prediction_401 = preds_401.astype(np.float32)
      6 

NameError: name 'model_T2_5' is not defined

## === cell 7


def create_sub(path_test, p401, p402, p403, p404, p405, p406):
    pred = (
        p401.astype(np.float32)
        + p402.astype(np.float32)
        + p403.astype(np.float32)
        + p404.astype(np.float32)
        + p405.astype(np.float32)
        + p406.astype(np.float32)
    ) / 6.0

    pred = np.clip(pred, 0.0, 1.0)

    sub = pd.read_csv(SAMPLE_SUB_CSV)
    pred_by_id = {int(i): float(p) for i, p in zip(test_ids, pred)}
    sub["MGMT_value"] = sub["BraTS21ID"].astype(int).map(pred_by_id).astype(float)

    sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5)

    return sub[["BraTS21ID", "MGMT_value"]]




## === cell 8
sub_df = create_sub(
    TEST_DIR,
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
)

sub_df.head(), sub_df.shape



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3567355904.py in <cell line: 0>()
      1 sub_df = create_sub(
      2     TEST_DIR,
----> 3     prediction_401,
      4     prediction_402,
      5     prediction_403,

NameError: name 'prediction_401' is not defined

## === cell 9
sub_df["MGMT_value"].describe()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2082861952.py in <cell line: 0>()
      1 # Quick sanity stats
----> 2 sub_df["MGMT_value"].describe()
      3 

NameError: name 'sub_df' is not defined

## === cell 10
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {sub_df.shape} and columns {list(sub_df.columns)}")
print(sub_df.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/360579785.py in <cell line: 0>()
      1 # Write submission
      2 sub_path = "submission.csv"
----> 3 sub_df.to_csv(sub_path, index=False)
      4 print(f"Wrote {sub_path} with shape {sub_df.shape} and columns {list(sub_df.columns)}")
      5 print(sub_df.head())

NameError: name 'sub_df' is not defined
