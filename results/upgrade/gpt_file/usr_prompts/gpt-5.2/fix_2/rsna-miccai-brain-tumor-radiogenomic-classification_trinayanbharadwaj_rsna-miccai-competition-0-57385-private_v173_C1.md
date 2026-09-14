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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize  # fixes NameError: resize is not defined

import tensorflow as tf
from tensorflow import keras

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
LABELS_CSV = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

labels_df = pd.read_csv(LABELS_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

print("Train labels:", labels_df.shape)
print("Sample submission:", sample_df.shape)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)



## === cell 2
IMG_PX_SIZE = 150


def _safe_norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.max(x)) if x.size else 0.0
    if mx <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def _read_dicom_pixel_array(dcm_path: str) -> np.ndarray:
    ds = dicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    return arr


def load_case_slices(
    case_dir: str,
    modality_idx: int,
    max_slices: int = 6,
    sum_thr: float = 100000.0,
    sum_norm_thr: float = 2000.0,
):
    """
    Mimics the user's original logic:
    - pick modality folder by sorted listing index
    - scan slices, keep up to 6 slices that pass thresholds
    - resize to 150x150, stack to 3-ch, normalize per-slice
    Returns list of (H,W,3) arrays length <= max_slices.
    """
    mri_type = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
    if len(mri_type) <= modality_idx:
        return []
    img_dir = mri_type[modality_idx]
    img_paths = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])
    out = []
    for p in img_paths:
        try:
            img = _read_dicom_pixel_array(p)
        except Exception:
            continue
        if float(np.sum(img)) <= sum_thr:
            continue
        resized_img = resize(
            img, (IMG_PX_SIZE, IMG_PX_SIZE), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        stacked = np.stack((resized_img,) * 3, axis=-1)
        stacked = _safe_norm01(stacked)
        if float(np.sum(stacked)) <= sum_norm_thr:
            continue
        out.append(stacked)
        if len(out) >= max_slices:
            break
    return out


def load_case_feature(case_dir: str):
    """
    Build one fixed-size tensor per case by taking up to 6 slices from each of:
      - flair (idx 0)
      - t1wce (idx 2)
      - t2w (idx 3)
    Then average all collected slices into a single (150,150,3) image.
    This is the smallest change to make the pipeline case-level and submission-valid.
    """
    slices = []
    slices += load_case_slices(
        case_dir, modality_idx=0, max_slices=6, sum_norm_thr=2000.0
    )  # flair
    slices += load_case_slices(
        case_dir, modality_idx=2, max_slices=6, sum_norm_thr=1900.0
    )  # t1wce
    slices += load_case_slices(
        case_dir, modality_idx=3, max_slices=6, sum_norm_thr=2000.0
    )  # t2w

    if len(slices) == 0:
        return np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    x = np.mean(np.stack(slices, axis=0), axis=0).astype(np.float32)
    x = _safe_norm01(x)
    return x


def list_case_dirs(path_root: str):
    return sorted([f.path for f in os.scandir(path_root) if f.is_dir()])


def case_id_from_dir(case_dir: str) -> int:
    return int(os.path.basename(case_dir))




## === cell 3
PRETRAIN_DIR = "/kaggle/input/trained-model-for-rsnamiccai"
PRETRAIN_FILES = [
    "rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_15_b400_flair_5k_0.73auc_imgs.h5",
    "rsna_miccai_20_b600_t1wce_7k_0.73auc_imgs.h5",
    "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
    "rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5",
    "rsna_miccai_20_b600_t1wce_7k_0.77auc_imgs.h5",
]
pretrained_paths = [os.path.join(PRETRAIN_DIR, f) for f in PRETRAIN_FILES]
available_pretrained = [p for p in pretrained_paths if os.path.exists(p)]

print("Found pretrained models:", len(available_pretrained))
USE_PRETRAINED = len(available_pretrained) == len(pretrained_paths)

models = []
if USE_PRETRAINED:
    for p in pretrained_paths:
        models.append(keras.models.load_model(p, compile=False))
    print("Loaded all pretrained models.")
else:
    print(
        "Pretrained models not available; will train a small fallback model on train/ and predict test/."
    )



## === cell 4
BAD_CASES = {109, 123, 709}

train_case_dirs = [
    d for d in list_case_dirs(TRAIN_DIR) if case_id_from_dir(d) not in BAD_CASES
]
test_case_dirs = list_case_dirs(TEST_DIR)

train_ids = [case_id_from_dir(d) for d in train_case_dirs]
test_ids = [case_id_from_dir(d) for d in test_case_dirs]

labels_map = dict(
    zip(
        labels_df["BraTS21ID"].astype(int).values,
        labels_df["MGMT_value"].astype(int).values,
    )
)
y_all = np.array([labels_map[i] for i in train_ids], dtype=np.float32)

print("Train cases:", len(train_case_dirs), "Test cases:", len(test_case_dirs))
print("y mean:", float(np.mean(y_all)))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1222466338.py in <cell line: 0>()
      2 BAD_CASES = {109, 123, 709}
      3 
----> 4 train_case_dirs = [
      5     d for d in list_case_dirs(TRAIN_DIR) if case_id_from_dir(d) not in BAD_CASES
      6 ]

/tmp/ipykernel_11/1222466338.py in <listcomp>(.0)
      3 
      4 train_case_dirs = [
----> 5     d for d in list_case_dirs(TRAIN_DIR) if case_id_from_dir(d) not in BAD_CASES
      6 ]
      7 test_case_dirs = list_case_dirs(TEST_DIR)

/tmp/ipykernel_11/3984156384.py in case_id_from_dir(case_dir)
     91 
     92 def case_id_from_dir(case_dir: str) -> int:
---> 93     return int(os.path.basename(case_dir))
     94 
     95 

ValueError: invalid literal for int() with base 10: 'train'

## === cell 5
from pathlib import Path

CACHE_DIR = Path("/kaggle/working/cache_rsna")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
X_train_cache = CACHE_DIR / "X_train.npy"
X_test_cache = CACHE_DIR / "X_test.npy"
train_ids_cache = CACHE_DIR / "train_ids.npy"
test_ids_cache = CACHE_DIR / "test_ids.npy"


def build_X(case_dirs):
    X = np.zeros((len(case_dirs), IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    for i, d in enumerate(case_dirs):
        X[i] = load_case_feature(d)
        if (i + 1) % 50 == 0 or (i + 1) == len(case_dirs):
            print(f"Processed {i+1}/{len(case_dirs)} cases")
    return X


if (
    X_train_cache.exists()
    and X_test_cache.exists()
    and train_ids_cache.exists()
    and test_ids_cache.exists()
):
    cached_train_ids = np.load(train_ids_cache).astype(int).tolist()
    cached_test_ids = np.load(test_ids_cache).astype(int).tolist()
    if cached_train_ids == train_ids and cached_test_ids == test_ids:
        X_train = np.load(X_train_cache)
        X_test = np.load(X_test_cache)
        print("Loaded cached features:", X_train.shape, X_test.shape)
    else:
        print("Cache mismatch; rebuilding features.")
        X_train = build_X(train_case_dirs)
        X_test = build_X(test_case_dirs)
        np.save(X_train_cache, X_train)
        np.save(X_test_cache, X_test)
        np.save(train_ids_cache, np.array(train_ids, dtype=int))
        np.save(test_ids_cache, np.array(test_ids, dtype=int))
else:
    X_train = build_X(train_case_dirs)
    X_test = build_X(test_case_dirs)
    np.save(X_train_cache, X_train)
    np.save(X_test_cache, X_test)
    np.save(train_ids_cache, np.array(train_ids, dtype=int))
    np.save(test_ids_cache, np.array(test_ids, dtype=int))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2588450027.py in <cell line: 0>()
     42         np.save(test_ids_cache, np.array(test_ids, dtype=int))
     43 else:
---> 44     X_train = build_X(train_case_dirs)
     45     X_test = build_X(test_case_dirs)
     46     np.save(X_train_cache, X_train)

NameError: name 'train_case_dirs' is not defined

## === cell 6
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

if not USE_PRETRAINED:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_all, test_size=0.2, random_state=SEED, stratify=y_all
    )

    inputs = keras.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    fallback_model = keras.Model(inputs, outputs)
    fallback_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    history = fallback_model.fit(
        X_tr, y_tr, validation_data=(X_va, y_va), epochs=6, batch_size=16, verbose=2
    )

    val_pred = fallback_model.predict(X_va, batch_size=32).reshape(-1)
    print("Validation AUC:", roc_auc_score(y_va, val_pred))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1296503381.py in <cell line: 0>()
      6 if not USE_PRETRAINED:
      7     X_tr, X_va, y_tr, y_va = train_test_split(
----> 8         X_train, y_all, test_size=0.2, random_state=SEED, stratify=y_all
      9     )
     10 

NameError: name 'X_train' is not defined

## === cell 7
if USE_PRETRAINED:
    preds_list = []
    for m in models:
        p = m.predict(X_test, batch_size=16, verbose=0)
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        else:
            p = p.reshape(-1)
        preds_list.append(p.astype(np.float32))
    test_pred = np.mean(np.stack(preds_list, axis=0), axis=0)
else:
    test_pred = (
        fallback_model.predict(X_test, batch_size=32, verbose=0)
        .reshape(-1)
        .astype(np.float32)
    )

test_pred = np.clip(test_pred, 0.0, 1.0)
print(
    "Pred stats:",
    float(test_pred.min()),
    float(test_pred.max()),
    float(test_pred.mean()),
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1600443966.py in <cell line: 0>()
     17 else:
     18     test_pred = (
---> 19         fallback_model.predict(X_test, batch_size=32, verbose=0)
     20         .reshape(-1)
     21         .astype(np.float32)

NameError: name 'fallback_model' is not defined

## === cell 8
pred_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_pred})

pred_df["BraTS21ID"] = pred_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}")

sub_df = sample_df[["BraTS21ID"]].merge(pred_df, on="BraTS21ID", how="left")

fill_value = float(np.mean(test_pred)) if len(test_pred) else 0.5
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(fill_value).astype(float)

print(sub_df.head())
print(
    "Submission shape:",
    sub_df.shape,
    "Missing preds:",
    int(sub_df["MGMT_value"].isna().sum()),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1289043331.py in <cell line: 0>()
      1 # Create submission aligned to sample_submission ordering (most robust)
----> 2 pred_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_pred})
      3 
      4 # Ensure correct dtype/format: IDs as zero-padded 5-digit strings like sample submission
      5 pred_df["BraTS21ID"] = pred_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}")

NameError: name 'test_ids' is not defined

## === cell 9
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", list(sub_df.columns))
print(
    "MGMT_value range:",
    float(sub_df["MGMT_value"].min()),
    float(sub_df["MGMT_value"].max()),
)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2177638988.py in <cell line: 0>()
      1 # Write submission
      2 out_path = "submission.csv"
----> 3 sub_df.to_csv(out_path, index=False)
      4 print("Wrote:", out_path)
      5 print("Columns:", list(sub_df.columns))

NameError: name 'sub_df' is not defined
