# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import warnings

import numpy as np
import pandas as pd

from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", str(SEED))




## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
LABELS_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Missing labels: {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int)
labels_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))




## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 6


def _safe_norm(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mx = float(np.max(img))
    if mx <= 0:
        return np.zeros_like(img, dtype=np.float32)
    return img / mx


def _stack3(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d.astype(np.float32)
    return np.stack([img2d, img2d, img2d], axis=-1)


def _list_cases(path_dir: str):
    """
    Bugfix: some environments may have an extra nested 'test/' or 'train/' folder.
    If the directory contains exactly one subdir named like the directory itself,
    descend into it. This prevents scanning '/.../test/test' style paths.
    """
    path_dir = os.path.abspath(path_dir)
    if not os.path.isdir(path_dir):
        raise FileNotFoundError(f"Missing directory: {path_dir}")

    subs = [f for f in os.scandir(path_dir) if f.is_dir()]
    base = os.path.basename(path_dir.rstrip("/"))
    if len(subs) == 1 and os.path.basename(subs[0].path.rstrip("/")) == base:
        path_dir = subs[0].path

    return sorted([f.path for f in os.scandir(path_dir) if f.is_dir()])


def _get_case_id_from_path(case_path: str) -> int:
    return int(os.path.basename(case_path))


def _get_modality_dir(case_path: str, modality: str) -> str:
    mod_dir = os.path.join(case_path, modality)
    if not os.path.exists(mod_dir):
        raise FileNotFoundError(f"Missing modality folder {modality} in {case_path}")
    return mod_dir


def _read_dcm_pixel(path: str) -> np.ndarray:
    b = tf.io.read_file(path)
    img = tf.io.decode_dicom_image(
        b,
        dtype=tf.uint16,
        color_dim=False,
        scale="auto",
        expand_animations=False,
    )
    arr = img.numpy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def load_case_t2_slices(case_path: str, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES):
    """
    Loads up to n_slices T2w slices, filters low-signal, resizes to img_px_size,
    normalizes to [0,1], and stacks into 3 channels.
    Pads with zeros if insufficient valid slices.
    """
    t2_dir = _get_modality_dir(case_path, "T2w")
    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(t2_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    picked = []
    for fp in dcm_files:
        try:
            px = _read_dcm_pixel(fp)
        except Exception:
            continue
        if px is None:
            continue
        if float(np.sum(px)) <= 100000:
            continue

        r = resize(
            px, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        r = _safe_norm(r)
        s = _stack3(r)
        if float(np.sum(s)) <= 2000:
            continue

        picked.append(s)
        if len(picked) >= n_slices:
            break

    if len(picked) < n_slices:
        pad = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
        picked = picked + [pad] * (n_slices - len(picked))

    return picked




## === cell 3
BAD_CASES = {109, 123, 709}  # known corrupted train subjects per competition note


def build_dataset_from_cases(case_paths, labels_map, max_cases=None):
    X_slices = [[] for _ in range(N_SLICES)]
    y = []

    used = 0
    for case_path in case_paths:
        case_id = _get_case_id_from_path(case_path)
        if case_id in BAD_CASES:
            continue
        if case_id not in labels_map:
            continue
        slices = load_case_t2_slices(case_path)
        for i in range(N_SLICES):
            X_slices[i].append(slices[i])
        y.append(labels_map[case_id])
        used += 1
        if max_cases is not None and used >= max_cases:
            break

    X_slices = [np.asarray(x, dtype=np.float32) for x in X_slices]
    y = np.asarray(y, dtype=np.float32)
    return X_slices, y


train_case_paths = _list_cases(TRAIN_DIR)

MAX_TRAIN_CASES = 320

X_slices_all, y_all = build_dataset_from_cases(
    train_case_paths, labels_map, max_cases=MAX_TRAIN_CASES
)

n = len(y_all)
assert n > 10, "Too few training examples loaded; dataset loading failed."
print("Loaded train cases:", n, " | slice shapes:", [x.shape for x in X_slices_all])




## === cell 4
idx = np.arange(n)
pos_idx = idx[y_all == 1]
neg_idx = idx[y_all == 0]
rng = np.random.default_rng(SEED)
rng.shuffle(pos_idx)
rng.shuffle(neg_idx)

val_frac = 0.2
n_pos_val = max(1, int(len(pos_idx) * val_frac))
n_neg_val = max(1, int(len(neg_idx) * val_frac))

val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])

rng.shuffle(train_idx)
rng.shuffle(val_idx)


def _split_slices(X_slices, train_idx, val_idx):
    X_tr = [x[train_idx] for x in X_slices]
    X_va = [x[val_idx] for x in X_slices]
    return X_tr, X_va


X_tr_slices, X_va_slices = _split_slices(X_slices_all, train_idx, val_idx)
y_tr, y_va = y_all[train_idx], y_all[val_idx]

print(
    "Train size:",
    len(y_tr),
    "Val size:",
    len(y_va),
    "Pos rate train/val:",
    float(y_tr.mean()),
    float(y_va.mean()),
)




## === cell 5
def build_small_cnn(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


EPOCHS_1, EPOCHS_2, EPOCHS_3 = 5, 7, 4
BATCH_SIZE = 16

model_T2 = build_small_cnn()
model_T2_2 = build_small_cnn()
model_T2_3 = build_small_cnn()




## === cell 6
def make_slice_training_data(X_slices, y):
    X = np.concatenate(X_slices, axis=0)
    y_rep = np.concatenate([y for _ in range(len(X_slices))], axis=0)
    return X, y_rep


Xtr_cat, ytr_cat = make_slice_training_data(X_tr_slices, y_tr.astype(np.float32))
Xva_cat, yva_cat = make_slice_training_data(X_va_slices, y_va.astype(np.float32))

perm_tr = np.random.permutation(len(ytr_cat))
perm_va = np.random.permutation(len(yva_cat))
Xtr_cat, ytr_cat = Xtr_cat[perm_tr], ytr_cat[perm_tr]
Xva_cat, yva_cat = Xva_cat[perm_va], yva_cat[perm_va]

history_1 = model_T2.fit(
    Xtr_cat,
    ytr_cat,
    validation_data=(Xva_cat, yva_cat),
    epochs=EPOCHS_1,
    batch_size=BATCH_SIZE,
    verbose=0,
)
history_2 = model_T2_2.fit(
    Xtr_cat,
    ytr_cat,
    validation_data=(Xva_cat, yva_cat),
    epochs=EPOCHS_2,
    batch_size=BATCH_SIZE,
    verbose=0,
)
history_3 = model_T2_3.fit(
    Xtr_cat,
    ytr_cat,
    validation_data=(Xva_cat, yva_cat),
    epochs=EPOCHS_3,
    batch_size=BATCH_SIZE,
    verbose=0,
)

print("Trained 3 models.")
print(
    "Final val AUCs:",
    history_1.history["val_auc"][-1],
    history_2.history["val_auc"][-1],
    history_3.history["val_auc"][-1],
)




## === cell 7
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    path_cases = _list_cases(path_test)
    for case_path in path_cases:
        try:
            _ = _get_case_id_from_path(case_path)
        except Exception:
            continue

        try:
            slices = load_case_t2_slices(case_path)
        except FileNotFoundError:
            pad = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
            slices = [pad] * N_SLICES

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

    def gnorm(a):
        mx = float(np.max(a))
        return a / mx if mx > 0 else a

    array_1, array_2, array_3 = gnorm(array_1), gnorm(array_2), gnorm(array_3)
    array_4, array_5, array_6 = gnorm(array_4), gnorm(array_5), gnorm(array_6)

    print(
        "Number of T2 images loaded are",
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


test = TEST_DIR
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)




## === cell 8
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




## === cell 9
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
):
    path_cases = _list_cases(path_test)
    cases = []
    for p in path_cases:
        try:
            cases.append(_get_case_id_from_path(p))
        except Exception:
            continue

    prediction = (
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
    ) / 18.0

    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)

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
)

sub_df.head()




## === cell 10
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
print(
    "MGMT_value range:",
    float(sub_df["MGMT_value"].min()),
    float(sub_df["MGMT_value"].max()),
)
