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
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

bad_ids = set(["00109", "00123", "00709"])  # per competition note

print("TensorFlow:", tf.__version__)
print("DATA_ROOT exists:", os.path.isdir(DATA_ROOT))




## === cell 1
def _safe_dcm_pixel_array(dcm_path: str):
    """
    Fix: Replace pydicom with TF decoder to avoid protobuf crash.
    Returns: 2D float32 numpy array or None.
    """
    try:
        b = tf.io.read_file(dcm_path)
        img = tf.io.decode_dicom_image(
            b,
            dtype=tf.uint16,
            color_dim=False,
            scale="auto",
            expand_animations=True,
        )
        img = tf.squeeze(img)
        if img.shape.rank == 3:
            img = tf.squeeze(img, axis=-1)

        arr = img.numpy().astype(np.float32)
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None


def _normalize_img(img2d: np.ndarray):
    mx = float(np.max(img2d)) if img2d is not None else 0.0
    if mx <= 0:
        return None
    img = img2d / mx
    return img


def load_case_slices_t2(case_dir: str, img_px_size: int = 150, n_slices: int = 6):
    """
    Loads up to n_slices T2w slices for one subject.
    Returns: list of (H,W,3) float32 in [0,1], length <= n_slices
    """
    t2_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2_dir):
        return []

    img_paths = sorted(
        [
            f.path
            for f in os.scandir(t2_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    out = []
    count = 0
    for p in img_paths:
        if count >= n_slices:
            break

        arr = _safe_dcm_pixel_array(p)
        if arr is None:
            continue
        if float(arr.sum()) <= 100000:
            continue

        t = tf.convert_to_tensor(arr[..., None], dtype=tf.float32)  # [H, W, 1]
        t = tf.image.resize(
            t, (img_px_size, img_px_size), method="bilinear", antialias=True
        )
        resized_img = tf.squeeze(t, axis=-1).numpy().astype(np.float32)

        norm2d = _normalize_img(resized_img)
        if norm2d is None:
            continue

        stacked = np.stack((norm2d,) * 3, axis=-1).astype(np.float32)
        if float(stacked.sum()) <= 2000:
            continue

        out.append(stacked)
        count += 1

    return out




## === cell 2
def load_dataset_t2(
    root_dir: str,
    labels_df: pd.DataFrame = None,
    img_px_size: int = 150,
    n_slices: int = 6,
):
    """
    Builds an image dataset by treating each selected slice as one example.
    Returns:
      X: (N, H, W, 3) float32
      y: (N,) float32 (if labels_df provided else None)
      meta: list of tuples (case_id_str, slice_index)
    """
    case_dirs = sorted([f.path for f in os.scandir(root_dir) if f.is_dir()])
    X_list, y_list, meta = [], [], []

    label_map = None
    if labels_df is not None:
        label_map = {
            f"{int(r.BraTS21ID):05d}": float(r.MGMT_value)
            for r in labels_df.itertuples(index=False)
        }

    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        if case_id in bad_ids and labels_df is not None:
            continue

        slices = load_case_slices_t2(
            case_dir, img_px_size=img_px_size, n_slices=n_slices
        )

        for si, img in enumerate(slices):
            if img is None or img.shape != (img_px_size, img_px_size, 3):
                continue

            if label_map is not None and case_id not in label_map:
                continue

            X_list.append(img)
            meta.append((case_id, si))
            if label_map is not None:
                y_list.append(label_map[case_id])

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32) if labels_df is not None else None
    return X, y, meta




## === cell 3
train_labels = pd.read_csv(TRAIN_LABELS_CSV)
train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(int)

IMG_PX_SIZE = 150
N_SLICES = 6

X_train, y_train, train_meta = load_dataset_t2(
    TRAIN_DIR, labels_df=train_labels, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
print("Train slice dataset:", X_train.shape, y_train.shape)

X_test, _, test_meta = load_dataset_t2(
    TEST_DIR, labels_df=None, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
print("Test slice dataset:", X_test.shape, len(test_meta))

assert X_train.shape[0] == y_train.shape[0], "X_train/y_train length mismatch"
assert X_test.shape[0] == len(test_meta), "X_test/test_meta length mismatch"




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
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 5
EPOCHS = 3
BATCH_SIZE = 32

case_ids_all = np.array([cid for (cid, _si) in train_meta])
unique_case_ids = np.array(sorted(set(case_ids_all.tolist())))

rng = np.random.RandomState(42)
rng.shuffle(unique_case_ids)
split_case = int(0.9 * len(unique_case_ids))
train_case_ids = set(unique_case_ids[:split_case].tolist())
valid_case_ids = set(unique_case_ids[split_case:].tolist())

tr_mask = np.array([cid in train_case_ids for cid in case_ids_all], dtype=bool)
va_mask = ~tr_mask

X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
X_va, y_va = X_train[va_mask], y_train[va_mask]

print("Train/valid slices:", X_tr.shape, X_va.shape)
print("Train/valid unique cases:", len(train_case_ids), len(valid_case_ids))

ensemble_models = []
seeds = [11, 22, 33, 44, 55]
for s in seeds:
    seed_everything(s)
    m = build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))
    m.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    ensemble_models.append(m)


## === cell 6
test_pred_models = []
if X_test.shape[0] > 0:
    for m in ensemble_models:
        p = m.predict(X_test, batch_size=64, verbose=0).reshape(-1).astype(np.float32)
        test_pred_models.append(p)
    test_pred_models = np.stack(test_pred_models, axis=0)  # (n_models, n_slices_total)
    print("Test pred stack:", test_pred_models.shape)
else:
    test_pred_models = np.zeros((len(ensemble_models), 0), dtype=np.float32)
    print("Warning: X_test is empty; will output 0.5 for all cases.")


## === cell 7
slice_pred_mean = (
    test_pred_models.mean(axis=0)
    if test_pred_models.shape[1] > 0
    else np.zeros((0,), dtype=np.float32)
)

case_to_preds = {}
for (case_id, slice_idx), pr in zip(test_meta, slice_pred_mean):
    case_to_preds.setdefault(case_id, []).append(float(pr))

test_case_ids = sorted([f.name for f in os.scandir(TEST_DIR) if f.is_dir()])
case_pred = {}
for cid in test_case_ids:
    preds = case_to_preds.get(cid, [])
    if len(preds) == 0:
        case_pred[cid] = 0.5
    else:
        case_pred[cid] = float(np.mean(preds))

print("Aggregated cases:", len(case_pred))


## === cell 8
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)

preds_out = []
for brats_id in sample_sub["BraTS21ID"].tolist():
    cid = f"{int(brats_id):05d}"
    preds_out.append(case_pred.get(cid, 0.5))

sub_df = pd.DataFrame(
    {
        "BraTS21ID": sample_sub["BraTS21ID"],
        "MGMT_value": np.asarray(preds_out, dtype=np.float32),
    }
)
sub_df["MGMT_value"] = sub_df["MGMT_value"].clip(0.0, 1.0)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.dtypes)
print(sub_df.head())
