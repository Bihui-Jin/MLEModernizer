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

0.61412

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61412) has done: 'I remove/guard the imports that trigger the protobuf `MessageFactory.GetPrototype` crash and keep only the libraries actually needed for DICOM loading and TensorFlow inference. Since the referenced pretrained `.h5` model file is not available in your environment, I keep the same CNN-based approach but build and train a small Keras model on-the-fly using the same T2w slice-extraction logic, so the notebook runs end-to-end. I also fix the missing `resize` symbol, correct the submission construction so predictions align 1:1 with test cases, and ensure `BraTS21ID` is written in the expected 5-digit format with a valid `submission.csv`. Finally, I exclude the known-bad train cases `[00109, 00123, 00709]` as suggested by the competition to avoid runtime/data issues and improve stability/score.'

# 9. Code solution

## === cell 0
import os
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

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

print("Train labels:", labels_df.shape, "Sample sub:", sample_sub.shape)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)




## === cell 2
def _safe_dcmread(path):
    """Robust DICOM read; returns None on failures."""
    try:
        return dicom.dcmread(path, force=True)
    except Exception:
        return None


def _normalize_img(x, eps=1e-6):
    x = x.astype(np.float32)
    x = x - np.min(x)
    denom = np.max(x)
    if denom < eps:
        return np.zeros_like(x, dtype=np.float32)
    return x / denom


def load_T2W_three_slices_for_case(case_dir, img_px_size=150, max_slices=3):
    """
    Core logic preserved: scan T2w series, pick first 3 "non-empty" slices by thresholds,
    resize to 150x150, stack to 3 channels, normalize.
    Returns list length up to 3 of (150,150,3) float32.
    """
    modalities = {
        os.path.basename(p): p
        for p in sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
    }
    if "T2w" not in modalities:
        return []

    t2_dir = modalities["T2w"]
    dcm_paths = sorted(
        [
            f.path
            for f in os.scandir(t2_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    if len(dcm_paths) == 0:
        return []

    picked = []
    count = 0
    for p in dcm_paths:
        ds = _safe_dcmread(p)
        if ds is None:
            continue
        try:
            arr = ds.pixel_array
        except Exception:
            continue

        if arr.sum() > 100000:
            resized_img = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked_img = np.stack((resized_img,) * 3, axis=-1)
            stacked_img = _normalize_img(stacked_img)
            if stacked_img.sum() > 2500:
                picked.append(stacked_img)
                count += 1
                if count >= max_slices:
                    break

    return picked


def load_T2W_three_slices_dataset(cases, base_dir, img_px_size=150):
    """
    Returns three arrays (N,150,150,3) for slice1/slice2/slice3.
    Only keeps cases where 3 slices are found, ensuring alignment.
    """
    x1, x2, x3, kept = [], [], [], []
    for case_id in cases:
        case_dir = os.path.join(base_dir, case_id)
        slices = load_T2W_three_slices_for_case(
            case_dir, img_px_size=img_px_size, max_slices=3
        )
        if len(slices) == 3:
            x1.append(slices[0])
            x2.append(slices[1])
            x3.append(slices[2])
            kept.append(case_id)
    if len(kept) == 0:
        return (
            np.zeros((0, img_px_size, img_px_size, 3), np.float32),
            np.zeros((0, img_px_size, img_px_size, 3), np.float32),
            np.zeros((0, img_px_size, img_px_size, 3), np.float32),
            [],
        )
    return (
        np.stack(x1).astype(np.float32),
        np.stack(x2).astype(np.float32),
        np.stack(x3).astype(np.float32),
        kept,
    )




## === cell 3
BAD_CASES = {"00109", "00123", "00709"}
train_case_ids = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
train_case_ids = [cid for cid in train_case_ids if cid not in BAD_CASES]

labels_map = dict(zip(labels_df["BraTS21ID"], labels_df["MGMT_value"].astype(int)))
train_case_ids = [cid for cid in train_case_ids if cid in labels_map]

test_case_ids = sorted([d.name for d in os.scandir(TEST_DIR) if d.is_dir()])

print("Train cases (after filtering):", len(train_case_ids))
print("Test cases:", len(test_case_ids))



## === cell 4
IMG_PX_SIZE = 150

x1_tr, x2_tr, x3_tr, kept_train = load_T2W_three_slices_dataset(
    train_case_ids, TRAIN_DIR, img_px_size=IMG_PX_SIZE
)
y_tr = np.array([labels_map[cid] for cid in kept_train], dtype=np.float32)

print(
    "Loaded train:",
    x1_tr.shape,
    x2_tr.shape,
    x3_tr.shape,
    "Labels:",
    y_tr.shape,
    "Pos rate:",
    float(y_tr.mean()) if len(y_tr) else None,
)

n = len(kept_train)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.2
n_val = int(round(n * val_frac))
val_idx = idx[:n_val]
tr_idx = idx[n_val:]

x1_train, x2_train, x3_train = x1_tr[tr_idx], x2_tr[tr_idx], x3_tr[tr_idx]
y_train = y_tr[tr_idx]
x1_val, x2_val, x3_val = x1_tr[val_idx], x2_tr[val_idx], x3_tr[val_idx]
y_val = y_tr[val_idx]

print("Split:", x1_train.shape[0], "train /", x1_val.shape[0], "val")




## === cell 5
def build_slice_model(input_shape=(150, 150, 3)):
    """
    Minimal CNN architecture to replace missing external pretrained model file.
    Keeps core semantics: image -> probability (binary classification).
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = build_slice_model((IMG_PX_SIZE, IMG_PX_SIZE, 3))
model_T2.summary()



## === cell 6

X_train_all = np.concatenate([x1_train, x2_train, x3_train], axis=0)
y_train_all = np.concatenate([y_train, y_train, y_train], axis=0)

X_val_all = np.concatenate([x1_val, x2_val, x3_val], axis=0)
y_val_all = np.concatenate([y_val, y_val, y_val], axis=0)

BATCH_SIZE = 16
EPOCHS = 5  # kept small to fit time limit; no early stopping

history = model_T2.fit(
    X_train_all,
    y_train_all,
    validation_data=(X_val_all, y_val_all),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 7
x1_te, x2_te, x3_te, kept_test = load_T2W_three_slices_dataset(
    test_case_ids, TEST_DIR, img_px_size=IMG_PX_SIZE
)
print(
    "Loaded test:",
    x1_te.shape,
    x2_te.shape,
    x3_te.shape,
    "Kept test cases:",
    len(kept_test),
)



## === cell 8
pred1 = model_T2.predict(x1_te, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
pred2 = model_T2.predict(x2_te, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
pred3 = model_T2.predict(x3_te, batch_size=BATCH_SIZE, verbose=0).reshape(-1)

prediction = (
    pred1.astype(np.float32) + pred2.astype(np.float32) + pred3.astype(np.float32)
) / 3.0
prediction = np.clip(prediction, 0.0, 1.0)

print(
    "Pred stats:",
    float(prediction.min()),
    float(prediction.max()),
    float(prediction.mean()),
)




## === cell 9
def create_sub(case_ids_5digit, preds):
    df = pd.DataFrame(
        {
            "BraTS21ID": pd.Series(case_ids_5digit, dtype=str).str.zfill(5),
            "MGMT_value": preds.astype(np.float32),
        }
    )
    return df


sub_df = create_sub(kept_test, prediction)

missing = sorted(set(sample_sub["BraTS21ID"]) - set(sub_df["BraTS21ID"]))
if len(missing) > 0:
    fill_value = float(prediction.mean()) if len(prediction) else 0.5
    sub_missing = pd.DataFrame({"BraTS21ID": missing, "MGMT_value": fill_value})
    sub_df = pd.concat([sub_df, sub_missing], axis=0, ignore_index=True)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(np.float32)

print(sub_df.head())
print(
    "Submission shape:", sub_df.shape, "Missing any:", sub_df["MGMT_value"].isna().any()
)



## === cell 10
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_df.columns))
print("submission.csv preview:\n", sub_df.head().to_string(index=False))
