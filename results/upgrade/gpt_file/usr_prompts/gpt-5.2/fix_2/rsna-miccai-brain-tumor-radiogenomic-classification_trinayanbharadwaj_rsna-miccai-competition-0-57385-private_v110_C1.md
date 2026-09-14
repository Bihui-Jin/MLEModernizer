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

0.55176

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.55176) has done: 'I remove/guard the imports that trigger the protobuf `MessageFactory.GetPrototype` crash and keep only the packages actually needed for loading DICOMs and running inference. Because the referenced pretrained `.h5` models are not available in your environment, I minimally replace that missing step by training the same kind of lightweight Keras CNN directly on the provided `train/` DICOMs (using the same “pick up to 6 informative slices” logic), then run predictions on `test/` and write `submission.csv`. I also fix logic bugs in `create_sub` (it was producing a single scalar prediction for all cases) and ensure IDs are formatted exactly as required. The result run end-to-end within the Kaggle environment and produce a valid submission CSV.'

# 9. Code solution

## === cell 0
import os
import random
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
labels_df = labels_df.set_index("BraTS21ID")

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

print("Train labels:", labels_df.shape, "Test sample:", sample_df.shape)



## === cell 2
MRI_TYPE_INDEX_T2W = 3  # original code assumes mri_type[3] is T2w after sorted()
IMG_PX_SIZE = 150
SLICES_PER_CASE = 6

BAD_TRAIN_IDS = set(["00109", "00123", "00709"])  # as per competition note


def _safe_read_dcm(path):
    try:
        d = dicom.dcmread(path)
        arr = d.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def load_case_slices(
    case_dir,
    mri_type_index=MRI_TYPE_INDEX_T2W,
    img_px_size=IMG_PX_SIZE,
    slices_per_case=SLICES_PER_CASE,
):
    """
    Returns: np.ndarray shape (slices_per_case, img_px_size, img_px_size, 3)
             If not enough valid slices, pads by repeating the last valid slice,
             else uses zeros if none valid (rare).
    """
    mri_types = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
    if len(mri_types) <= mri_type_index:
        t2_candidates = [p for p in mri_types if os.path.basename(p).lower() == "t2w"]
        if t2_candidates:
            mri_path = t2_candidates[0]
        else:
            return None
    else:
        mri_path = mri_types[mri_type_index]

    dcm_paths = sorted(
        [
            f.path
            for f in os.scandir(mri_path)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    selected = []
    for p in dcm_paths:
        arr = _safe_read_dcm(p)
        if arr is None:
            continue
        if arr.sum() > 100000:  # original heuristic
            r = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked = np.stack((r,) * 3, axis=-1)
            mx = float(np.max(stacked))
            if mx > 0:
                stacked = stacked / mx
            if stacked.sum() > 2000:  # original heuristic
                selected.append(stacked)
                if len(selected) >= slices_per_case:
                    break

    if len(selected) == 0:
        selected = [np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)]

    while len(selected) < slices_per_case:
        selected.append(selected[-1].copy())

    return np.stack(selected[:slices_per_case], axis=0)


def load_dataset(subject_ids, base_dir, labels_indexed=None):
    """
    X shape: (N, SLICES_PER_CASE, IMG_PX_SIZE, IMG_PX_SIZE, 3)
    y shape: (N,) if labels_indexed provided else None
    """
    X_list, y_list, kept_ids = [], [], []
    for sid in subject_ids:
        case_dir = os.path.join(base_dir, sid)
        if not os.path.isdir(case_dir):
            continue
        vol = load_case_slices(case_dir)
        if vol is None:
            continue
        X_list.append(vol)
        kept_ids.append(sid)
        if labels_indexed is not None:
            y_list.append(int(labels_indexed.loc[sid, "MGMT_value"]))
    X = (
        np.stack(X_list, axis=0)
        if len(X_list)
        else np.empty(
            (0, SLICES_PER_CASE, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32
        )
    )
    y = np.array(y_list, dtype=np.float32) if labels_indexed is not None else None
    return X, y, kept_ids




## === cell 3
all_train_ids = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
all_train_ids = [sid for sid in all_train_ids if sid not in BAD_TRAIN_IDS]
all_train_ids = [sid for sid in all_train_ids if sid in labels_df.index]

MAX_TRAIN_SUBJECTS = 240  # pragmatic cap for runtime
if len(all_train_ids) > MAX_TRAIN_SUBJECTS:
    rng = np.random.default_rng(SEED)
    all_train_ids = sorted(
        rng.choice(all_train_ids, size=MAX_TRAIN_SUBJECTS, replace=False).tolist()
    )

rng = np.random.default_rng(SEED)
rng.shuffle(all_train_ids)

split = int(0.85 * len(all_train_ids))
train_ids = all_train_ids[:split]
val_ids = all_train_ids[split:]

print("Subjects train/val:", len(train_ids), len(val_ids))

X_train, y_train, kept_train_ids = load_dataset(train_ids, TRAIN_DIR, labels_df)
X_val, y_val, kept_val_ids = load_dataset(val_ids, TRAIN_DIR, labels_df)

print("Loaded X_train:", X_train.shape, "y_train:", y_train.shape)
print("Loaded X_val:", X_val.shape, "y_val:", y_val.shape)




## === cell 4
def build_slice_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    return model


slice_model = build_slice_model()
slice_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)
slice_model.summary()



## === cell 5
Xtr = X_train.reshape((-1, IMG_PX_SIZE, IMG_PX_SIZE, 3))
ytr = np.repeat(y_train, SLICES_PER_CASE)

Xva = X_val.reshape((-1, IMG_PX_SIZE, IMG_PX_SIZE, 3))
yva = np.repeat(y_val, SLICES_PER_CASE)

EPOCHS = 3
BATCH_SIZE = 32

history = slice_model.fit(
    Xtr,
    ytr,
    validation_data=(Xva, yva),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)




## === cell 6
def predict_subject_proba(model, X_subjects, batch_size=64):
    X_flat = X_subjects.reshape((-1, IMG_PX_SIZE, IMG_PX_SIZE, 3))
    p_flat = model.predict(X_flat, batch_size=batch_size, verbose=0).reshape(
        (-1, SLICES_PER_CASE)
    )
    return p_flat.mean(axis=1)


val_pred = predict_subject_proba(slice_model, X_val)
val_auc = tf.keras.metrics.AUC()
val_auc.update_state(y_val, val_pred)
print("Validation subject-level AUC:", float(val_auc.result().numpy()))



## === cell 7
test_ids = sample_df["BraTS21ID"].tolist()
X_test, _, kept_test_ids = load_dataset(test_ids, TEST_DIR, labels_indexed=None)

test_pred_map = {}
if X_test.shape[0] > 0:
    preds = predict_subject_proba(slice_model, X_test)
    for sid, p in zip(kept_test_ids, preds):
        test_pred_map[sid] = float(p)

sub = sample_df.copy()
sub["MGMT_value"] = sub["BraTS21ID"].map(test_pred_map).fillna(0.5).astype(np.float32)

sub["MGMT_value"] = sub["MGMT_value"].clip(0.0, 1.0)

sub.head(), sub.shape



## === cell 8
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.describe(include="all"))
