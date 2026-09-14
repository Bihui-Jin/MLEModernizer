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

0.56

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.56) has done: 'You’re failing early due to (1) an incompatible `pympler`/protobuf import chain causing the `MessageFactory` error, and (2) missing external pretrained `.h5` files, so no model exists at inference time. To keep the core “CNN on selected DICOM slices” logic but make the notebook self-contained, I remove the problematic/unused imports and replace the missing model loads with a small Keras CNN trained quickly on-the-fly using the same kind of 150×150×3 inputs you already build. I also fix the `resize` NameError by importing it reliably (and by avoiding a brittle dependency), and fix the submission creation bug where the prediction vector wasn’t aligned per-case. Finally, the code writes a valid `submission.csv` with the required columns and ordering based on `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pydicom as dicom
import cv2

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
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

IMG_PX_SIZE = 150

BAD_CASES = set(["00109", "00123", "00709"])


def _sorted_case_dirs(root_dir):
    return sorted([f.path for f in os.scandir(root_dir) if f.is_dir()])


def _safe_max_normalize(x: np.ndarray) -> np.ndarray:
    mx = float(np.max(x))
    if mx <= 0 or not np.isfinite(mx):
        return x.astype(np.float32)
    return (x / mx).astype(np.float32)


def _read_resize_stack(dcm_path, img_px_size=150):
    ds = dicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    arr = cv2.resize(arr, (img_px_size, img_px_size), interpolation=cv2.INTER_AREA)

    stacked = np.stack([arr, arr, arr], axis=-1)
    stacked = _safe_max_normalize(stacked)
    return stacked


def load_one_slice_per_case(
    path_dir, modality_index, min_sum=100000, min_norm_sum=10000
):
    """
    Loads 1 representative slice per case for a given modality index in sorted modality folders.
    This preserves the original approach of picking the first slice that passes simple intensity thresholds.
    """
    X = []
    case_ids = []
    for case_path in _sorted_case_dirs(path_dir):
        case_id = os.path.basename(case_path)
        if path_dir.endswith("train") and case_id in BAD_CASES:
            continue

        mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if modality_index >= len(mri_types):
            continue

        img_paths = sorted(
            [f.path for f in os.scandir(mri_types[modality_index]) if f.is_file()]
        )
        picked = None
        for p in img_paths:
            ds = dicom.dcmread(p)
            arr = ds.pixel_array
            if arr.sum() > min_sum:
                stacked = _read_resize_stack(p, IMG_PX_SIZE)
                if stacked.sum() > min_norm_sum:
                    picked = stacked
                    break

        if picked is None and len(img_paths) > 0:
            mid = img_paths[len(img_paths) // 2]
            try:
                picked = _read_resize_stack(mid, IMG_PX_SIZE)
            except Exception:
                picked = None

        if picked is not None:
            X.append(picked)
            case_ids.append(int(case_id))

    X = np.asarray(X, dtype=np.float32)
    return X, case_ids


def load_k_slices_per_case(
    path_dir, modality_index, k=6, min_sum=100000, min_norm_sum=3000
):
    """
    Loads up to k slices per case for a given modality index.
    Matches the original code’s intent (arrays 1..6) but returns a single tensor (N, k, H, W, C)
    plus case_ids. If fewer than k slices are found, repeats the last found slice.
    """
    X = []
    case_ids = []
    for case_path in _sorted_case_dirs(path_dir):
        case_id = os.path.basename(case_path)
        if path_dir.endswith("train") and case_id in BAD_CASES:
            continue

        mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if modality_index >= len(mri_types):
            continue

        img_paths = sorted(
            [f.path for f in os.scandir(mri_types[modality_index]) if f.is_file()]
        )

        slices = []
        for p in img_paths:
            ds = dicom.dcmread(p)
            arr = ds.pixel_array
            if arr.sum() > min_sum:
                stacked = _read_resize_stack(p, IMG_PX_SIZE)
                if stacked.sum() > min_norm_sum:
                    slices.append(stacked)
                    if len(slices) >= k:
                        break

        if len(slices) == 0 and len(img_paths) > 0:
            idxs = np.linspace(
                0, len(img_paths) - 1, num=min(k, len(img_paths)), dtype=int
            )
            for j in idxs:
                try:
                    slices.append(_read_resize_stack(img_paths[j], IMG_PX_SIZE))
                except Exception:
                    pass

        if len(slices) == 0:
            continue

        while len(slices) < k:
            slices.append(slices[-1])

        X.append(np.stack(slices[:k], axis=0))
        case_ids.append(int(case_id))

    X = np.asarray(X, dtype=np.float32)  # (N, k, H, W, C)
    return X, case_ids




## === cell 2
labels_df = pd.read_csv(TRAIN_LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int)
id_to_label = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))

X_train, train_case_ids = load_one_slice_per_case(
    TRAIN_DIR, modality_index=3, min_sum=100000, min_norm_sum=3000
)

y_train = np.array([id_to_label[i] for i in train_case_ids], dtype=np.float32)

print(
    "Train tensor:",
    X_train.shape,
    "Labels:",
    y_train.shape,
    "Pos rate:",
    float(y_train.mean()),
)



## === cell 3


def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0)(inputs)  # already normalized but keep harmless
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_model()
model.summary()



## === cell 4
from sklearn.model_selection import train_test_split

X_tr, X_va, y_tr, y_va = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)

history = model.fit(
    X_tr, y_tr, validation_data=(X_va, y_va), epochs=5, batch_size=16, verbose=2
)



## === cell 5
X_test, test_case_ids = load_one_slice_per_case(
    TEST_DIR, modality_index=3, min_sum=100000, min_norm_sum=3000
)
print("Test tensor:", X_test.shape, "Num test ids:", len(test_case_ids))

test_pred = (
    model.predict(X_test, batch_size=16, verbose=0).reshape(-1).astype(np.float32)
)

test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)



## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)

pred_map = dict(zip(test_case_ids, test_pred))

sample_sub["MGMT_value"] = (
    sample_sub["BraTS21ID"].map(pred_map).fillna(0.5).astype(float)
)

sub_df = sample_sub[["BraTS21ID", "MGMT_value"]].copy()
print(sub_df.head())
print(
    "Submission rows:",
    len(sub_df),
    "Missing preds filled:",
    int(sub_df["MGMT_value"].isna().sum()),
)



## === cell 7
try:
    sns.displot(sub_df["MGMT_value"])
    plt.show()
except Exception as e:
    print("Plot skipped:", e)



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
