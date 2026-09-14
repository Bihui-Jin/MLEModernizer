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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the immediate import-time crash caused by an incompatible `protobuf`/`tensorflow` interaction by removing unused heavy imports (notably `pympler`) and using `tensorflow.keras` consistently. Since the referenced pretrained models are not present in your provided input paths, I add a minimal fallback that trains a small CNN on the available train DICOMs using the same “7 slices per case” loading idea, so the notebook runs end-to-end and produces a valid `submission.csv`. I also fix `resize`/array normalization bugs (lists divided by scalars, division by zero) and correct the submission creation logic so predictions align 1-to-1 with test cases. These changes preserve the overall approach (CNN on resized DICOM slices, averaged per-case probability) while making it executable in this Kaggle environment.'

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
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())



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

assert os.path.exists(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Missing {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print(labels_df.shape, sample_sub.shape)
labels_df.head()



## === cell 2
IMG_PX_SIZE = 150
N_SLICES = 7


def _safe_norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.max(x))
    if mx <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def _read_dicom_pixel(path: str) -> np.ndarray:
    ds = dicom.dcmread(path)
    arr = ds.pixel_array
    return arr


def _load_case_sequence_slices(
    case_dir: str,
    seq_name: str,
    img_px_size: int = IMG_PX_SIZE,
    n_slices: int = N_SLICES,
):
    seq_dir = os.path.join(case_dir, seq_name)
    if not os.path.isdir(seq_dir):
        return []
    dcm_paths = sorted(
        [
            os.path.join(seq_dir, f)
            for f in os.listdir(seq_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    if not dcm_paths:
        return []
    out = []
    count = 0
    for p in dcm_paths:
        try:
            px = _read_dicom_pixel(p)
        except Exception:
            continue
        if float(np.sum(px)) <= 100000:
            continue
        px_r = resize(
            px, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        stacked = np.stack([px_r, px_r, px_r], axis=-1)
        stacked = _safe_norm01(stacked)
        if float(np.sum(stacked)) <= 2000:
            continue
        out.append(stacked)
        count += 1
        if count >= n_slices:
            break
    return out


def load_cases_7slices_by_sequence(
    root_dir: str,
    seq_name: str,
    img_px_size: int = IMG_PX_SIZE,
    n_slices: int = N_SLICES,
):
    case_dirs = sorted(
        [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
    )
    return case_dirs


def load_dataset_for_training(
    train_dir: str,
    labels: pd.DataFrame,
    seq_name: str = "T2w",
    img_px_size: int = IMG_PX_SIZE,
    n_slices: int = N_SLICES,
):
    bad_ids = set([109, 123, 709])  # per competition note (00109, 00123, 00709)
    id_to_label = dict(
        zip(
            labels["BraTS21ID"].astype(int).values,
            labels["MGMT_value"].astype(int).values,
        )
    )

    X_list = []
    y_list = []

    case_ids = sorted(
        [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    )
    for cid_str in case_ids:
        cid = int(cid_str)
        if cid in bad_ids:
            continue
        if cid not in id_to_label:
            continue
        case_dir = os.path.join(train_dir, cid_str)
        slices = _load_case_sequence_slices(case_dir, seq_name, img_px_size, n_slices)
        if len(slices) == 0:
            continue
        while len(slices) < n_slices:
            slices.append(slices[-1].copy())
        X_list.append(np.stack(slices, axis=0))  # (n_slices, H, W, 3)
        y_list.append(id_to_label[cid])

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, n_slices, H, W, 3)
    y = np.array(y_list).astype(np.float32)
    return X, y


def load_dataset_for_inference(
    test_dir: str,
    seq_name: str = "T2w",
    img_px_size: int = IMG_PX_SIZE,
    n_slices: int = N_SLICES,
):
    case_ids = sorted(
        [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    )
    X_list = []
    kept_ids = []
    for cid_str in case_ids:
        case_dir = os.path.join(test_dir, cid_str)
        slices = _load_case_sequence_slices(case_dir, seq_name, img_px_size, n_slices)
        if len(slices) == 0:
            slices = [
                np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                for _ in range(n_slices)
            ]
        while len(slices) < n_slices:
            slices.append(slices[-1].copy())
        X_list.append(np.stack(slices, axis=0))
        kept_ids.append(int(cid_str))
    X = np.stack(X_list, axis=0).astype(np.float32)
    return kept_ids, X




## === cell 3
def build_slice_cnn(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model




## === cell 4
X_cases, y_cases = load_dataset_for_training(
    TRAIN_DIR, labels_df, seq_name="T2w", img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
print(
    "Train cases:", X_cases.shape, "labels:", y_cases.shape, "pos rate:", y_cases.mean()
)

X_slices = X_cases.reshape((-1, IMG_PX_SIZE, IMG_PX_SIZE, 3))
y_slices = np.repeat(y_cases, N_SLICES)
print("Train slices:", X_slices.shape, y_slices.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1782194121.py in <cell line: 0>()
      1 # Prepare training data from available files (fallback for missing pretrained models).
      2 # Core idea: use T2w slices (the original code heavily used T2w) and train on slices.
----> 3 X_cases, y_cases = load_dataset_for_training(
      4     TRAIN_DIR, labels_df, seq_name="T2w", img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
      5 )

/tmp/ipykernel_11/3724143048.py in load_dataset_for_training(train_dir, labels, seq_name, img_px_size, n_slices)
     99     )
    100     for cid_str in case_ids:
--> 101         cid = int(cid_str)
    102         if cid in bad_ids:
    103             continue

ValueError: invalid literal for int() with base 10: 'train'

## === cell 5
from sklearn.model_selection import train_test_split

idx = np.arange(len(y_cases))
train_idx, val_idx = train_test_split(
    idx, test_size=0.2, random_state=SEED, stratify=y_cases
)

X_train = X_cases[train_idx].reshape((-1, IMG_PX_SIZE, IMG_PX_SIZE, 3))
y_train = np.repeat(y_cases[train_idx], N_SLICES)

X_val = X_cases[val_idx].reshape((-1, IMG_PX_SIZE, IMG_PX_SIZE, 3))
y_val = np.repeat(y_cases[val_idx], N_SLICES)

model = build_slice_cnn((IMG_PX_SIZE, IMG_PX_SIZE, 3))
history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32, verbose=2
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1823890424.py in <cell line: 0>()
      2 from sklearn.model_selection import train_test_split
      3 
----> 4 idx = np.arange(len(y_cases))
      5 train_idx, val_idx = train_test_split(
      6     idx, test_size=0.2, random_state=SEED, stratify=y_cases

NameError: name 'y_cases' is not defined

## === cell 6
test_ids, X_test_cases = load_dataset_for_inference(
    TEST_DIR, seq_name="T2w", img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
X_test_slices = X_test_cases.reshape((-1, IMG_PX_SIZE, IMG_PX_SIZE, 3))

p_slice = model.predict(X_test_slices, batch_size=64, verbose=0).reshape((-1,))
p_case = p_slice.reshape((-1, N_SLICES)).mean(axis=1)

print(
    "Test cases:",
    len(test_ids),
    "Pred shape:",
    p_case.shape,
    "Pred range:",
    (float(p_case.min()), float(p_case.max())),
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/436272764.py in <cell line: 0>()
      1 # Inference on test (predict per-slice, then average per-case).
----> 2 test_ids, X_test_cases = load_dataset_for_inference(
      3     TEST_DIR, seq_name="T2w", img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
      4 )
      5 X_test_slices = X_test_cases.reshape((-1, IMG_PX_SIZE, IMG_PX_SIZE, 3))

/tmp/ipykernel_11/3724143048.py in load_dataset_for_inference(test_dir, seq_name, img_px_size, n_slices)
    142             slices.append(slices[-1].copy())
    143         X_list.append(np.stack(slices, axis=0))
--> 144         kept_ids.append(int(cid_str))
    145     X = np.stack(X_list, axis=0).astype(np.float32)
    146     return kept_ids, X

ValueError: invalid literal for int() with base 10: 'test'

## === cell 7
sub = sample_sub.copy()
sub["BraTS21ID"] = sub["BraTS21ID"].astype(int)

pred_map = {i: float(p) for i, p in zip(test_ids, p_case)}
sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map)

sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(float)

sub["MGMT_value"] = sub["MGMT_value"].clip(0.0, 1.0)

sub.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/427606656.py in <cell line: 0>()
      3 sub["BraTS21ID"] = sub["BraTS21ID"].astype(int)
      4 
----> 5 pred_map = {i: float(p) for i, p in zip(test_ids, p_case)}
      6 sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map)
      7 

NameError: name 'test_ids' is not defined

## === cell 8
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.describe(include="all"))
