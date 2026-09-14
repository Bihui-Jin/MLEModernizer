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

0.60235

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.60235) has done: 'I fix the import-time crash by removing the unused `pympler` dependency that triggers the protobuf `MessageFactory` error in this environment, and I also remove other unused imports that can break unexpectedly. Since the referenced pretrained `.h5` models are not present, I keep the same “predict probabilities per case and write submission.csv” pipeline but train a lightweight TensorFlow/Keras CNN on-the-fly from the provided train DICOMs using the same kind of T2w slice sampling logic as your loader. I also fix the `resize` NameError by using an always-available resize implementation, and I fix the submission creation logic so it aligns predictions to the correct `BraTS21ID` order and produces exactly one prediction per test subject. Finally, I ensure the code writes a valid `submission.csv` with columns `BraTS21ID,MGMT_value`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom as dicom

import tensorflow as tf
from tensorflow import keras
from keras import layers

SEED = 42
random.seed(SEED)
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
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

BAD_CASES = {"00109", "00123", "00709"}

IMG_PX_SIZE = 150
SLICES_PER_CASE = 4




## === cell 2
def _resize_nn(img2d: np.ndarray, out_hw=(IMG_PX_SIZE, IMG_PX_SIZE)) -> np.ndarray:
    """Nearest-neighbor resize using TensorFlow (available), avoids skimage dependency."""
    x = tf.convert_to_tensor(img2d, dtype=tf.float32)
    x = x[None, ..., None]  # (1,H,W,1)
    x = tf.image.resize(x, out_hw, method="nearest")
    x = tf.squeeze(x, axis=(0, 3))  # (H,W)
    return x.numpy()


def _normalize_to_0_1(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _get_modality_dir(case_dir: str, modality: str = "T2w") -> str:
    mdir = os.path.join(case_dir, modality)
    if not os.path.isdir(mdir):
        raise FileNotFoundError(f"Missing modality directory: {mdir}")
    return mdir


def _list_dcm_files(modality_dir: str):
    files = [
        os.path.join(modality_dir, f)
        for f in os.listdir(modality_dir)
        if f.lower().endswith(".dcm")
    ]
    files.sort()
    return files


def _load_case_slices(
    case_dir: str,
    modality: str = "T2w",
    max_slices: int = SLICES_PER_CASE,
    sum_thresh: float = 100000.0,
    norm_sum_thresh: float = 2500.0,
):
    """
    Load up to `max_slices` informative slices for a case.
    Mimics the original idea: select slices with sufficient pixel intensity sum,
    resize to IMG_PX_SIZE, stack to 3 channels, normalize.
    Returns: list of (H,W,3) float32 in [0,1]
    """
    modality_dir = _get_modality_dir(case_dir, modality)
    dcm_files = _list_dcm_files(modality_dir)
    if not dcm_files:
        return []

    selected = []
    for fp in dcm_files:
        try:
            ds = dicom.dcmread(fp)
            arr = ds.pixel_array
        except Exception:
            continue

        if arr is None:
            continue
        if float(np.sum(arr)) <= sum_thresh:
            continue

        arr_rs = _resize_nn(arr, (IMG_PX_SIZE, IMG_PX_SIZE))
        arr_n = _normalize_to_0_1(arr_rs)
        stacked = np.stack([arr_n, arr_n, arr_n], axis=-1).astype(np.float32)

        if float(np.sum(stacked)) <= norm_sum_thresh:
            continue

        selected.append(stacked)
        if len(selected) >= max_slices:
            break

    return selected




## === cell 3
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

train_case_ids = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
train_case_ids = [cid for cid in train_case_ids if cid not in BAD_CASES]
labels_df = labels_df[labels_df["BraTS21ID"].isin(train_case_ids)].reset_index(
    drop=True
)

sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

len_train_cases = len(labels_df)
len_test_cases = len(sample_sub)
print("Train cases:", len_train_cases, "Test cases:", len_test_cases)




## === cell 4
def build_training_dataset(
    labels_df: pd.DataFrame, train_dir: str, modality: str = "T2w"
):
    """
    Create a slice-level dataset:
    each selected slice inherits the case label.
    This preserves the original pipeline semantics (slice predictions averaged per case).
    """
    X = []
    y = []
    case_counts = 0
    slice_counts = 0

    for _, row in labels_df.iterrows():
        cid = row["BraTS21ID"]
        case_dir = os.path.join(train_dir, cid)
        slices = _load_case_slices(
            case_dir, modality=modality, max_slices=SLICES_PER_CASE
        )
        if not slices:
            continue
        lab = float(row["MGMT_value"])
        for sl in slices:
            X.append(sl)
            y.append(lab)
        case_counts += 1
        slice_counts += len(slices)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    print(
        f"Built training slice dataset: {slice_counts} slices from {case_counts} cases. X shape={X.shape}"
    )
    return X, y


X_all, y_all = build_training_dataset(labels_df, TRAIN_DIR, modality="T2w")

if X_all.shape[0] == 0:
    raise RuntimeError("No training slices were loaded; cannot train model.")



## === cell 5
rng = np.random.RandomState(SEED)
idx = np.arange(len(X_all))
rng.shuffle(idx)

val_frac = 0.15
n_val = max(1, int(len(idx) * val_frac))
val_idx = idx[:n_val]
tr_idx = idx[n_val:]

X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]

print("Train slices:", X_tr.shape, "Val slices:", X_val.shape)




## === cell 6
def make_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = make_model()
model_T2.summary()



## === cell 7
BATCH_SIZE = 32
EPOCHS = 5

history = model_T2.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)

model_inception_v3 = model_T2




## === cell 8
def predict_case_probability(model, case_dir: str, modality: str = "T2w") -> float:
    slices = _load_case_slices(case_dir, modality=modality, max_slices=SLICES_PER_CASE)
    if not slices:
        return 0.5
    X = np.asarray(slices, dtype=np.float32)
    p = model.predict(X, batch_size=16, verbose=0).reshape(-1)
    return float(np.mean(p))


def predict_test(
    model, test_dir: str, sample_sub: pd.DataFrame, modality: str = "T2w"
) -> pd.DataFrame:
    preds = []
    for cid in sample_sub["BraTS21ID"].tolist():
        case_dir = os.path.join(test_dir, cid)
        preds.append(predict_case_probability(model, case_dir, modality=modality))
    out = sample_sub.copy()
    out["MGMT_value"] = np.asarray(preds, dtype=np.float32)
    return out


sub_df = predict_test(model_T2, TEST_DIR, sample_sub, modality="T2w")
sub_df.head()



## === cell 9
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

assert os.path.isfile(sub_path), "submission.csv was not created"
assert list(sub_df.columns) == [
    "BraTS21ID",
    "MGMT_value",
], "Submission columns are incorrect"
assert len(sub_df) == len(sample_sub), "Submission row count mismatch"
print("Wrote", sub_path, "with shape", sub_df.shape)
print(sub_df.describe(include="all"))
