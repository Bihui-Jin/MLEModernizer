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

0.55294

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.55294) has done: 'I fix the import/runtime crash by removing incompatible optional imports (pympler/seaborn/matplotlib/cv2) and by using `tensorflow.keras` consistently to avoid Keras/TensorFlow version mismatches that trigger the protobuf `MessageFactory` error. Because the referenced pretrained `.h5` models are not present in your `/kaggle/input` tree, I keep the same ensemble-style “predict-on-6-slices then average” core logic but train a small CNN locally on the provided training data and then run inference on the test set. I also fix the missing `resize` symbol and ensure arrays are converted to NumPy before normalization to prevent list/float errors. Finally, I correct submission alignment/format (5-digit `BraTS21ID`, one prediction per case) and always write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pydicom
from skimage.transform import resize

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

BAD_CASES = {"00109", "00123", "00709"}

IMG_PX_SIZE = 150
N_SLICES = 6  # keep same "6 representative images per case" semantics as original

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

sample_sub_df = pd.read_csv(SAMPLE_SUB)
sample_sub_df["BraTS21ID"] = sample_sub_df["BraTS21ID"].astype(str).str.zfill(5)

print("Train cases:", len(labels_df), " Test cases:", len(sample_sub_df))




## === cell 2
def _safe_dcm_pixel_array(dcm_path: str):
    """Read DICOM and return pixel_array as float32, or None if unreadable."""
    try:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def _normalize_to_unit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.max(x))
    if not np.isfinite(mx) or mx < eps:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def _case_modality_dir(case_dir: str, modality: str = "T2w") -> str:
    return os.path.join(case_dir, modality)


def load_case_slices(
    case_dir: str, modality: str = "T2w", img_px_size: int = 150, n_slices: int = 6
):
    """
    Load up to n_slices "informative" slices from a case/modality folder.
    Fallbacks ensure exactly n_slices are returned (by padding with last/zeros).
    Output shape: (n_slices, img_px_size, img_px_size, 3) in [0,1].
    """
    modality_dir = _case_modality_dir(case_dir, modality)
    if not os.path.isdir(modality_dir):
        return np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)

    dcm_files = sorted(
        [
            os.path.join(modality_dir, f)
            for f in os.listdir(modality_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    if len(dcm_files) == 0:
        return np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)

    chosen = []
    count = 0
    for fp in dcm_files:
        arr = _safe_dcm_pixel_array(fp)
        if arr is None:
            continue
        if float(np.sum(arr)) > 100000.0:
            resized = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked = np.stack([resized, resized, resized], axis=-1)
            stacked = _normalize_to_unit(stacked)
            if float(np.sum(stacked)) > 2000.0:
                chosen.append(stacked)
                count += 1
                if count >= n_slices:
                    break

    if len(chosen) == 0:
        idxs = np.linspace(
            0, len(dcm_files) - 1, num=min(n_slices, len(dcm_files)), dtype=int
        )
        for idx in idxs:
            arr = _safe_dcm_pixel_array(dcm_files[int(idx)])
            if arr is None:
                continue
            resized = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked = np.stack([resized, resized, resized], axis=-1)
            chosen.append(_normalize_to_unit(stacked))
            if len(chosen) >= n_slices:
                break

    if len(chosen) == 0:
        chosen = [np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)]

    while len(chosen) < n_slices:
        chosen.append(chosen[-1].copy())

    return np.stack(chosen[:n_slices], axis=0).astype(np.float32)




## === cell 3
def build_model(img_px_size: int = 150):
    inputs = keras.Input(shape=(img_px_size, img_px_size, 3))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_model(IMG_PX_SIZE)
model.summary()




## === cell 4
def make_training_arrays(
    labels_dataframe: pd.DataFrame, train_dir: str, modality: str = "T2w"
):
    """
    Convert case-level labels into slice-level training data:
    X shape: (num_cases*n_slices, H, W, 3)
    y shape: (num_cases*n_slices,)
    """
    X_list = []
    y_list = []
    missing = 0
    for _, row in labels_dataframe.iterrows():
        case_id = str(row["BraTS21ID"]).zfill(5)
        case_dir = os.path.join(train_dir, case_id)
        if not os.path.isdir(case_dir):
            missing += 1
            continue
        slices = load_case_slices(
            case_dir, modality=modality, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
        )
        y = float(row["MGMT_value"])
        X_list.append(slices)
        y_list.append(np.full((N_SLICES,), y, dtype=np.float32))

    if len(X_list) == 0:
        raise RuntimeError("No training data found. Check paths and dataset mounting.")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    print("Built training arrays:", X.shape, y.shape, " missing_cases:", missing)
    return X, y


X, y = make_training_arrays(labels_df, TRAIN_DIR, modality="T2w")



## === cell 5
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
)
val_datagen = keras.preprocessing.image.ImageDataGenerator()

train_gen = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True, seed=SEED)
val_gen = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)

EPOCHS = 3
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, verbose=2)




## === cell 6
def predict_case_probability(case_dir: str, modality: str = "T2w") -> float:
    slices = load_case_slices(
        case_dir, modality=modality, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
    )
    preds = model.predict(slices, verbose=0).reshape(-1)  # per-slice probs
    p = float(np.mean(preds))
    if not np.isfinite(p):
        p = 0.5
    return float(np.clip(p, 0.0, 1.0))


test_case_ids = sample_sub_df["BraTS21ID"].tolist()
preds = []
for cid in test_case_ids:
    case_dir = os.path.join(TEST_DIR, cid)
    preds.append(predict_case_probability(case_dir, modality="T2w"))

preds = np.asarray(preds, dtype=np.float32)
print("Pred summary:", float(preds.min()), float(preds.max()), float(preds.mean()))



## === cell 7
sub_df = pd.DataFrame({"BraTS21ID": test_case_ids, "MGMT_value": preds})
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample_sub_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape", sub_df.shape)
print(sub_df.head())



## === cell 8
assert os.path.isfile("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["BraTS21ID", "MGMT_value"]
assert chk["MGMT_value"].between(0, 1).all()
assert chk.shape[0] == sample_sub_df.shape[0]
print("Submission looks valid.")
