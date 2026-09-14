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

0.48235

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.48235) has done: 'I remove/guard problematic imports that trigger the protobuf `MessageFactory` error and keep only what’s needed for inference. Because the referenced pretrained `.h5` file is not present in your environment, I add a minimal fallback that trains the same simple pipeline idea (single-slice per case → 3-channel image → CNN classifier) directly from the provided training data, while excluding the known-bad cases. I fix the missing `resize` import by implementing resizing via PIL (no extra dependencies) and make the test image loading deterministic and robust to empty/invalid DICOM slices. Finally, I ensure the submission has the exact required columns, correct `BraTS21ID` formatting (5-digit strings), and is written to `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import pydicom
from PIL import Image

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
def resize_to_square(arr2d: np.ndarray, size: int) -> np.ndarray:
    arr2d = np.asarray(arr2d)
    if arr2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr2d.shape}")
    img = Image.fromarray(arr2d.astype(np.float32))
    img = img.resize((size, size), resample=Image.BILINEAR)
    out = np.asarray(img, dtype=np.float32)
    return out


def normalize_01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)




## === cell 2
IMG_PX_SIZE = 299


def load_one_case_image(
    case_dir: str, img_size: int = IMG_PX_SIZE, prefer_sequence: str = "FLAIR"
) -> np.ndarray:
    """
    Core logic preserved: pick first MRI sequence folder, scan slices until a non-empty slice is found,
    then resize to 299x299 and return.
    Robustness improvements: handle DICOM read errors and empty sequences.
    """
    seq_dirs = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
    if not seq_dirs:
        raise FileNotFoundError(f"No modality folders in {case_dir}")

    chosen = None
    for d in seq_dirs:
        if os.path.basename(d).upper() == prefer_sequence.upper():
            chosen = d
            break
    if chosen is None:
        chosen = seq_dirs[0]

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(chosen)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files in {chosen}")

    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp, force=True)
            pix = ds.pixel_array.astype(np.float32)
        except Exception:
            continue

        if np.nansum(pix) > 1000:
            pix = normalize_01(pix)
            pix = resize_to_square(pix, img_size)
            pix = normalize_01(pix)
            return pix

    mid_fp = dcm_files[len(dcm_files) // 2]
    ds = pydicom.dcmread(mid_fp, force=True)
    pix = normalize_01(ds.pixel_array.astype(np.float32))
    pix = resize_to_square(pix, img_size)
    pix = normalize_01(pix)
    return pix


def load_images_for_ids(
    root_dir: str, ids: list, img_size: int = IMG_PX_SIZE
) -> np.ndarray:
    arr = []
    for brats_id in ids:
        case_dir = os.path.join(root_dir, f"{int(brats_id):05d}")
        img = load_one_case_image(case_dir, img_size=img_size)
        arr.append(img)
    arr = np.stack(arr, axis=0).astype(np.float32)
    return arr




## === cell 3
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

labels_df = pd.read_csv(LABELS_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

bad_ids = {109, 123, 709}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

train_ids = labels_df["BraTS21ID"].tolist()
train_y = labels_df["MGMT_value"].astype(np.float32).values

test_ids = sample_df["BraTS21ID"].tolist()



## === cell 4
pretrained_path = (
    "/kaggle/input/trained-model-for-rsnamiccai/model_rsna_miccai_100epochs.h5"
)
model = None

if os.path.exists(pretrained_path):
    model = keras.models.load_model(pretrained_path)
else:
    def build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)):
        inputs = keras.Input(shape=input_shape)
        x = layers.Rescaling(1.0)(inputs)
        x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.25)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        m = keras.Model(inputs, outputs)
        m.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )
        return m

    model = build_model()




## === cell 5
def to_rgb_batch(gray_batch: np.ndarray) -> np.ndarray:
    gray_batch = gray_batch.reshape((-1, IMG_PX_SIZE, IMG_PX_SIZE, 1)).astype(
        np.float32
    )
    rgb = np.repeat(gray_batch, 3, axis=-1)
    return rgb


if not os.path.exists(pretrained_path):
    n = len(train_ids)
    idx = np.arange(n)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    split = int(n * 0.85)
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_ids = [train_ids[i] for i in tr_idx]
    va_ids = [train_ids[i] for i in va_idx]
    y_tr = train_y[tr_idx]
    y_va = train_y[va_idx]

    x_tr_gray = load_images_for_ids(TRAIN_DIR, tr_ids, img_size=IMG_PX_SIZE)
    x_va_gray = load_images_for_ids(TRAIN_DIR, va_ids, img_size=IMG_PX_SIZE)

    x_tr = to_rgb_batch(x_tr_gray)
    x_va = to_rgb_batch(x_va_gray)

    model.fit(
        x_tr, y_tr, validation_data=(x_va, y_va), epochs=3, batch_size=8, verbose=2
    )



## === cell 6
x_test_gray = load_images_for_ids(TEST_DIR, test_ids, img_size=IMG_PX_SIZE)
rgb_batch_test = to_rgb_batch(x_test_gray)

preds = model.predict(rgb_batch_test, batch_size=8, verbose=0)
preds = preds.reshape(-1).astype(np.float32)



## === cell 7
sub_df = pd.DataFrame(
    {"BraTS21ID": [f"{int(i):05d}" for i in test_ids], "MGMT_value": preds}
)

sub_df["MGMT_value"] = sub_df["MGMT_value"].clip(0.0, 1.0)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

print(sub_df.head())
print(f"Saved submission to: {sub_path}, shape={sub_df.shape}")
