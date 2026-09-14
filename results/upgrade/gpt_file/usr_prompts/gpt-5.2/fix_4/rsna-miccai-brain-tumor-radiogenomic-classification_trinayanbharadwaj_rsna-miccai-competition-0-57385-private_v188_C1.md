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

# 8. Previous improvement plans

- What this solution (achieved 0.44941) has done: 'I fix the initial import/runtime crash by removing the unnecessary `pympler` import that triggers the protobuf `MessageFactory.GetPrototype` error, and make the `resize` function available reliably at runtime. Because the referenced pre-trained model files are not present in this Kaggle environment, I keep the same “predict-probabilities then average” core logic but switch to a lightweight fallback that trains a small CNN on-the-fly using the same DICOM slice sampling approach, so the notebook can run end-to-end and output `submission.csv`. I also fix the submission-creation logic bug where `prediction` was overwritten inside the loop (and lengths could mismatch), ensuring predictions align 1:1 with test IDs. Finally, I ensure the saved CSV has the exact required columns (`BraTS21ID`, `MGMT_value`) and uses the provided sample submission ordering.'
- What this solution (achieved 0.62706) has done: 'I fix the current runtime crash by avoiding the protobuf-incompatible `pydicom` import path and using `pydicom.dcmread(..., force=True)` with a safe fallback when pixel decoding fails, so the pipeline always runs end-to-end. I also remove the hard dependency on `sklearn` (not guaranteed installed) by implementing a tiny stratified split with NumPy, keeping the same training/validation semantics. Finally, I keep your same “per-slice predict then average across modalities” logic, but make DICOM file ordering numeric (Image-2 before Image-10), which is a score-positive but minimal correctness fix for consistent slice sampling and typically improves AUC.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash (`MessageFactory` protobuf incompatibility) by removing the direct `pydicom` dependency and switching DICOM reading to OpenCV’s built-in DICOM loader (`cv2.dcmread`), which is available since you already use `cv2`. This keeps your core logic identical: sample a fixed number of slices per modality, resize/normalize to RGB, train the same small CNN, and average per-slice probabilities into a per-case probability. I also add a tiny safety guard in case a DICOM file cannot be decoded, so the pipeline never fails mid-run and always writes `submission.csv` with the required columns/order. No score-target tuning is applied because the provided target score (-1.0) is not a meaningful AUC destination; changes are focused on correctness/stability while preserving semantics.'

# 9. Code solution

## === cell 0
import os
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers

import cv2

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)



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

train_labels = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

bad_ids = {109, 123, 709}
train_labels = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].reset_index(
    drop=True
)

train_labels.head(), sample_sub.head()




## === cell 2
def _natural_key(path):
    """
    Ensure Image-2.dcm comes before Image-10.dcm.
    This stabilizes slice selection vs lexicographic sorting.
    """
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    return int(nums[-1]) if nums else 0


def _safe_read_dicom_pixel(path):
    """
    Bug fix: remove pydicom dependency that can crash in this environment due to
    protobuf incompatibilities. Use OpenCV DICOM reader instead.

    Returns float32 2D pixel array or None.
    """
    try:
        arr = cv2.dcmread(path)
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                return None
        arr = arr.astype(np.float32, copy=False)
        return arr
    except Exception:
        return None


def _resize_to_rgb(arr2d, img_px_size=150):
    """Resize 2D array to (img_px_size, img_px_size, 3) normalized to [0,1]."""
    resized = cv2.resize(
        arr2d, (img_px_size, img_px_size), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    mx = float(np.max(resized)) if resized.size else 0.0
    if mx > 0:
        resized = resized / mx
    rgb = np.stack([resized, resized, resized], axis=-1)
    return rgb


def load_case_slices(path_case, series_name, n_slices=7, img_px_size=150):
    """
    Load up to n_slices from a specific MRI series folder under a case, using similar
    filtering logic (pixel sum threshold). Pads with zeros if not enough slices.
    """
    series_dir = os.path.join(path_case, series_name)
    zero = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
    if not os.path.isdir(series_dir):
        return [zero] * n_slices

    files = [
        os.path.join(series_dir, f)
        for f in os.listdir(series_dir)
        if f.lower().endswith(".dcm")
    ]
    files = sorted(files, key=_natural_key)

    selected = []
    for fp in files:
        arr = _safe_read_dicom_pixel(fp)
        if arr is None:
            continue
        if float(arr.sum()) <= 100000:
            continue
        img = _resize_to_rgb(arr, img_px_size=img_px_size)
        if float(img.sum()) <= 2000:
            continue
        selected.append(img)
        if len(selected) >= n_slices:
            break

    if len(selected) < n_slices:
        selected = selected + [zero] * (n_slices - len(selected))
    return selected[:n_slices]




## === cell 3
def build_slice_model(input_shape=(150, 150, 3)):
    """
    Minimal CNN (fallback) to produce a probability.
    Preserves the same semantics: predict per-slice probability then average.
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
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model




## === cell 4
IMG_PX_SIZE = 150
N_SLICES = 7
MODALITIES = ["T2w", "FLAIR", "T1wCE"]


def make_training_data(train_dir, labels_df, max_cases=None):
    X = []
    y = []
    used = 0
    for brats_id, target in labels_df[["BraTS21ID", "MGMT_value"]].itertuples(
        index=False
    ):
        case_dir = os.path.join(train_dir, f"{int(brats_id):05d}")
        if not os.path.isdir(case_dir):
            continue
        for mod in MODALITIES:
            slices = load_case_slices(
                case_dir, mod, n_slices=N_SLICES, img_px_size=IMG_PX_SIZE
            )
            X.extend(slices)
            y.extend([float(target)] * len(slices))
        used += 1
        if max_cases is not None and used >= max_cases:
            break
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y, used


X_train, y_train, used_cases = make_training_data(
    TRAIN_DIR, train_labels, max_cases=160
)
X_train.shape, y_train.shape, used_cases




## === cell 5
def stratified_train_val_split(X, y, test_size=0.2, seed=42):
    """
    Minimal stratified split preserving original intent.
    """
    rng = np.random.default_rng(seed)
    y_int = (y >= 0.5).astype(int)

    idx0 = np.where(y_int == 0)[0]
    idx1 = np.where(y_int == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_val = int(round(len(idx0) * test_size))
    n1_val = int(round(len(idx1) * test_size))

    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    tr_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])
    rng.shuffle(val_idx)
    rng.shuffle(tr_idx)

    return X[tr_idx], X[val_idx], y[tr_idx], y[val_idx]


X_tr, X_va, y_tr, y_va = stratified_train_val_split(
    X_train, y_train, test_size=0.2, seed=SEED
)

model = build_slice_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))

history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_va, y_va),
    epochs=3,
    batch_size=32,
    verbose=2,
)




## === cell 6
def predict_case_probability(model, path_case):
    """
    Predict per-slice probabilities for each modality and average them.
    """
    probs = []
    for mod in MODALITIES:
        slices = load_case_slices(
            path_case, mod, n_slices=N_SLICES, img_px_size=IMG_PX_SIZE
        )
        X = np.asarray(slices, dtype=np.float32)
        p = model.predict(X, verbose=0).reshape(-1)
        probs.append(p)
    probs = np.concatenate(probs, axis=0)
    return float(np.mean(probs))




## === cell 7
test_ids = sample_sub["BraTS21ID"].tolist()

preds = []
for brats_id in test_ids:
    case_dir = os.path.join(TEST_DIR, f"{int(brats_id):05d}")
    preds.append(predict_case_probability(model, case_dir))

sub_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": preds})
sub_df.head(), sub_df.shape



## === cell 8
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).clip(0.0, 1.0)
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].apply(lambda x: f"{int(x):05d}")

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
