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

0.31882

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.31882) has done: 'I (1) remove the import(s) that trigger the protobuf `MessageFactory.GetPrototype` crash, (2) replace the missing external pretrained model load with a small fallback Keras CNN that can be trained from the provided training DICOMs (same “predict probability” semantics), and (3) fix the image-loading pipeline so it deterministically loads T2w slices, returns proper NumPy arrays, and never references undefined names. I also fix `create_sub` so it averages per-case predictions correctly (previously it overwrote `prediction` inside the loop and produced a length mismatch) and ensure the submission uses the exact required columns and ID formatting. These changes are the minimum needed to run end-to-end and produce a valid `submission.csv`, and the trained model should score better than “no submission”.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
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
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

bad_cases = set(["00109", "00123", "00709"])

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)
labels_df.head()




## === cell 2
def _list_case_ids(base_dir):
    case_ids = []
    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name):
        if entry.is_dir():
            case_ids.append(entry.name)
    return case_ids


def _safe_dcmread(path):
    try:
        ds = dicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def load_case_slices_t2w(case_dir, img_px_size=150, n_slices=6):
    """
    Minimal fix of the original logic:
    - Use the T2w folder explicitly (instead of mri_type[3], which is order-dependent).
    - Select evenly spaced slices across the series for robustness and deterministic behavior.
    - Return exactly n_slices RGB(3-channel) images normalized to [0,1].
    """
    t2_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2_dir):
        candidates = [
            f.path
            for f in os.scandir(case_dir)
            if f.is_dir() and "t2" in f.name.lower()
        ]
        if not candidates:
            return None
        t2_dir = sorted(candidates)[0]

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(t2_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    if len(dcm_files) == 0:
        return None

    idxs = np.linspace(0, len(dcm_files) - 1, num=n_slices).round().astype(int)

    imgs = []
    for idx in idxs:
        arr = _safe_dcmread(dcm_files[idx])
        if arr is None:
            return None
        arr_resized = resize(
            arr, (img_px_size, img_px_size), anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
        mx = float(arr_resized.max())
        if mx > 0:
            arr_resized = arr_resized / mx
        arr_rgb = np.stack([arr_resized, arr_resized, arr_resized], axis=-1)  # (H,W,3)
        imgs.append(arr_rgb)

    return np.stack(imgs, axis=0)  # (n_slices,H,W,3)


def load_dataset_from_ids(
    base_dir, case_ids, labels_df=None, img_px_size=150, n_slices=6
):
    """
    Returns:
      X: (N, n_slices, H, W, 3)
      y: (N,) if labels_df is provided else None
      kept_ids: list of case IDs kept
    """
    X_list = []
    y_list = []
    kept_ids = []

    y_map = None
    if labels_df is not None:
        y_map = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))

    for cid in case_ids:
        if cid in bad_cases and base_dir.endswith("train"):
            continue
        case_dir = os.path.join(base_dir, cid)
        vol = load_case_slices_t2w(case_dir, img_px_size=img_px_size, n_slices=n_slices)
        if vol is None:
            continue
        X_list.append(vol)
        kept_ids.append(cid)
        if y_map is not None:
            y_list.append(float(y_map[cid]))

    X = (
        np.stack(X_list, axis=0).astype(np.float32)
        if len(X_list)
        else np.zeros((0, n_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    )
    y = np.array(y_list, dtype=np.float32) if y_map is not None else None
    return X, y, kept_ids




## === cell 3
IMG_PX_SIZE = 150
N_SLICES = 6


def build_model(img_px_size=150, n_slices=6):
    inp = keras.Input(shape=(n_slices, img_px_size, img_px_size, 3))
    x = layers.TimeDistributed(layers.Conv2D(16, 3, padding="same", activation="relu"))(
        inp
    )
    x = layers.TimeDistributed(layers.MaxPool2D())(x)
    x = layers.TimeDistributed(layers.Conv2D(32, 3, padding="same", activation="relu"))(
        x
    )
    x = layers.TimeDistributed(layers.MaxPool2D())(x)
    x = layers.TimeDistributed(layers.Conv2D(64, 3, padding="same", activation="relu"))(
        x
    )
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(
        x
    )  # (batch, n_slices, channels)
    x = layers.GlobalAveragePooling1D()(x)  # aggregate slices
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = build_model(IMG_PX_SIZE, N_SLICES)
model_T2.summary()



## === cell 4
train_ids_all = labels_df["BraTS21ID"].tolist()

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_ids_all))
split = int(0.85 * len(train_ids_all))
train_ids = [train_ids_all[i] for i in perm[:split]]
val_ids = [train_ids_all[i] for i in perm[split:]]

X_train, y_train, kept_train = load_dataset_from_ids(
    TRAIN_DIR,
    train_ids,
    labels_df=labels_df,
    img_px_size=IMG_PX_SIZE,
    n_slices=N_SLICES,
)
X_val, y_val, kept_val = load_dataset_from_ids(
    TRAIN_DIR, val_ids, labels_df=labels_df, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)

print("Train:", X_train.shape, y_train.shape, "Val:", X_val.shape, y_val.shape)



## === cell 5
BATCH_SIZE = 4
EPOCHS = 3  # minimal epochs to finish in the 600s limit while producing non-trivial predictions

if len(X_train) == 0 or len(X_val) == 0:
    raise RuntimeError(
        "No training/validation data was loaded. Check DICOM reading and paths."
    )

history = model_T2.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 6
test_ids = _list_case_ids(TEST_DIR)
X_test, _, kept_test_ids = load_dataset_from_ids(
    TEST_DIR, test_ids, labels_df=None, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)

print("Test:", X_test.shape, "Kept:", len(kept_test_ids), "of", len(test_ids))

test_pred = (
    model_T2.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    .reshape(-1)
    .astype(np.float64)
)

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 7
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

pred_map = dict(zip(kept_test_ids, test_pred))

mgmt_vals = []
for cid in sample_sub["BraTS21ID"].tolist():
    mgmt_vals.append(float(pred_map.get(cid, 0.5)))

sub_df = pd.DataFrame({"BraTS21ID": sample_sub["BraTS21ID"], "MGMT_value": mgmt_vals})
sub_df.head()



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.describe(include="all"))
