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

0.50118

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.54118) has done: 'I fix the import-time crash by removing the unused `pympler` dependency that triggers the protobuf `MessageFactory.GetPrototype` error in this environment. Since the external pretrained `.h5` model file is missing, I keep the same “single CNN model → predict 7 slices → average” core logic but train a small Keras CNN on-the-fly using the provided training data so `model` is always defined. I also fix the `resize` NameError by using a local import, correct the submission creation logic so each case gets its own averaged probability (not a vector reused for all rows), and ensure `BraTS21ID` formatting matches the sample submission. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.53647) has done: 'I remove the protobuf-triggering import-time crash by avoiding the `keras` standalone import and using only `tf.keras` throughout (this is score-neutral and fixes the environment error). I fix the training-time `AUC` metric shape error by switching to a binary head (`sigmoid` + `binary_crossentropy`) so the metric receives consistent 1D probabilities. I also fix the submission merge dtype mismatch by forcing both `sample_sub` and predictions to use the same zero-padded string `BraTS21ID` before merging. These changes keep the same core approach (7 T2w slices → CNN → average slice probabilities per case) while making the notebook run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.50118) has done: 'I fix the import-time crash causing the protobuf `MessageFactory.GetPrototype` error by removing the standalone `keras` import/usage and relying exclusively on `tf.keras` (this is score-neutral and unblocks execution). I also remove the dependency on `skimage` by implementing resizing via `tf.image.resize`, since `skimage` may not be available in the Kaggle environment you described. The rest of the pipeline (7 T2w slices → small CNN → average slice probabilities per case → submission merge with sample_submission order) is kept the same to preserve evaluation semantics and keep the score behavior stable while ensuring a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom as dicom

import tensorflow as tf
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



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

labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(int)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

id_to_label = dict(zip(labels_df["BraTS21ID"].values, labels_df["MGMT_value"].values))

BAD_CASES = {109, 123, 709}

print("Train labels:", labels_df.shape, " Sample submission:", sample_sub.shape)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    " Test dir exists:",
    os.path.isdir(TEST_DIR),
)




## === cell 2
def _list_case_dirs(base_dir):
    case_dirs = []
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            name = os.path.basename(entry.path)
            if name.isdigit():
                case_dirs.append(entry.path)
    return sorted(case_dirs)


def _pick_series_dir(case_dir, prefer="T2w"):
    candidates = []
    for entry in os.scandir(case_dir):
        if entry.is_dir():
            candidates.append(entry.path)
    candidates = sorted(candidates)
    for c in candidates:
        if os.path.basename(c).lower() == prefer.lower():
            return c
    for c in candidates:
        if "t2" in os.path.basename(c).lower():
            return c
    return candidates[0] if candidates else None


def _read_dcm_pixel(path):
    ds = dicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    return arr


def _resize_np_image(px2d, out_hw):
    t = tf.convert_to_tensor(px2d[..., None], dtype=tf.float32)  # (H,W,1)
    t = tf.image.resize(t, out_hw, method="bilinear", antialias=True)
    t = tf.squeeze(t, axis=-1)  # (H,W)
    return t.numpy().astype(np.float32)


def load_case_t2_slices(
    case_dir, img_px_size=150, n_slices=7, min_sum=100000, min_norm_sum=2000
):
    """
    Returns list of up to n_slices images (H,W,3) normalized to [0,1].
    Mirrors the original logic: scan all T2w dicoms and pick first 7 passing thresholds.
    """
    series_dir = _pick_series_dir(case_dir, prefer="T2w")
    if series_dir is None or (not os.path.isdir(series_dir)):
        return []

    img_files = sorted(
        [
            f.path
            for f in os.scandir(series_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    out = []
    for fp in img_files:
        try:
            px = _read_dcm_pixel(fp)
        except Exception:
            continue

        if float(np.sum(px)) <= min_sum:
            continue

        px_r = _resize_np_image(px, (img_px_size, img_px_size))
        stacked = np.stack([px_r, px_r, px_r], axis=-1).astype(np.float32)

        denom = float(np.max(stacked))
        if denom <= 0:
            continue
        stacked_norm = stacked / denom

        if float(np.sum(stacked_norm)) <= min_norm_sum:
            continue

        out.append(stacked_norm.astype(np.float32))
        if len(out) >= n_slices:
            break
    return out




## === cell 3
def build_model(input_shape=(150, 150, 3)):
    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 4
IMG_PX_SIZE = 150
N_SLICES = 7

train_case_dirs = _list_case_dirs(TRAIN_DIR)
train_ids = [int(os.path.basename(p)) for p in train_case_dirs]
train_ids = [i for i in train_ids if i in id_to_label and i not in BAD_CASES]

X_list, y_list = [], []
kept_cases = 0

for cid in train_ids:
    case_dir = os.path.join(TRAIN_DIR, f"{cid:05d}")
    slices = load_case_t2_slices(case_dir, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES)
    if len(slices) == 0:
        continue
    lab = float(id_to_label[cid])  # binary_crossentropy expects float labels in [0,1]
    for sl in slices:
        X_list.append(sl)
        y_list.append(lab)
    kept_cases += 1

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.float32)

print(
    "Kept cases:",
    kept_cases,
    " Training slices:",
    X.shape,
    " Labels:",
    y.shape,
    " Pos rate:",
    float(y.mean()) if len(y) else None,
)

can_train = X.shape[0] > 50



## === cell 5
from sklearn.model_selection import train_test_split

if can_train:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=(y.astype(int) if len(np.unique(y)) > 1 else None),
    )

    model = build_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3))

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=3,
        batch_size=32,
        verbose=2,
    )
else:
    model = None
    print(
        "WARNING: Not enough training data extracted to train a model; will fallback to constant predictions."
    )




## === cell 6
def predict_test_cases(model, path_test, img_px_size=150, n_slices=7):
    test_case_dirs = _list_case_dirs(path_test)
    test_ids = [int(os.path.basename(p)) for p in test_case_dirs]

    preds = []
    for cid in test_ids:
        case_dir = os.path.join(path_test, f"{cid:05d}")
        slices = load_case_t2_slices(
            case_dir, img_px_size=img_px_size, n_slices=n_slices
        )

        if model is None or len(slices) == 0:
            preds.append(0.5)
            continue

        arr = np.asarray(slices, dtype=np.float32)
        p = model.predict(arr, verbose=0).reshape(-1)  # per-slice prob
        preds.append(float(np.mean(p)))

    return pd.DataFrame(
        {"BraTS21ID": [f"{i:05d}" for i in test_ids], "MGMT_value": preds}
    )


sub_df = predict_test_cases(model, TEST_DIR, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES)
print(sub_df.head(), sub_df.shape)



## === cell 7
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

print("Final submission shape:", sub_df.shape)
print(sub_df.head())



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_df.columns))
print("submission.csv preview:\n", sub_df.head().to_string(index=False))
