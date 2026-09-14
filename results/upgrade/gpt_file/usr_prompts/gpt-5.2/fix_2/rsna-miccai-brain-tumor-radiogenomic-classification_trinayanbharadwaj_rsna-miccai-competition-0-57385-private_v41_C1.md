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

0.59647

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.59647) has done: 'I remove/guard the imports that trigger the protobuf `MessageFactory.GetPrototype` crash and also drop dependencies that aren’t needed to generate a submission. Since the referenced pre-trained model files don’t exist in your environment, I keep the same “image → model.predict → average → submission” pipeline but build and quickly train the same small CNN architecture directly on the provided training set, then run inference on test. I also fix the missing `resize`/`randrange`/`model_*` NameErrors, correct the submission ID formatting to match `sample_submission.csv`, and ensure predictions are aligned 1:1 with test cases. Finally, the script always write a valid `submission.csv` with the required columns.'

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
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

IMG_PX_SIZE = 299  # preserve original image size expectation
CHANNELS = 3

BAD_CASES = {"00109", "00123", "00709"}


def list_case_dirs(path_dir):
    return sorted([f.path for f in os.scandir(path_dir) if f.is_dir()])


def case_id_from_path(p):
    return os.path.basename(p)


def safe_normalize(img):
    mx = np.max(img)
    if mx <= 0:
        return img.astype(np.float32)
    return (img / mx).astype(np.float32)


def load_one_series_rep(case_path, series_name):
    """
    Load a single representative DICOM slice from a specific series folder.
    Preserves core logic: choose the first slice whose pixel sum > 100000,
    then resize to 299x299 and stack into 3 channels, normalized.
    """
    series_path = os.path.join(case_path, series_name)
    if not os.path.isdir(series_path):
        return None

    dcm_paths = sorted(
        [
            f.path
            for f in os.scandir(series_path)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    for p in dcm_paths:
        try:
            img = dicom.dcmread(p)
            arr = img.pixel_array
        except Exception:
            continue

        if arr is None:
            continue
        if np.sum(arr) > 100000:
            resized_img = resize(
                arr, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True, preserve_range=True
            )
            resized_img = np.array(resized_img, dtype=np.float32)
            stacked = np.stack((resized_img,) * 3, axis=-1)
            stacked = safe_normalize(stacked)
            if np.sum(stacked) > 10000:
                return stacked
    return None


def load_dataset_images(base_dir, series_name, allowed_ids=None, max_cases=None):
    """
    Returns:
      X: float32 array [N, 299, 299, 3]
      ids: list of case ids strings (zero-padded)
    """
    X = []
    ids = []
    for case_path in list_case_dirs(base_dir):
        cid = case_id_from_path(case_path)
        if allowed_ids is not None and cid not in allowed_ids:
            continue
        if base_dir == TRAIN_DIR and cid in BAD_CASES:
            continue

        img = load_one_series_rep(case_path, series_name)
        if img is None:
            series_path = os.path.join(case_path, series_name)
            if os.path.isdir(series_path):
                dcm_paths = sorted(
                    [
                        f.path
                        for f in os.scandir(series_path)
                        if f.is_file() and f.name.lower().endswith(".dcm")
                    ]
                )
                for p in dcm_paths:
                    try:
                        arr = dicom.dcmread(p).pixel_array
                        resized_img = resize(
                            arr,
                            (IMG_PX_SIZE, IMG_PX_SIZE),
                            anti_aliasing=True,
                            preserve_range=True,
                        )
                        resized_img = np.array(resized_img, dtype=np.float32)
                        stacked = np.stack((resized_img,) * 3, axis=-1)
                        img = safe_normalize(stacked)
                        break
                    except Exception:
                        continue

        if img is None:
            img = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS), dtype=np.float32)

        X.append(img)
        ids.append(cid)

        if max_cases is not None and len(ids) >= max_cases:
            break

    X = np.stack(X, axis=0).astype(np.float32)
    gmx = np.max(X)
    if gmx > 0:
        X = X / gmx
    return X, ids




## === cell 2
labels_df = pd.read_csv(TRAIN_LABELS_CSV, dtype={"BraTS21ID": str})
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

train_ids_set = set(labels_df["BraTS21ID"].tolist())

X_flair, ids_flair = load_dataset_images(TRAIN_DIR, "FLAIR", allowed_ids=train_ids_set)
X_t2, ids_t2 = load_dataset_images(TRAIN_DIR, "T2w", allowed_ids=train_ids_set)

flair_map = {cid: X_flair[i] for i, cid in enumerate(ids_flair)}
t2_map = {cid: X_t2[i] for i, cid in enumerate(ids_t2)}

common_ids = sorted(list(set(flair_map.keys()) & set(t2_map.keys())))
y = (
    labels_df.set_index("BraTS21ID")
    .loc[common_ids, "MGMT_value"]
    .values.astype(np.float32)
)

X = np.stack(
    [np.concatenate([flair_map[cid], t2_map[cid]], axis=-1) for cid in common_ids],
    axis=0,
).astype(np.float32)

print("Train shape:", X.shape, "Labels shape:", y.shape)



## === cell 3
from sklearn.model_selection import train_test_split

strat = y if len(np.unique(y)) > 1 else None
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=strat
)

inputs = keras.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, X.shape[-1]))

x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=8, verbose=2
)



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB_CSV, dtype={"BraTS21ID": str})
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].str.zfill(5)
test_ids_order = sample_sub["BraTS21ID"].tolist()
test_ids_set = set(test_ids_order)

X_test_flair, ids_test_flair = load_dataset_images(
    TEST_DIR, "FLAIR", allowed_ids=test_ids_set
)
X_test_t2, ids_test_t2 = load_dataset_images(TEST_DIR, "T2w", allowed_ids=test_ids_set)

flair_map_t = {cid: X_test_flair[i] for i, cid in enumerate(ids_test_flair)}
t2_map_t = {cid: X_test_t2[i] for i, cid in enumerate(ids_test_t2)}

X_test = []
for cid in test_ids_order:
    f = flair_map_t.get(cid, np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32))
    t = t2_map_t.get(cid, np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32))
    X_test.append(np.concatenate([f, t], axis=-1))
X_test = np.stack(X_test, axis=0).astype(np.float32)

pred = model.predict(X_test, batch_size=8, verbose=0).reshape(-1).astype(np.float32)
pred = np.clip(pred, 0.0, 1.0)

sub_df = pd.DataFrame({"BraTS21ID": test_ids_order, "MGMT_value": pred})
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
