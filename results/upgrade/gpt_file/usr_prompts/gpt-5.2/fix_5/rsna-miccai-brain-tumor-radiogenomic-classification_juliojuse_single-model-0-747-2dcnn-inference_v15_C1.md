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

- What this solution (achieved 0.5) has done: 'I fix the runtime crash caused by the `pydicom` import (it’s triggering a protobuf incompatibility) by replacing DICOM loading with a TensorFlow-only PNG decode path that reads the already-extracted PNG slices in the competition folders. Then I fix the submission merge bug that drops/renames `MGMT_value` (causing the `KeyError`) by explicitly controlling suffixes and writing into a new prediction column before finalizing `MGMT_value`. These changes keep your core approach identical (single-sequence slice model → slice predictions → patient mean) while making the notebook run end-to-end and always produce `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I fix the runtime crash coming from an indirect protobuf/TensorFlow compatibility issue by removing the dependency on `sklearn` (which can trigger that protobuf `MessageFactory` error in some Kaggle images) and replacing the split with a small, deterministic, stratified split implemented in pure pandas/numpy. This keeps your core pipeline identical: same slice extraction, same model, same training loop, same patient-mean aggregation, and the same submission writing. I also add a tiny guard to ensure we always train on at least one slice and always write a valid `submission.csv` with the required columns. These changes are score-neutral in intent but make the notebook run end-to-end reliably; training/validation split remains stratified and deterministic.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

train_df = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)

IMAGE_SIZE = 128
SEQ = "T1wCE"  # keep same single-sequence intent as original code


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Uses extracted PNG slices (preferred) present in the Kaggle dataset.
    Falls back to listing .dcm if no PNG exists (but DICOM decode is not attempted later).
    Keeps original intent: take middle 50% and sample by interval.
    """
    assert image_type in TYPES
    base = TRAIN_DIR if folder == "train" else TEST_DIR
    patient_path = os.path.join(base, str(int(brats21id)).zfill(5), image_type)

    png_paths = sorted(
        glob.glob(os.path.join(patient_path, "*.png")),
        key=lambda x: int(os.path.basename(x)[:-4].split("-")[-1]),
    )
    paths = png_paths
    if len(paths) == 0:
        dcm_paths = sorted(
            glob.glob(os.path.join(patient_path, "*.dcm")),
            key=lambda x: int(os.path.basename(x)[:-4].split("-")[-1]),
        )
        paths = dcm_paths

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)
    interval = 3 if num_images >= 10 else 1
    return np.array(paths[start:end:interval], dtype=object)


def load_image_tf(path, size=128):
    """
    Decode PNG/JPEG via TF; if a .dcm path appears, return zeros instead of crashing.
    """
    path = tf.convert_to_tensor(path)
    is_png = tf.strings.regex_full_match(tf.strings.lower(path), ".*\\.png$")
    is_jpg = tf.strings.regex_full_match(tf.strings.lower(path), ".*\\.(jpg|jpeg)$")

    def _decode_png():
        b = tf.io.read_file(path)
        img = tf.image.decode_png(b, channels=1)  # (H,W,1), uint8
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, (size, size), method="bilinear")
        img = tf.squeeze(img, axis=-1)  # (H,W)
        return img

    def _decode_jpg():
        b = tf.io.read_file(path)
        img = tf.image.decode_jpeg(b, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (size, size), method="bilinear")
        img = tf.squeeze(img, axis=-1)
        return img

    def _fallback_zero():
        return tf.zeros((size, size), dtype=tf.float32)

    img = tf.case(
        [(is_png, _decode_png), (is_jpg, _decode_jpg)],
        default=_fallback_zero,
        exclusive=True,
    )
    img = tf.ensure_shape(img, (size, size))
    return img




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_rows = []
for _, r in train_df.iterrows():
    brats = int(r.BraTS21ID)
    y = float(r.MGMT_value)
    paths = get_all_image_paths(brats, SEQ, folder="train")
    for p in paths:
        train_rows.append((brats, p, y))

train_slices = pd.DataFrame(train_rows, columns=["BraTS21ID", "path", "MGMT_value"])
train_slices = train_slices[train_slices["path"].notnull()].reset_index(drop=True)

test_rows = []
for brats in sample_sub.BraTS21ID.astype(int).tolist():
    paths = get_all_image_paths(brats, SEQ, folder="test")
    for p in paths:
        test_rows.append((brats, p))
test_slices = pd.DataFrame(test_rows, columns=["BraTS21ID", "path"])
test_slices = test_slices[test_slices["path"].notnull()].reset_index(drop=True)

test_ids_with_slices = set(test_slices.BraTS21ID.astype(int).tolist())

train_slices.head(), test_slices.head(), len(train_slices), len(test_slices)




## === cell 2
def make_dataset(df, training=True, batch_size=32):
    paths = df["path"].values
    if training:
        labels = df["MGMT_value"].values.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_train(p, y):
        img = load_image_tf(p, size=IMAGE_SIZE)
        img = tf.stack([img, img, img], axis=-1)  # (H,W,3)
        return img, tf.reshape(y, (1,))

    def _map_test(p):
        img = load_image_tf(p, size=IMAGE_SIZE)
        img = tf.stack([img, img, img], axis=-1)
        return img

    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True).map(
            _map_train, num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        ds = ds.map(_map_test, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def stratified_split_ids(ids, labels, test_size=0.2, seed=SEED):
    df = pd.DataFrame({"id": ids, "y": labels}).drop_duplicates("id")
    rng = np.random.default_rng(seed)

    tr_ids = []
    va_ids = []
    for cls, g in df.groupby("y"):
        g_ids = g["id"].to_numpy()
        rng.shuffle(g_ids)
        n_val = int(np.round(len(g_ids) * test_size))
        n_val = min(max(n_val, 1), max(len(g_ids) - 1, 1)) if len(g_ids) > 1 else 0
        va_ids.extend(g_ids[:n_val].tolist())
        tr_ids.extend(g_ids[n_val:].tolist())

    return tr_ids, va_ids


unique_ids = train_df.BraTS21ID.astype(int).tolist()
labels_by_id = train_df.set_index(train_df.BraTS21ID.astype(int))["MGMT_value"]
id_labels = [int(labels_by_id.loc[i]) for i in unique_ids]
tr_ids, va_ids = stratified_split_ids(unique_ids, id_labels, test_size=0.2, seed=SEED)

tr_slices = train_slices[train_slices.BraTS21ID.isin(tr_ids)].reset_index(drop=True)
va_slices = train_slices[train_slices.BraTS21ID.isin(va_ids)].reset_index(drop=True)

if len(tr_slices) == 0:
    tr_slices = train_slices.copy()
if len(va_slices) == 0:
    va_slices = train_slices.sample(
        n=min(len(train_slices), 1024), random_state=SEED
    ).reset_index(drop=True)

train_ds = make_dataset(tr_slices, training=True, batch_size=32)
val_ds = make_dataset(va_slices, training=True, batch_size=32)
test_ds = make_dataset(test_slices, training=False, batch_size=32)

len(tr_slices), len(va_slices)




## === cell 3
def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
    )
    return model


model = build_model()
model.summary()




## === cell 4
EPOCHS = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 5
test_pred_slice = model.predict(test_ds, verbose=1).reshape(-1)

test_slice_ids = test_slices.BraTS21ID.astype(int).values
pred_df = pd.DataFrame({"BraTS21ID": test_slice_ids, "MGMT_value": test_pred_slice})
pred_patient = pred_df.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

sub = sample_sub.copy()
sub["BraTS21ID_int"] = sub["BraTS21ID"].astype(int)

sub = sub.merge(
    pred_patient.rename(columns={"BraTS21ID": "BraTS21ID_int", "MGMT_value": "pred"}),
    on="BraTS21ID_int",
    how="left",
)

sub["MGMT_value"] = sub["pred"].fillna(0.5).astype(float)

sub["BraTS21ID"] = sub["BraTS21ID_int"].apply(lambda x: str(int(x)).zfill(5))
sub = sub[["BraTS21ID", "MGMT_value"]]

sub.to_csv("submission.csv", index=False)
sub.head(), sub.shape
