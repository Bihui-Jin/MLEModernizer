# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

COMP_ROOT = "/kaggle/input/aerial-cactus-identification"
if not os.path.exists(COMP_ROOT):
    COMP_ROOT = "/kaggle/input"

print("COMP_ROOT =", COMP_ROOT)
print("Exists train.zip:", os.path.exists(os.path.join(COMP_ROOT, "train.zip")))
print("Exists test.zip:", os.path.exists(os.path.join(COMP_ROOT, "test.zip")))
print("Exists train.csv:", os.path.exists(os.path.join(COMP_ROOT, "train.csv")))
print(
    "Exists sample_submission.csv:",
    os.path.exists(os.path.join(COMP_ROOT, "sample_submission.csv")),
)



## === cell 1
import zipfile

extract_dir = "/kaggle/working"

train_zip_path = os.path.join(COMP_ROOT, "train.zip")
test_zip_path = os.path.join(COMP_ROOT, "test.zip")


def _already_extracted(base_dir: str, subdir: str) -> bool:
    d = os.path.join(base_dir, subdir)
    if not os.path.isdir(d):
        return False
    try:
        return any(name.lower().endswith(".jpg") for name in os.listdir(d))
    except Exception:
        return False


if not (
    _already_extracted(extract_dir, "train")
    or _already_extracted(extract_dir, "aerial-cactus-identification/train")
):
    with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

if not (
    _already_extracted(extract_dir, "test")
    or _already_extracted(extract_dir, "aerial-cactus-identification/test")
):
    with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

print("Extraction done to:", extract_dir)



## === cell 2
pass




## === cell 3
def find_dir_with_jpgs(base, dirname_candidates):
    for d in dirname_candidates:
        cand = os.path.join(base, d)
        if os.path.isdir(cand):
            try:
                files = os.listdir(cand)
            except Exception:
                continue
            if any(f.lower().endswith(".jpg") for f in files):
                return cand
    return None


train_dir = find_dir_with_jpgs(
    "/kaggle/working", ["train", "aerial-cactus-identification/train"]
) or find_dir_with_jpgs("/kaggle/working/aerial-cactus-identification", ["train"])

test_dir = find_dir_with_jpgs(
    "/kaggle/working", ["test", "aerial-cactus-identification/test"]
) or find_dir_with_jpgs("/kaggle/working/aerial-cactus-identification", ["test"])

if train_dir is None or test_dir is None:
    for root, dirs, files in os.walk("/kaggle/working"):
        if root.count(os.sep) > "/kaggle/working".count(os.sep) + 5:
            continue
        base = os.path.basename(root)
        if base == "train" and any(f.lower().endswith(".jpg") for f in files):
            train_dir = root
        if base == "test" and any(f.lower().endswith(".jpg") for f in files):
            test_dir = root
    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Could not locate extracted train/test image folders. train_dir={train_dir}, test_dir={test_dir}"
        )

print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)

train_csv_path = os.path.join(COMP_ROOT, "train.csv")
sample_sub_path = os.path.join(COMP_ROOT, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
print(train_df.head())
print(
    "train_dir exists:",
    os.path.exists(train_dir),
    "n_files:",
    len(os.listdir(train_dir)) if os.path.exists(train_dir) else None,
)
print(
    "test_dir exists:",
    os.path.exists(test_dir),
    "n_files:",
    len(os.listdir(test_dir)) if os.path.exists(test_dir) else None,
)




## === cell 4
def count_files(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images: {test_count}")



## === cell 5
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("Using tensorflow version:", tf.__version__)



## === cell 10
train_df["has_cactus"] = train_df["has_cactus"].astype(str)



## === cell 11

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (32, 32)
BATCH_TRAIN = 1024
BATCH_VAL = 256

val_split = 0.10
n_total = len(train_df)
n_val = int(np.floor(n_total * val_split))
n_train = n_total - n_val

train_df_split = train_df.iloc[:n_train].reset_index(drop=True)
val_df_split = train_df.iloc[n_train:].reset_index(drop=True)


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)  # keep in 0..255 prior to scaling
    return img


def _augment_like_original(img):
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=SEED)
    img = tf.image.rot90(img, k)

    do_lr = tf.random.uniform([], 0.0, 1.0, seed=SEED) > 0.5
    img = tf.cond(do_lr, lambda: tf.image.flip_left_right(img), lambda: img)

    do_ud = tf.random.uniform([], 0.0, 1.0, seed=SEED) > 0.5
    img = tf.cond(do_ud, lambda: tf.image.flip_up_down(img), lambda: img)

    factor = tf.random.uniform([], minval=0.8, maxval=1.2, dtype=tf.float32, seed=SEED)
    img = tf.clip_by_value(img * factor, 0.0, 255.0)

    img = img / 255.0
    return img


def _make_labeled_ds(df, batch_size, training: bool):
    paths = tf.constant([os.path.join(train_dir, x) for x in df["id"].values])
    labels = tf.constant(df["has_cactus"].astype(np.float32).values)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img = _decode_resize(p)
        if training:
            img = _augment_like_original(img)
        else:
            img = img / 255.0
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = _make_labeled_ds(train_df_split, BATCH_TRAIN, training=True)
val_ds = _make_labeled_ds(val_df_split, BATCH_VAL, training=False)

train_batches = int(np.ceil(len(train_df_split) / BATCH_TRAIN))
val_batches = int(np.ceil(len(val_df_split) / BATCH_VAL))
print("train batches:", train_batches, "val batches:", val_batches)
if train_batches == 0 or val_batches == 0:
    raise ValueError(
        f"Empty dataset(s). train_batches={train_batches} val_batches={val_batches}"
    )



## === cell 12
pass



## === cell 13
sample_sub = pd.read_csv(sample_sub_path)

BATCH_TEST = 256

test_paths = tf.constant([os.path.join(test_dir, x) for x in sample_sub["id"].values])
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _map_test(p):
    img = _decode_resize(p)
    img = img / 255.0
    return img


test_ds = (
    test_ds.map(_map_test, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_TEST)
    .prefetch(AUTOTUNE)
)

test_batches = int(np.ceil(len(sample_sub) / BATCH_TEST))
print("test batches:", test_batches, "n_test:", len(sample_sub))
if test_batches == 0:
    raise ValueError("Empty test dataset. Check test_dir and sample_submission ids.")



## === cell 14
efficient_net = EfficientNetB3(
    weights="imagenet", input_shape=(32, 32, 3), include_top=False, pooling="max"
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.summary()



## === cell 15
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 16
history = model.fit(
    train_ds,
    epochs=50,
    steps_per_epoch=15,
    validation_data=val_ds,
    validation_steps=7,
    verbose=2,
)



## === cell 17
acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])
print(
    "Last epoch metrics:",
    {
        "accuracy": acc[-1] if acc else None,
        "val_accuracy": val_acc[-1] if val_acc else None,
        "loss": loss[-1] if loss else None,
        "val_loss": val_loss[-1] if val_loss else None,
    },
)



## === cell 18
preds = model.predict(test_ds, steps=test_batches, verbose=1)



## === cell 19
predictions = preds.reshape(-1)
submission = sample_sub.copy()
submission["has_cactus"] = predictions[: len(submission)]
submission["has_cactus"] = submission["has_cactus"].astype(float).clip(0.0, 1.0)
print(submission.head())
print(submission.shape)



## === cell 20
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)



## === cell 21
print(os.listdir("/kaggle/working"))
print("submission.csv exists:", os.path.exists("/kaggle/working/submission.csv"))
print(
    "submission.csv size:",
    (
        os.path.getsize("/kaggle/working/submission.csv")
        if os.path.exists("/kaggle/working/submission.csv")
        else None
    ),
)
