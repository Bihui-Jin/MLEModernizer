# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
print(train.head(2))
print(submissions.head(2))



## === cell 2
h_target = 256
w_target = 256
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

label_split = train["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)
num_classes = len(class_names)

print("num_classes:", num_classes)
print("classes:", class_names)



## === cell 3
idx = np.arange(len(train))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

val_frac = 0.15
val_size = int(len(train) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

train_df = train.iloc[tr_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

Y_train = Y[tr_idx]
Y_val = Y[val_idx]

print("train_df:", train_df.shape, "val_df:", val_df.shape)




## === cell 4
def load_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(
        img_bytes,
        channels=3,
        dct_method="INTEGER_FAST",  # faster, negligible numeric differences only
    )  # uint8
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [h_target, w_target], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _map_train(p, yy):
    return load_image(p), yy


def _map_test(p):
    return load_image(p)


def make_ds(df, y=None, shuffle=False, batch_size=32, cache=None):
    img_dir = TRAIN_IMG_DIR if y is not None else TEST_IMG_DIR
    img_names = tf.constant(df["image"].values.astype(str))
    paths = tf.strings.join([img_dir, os.sep, img_names])

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    try:
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.map_fusion = True
    except Exception:
        pass

    if y is not None:
        y = tf.constant(y.astype(np.float32))
        ds = tf.data.Dataset.from_tensor_slices((paths, y)).with_options(options)
        if shuffle:
            buf = min(len(df), 8192)
            ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_map_train, num_parallel_calls=AUTOTUNE, deterministic=False)
        if cache is not None:
            ds = ds.cache() if cache == "" else ds.cache(cache)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
        ds = ds.map(_map_test, num_parallel_calls=AUTOTUNE, deterministic=False)
        if cache is not None:
            ds = ds.cache() if cache == "" else ds.cache(cache)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

train_cache = os.path.join(
    CACHE_DIR, f"train_{h_target}x{w_target}_bs{batch_size}.cache"
)
val_cache = os.path.join(CACHE_DIR, f"val_{h_target}x{w_target}_bs{batch_size}.cache")

train_ds = make_ds(
    train_df, Y_train, shuffle=True, batch_size=batch_size, cache=train_cache
)
val_ds = make_ds(val_df, Y_val, shuffle=False, batch_size=batch_size, cache=val_cache)



## === cell 5
base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
    pooling="avg",
)
base.trainable = False  # keep it light and stable in <600s

inputs = keras.Input(shape=(h_target, w_target, 3))
x = inputs
x = tf.keras.applications.resnet.preprocess_input(x * 255.0)
x = base(x, training=False)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy"
)

model.summary()



## === cell 6
epochs = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)



## === cell 7
val_pred = model.predict(val_ds, verbose=1)

y_true = Y_val.astype(np.int32)


def f1_micro(y_true_bin, y_pred_bin):
    tp = np.logical_and(y_true_bin == 1, y_pred_bin == 1).sum()
    fp = np.logical_and(y_true_bin == 0, y_pred_bin == 1).sum()
    fn = np.logical_and(y_true_bin == 1, y_pred_bin == 0).sum()
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


grid = np.linspace(0.1, 0.9, 17).astype(np.float32)  # (T,)
T = grid.shape[0]
C = num_classes

y_true_i = y_true.astype(np.int8)  # (N,C)
pred_f = val_pred.astype(np.float32)  # (N,C)

pos_cnt = y_true_i.sum(axis=0).astype(np.int32)  # (C,)

tp = np.empty((C, T), dtype=np.int32)
fp = np.empty((C, T), dtype=np.int32)
for j, thr in enumerate(grid):
    pred_pos = pred_f >= thr  # (N,C) bool
    tp[:, j] = np.logical_and(pred_pos, y_true_i == 1).sum(axis=0, dtype=np.int32)
    fp[:, j] = np.logical_and(pred_pos, y_true_i == 0).sum(axis=0, dtype=np.int32)

fn = (pos_cnt[:, None] - tp).astype(np.int32)

denom = (2 * tp + fp + fn).astype(np.float32)
f1 = np.where(denom > 0, (2.0 * tp) / denom, 0.0).astype(np.float32)  # (C,T)

best_idx = f1.argmax(axis=1)  # (C,)
best_thr = grid[best_idx].astype(np.float32)

val_pred_bin = (val_pred >= best_thr.reshape(1, -1)).astype(np.int32)
print("Val micro-F1 (sanity):", f1_micro(y_true, val_pred_bin))
print("Thresholds:", dict(zip(class_names, np.round(best_thr, 3))))



## === cell 8
test_ds = make_ds(submissions, y=None, shuffle=False, batch_size=batch_size, cache=None)
preds = model.predict(test_ds, verbose=1)

print("preds shape:", preds.shape)



## === cell 9
pred_bin = preds >= best_thr.reshape(1, -1)

class_names_arr = np.asarray(class_names, dtype=object)
healthy_idx = (
    int(np.where(class_names_arr == "healthy")[0][0])
    if "healthy" in class_names
    else -1
)

N, C = pred_bin.shape
chosen_mask = pred_bin.copy()

none_mask = ~chosen_mask.any(axis=1)
if none_mask.any():
    argm = np.argmax(preds[none_mask], axis=1)
    chosen_mask[none_mask] = False
    chosen_mask[np.flatnonzero(none_mask), argm] = True

if healthy_idx != -1:
    has_healthy = chosen_mask[:, healthy_idx]
    more_than_one = chosen_mask.sum(axis=1) > 1
    drop_mask = has_healthy & more_than_one
    if drop_mask.any():
        chosen_mask[drop_mask, healthy_idx] = False
        became_empty = drop_mask & (~chosen_mask.any(axis=1))
        if became_empty.any():
            chosen_mask[became_empty, healthy_idx] = True

idxs = [np.flatnonzero(row).tolist() for row in chosen_mask]
out_labels = [" ".join(class_names_arr[i].tolist()) for i in idxs]

submission_df = pd.DataFrame(
    {"image": submissions["image"].values, "labels": out_labels}
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head(10))



## === cell 10
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(submissions)
print("submission.csv OK:", chk.shape)
