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

# 5. Target score

0.7600184672206839

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 8))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
        dct_method="INTEGER_FAST",
        ratio=2,
    )  # uint8
    img = tf.image.resize(img, [h_target, w_target], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([h_target, w_target, 3])
    return img


@tf.function
def _map_train(p, yy):
    return load_image(p), yy


@tf.function
def _map_test(p):
    return load_image(p)


def make_ds(df, y=None, shuffle=False, batch_size=32, cache_in_memory=False):
    img_dir = TRAIN_IMG_DIR if y is not None else TEST_IMG_DIR

    img_names = df["image"].astype(str).to_numpy()
    prefix = img_dir.rstrip("/") + "/"
    paths_np = np.char.add(prefix, img_names).astype("U")
    paths = tf.convert_to_tensor(paths_np, dtype=tf.string)

    if y is not None:
        y = tf.convert_to_tensor(y.astype(np.float32))
        ds = tf.data.Dataset.from_tensor_slices((paths, y))
        if shuffle:
            buf = min(len(df), 8192)
            ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_map_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(_map_test, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache_in_memory:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(
    train_df, Y_train, shuffle=True, batch_size=batch_size, cache_in_memory=False
)
val_ds = make_ds(
    val_df, Y_val, shuffle=False, batch_size=batch_size, cache_in_memory=False
)

steps_per_epoch = (len(train_df) + batch_size - 1) // batch_size
validation_steps = (len(val_df) + batch_size - 1) // batch_size
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3925140225.py in <cell line: 0>()
     67 
     68 
---> 69 train_ds = make_ds(
     70     train_df, Y_train, shuffle=True, batch_size=batch_size, cache_in_memory=False
     71 )

/tmp/ipykernel_11/3925140225.py in make_ds(df, y, shuffle, batch_size, cache_in_memory)
     36     img_names = df["image"].astype(str).to_numpy()
     37     prefix = img_dir.rstrip("/") + "/"
---> 38     paths_np = np.char.add(prefix, img_names).astype("U")
     39     paths = tf.convert_to_tensor(paths_np, dtype=tf.string)
     40 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U49' and 'object' (the few cases where this used to work often lead to incorrect results).

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
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3816630612.py in <cell line: 0>()
      1 epochs = 3
      2 history = model.fit(
----> 3     train_ds,
      4     validation_data=val_ds,
      5     epochs=epochs,

NameError: name 'train_ds' is not defined

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




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4033582737.py in <cell line: 0>()
----> 1 val_pred = model.predict(val_ds, verbose=1)
      2 y_true = Y_val.astype(np.int32)
      3 
      4 
      5 def f1_micro(y_true_bin, y_pred_bin):

NameError: name 'val_ds' is not defined

## === cell 8
test_ds = make_ds(
    submissions, y=None, shuffle=False, batch_size=batch_size, cache_in_memory=False
)
preds = model.predict(test_ds, verbose=1)

print("preds shape:", preds.shape)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1664398900.py in <cell line: 0>()
----> 1 test_ds = make_ds(
      2     submissions, y=None, shuffle=False, batch_size=batch_size, cache_in_memory=False
      3 )
      4 preds = model.predict(test_ds, verbose=1)
      5 

/tmp/ipykernel_11/3925140225.py in make_ds(df, y, shuffle, batch_size, cache_in_memory)
     36     img_names = df["image"].astype(str).to_numpy()
     37     prefix = img_dir.rstrip("/") + "/"
---> 38     paths_np = np.char.add(prefix, img_names).astype("U")
     39     paths = tf.convert_to_tensor(paths_np, dtype=tf.string)
     40 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U48' and 'object' (the few cases where this used to work often lead to incorrect results).

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

assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(submissions)
print("submission.csv OK:", chk.shape)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1449801971.py in <cell line: 0>()
----> 1 pred_bin = preds >= best_thr.reshape(1, -1)
      2 
      3 class_names_arr = np.asarray(class_names, dtype=object)
      4 healthy_idx = (
      5     int(np.where(class_names_arr == "healthy")[0][0])

NameError: name 'preds' is not defined
