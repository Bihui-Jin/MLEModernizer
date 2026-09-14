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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf

tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)
np.random.seed(42)

INPUT_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
WORK_ROOT = "/kaggle/working"

train_csv_path = f"{INPUT_ROOT}/train.csv"
sample_sub_path = f"{INPUT_ROOT}/sample_submission.csv"
train_img_dir = f"{INPUT_ROOT}/train_images"
test_img_dir = f"{INPUT_ROOT}/test_images"

print("Exists train_csv:", os.path.exists(train_csv_path))
print("Exists sample_submission:", os.path.exists(sample_sub_path))
print("Exists train_img_dir:", os.path.isdir(train_img_dir))
print("Exists test_img_dir:", os.path.isdir(test_img_dir))

sub_df = pd.read_csv(sample_sub_path)
train_df = pd.read_csv(train_csv_path)

print("sample_submission head:\n", sub_df.head())
print("n_test:", len(sub_df))
print("train head:\n", train_df.head())
print("n_train:", len(train_df))



## === cell 1
LABELS = ["complex", "frog_eye_leaf_spot", "healthy", "powdery_mildew", "rust", "scab"]
label_to_idx = {l: i for i, l in enumerate(LABELS)}
LABELS_ARR = np.array(LABELS, dtype=object)


def encode_labels_df(series: pd.Series) -> np.ndarray:
    n = len(series)
    y = np.zeros((n, len(LABELS)), dtype=np.float32)
    s = series.fillna("").astype(str).to_numpy()

    for i, lab_str in enumerate(s):
        if not lab_str:
            continue
        for tok in lab_str.split(" "):
            j = label_to_idx.get(tok)
            if j is not None:
                y[i, j] = 1.0
    return y


train_paths = (Path(train_img_dir) / train_df["image"]).astype(str).values
train_targets = encode_labels_df(train_df["labels"]).astype(np.float32)

print("Train paths shape:", train_paths.shape)
print("Train targets shape:", train_targets.shape)
print("Target sample (first row):", train_df.loc[0, "labels"], train_targets[0])



## === cell 2
IMG_SIZE = (380, 380)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

data_opts = tf.data.Options()
data_opts.autotune.enabled = True
data_opts.threading.private_threadpool_size = 0
data_opts.threading.max_intra_op_parallelism = 0
data_opts.deterministic = True
data_opts.experimental_slack = True

data_opts.experimental_optimization.apply_default_optimizations = True


@tf.function
def _decode_resize_uint8(p):
    img = tf.io.read_file(p)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.cast(img, tf.uint8)
    return img


@tf.function
def _preprocess_effnet_uint8(img_uint8):
    img = tf.cast(img_uint8, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(
        img
    )  # expects [0..255] float
    return img


@tf.function
def train_map_fn(p, y):
    img_uint8 = _decode_resize_uint8(p)
    img_uint8 = tf.image.random_flip_left_right(img_uint8)
    img_uint8 = tf.image.random_flip_up_down(img_uint8)
    img = _preprocess_effnet_uint8(img_uint8)
    return img, y


@tf.function
def val_map_fn(p, y):
    img_uint8 = _decode_resize_uint8(p)
    img = _preprocess_effnet_uint8(img_uint8)
    return img, y


n = len(train_paths)
idx = np.arange(n)
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_size = int(0.1 * n)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_paths, trn_y = train_paths[trn_idx], train_targets[trn_idx]
val_paths, val_y = train_paths[val_idx], train_targets[val_idx]

ds_train = tf.data.Dataset.from_tensor_slices((trn_paths, trn_y)).with_options(
    data_opts
)
ds_train = ds_train.shuffle(
    min(len(trn_paths), 4096), seed=42, reshuffle_each_iteration=True
)
ds_train = ds_train.map(train_map_fn, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=False)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_y)).with_options(data_opts)
ds_val = ds_val.map(val_map_fn, num_parallel_calls=AUTOTUNE)

ds_val = ds_val.cache()
ds_val = ds_val.batch(BATCH_SIZE, drop_remainder=False)
ds_val = ds_val.prefetch(AUTOTUNE)

print("Train batches:", tf.data.experimental.cardinality(ds_train).numpy())
print("Val batches:", tf.data.experimental.cardinality(ds_val).numpy())



## === cell 3
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)

x_in = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(x_in, training=False)
x = tf.keras.layers.Dropout(0.3)(x)
x_out = tf.keras.layers.Dense(len(LABELS), activation="sigmoid")(x)
model = tf.keras.Model(x_in, x_out)

base.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=32,
)

print(model.summary())



## === cell 4
EPOCHS_HEAD = 2
history1 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_HEAD, verbose=1)

base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    steps_per_execution=32,
)

EPOCHS_FT = 1
history2 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_FT, verbose=1)



## === cell 5
test_paths = (Path(test_img_dir) / sub_df["image"]).astype(str).values
ds_test = tf.data.Dataset.from_tensor_slices(test_paths)


@tf.function
def test_map_fn(p):
    img_uint8 = _decode_resize_uint8(p)
    img = _preprocess_effnet_uint8(img_uint8)
    return img


ds_test = ds_test.with_options(data_opts)
ds_test = ds_test.map(test_map_fn, num_parallel_calls=AUTOTUNE)
ds_test = (
    ds_test.cache()
)  # safe: test set is only ~3.7k images, avoids re-decode if predict iterates internally
ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=False)
ds_test = ds_test.prefetch(AUTOTUNE)

pred = model.predict(ds_test, verbose=1)
print("Pred shape:", pred.shape)
print("Pred sample row:", pred[0])



## === cell 6
threshold = 0.5

pred_bin = pred >= threshold

any_on = pred_bin.any(axis=1)
argmax_idx = np.argmax(pred, axis=1)

rows_no_on = np.where(~any_on)[0]
pred_bin_fixed = pred_bin.copy()
pred_bin_fixed[rows_no_on, :] = False
pred_bin_fixed[rows_no_on, argmax_idx[rows_no_on]] = True

pred_labels = [" ".join(LABELS_ARR[row].tolist()) for row in pred_bin_fixed]

out_df = pd.DataFrame({"image": sub_df["image"].values, "labels": pred_labels})

out_path = "submission.csv"
out_df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(out_df.head())
print("Submission columns:", list(pd.read_csv(out_path, nrows=1).columns))
print("Submission rows:", len(out_df))
assert out_df.shape[0] == sub_df.shape[0]
assert list(out_df.columns) == ["image", "labels"]
assert out_path.endswith(".csv")
