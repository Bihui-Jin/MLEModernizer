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
import gc
import sys
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True
try:
    DATASET_OPTIONS.experimental_optimization.map_parallelization = True
    DATASET_OPTIONS.experimental_optimization.parallel_batch = True
except Exception:
    pass




## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3
TOTAL_INPUTS = NR_CHANNELS * IMG_HEIGHT * IMG_WIDTH




## === cell 2
BASE_INPUT = "/kaggle/input/plant-pathology-2021-fgvc8"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input/plant-pathology-2021-fgvc8"

TRAIN_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.exists(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing TRAIN_CSV_PATH: {TRAIN_CSV_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing SAMPLE_SUB_PATH: {SAMPLE_SUB_PATH}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image"].tolist()

imglist_test = np.array(
    [os.path.join(TEST_DIR, fn) for fn in test_images], dtype=object
)
missing = [p for p in imglist_test.tolist() if not os.path.exists(p)]
print("test images:", len(imglist_test), "missing:", len(missing))
if len(missing) > 0:
    print("First missing:", missing[0])




## === cell 3
AUTOTUNE = tf.data.AUTOTUNE


def _load_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(tf.constant(imglist_test))
    .with_options(DATASET_OPTIONS)
    .map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(32, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

_first_batch = next(iter(test_ds))
print((len(imglist_test), IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))




## === cell 4
print(_first_batch[0, :2, :2, :].numpy())




## === cell 5
training_csv = pd.read_csv(TRAIN_CSV_PATH)

all_tokens = training_csv["labels"].astype(str).str.split()
tagnames = np.unique(np.concatenate(all_tokens.values))
num_classes = len(tagnames)
print("num_classes:", num_classes)
print("tagnames:", tagnames)

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
    pooling="avg",
)
base.trainable = False

inp = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
x = base(inp, training=False)
out = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="pred")(x)
model_f = tf.keras.Model(inp, out)




## === cell 6
def labels_to_multihot(labels_series, tagnames):
    s = labels_series.astype(str).str.split()
    exploded = s.explode()
    d = pd.get_dummies(exploded)
    y_df = (
        d.groupby(level=0).sum().reindex(columns=tagnames, fill_value=0).clip(upper=1)
    )
    return y_df.to_numpy(dtype=np.int8)


def build_ds_from_paths(paths, y=None, batch_size=32, shuffle=False, cache=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths))
    else:
        ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(y)))

    ds = ds.with_options(DATASET_OPTIONS)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(int(len(paths)), 4096),
            seed=42,
            reshuffle_each_iteration=False,
        )

    if y is None:
        ds = ds.map(
            _load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:

        def _map_xy(p, yy):
            return _load_and_preprocess(p), tf.cast(yy, tf.float32)

        ds = ds.map(_map_xy, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache:
        pass

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def f1_per_class(y_true, y_pred_bool):
    tp = np.sum((y_true == 1) & (y_pred_bool == 1), axis=0).astype(np.float64)
    fp = np.sum((y_true == 0) & (y_pred_bool == 1), axis=0).astype(np.float64)
    fn = np.sum((y_true == 1) & (y_pred_bool == 0), axis=0).astype(np.float64)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1


val_frac = 0.10
idx = np.arange(len(training_csv))
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

val_df = training_csv.iloc[val_idx].reset_index(drop=True)
trn_df = training_csv.iloc[trn_idx].reset_index(drop=True)

trn_paths = np.array(
    [os.path.join(TRAIN_DIR, fn) for fn in trn_df["image"].tolist()], dtype=object
)
val_paths = np.array(
    [os.path.join(TRAIN_DIR, fn) for fn in val_df["image"].tolist()], dtype=object
)

y_trn = labels_to_multihot(trn_df["labels"], tagnames).astype(np.float32)
y_val = labels_to_multihot(val_df["labels"], tagnames).astype(np.int8)

trn_ds = build_ds_from_paths(
    trn_paths, y=y_trn, batch_size=32, shuffle=True, cache=True
)
val_ds = build_ds_from_paths(
    val_paths, y=y_val.astype(np.float32), batch_size=32, shuffle=False, cache=True
)

model_f.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

_ = model_f.fit(trn_ds, epochs=3, verbose=1)

val_x_ds = val_ds.map(
    lambda x, y: x, num_parallel_calls=AUTOTUNE, deterministic=True
).prefetch(AUTOTUNE)
p_val = model_f.predict(val_x_ds, verbose=1)

thr_grid = np.linspace(0.05, 0.60, 12)
best_thr = np.full((num_classes,), 0.26, dtype=np.float32)
best_f1 = np.full((num_classes,), -1.0, dtype=np.float64)

for thr in thr_grid:
    pred_bool = p_val >= thr
    f1c = f1_per_class(y_val, pred_bool)
    better = f1c > best_f1
    best_f1[better] = f1c[better]
    best_thr[better] = thr

print(
    "Calibrated thresholds summary:",
    "min",
    float(best_thr.min()),
    "max",
    float(best_thr.max()),
    "mean",
    float(best_thr.mean()),
)

del trn_ds, trn_paths, trn_df, y_trn, val_ds, val_x_ds, val_paths, val_df, y_val, p_val
gc.collect()




## === cell 7
X_test = model_f.predict(test_ds, verbose=1)
print(X_test.shape)




## === cell 8
print(X_test[:2])




## === cell 9
pass




## === cell 10
def class2tags(classes_bool, tagnames):
    idx = [np.flatnonzero(row) for row in classes_bool]
    return [" ".join(tagnames[i]) for i in idx]


def probs_to_tags_with_fallback_perclass(probs, tagnames, per_class_thresh):
    pred = probs >= per_class_thresh.reshape(1, -1)
    empty = pred.sum(axis=1) == 0
    if np.any(empty):
        top1 = np.argmax(probs[empty], axis=1)
        pred[empty, :] = False
        pred[np.where(empty)[0], top1] = True
    return class2tags(pred, tagnames)




## === cell 11
test_predtags = probs_to_tags_with_fallback_perclass(X_test, tagnames, best_thr)
print("Example tags:", test_predtags[:5])




## === cell 12
del _first_batch
gc.collect()




## === cell 13
def write_path_from_sample(sample_df):
    return sample_df["image"].values




## === cell 14
df1 = pd.DataFrame(write_path_from_sample(sample_sub), columns=["image"])
df1.head()




## === cell 15
df2 = pd.DataFrame(test_predtags, columns=["labels"])
df2.head()




## === cell 16
sub_df = pd.concat([df1, df2], axis=1)
sub_df = sub_df[["image", "labels"]]
assert sub_df.shape[0] == sample_sub.shape[0], "Submission row count mismatch."

sub_path = "./submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub_df.shape)
print(sub_df.columns.tolist())
print(sub_df.head())
