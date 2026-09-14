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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd



## === cell 1
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass



## === cell 2
from tensorflow.keras import layers, models, optimizers



## === cell 3
sam_sub = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
sam_sub.head()



## === cell 4
train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"

train = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
print(train.shape)
print(train.head())



## === cell 5
test_df = sam_sub[["image"]].copy()
print("test_df:", test_df.shape)
test_df.head()



## === cell 6
all_labels = sorted(
    {lab for s in train["labels"].fillna("").astype(str) for lab in s.split() if lab}
)
print("Num unique labels:", len(all_labels))
print("Labels:", all_labels)

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=all_labels)
label_lists = train["labels"].fillna("").astype(str).str.split()
Y = mlb.fit_transform(label_lists)

train_ml = train.copy()
train_ml[all_labels] = Y
train_ml.head()



## === cell 7
IMG_SIZE = (432, 648)
BATCH_SIZE = 16

num_classes = len(all_labels)
print("Num classes:", num_classes)

AUTOTUNE = tf.data.AUTOTUNE

idx = np.arange(len(train_ml))
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_size = int(round(0.1 * len(idx)))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_split_df = train_ml.iloc[trn_idx].reset_index(drop=True)
valid_split_df = train_ml.iloc[val_idx].reset_index(drop=True)

train_paths = (train_dir + "/" + train_split_df["image"].astype(str)).to_numpy()
valid_paths = (train_dir + "/" + valid_split_df["image"].astype(str)).to_numpy()
test_paths = (test_dir + "/" + test_df["image"].astype(str)).to_numpy()

train_labels = train_split_df[all_labels].to_numpy(np.float32, copy=False)
valid_labels = valid_split_df[all_labels].to_numpy(np.float32, copy=False)


@tf.function
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize_with_pad(
        img,
        IMG_SIZE[0],
        IMG_SIZE[1],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


augment = tf.keras.Sequential(
    [
        layers.RandomRotation(factor=20.0 / 360.0, seed=42),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=42),
        layers.RandomFlip(mode="horizontal", seed=42),
    ],
    name="augment",
)


@tf.function
def _train_map_batch(x, y):
    x = augment(x, training=True)
    return x, y


@tf.function
def _decode_xy(path, y):
    return _decode_resize(path), y


@tf.function
def _decode_x(path):
    return _decode_resize(path)


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.threading.private_threadpool_size = 0
options.threading.max_intra_op_parallelism = 0

SHUFFLE_BUFFER = 4096

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).with_options(
    options
)
train_ds = train_ds.shuffle(
    buffer_size=SHUFFLE_BUFFER, seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_decode_xy, num_parallel_calls=AUTOTUNE, deterministic=True)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.map(
    _train_map_batch, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels)).with_options(
    options
)
valid_ds = valid_ds.map(_decode_xy, num_parallel_calls=AUTOTUNE, deterministic=True)
valid_ds = valid_ds.cache()
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False)
valid_ds = valid_ds.prefetch(AUTOTUNE)

TEST_BATCH_SIZE = 32
test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
test_ds = test_ds.map(_decode_x, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.cache()
test_ds = test_ds.batch(TEST_BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)



## === cell 8
trained_model_sub = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="sigmoid"),
    ]
)

trained_model_sub.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
    steps_per_execution=20,
)

trained_model_sub.summary()



## === cell 9
EPOCHS = 3

history = trained_model_sub.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 10
from sklearn.metrics import f1_score

valid_prob = trained_model_sub.predict(valid_ds, verbose=1)
y_true = valid_labels.astype(np.int32)

grid = np.array(
    [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50], dtype=np.float32
)
best_thr = np.full((num_classes,), 0.30, dtype=np.float32)

P = valid_prob.astype(np.float32, copy=False)
Yt = y_true  # int32

for c in range(num_classes):
    yt = Yt[:, c]
    pos = int(yt.sum())
    if pos == 0:
        best_thr[c] = 0.50
        continue

    pc = P[:, c]
    preds = (pc[None, :] >= grid[:, None]).astype(np.int32)
    tp = (preds & yt[None, :]).sum(axis=1)
    pred_pos = preds.sum(axis=1)

    fp = pred_pos - tp
    fn = pos - tp

    denom = (2 * tp + fp + fn).astype(np.float32)
    f1s = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    best_thr[c] = float(grid[int(np.argmax(f1s))])

val_pred = (valid_prob >= best_thr[None, :]).astype(np.int32)
micro_f1 = f1_score(y_true.ravel(), val_pred.ravel(), zero_division=0)
print("Validation micro-F1 (thresholded):", micro_f1)
print(
    "Threshold summary: min/mean/max =",
    float(best_thr.min()),
    float(best_thr.mean()),
    float(best_thr.max()),
)



## === cell 11
test_prob = trained_model_sub.predict(test_ds, verbose=1)
print("test_prob shape:", test_prob.shape)

test_bin = (test_prob >= best_thr[None, :]).astype(np.int32)
ordered_test_ids = (
    test_df["image"].astype(str).tolist()
)  # preserve sample_submission order

argmax_idx = np.argmax(test_prob, axis=1).astype(np.int64, copy=False)
label_arr = np.asarray(all_labels, dtype=object)

pred_strings = []
append_ps = pred_strings.append
for i in range(test_bin.shape[0]):
    cols_i = np.flatnonzero(test_bin[i])
    if cols_i.size == 0:
        append_ps(str(label_arr[int(argmax_idx[i])]))
    else:
        append_ps(" ".join(label_arr[cols_i]))

sub = pd.DataFrame({"image": ordered_test_ids, "labels": pred_strings})
sub = sub[["image", "labels"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Submission path:", os.path.abspath("submission.csv"))
print(
    "Unique predicted labels (sample):",
    sorted({l for s in sub["labels"].head(100).tolist() for l in s.split()}),
)
