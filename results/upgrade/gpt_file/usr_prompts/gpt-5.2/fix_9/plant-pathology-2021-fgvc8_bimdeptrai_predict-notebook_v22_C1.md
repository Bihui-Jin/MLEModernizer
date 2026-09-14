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
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True
DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32



## === cell 4
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)
n_classes = len(class_names)

idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(train))
tr_idx, va_idx = idx[:split], idx[split:]

train_df = train.iloc[tr_idx].copy()
valid_df = train.iloc[va_idx].copy()

y_train = y[tr_idx].astype(np.float32, copy=False)
y_valid = y[va_idx].astype(np.float32, copy=False)

train_df["__idx__"] = np.arange(len(train_df))
valid_df["__idx__"] = np.arange(len(valid_df))

train_df.head(), n_classes, class_names[:5]



## === cell 5
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train_paths = (TRAIN_DIR + "/" + train_df["image"].values).astype(str)
valid_paths = (TRAIN_DIR + "/" + valid_df["image"].values).astype(str)
test_paths = (TEST_DIR + "/" + submissions["image"].values).astype(str)

augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=15.0 / 360.0, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="nearest",
            seed=SEED,
        ),
        tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="augmenter",
)


@tf.function
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _decode_resize_with_label(path, y_):
    return _decode_resize(path), y_


def _make_ds(paths_np, labels_np=None, training=False):
    """
    Timeout fix (correctness-preserving, core logic unchanged):
      - Cache decoded+resized+normalized tensors in RAM (cache() with no filename).
        This removes repeated JPEG decode/resize cost across epochs while producing identical
        model inputs for the non-augmented stage.
      - Keep augmentation AFTER cache so each epoch still gets fresh randomized augmentation
        (same semantics as before: augmentation is only applied when training=True).
      - Keep deterministic behavior and file paths unchanged.
    """
    if labels_np is None:
        ds = tf.data.Dataset.from_tensor_slices(paths_np)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths_np, labels_np))

    ds = ds.with_options(DATASET_OPTIONS)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths_np), 4096),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    if labels_np is None:
        ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.cache()
    else:
        ds = ds.map(
            _decode_resize_with_label, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.cache()
        if training:
            ds = ds.map(
                lambda x, y_: (augmenter(x, training=True), y_),
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_dataset = _make_ds(train_paths, y_train, training=True)
valid_dataset = _make_ds(valid_paths, y_valid, training=False)
test_dataset = _make_ds(test_paths, labels_np=None, training=False)

train_generator = train_dataset
valid_generator = valid_dataset
test_generator = test_dataset



## === cell 6
base = tf.keras.applications.EfficientNetB4(
    include_top=False, weights="imagenet", input_shape=(h_target, w_target, 3)
)
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(n_classes, activation="sigmoid")(x)
model = tf.keras.Model(inputs=base.input, outputs=out)

for layer in base.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=32,
)

model.summary()



## === cell 7
epochs_stage1 = 2
history1 = model.fit(
    train_generator, validation_data=valid_generator, epochs=epochs_stage1, verbose=1
)

for layer in base.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    steps_per_execution=32,
)

epochs_stage2 = 1
history2 = model.fit(
    train_generator, validation_data=valid_generator, epochs=epochs_stage2, verbose=1
)



## === cell 8
preds = model.predict(test_generator, verbose=1)
preds.shape



## === cell 9
thresh = 0.25
class_names_arr = np.asarray(class_names, dtype=object)

top_idx = preds.argmax(axis=1).astype(np.int32)
chosen_mask = preds >= thresh

healthy_idx = class_names.index("healthy") if "healthy" in class_names else None
if healthy_idx is not None:
    is_top_healthy = top_idx == healthy_idx
    chosen_has_healthy = chosen_mask[:, healthy_idx]
else:
    is_top_healthy = np.zeros(preds.shape[0], dtype=bool)
    chosen_has_healthy = np.zeros(preds.shape[0], dtype=bool)

chosen_count = chosen_mask.sum(axis=1)
use_top_only = (chosen_count == 0) | chosen_has_healthy

pred_labels = np.empty(preds.shape[0], dtype=object)
pred_labels[is_top_healthy] = "healthy"

idx_top_only = np.flatnonzero((~is_top_healthy) & use_top_only)
pred_labels[idx_top_only] = class_names_arr[top_idx[idx_top_only]]

idx_multi = np.flatnonzero((~is_top_healthy) & (~use_top_only))
for i in idx_multi:
    idxs = np.flatnonzero(chosen_mask[i])
    pred_labels[i] = " ".join(class_names_arr[idxs].tolist())

submissions = submissions.copy()
submissions["labels"] = pred_labels.tolist()
submissions.to_csv("submission.csv", index=False)

submissions.head()



## === cell 10
assert os.path.exists("submission.csv")
assert list(submissions.columns) == ["image", "labels"]
assert len(submissions) == len(
    pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
)
print("Wrote submission.csv with shape:", submissions.shape)
print(submissions.sample(5, random_state=SEED))
