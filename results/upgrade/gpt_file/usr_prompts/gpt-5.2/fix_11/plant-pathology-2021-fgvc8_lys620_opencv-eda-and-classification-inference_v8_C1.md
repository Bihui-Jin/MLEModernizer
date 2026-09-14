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
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

df = pd.read_csv(TRAIN_CSV)
sub = pd.read_csv(SAMPLE_SUB)

print(df.shape, sub.shape)
df.head()




## === cell 2
df["labels_list"] = df["labels"].fillna("").str.split(" ")

all_labels = sorted({lab for labs in df["labels_list"] for lab in labs if lab != ""})
label2idx = {l: i for i, l in enumerate(all_labels)}
idx2label = {i: l for l, i in label2idx.items()}

NUM_CLASSES = len(all_labels)
print("NUM_CLASSES:", NUM_CLASSES)
print(all_labels)

if NUM_CLASSES > 0:
    targets_df = df["labels"].fillna("").str.get_dummies(sep=" ")
    targets_df = targets_df.reindex(columns=all_labels, fill_value=0)
    targets = targets_df.to_numpy(dtype=np.float32, copy=False)
else:
    targets = np.zeros((len(df), 0), dtype=np.float32)

df["target"] = list(targets)




## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 16

train_paths = (TRAIN_DIR + "/" + df["image"].astype(str)).to_numpy()
train_targets = targets  # already a contiguous float32 array

n = len(df)
idx = np.arange(n)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_size = int(0.1 * n)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_paths, tr_y = train_paths[tr_idx], train_targets[tr_idx]
va_paths, va_y = train_paths[val_idx], train_targets[val_idx]

eff_preprocess = tf.keras.applications.efficientnet.preprocess_input

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True

opt = options.experimental_optimization
for name, value in [
    ("map_fusion", True),
    ("parallel_batch", True),
    ("map_parallelization", True),
]:
    try:
        setattr(opt, name, value)
    except Exception:
        pass


def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32)
    img = eff_preprocess(img)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


def load_preprocess(path, y):
    return _decode_resize_preprocess(path), y


def load_preprocess_test(path):
    return _decode_resize_preprocess(path)


cache_dir = "./tfdata_cache_pp2021"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(
    cache_dir, f"train_img224_b{BATCH_SIZE}_seed{SEED}.cache"
)
val_cache_path = os.path.join(cache_dir, f"val_img224_b{BATCH_SIZE}_seed{SEED}.cache")
test_cache_path = os.path.join(cache_dir, f"test_img224_b{BATCH_SIZE}_seed{SEED}.cache")

train_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_y))
train_ds = train_ds.with_options(options)
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(
    load_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((va_paths, va_y))
val_ds = val_ds.with_options(options)
val_ds = val_ds.map(load_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
val_ds = val_ds.cache(val_cache_path)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
val_ds = val_ds.prefetch(AUTOTUNE)




## === cell 4
base = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # stability/speed

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()




## === cell 5
EPOCHS = 3

steps_per_epoch = int(np.ceil(len(tr_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(va_paths) / BATCH_SIZE))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 6
test_images = sub["image"].to_numpy()
test_paths = (TEST_DIR + "/" + sub["image"].astype(str)).to_numpy()

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.with_options(options)
test_ds = test_ds.map(
    load_preprocess_test, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.cache(test_cache_path)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
print("Pred probs shape:", probs.shape)




## === cell 7
THRESH = 0.5

mask = probs >= THRESH
argmax_idx = probs.argmax(axis=1)

idx2label_arr = np.array([idx2label[i] for i in range(NUM_CLASSES)], dtype=object)

pred_labels = []
for i in range(probs.shape[0]):
    chosen = np.flatnonzero(mask[i])
    if chosen.size == 0:
        chosen = np.array([argmax_idx[i]], dtype=np.int64)
    pred_labels.append(" ".join(idx2label_arr[chosen].tolist()))

submission = pd.DataFrame({"image": test_images, "labels": pred_labels})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)




## === cell 8
print("Competition Complete!!")
