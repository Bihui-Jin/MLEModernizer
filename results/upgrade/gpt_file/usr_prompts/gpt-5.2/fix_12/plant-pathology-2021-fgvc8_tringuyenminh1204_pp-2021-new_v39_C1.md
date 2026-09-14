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
import os, re, math, random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model

print("tf:", tf.__version__)

try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception as e:
    print("GPU memory growth not set:", repr(e))

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



## === cell 1
import pathlib


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


DATA_ROOT = _first_existing(
    [
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "/kaggle/data/plant-pathology-2021-fgvc8",
        "../input/plant-pathology-2021-fgvc8",
        "../kaggle/input/plant-pathology-2021-fgvc8",
    ]
)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate dataset root 'plant-pathology-2021-fgvc8' in expected locations."
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

for p in [TRAIN_CSV, SAMPLE_SUB_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected path: {p}")

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV:", TRAIN_CSV)
print("TEST_IMG_DIR:", TEST_IMG_DIR)




## === cell 2
@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ],
)
def _decode_image_labeled(filename, label):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (512, 512), antialias=False)
    return image, label


@tf.function(
    reduce_retracing=True,
    input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)],
)
def _decode_image_unlabeled(filename):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (512, 512), antialias=False)
    return image


def decode_image(filename, label=None, image_size=(512, 512)):
    if label is None:
        return _decode_image_unlabeled(filename)
    return _decode_image_labeled(filename, label)




## === cell 3
BATCH_SIZE = 32
IMAGE_SIZE = (512, 512)



## === cell 4
source = TEST_IMG_DIR
_valid_ext = (".jpg", ".jpeg", ".png")
IMAGE_PATHS = sorted(
    [
        entry.path
        for entry in os.scandir(source)
        if entry.is_file() and entry.name.lower().endswith(_valid_ext)
    ]
)
TEST_FILENAMES = [os.path.basename(p) for p in IMAGE_PATHS]

print("Num test images:", len(IMAGE_PATHS))
print("First test image:", TEST_FILENAMES[0] if TEST_FILENAMES else None)



## === cell 5
IMAGE_PATHS[:5]



## === cell 6
train_df = pd.read_csv(TRAIN_CSV)
print(train_df.head())
print("Train rows:", len(train_df))

labels_series = train_df["labels"].astype(str)
all_labels = sorted(
    {lab for s in labels_series.tolist() for lab in s.split(" ") if lab}
)
label2idx = {l: i for i, l in enumerate(all_labels)}
idx2label = {i: l for l, i in label2idx.items()}
NUM_CLASSES = len(all_labels)

print("NUM_CLASSES:", NUM_CLASSES)
print("Classes:", all_labels)

targets_df = labels_series.str.get_dummies(sep=" ").reindex(
    columns=all_labels, fill_value=0
)
targets = targets_df.to_numpy(dtype=np.float32, copy=False)

train_df["filepath"] = TRAIN_IMG_DIR + "/" + train_df["image"].astype(str)



## === cell 7
AUTO = tf.data.experimental.AUTOTUNE



## === cell 8
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Threading config not applied:", repr(e))

_HAS_GPU = len(tf.config.list_physical_devices("GPU")) > 0

test_options = tf.data.Options()
test_options.autotune.enabled = True
test_options.threading.private_threadpool_size = 0
test_options.deterministic = True
test_options.experimental_optimization.map_parallelization = True
test_options.experimental_optimization.parallel_batch = True
test_options.experimental_optimization.map_and_batch_fusion = True

test_dataset = (
    tf.data.Dataset.from_tensor_slices(np.asarray(IMAGE_PATHS, dtype=np.str_))
    .with_options(test_options)
    .map(
        lambda x: decode_image(x, label=None, image_size=IMAGE_SIZE),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
if _HAS_GPU:
    test_dataset = test_dataset.apply(
        tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTO)
    )



## === cell 9
idx = np.arange(len(train_df))
np.random.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr = train_df.iloc[tr_idx].reset_index(drop=True)
va = train_df.iloc[va_idx].reset_index(drop=True)

y_tr = targets[tr_idx]
y_va = targets[va_idx]


@tf.function(reduce_retracing=True)
def _aug(img, lab):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED)
    return img, lab


def make_ds(paths, y, training=True, cache_name="cache"):
    paths = np.asarray(paths, dtype=np.str_)
    y = np.asarray(y, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    ds_options = tf.data.Options()
    ds_options.autotune.enabled = True
    ds_options.threading.private_threadpool_size = 0
    ds_options.experimental_optimization.map_parallelization = True
    ds_options.experimental_optimization.parallel_batch = True
    ds_options.experimental_optimization.map_and_batch_fusion = True
    ds_options.deterministic = not training
    ds = ds.with_options(ds_options)

    if training:
        ds = ds.shuffle(
            min(int(paths.shape[0]), 4096), seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda p, t: decode_image(p, t, image_size=IMAGE_SIZE),
        num_parallel_calls=AUTO,
        deterministic=not training,
    )

    cache_path = os.path.join("/kaggle/working", cache_name)
    ds = ds.cache(cache_path)

    if training:
        ds = ds.map(_aug, num_parallel_calls=AUTO, deterministic=False)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    if _HAS_GPU:
        ds = ds.apply(
            tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTO)
        )
    return ds


train_ds = make_ds(tr["filepath"].values, y_tr, training=True, cache_name="train.cache")
val_ds = make_ds(va["filepath"].values, y_va, training=False, cache_name="val.cache")




## === cell 10
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 11
try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))

inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = FixedDropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    jit_compile=True,
)

EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## === cell 12
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs
print("Pred probs shape:", temp_probs.shape)



## === cell 13
temp_probs[:2]



## === cell 14
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {0: 0.25, 1: 0.35, 2: 0.25, 3: 0.35, 4: 0.35}

needed = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
missing = [c for c in needed if c not in label2idx]
if missing:
    raise ValueError(f"Missing expected classes in train.csv label space: {missing}")

idx_scab = label2idx["scab"]
idx_fels = label2idx["frog_eye_leaf_spot"]
idx_complex = label2idx["complex"]
idx_rust = label2idx["rust"]
idx_pm = label2idx["powdery_mildew"]
idx_healthy = label2idx["healthy"]

disease_indices = np.array(
    [idx_scab, idx_fels, idx_complex, idx_rust, idx_pm], dtype=np.int32
)
disease_names = np.array(
    ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"], dtype=object
)
thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)

p5 = temp_probs[:, disease_indices]
hits = p5 > thr  # (N,5) boolean

counts = hits.sum(axis=1)
has_complex = hits[:, 2]
need_add_complex = (counts >= 2) & (~has_complex)

picked = [disease_names[row].tolist() for row in hits]
if np.any(need_add_complex):
    for i in np.flatnonzero(need_add_complex):
        picked[i].append("complex")
pred_string = [" ".join(parts) if parts else "healthy" for parts in picked]

print("Example preds:", pred_string[:5])



## === cell 15
pred_string[:10]



## === cell 16
df = pd.DataFrame({"image": TEST_FILENAMES, "labels": pred_string})
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with rows:", len(df))
