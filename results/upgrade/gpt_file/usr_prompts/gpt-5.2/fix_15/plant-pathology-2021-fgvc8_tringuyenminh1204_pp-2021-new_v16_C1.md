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
import os, random, math, re

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Could not enable op determinism (ok):", repr(e))

try:
    tf.config.optimizer.set_jit(False)
except Exception as e:
    print("Could not set jit (ok):", repr(e))

os.environ["TF_DETERMINISTIC_OPS"] = "1"

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if gpus:
        print("GPUs:", gpus)
except Exception as e:
    print("Could not set memory growth (ok):", repr(e))




## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(path + "train.csv")
sub = pd.read_csv(path + "sample_submission.csv")

print(train.shape, sub.shape)
print(train.columns.tolist(), sub.columns.tolist())
train.head()




## === cell 2
AUTO = tf.data.experimental.AUTOTUNE

TRAIN_IMG_DIR = os.path.join(path, "train_images")
TEST_IMG_DIR = os.path.join(path, "test_images")

train_paths = np.asarray(
    [os.path.join(TRAIN_IMG_DIR, fn) for fn in train["image"].values], dtype=str
)
test_paths = np.asarray(
    [os.path.join(TEST_IMG_DIR, fn) for fn in sub["image"].values], dtype=str
)

missing_train = int(not os.path.exists(train_paths[0]))
missing_test = int(not os.path.exists(test_paths[0]))
print(
    "missing train (first 1):", missing_train, "missing test (first 1):", missing_test
)




## === cell 3
CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def labels_to_multihot(s: str) -> np.ndarray:
    y = np.zeros(len(CLASSES), dtype=np.float32)
    for lab in str(s).split():
        if lab in CLASS_TO_IDX:
            y[CLASS_TO_IDX[lab]] = 1.0
    return y


y_all = np.stack([labels_to_multihot(s) for s in train["labels"].values])
print("y shape:", y_all.shape, "positives per class:", y_all.sum(axis=0))




## === cell 4
IMG_SIZE = (224, 224)  # ResNet50 default; keeps runtime reasonable
BATCH_SIZE = 32
EPOCHS = 2  # keep as-is per original intent

IMG_SIZE_T = tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32)


def decode_image_xy(filename, label):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(image, IMG_SIZE_T)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label


def decode_image_x(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(image, IMG_SIZE_T)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image




## === cell 5
idx = np.arange(len(train_paths))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_paths = train_paths[tr_idx]
va_paths = train_paths[va_idx]
tr_y = y_all[tr_idx]
va_y = y_all[va_idx]

opt = tf.data.Options()
opt.experimental_deterministic = True
try:
    opt.experimental_optimization.apply_default_optimizations = True
except Exception as e:
    print("Could not set apply_default_optimizations (ok):", repr(e))

try:
    opt.experimental_optimization.map_parallelization = True
    opt.experimental_optimization.parallel_batch = True
    opt.experimental_optimization.map_and_batch_fusion = True
except Exception as e:
    print("Could not enable extra tf.data optimizations (ok):", repr(e))

train_steps = int(math.ceil(len(tr_paths) / BATCH_SIZE))
val_steps = int(math.ceil(len(va_paths) / BATCH_SIZE))

HAS_GPU = len(tf.config.list_physical_devices("GPU")) > 0


def make_train_ds(paths, y):
    ds = tf.data.Dataset.from_tensor_slices((paths, y)).with_options(opt)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_image_xy, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.ignore_errors()
    ds = ds.prefetch(AUTO)
    return ds


def make_eval_ds(paths, y=None):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(opt)
        ds = ds.map(decode_image_x, num_parallel_calls=AUTO, deterministic=True)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, y)).with_options(opt)
        ds = ds.map(decode_image_xy, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.ignore_errors()
    ds = ds.prefetch(AUTO)
    return ds


train_ds = make_train_ds(tr_paths, tr_y)
val_ds = make_eval_ds(va_paths, va_y)
test_ds = make_eval_ds(test_paths)

print(
    "Datasets ready:",
    "train steps/epoch:",
    train_steps,
    "val steps:",
    val_steps,
    "test batches:",
    tf.data.experimental.cardinality(test_ds).numpy(),
    "HAS_GPU:",
    HAS_GPU,
)




## === cell 6
base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)

base.trainable = False

x = base.output
out = tf.keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)
model = tf.keras.Model(inputs=base.input, outputs=out)

model.summary()




## === cell 7
spe = max(1, train_steps)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    steps_per_execution=spe,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    validation_freq=EPOCHS,
    verbose=1,
)




## === cell 8
probs = model.predict(test_ds, verbose=1)
print("probs shape:", probs.shape)

assert probs.shape[0] == len(sub), (probs.shape[0], len(sub))

name = np.array(CLASSES, dtype=object)

thr = np.array([0.30, 0.50, 0.30, 0.50, 0.50, 0.50], dtype=np.float32)
sel = probs > thr[None, :]

rows, cols = np.nonzero(sel)
parts = [[] for _ in range(sel.shape[0])]
for r, c in zip(rows.tolist(), cols.tolist()):
    parts[r].append(name[c])

pred_string = ["healthy" if len(p) == 0 else " ".join(p) for p in parts]

submission = sub.copy()
submission["labels"] = pred_string

submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
