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

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)




## === cell 1
BASE = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

print(train_df.head())
print(sub_df.head())
print("train rows:", len(train_df), "test rows:", len(sub_df))




## === cell 2
IMAGE_SIZE = (512, 512)


@tf.function
def _decode_and_resize(bits):
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)  # == cast/255
    image = tf.image.resize(image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.ensure_shape(image, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    return image


def decode_image(filename, label=None, image_size=IMAGE_SIZE):
    bits = tf.io.read_file(filename)
    image = _decode_and_resize(bits)
    if label is None:
        return image
    return image, label




## === cell 3
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
SEED = 1337

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        max(2, (os.cpu_count() or 2) // 2)
    )
except Exception as e:
    print("Threading config skipped:", repr(e))

print("Global XLA JIT: not explicitly enabled (model jit_compile=True is used)")




## === cell 4
test_images = sub_df["image"].astype(str).tolist()
IMAGE_PATHS = [os.path.join(TEST_IMG_DIR, fn) for fn in test_images]

sample_check = IMAGE_PATHS[: min(10, len(IMAGE_PATHS))]
missing = [p for p in sample_check if not os.path.exists(p)]
print("Missing test images (sample check):", len(missing))




## === cell 5
data_opts = tf.data.Options()
data_opts.experimental_deterministic = True

try:
    data_opts.threading.private_threadpool_size = min(32, max(8, (os.cpu_count() or 8)))
    data_opts.threading.max_intra_op_parallelism = 0
except Exception:
    pass

try:
    data_opts.experimental_optimization.map_and_batch_fusion = True
    data_opts.experimental_optimization.parallel_batch = True
    data_opts.experimental_optimization.autotune_buffers = True
except Exception:
    pass

try:
    data_opts.experimental_slack = True
except Exception:
    pass

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(data_opts)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)




## === cell 6
CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

train_paths = [
    os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image"].astype(str).tolist()
]

label_lists = train_df["labels"].astype(str).str.split()

y = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
for cls, j in class_to_idx.items():
    y[:, j] = label_lists.apply(lambda toks, c=cls: float(c in toks)).to_numpy(
        dtype=np.float32
    )

print("Train paths:", len(train_paths), "y shape:", y.shape)




## === cell 7
idx = np.arange(len(train_paths))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

val_frac = 0.1
n_val = int(len(idx) * val_frac)
val_idx = idx[:n_val]
trn_idx = idx[n_val:]

trn_paths = np.asarray(train_paths, dtype=object)[trn_idx].tolist()
val_paths = np.asarray(train_paths, dtype=object)[val_idx].tolist()
y_trn = y[trn_idx]
y_val = y[val_idx]

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, "train_decode_resize.cache")
VAL_CACHE = os.path.join(CACHE_DIR, "val_decode_resize.cache")


def make_ds(paths, labels=None, training=False, cache_path=None):
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(decode_image, num_parallel_calls=AUTO, deterministic=True)
        if cache_path is not None:
            ds = ds.cache(cache_path)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(decode_image, num_parallel_calls=AUTO, deterministic=True)
        if cache_path is not None:
            ds = ds.cache(cache_path)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.prefetch(AUTO)
    return ds.with_options(data_opts)


train_ds = make_ds(trn_paths, y_trn, training=True, cache_path=TRAIN_CACHE)
val_ds = make_ds(val_paths, y_val, training=False, cache_path=VAL_CACHE)

steps_per_epoch = int(math.ceil(len(trn_paths) / BATCH_SIZE))
validation_steps = int(math.ceil(len(val_paths) / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)
print("Cache dir:", CACHE_DIR)




## === cell 8
def _warm_cache(ds, steps, name):
    it = iter(ds)
    for _ in range(steps):
        next(it)
    print(f"Warmed cache for {name}: {steps} batches")


need_train_warm = not (
    os.path.exists(TRAIN_CACHE) or os.path.exists(TRAIN_CACHE + ".index")
)
need_val_warm = not (os.path.exists(VAL_CACHE) or os.path.exists(VAL_CACHE + ".index"))

if need_train_warm:
    _warm_cache(
        make_ds(trn_paths, y_trn, training=False, cache_path=TRAIN_CACHE),
        steps_per_epoch,
        "train",
    )
if need_val_warm:
    _warm_cache(
        make_ds(val_paths, y_val, training=False, cache_path=VAL_CACHE),
        validation_steps,
        "val",
    )




## === cell 9
base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    pooling="avg",
)
inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = inputs
x = keras.applications.resnet50.preprocess_input(x * 255.0)
x = base(x, training=False)
outputs = layers.Dense(len(CLASSES), activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    jit_compile=True,
    steps_per_execution=16,
)

model.summary()




## === cell 10
EPOCHS = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 11
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs
print("probs shape:", probs.shape)

name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
threshold2 = {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3}

thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)
thr2 = np.array([threshold2[i] for i in range(5)], dtype=np.float32)

above_thr = temp_probs[:, :5] > thr[None, :]
above_thr2 = temp_probs[:, :5] > thr2[None, :]
cnt2 = above_thr2.sum(axis=1)

pred_string = []
for row_idx in range(temp_probs.shape[0]):
    hits = np.nonzero(above_thr[row_idx])[0].tolist()
    labels = [name[i] for i in hits]

    if cnt2[row_idx] > 2:
        if 2 not in hits:
            labels.append("complex")

    if not labels:
        pred_string.append(name[6])
    else:
        pred_string.append(" ".join(labels))

print("Example preds:", pred_string[:10])

out_df = pd.DataFrame(
    {
        "image": test_images[: len(pred_string)],
        "labels": pred_string[: len(test_images)],
    }
)
out_df.to_csv("submission.csv", index=False)
print(out_df.head())
print("Wrote submission.csv with rows:", len(out_df))
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0
