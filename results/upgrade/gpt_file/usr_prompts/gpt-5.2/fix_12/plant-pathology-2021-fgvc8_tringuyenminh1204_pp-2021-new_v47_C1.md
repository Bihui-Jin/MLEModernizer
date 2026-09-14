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

print("tf:", tf.__version__)
print("keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    from tensorflow.keras import mixed_precision

    if tf.config.list_physical_devices("GPU"):
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled: mixed_float16")
    else:
        print("Mixed precision not enabled (no GPU detected).")
except Exception as e:
    print("Mixed precision setup skipped:", repr(e))

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass




## === cell 1
import pathlib

BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"




## === cell 2
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
    ]
)
def decode_file_nolabel(filename, image_size_hw):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)  # equivalent to cast/255
    image = tf.image.resize(
        image,
        (image_size_hw[0], image_size_hw[1]),
        method=tf.image.ResizeMethod.BILINEAR,
    )
    return image


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(6,), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
    ]
)
def decode_file_with_label(filename, label, image_size_hw):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(
        image,
        (image_size_hw[0], image_size_hw[1]),
        method=tf.image.ResizeMethod.BILINEAR,
    )
    return image, label




## === cell 3
BATCH_SIZE = 32
IMAGE_SIZE = (512, 512)
AUTO = tf.data.AUTOTUNE

DATASET_OPTS = tf.data.Options()
DATASET_OPTS.experimental_deterministic = False
try:
    DATASET_OPTS.experimental_optimization.apply_default_optimizations = True
    DATASET_OPTS.experimental_optimization.map_parallelization = True
    DATASET_OPTS.experimental_optimization.parallel_batch = True
    DATASET_OPTS.experimental_optimization.autotune = True
except Exception:
    pass

image_size_hw = tf.constant([IMAGE_SIZE[0], IMAGE_SIZE[1]], dtype=tf.int32)




## === cell 4
sub = pd.read_csv(SAMPLE_SUB)
test_images = sub["image"].astype(str).tolist()
test_paths = [os.path.join(TEST_IMG_DIR, fn) for fn in test_images]

if test_paths and (not os.path.exists(test_paths[0])):
    raise FileNotFoundError(f"Test image not found: {test_paths[0]}")

test_dataset = tf.data.Dataset.from_tensor_slices(test_paths).with_options(DATASET_OPTS)
test_dataset = test_dataset.map(
    lambda p: decode_file_nolabel(p, image_size_hw),
    num_parallel_calls=AUTO,
    deterministic=False,
)

try:
    test_dataset = test_dataset.cache()
except Exception:
    pass

test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)
test_dataset = test_dataset.prefetch(AUTO)




## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
train_df["image"] = train_df["image"].astype(str)
train_df["labels"] = train_df["labels"].astype(str)

classes = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
class_to_idx = {c: i for i, c in enumerate(classes)}

train_paths = [os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image"].tolist()]
if train_paths and (not os.path.exists(train_paths[0])):
    raise FileNotFoundError(f"Training image not found: {train_paths[0]}")

label_dummies = train_df["labels"].str.get_dummies(sep=" ")
label_dummies = label_dummies.reindex(columns=classes, fill_value=0)
y = label_dummies.to_numpy(dtype=np.float32)




## === cell 6
n = len(train_paths)
idx = np.arange(n)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_n = int(n * val_frac)

val_idx = idx[:val_n]
tr_idx = idx[val_n:]

tr_paths = [train_paths[i] for i in tr_idx]
va_paths = [train_paths[i] for i in val_idx]

tr_y = y[tr_idx]
va_y = y[val_idx]

train_steps = math.ceil(len(tr_paths) / BATCH_SIZE)
val_steps = math.ceil(len(va_paths) / BATCH_SIZE)


def make_ds(paths, labels, training, cache_path):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(DATASET_OPTS)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, lab: decode_file_with_label(p, lab, image_size_hw),
        num_parallel_calls=AUTO,
        deterministic=False,
    )

    try:
        ds = ds.cache()
    except Exception:
        pass

    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.prefetch(AUTO)
    return ds


train_ds = make_ds(
    tr_paths, tr_y, training=True, cache_path="/kaggle/working/cache_train.tfdata"
)
val_ds = make_ds(
    va_paths, va_y, training=False, cache_path="/kaggle/working/cache_val.tfdata"
)




## === cell 7
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




## === cell 8
inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs,
    pooling="avg",
)
x = base.output
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = FixedDropout(0.2)(x)

outputs = tf.keras.layers.Dense(len(classes), activation="sigmoid", dtype="float32")(x)

model = tf.keras.Model(inputs, outputs)

base.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)




## === cell 9
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1,
)

base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
)
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 10
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs




## === cell 11
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.4, 1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}
healthy_threshold = 0.5  # used only when no disease exceeds threshold

disease_probs = temp_probs[:, :5]
thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)[None, :]
hits = disease_probs > thr  # (N,5) boolean
disease_names = np.array([name[i] for i in range(5)], dtype=object)

rows = []
append = rows.append
for r in range(hits.shape[0]):
    idxs = np.flatnonzero(hits[r])
    if idxs.size == 0:
        append(name[5])
    else:
        append(" ".join(disease_names[idxs].tolist()))
pred_string = rows

submission = pd.DataFrame({"image": test_images, "labels": pred_string})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[0] == len(test_images)
assert list(submission.columns) == ["image", "labels"]
assert submission["labels"].isna().sum() == 0
