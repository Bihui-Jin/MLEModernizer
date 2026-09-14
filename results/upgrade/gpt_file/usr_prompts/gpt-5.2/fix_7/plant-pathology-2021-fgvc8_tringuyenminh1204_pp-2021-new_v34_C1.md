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

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Warning: could not enable XLA JIT:", e)

tf.config.experimental.enable_op_determinism()
tf.keras.backend.set_floatx("float32")

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Warning: could not set threading config:", e)



## === cell 1
BASE = "/kaggle/input/plant-pathology-2021-fgvc8"
if not os.path.exists(BASE):
    BASE = "/kaggle/input"  # fallback

TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV), TRAIN_CSV)
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB_CSV), SAMPLE_SUB_CSV)
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR), TRAIN_IMG_DIR)
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR), TEST_IMG_DIR)



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB_CSV)

print(train_df.head())
print(sub_df.head())
print("Train rows:", len(train_df), "Test rows:", len(sub_df))

CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


def labels_to_multihot(s: str):
    labs = str(s).strip().split()
    y = np.zeros(len(CLASSES), dtype=np.float32)
    for lab in labs:
        if lab in CLASS2IDX:
            y[CLASS2IDX[lab]] = 1.0
    return y


y = np.stack([labels_to_multihot(s) for s in train_df["labels"].values])
train_paths = np.asarray(
    [os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image"].values], dtype=np.str_
)

test_images = sub_df["image"].values.tolist()
test_paths = np.asarray(
    [os.path.join(TEST_IMG_DIR, fn) for fn in test_images], dtype=np.str_
)

print(
    "Prepared paths. Train paths:", train_paths.shape, "Test paths:", test_paths.shape
)



## === cell 3
IMG_SIZE = (512, 512)
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE


@tf.function
def decode_image(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)  # exact /255 conversion
    image.set_shape([None, None, 3])
    image = tf.image.resize(
        image, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    if label is None:
        return image
    return image, label




## === cell 4
idx = np.arange(len(train_paths))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_n = int(len(idx) * val_frac)

val_idx = idx[:val_n]
trn_idx = idx[val_n:]

trn_paths = train_paths[trn_idx]
trn_y = y[trn_idx]
val_paths = train_paths[val_idx]
val_y = y[val_idx]

options = tf.data.Options()
options.experimental_deterministic = True

try:
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass

train_ds = (
    tf.data.Dataset.from_tensor_slices((trn_paths, trn_y))
    .with_options(options)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_y))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

steps_per_epoch = int(math.ceil(len(trn_paths) / BATCH_SIZE))
val_steps = int(math.ceil(len(val_paths) / BATCH_SIZE))
test_steps = int(math.ceil(len(test_paths) / BATCH_SIZE))

print(train_ds, val_ds, test_ds)
print(
    "steps_per_epoch:",
    steps_per_epoch,
    "val_steps:",
    val_steps,
    "test_steps:",
    test_steps,
)



## === cell 6
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_tensor=inputs
)
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=32,
)

model.summary()



## === cell 7
EPOCHS = 3
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
)



## === cell 8
probs = model.predict(test_ds, verbose=1)
print("probs shape:", probs.shape)

temp_probs = probs



## === cell 9
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.5, 1: 0.4, 2: 0.5, 3: 0.5, 4: 0.5}

p = np.asarray(temp_probs, dtype=np.float32)
thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)
sel = p[:, :5] > thr[None, :]  # (N,5) boolean

cnt = sel.sum(axis=1)
complex_selected = sel[:, 2]
add_complex = (cnt >= 2) & (~complex_selected)

labels_per_i = [name[i] for i in range(5)]
pred_string = []
for r in range(sel.shape[0]):
    parts = [labels_per_i[i] for i in range(5) if sel[r, i]]
    if add_complex[r]:
        parts.append("complex")
    if not parts:
        parts = [name[5]]
    pred_string.append(" ".join(parts))

print("Example preds:", pred_string[:10])

submission = pd.DataFrame({"image": test_images, "labels": pred_string})

assert len(submission) == len(sub_df), (len(submission), len(sub_df))
assert list(submission.columns) == ["image", "labels"]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
