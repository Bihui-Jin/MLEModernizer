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
print("tf.keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Thread config skipped:", e)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism config skipped:", e)

try:
    tf.config.optimizer.set_jit(False)
except Exception as e:
    print("XLA JIT config skipped:", e)

GPUS = tf.config.list_physical_devices("GPU")
print("Num GPUs:", len(GPUS))




## === cell 1
import pathlib


def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


DATA_ROOT = find_existing_path(
    [
        "../input/plant-pathology-2021-fgvc8",
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "/kaggle/data/plant-pathology-2021-fgvc8",
        "../input",
        "/kaggle/input",
        "/kaggle/data",
    ]
)

if DATA_ROOT is None:
    raise FileNotFoundError("Could not locate Kaggle input data root.")

if os.path.basename(DATA_ROOT) != "plant-pathology-2021-fgvc8":
    maybe = os.path.join(DATA_ROOT, "plant-pathology-2021-fgvc8")
    if os.path.exists(maybe):
        DATA_ROOT = maybe

TRAIN_CSV = find_existing_path(
    [
        os.path.join(DATA_ROOT, "train.csv"),
        "../input/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)
SAMPLE_SUB = find_existing_path(
    [
        os.path.join(DATA_ROOT, "sample_submission.csv"),
        "../input/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

TRAIN_IMG_DIR = find_existing_path(
    [
        os.path.join(DATA_ROOT, "train_images"),
        "../input/plant-pathology-2021-fgvc8/train_images",
        "/kaggle/input/plant-pathology-2021-fgvc8/train_images",
        "/kaggle/data/plant-pathology-2021-fgvc8/train_images",
    ]
)
TEST_IMG_DIR = find_existing_path(
    [
        os.path.join(DATA_ROOT, "test_images"),
        "../input/plant-pathology-2021-fgvc8/test_images",
        "/kaggle/input/plant-pathology-2021-fgvc8/test_images",
        "/kaggle/data/plant-pathology-2021-fgvc8/test_images",
    ]
)

if (
    TRAIN_CSV is None
    or SAMPLE_SUB is None
    or TRAIN_IMG_DIR is None
    or TEST_IMG_DIR is None
):
    raise FileNotFoundError(
        f"Missing required files/dirs. "
        f"TRAIN_CSV={TRAIN_CSV}, SAMPLE_SUB={SAMPLE_SUB}, "
        f"TRAIN_IMG_DIR={TRAIN_IMG_DIR}, TEST_IMG_DIR={TEST_IMG_DIR}"
    )

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)




## === cell 2
@tf.function
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image.set_shape([None, None, 3])  # stable tracing
    image = tf.image.convert_image_dtype(image, tf.float32)  # deterministic cast/255
    image = tf.image.resize(image, image_size, method="bilinear", antialias=False)
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32
IMG_SIZE = (512, 512)




## === cell 4
source = TEST_IMG_DIR


def list_images_sorted(folder):
    exts = (".jpg", ".jpeg", ".png")
    files = []
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file():
                n = entry.name
                if n.lower().endswith(exts):
                    files.append(n)
    files.sort()
    return files


test_files = list_images_sorted(source)
IMAGE_PATHS = [os.path.join(source, f) for f in test_files]

print("num test images:", len(IMAGE_PATHS))
print("first 3:", IMAGE_PATHS[:3])




## === cell 5
IMAGE_PATHS[:10]




## === cell 6
AUTO = tf.data.experimental.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(options)
    .map(
        lambda x: decode_image(x, label=None, image_size=IMG_SIZE),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .apply(tf.data.experimental.ignore_errors())
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)




## === cell 7
from tensorflow import keras


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
PRETRAINED_CANDIDATES = [
    "../input/4smresnet50/4SMResNet50.h5",
    "/kaggle/input/4smresnet50/4SMResNet50.h5",
    "/kaggle/data/4smresnet50/4SMResNet50.h5",
]
PRETRAINED_PATH = find_existing_path(PRETRAINED_CANDIDATES)
print("PRETRAINED_PATH:", PRETRAINED_PATH)

model = None
if PRETRAINED_PATH is not None:
    model = tf.keras.models.load_model(
        PRETRAINED_PATH, compile=False, custom_objects={"FixedDropout": FixedDropout}
    )
    print("Loaded pretrained model:", PRETRAINED_PATH)




## === cell 9
if model is None:
    train_df = pd.read_csv(TRAIN_CSV)
    print("train_df:", train_df.shape)
    print(train_df.head())

    class_names = [
        "scab",
        "frog_eye_leaf_spot",
        "complex",
        "rust",
        "powdery_mildew",
        "healthy",
    ]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    def encode_labels(label_str):
        y = np.zeros(len(class_names), dtype=np.float32)
        for token in str(label_str).split():
            if token in class_to_idx:
                y[class_to_idx[token]] = 1.0
        return y

    train_img_dir = TRAIN_IMG_DIR
    train_df["filepath"] = train_df["image"].apply(
        lambda x: os.path.join(train_img_dir, x)
    )
    train_df["y"] = train_df["labels"].apply(encode_labels)

    paths = train_df["filepath"].values
    ys = np.stack(train_df["y"].values)

    idx = np.arange(len(paths))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    paths = paths[idx]
    ys = ys[idx]

    split = int(0.95 * len(paths))
    tr_paths, va_paths = paths[:split], paths[split:]
    tr_ys, va_ys = ys[:split], ys[split:]

    def decode_with_label(p, y):
        return decode_image(p, y, image_size=IMG_SIZE)

    train_options = tf.data.Options()
    train_options.experimental_deterministic = True

    train_ds = (
        tf.data.Dataset.from_tensor_slices((tr_paths, tr_ys))
        .with_options(train_options)
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .map(decode_with_label, num_parallel_calls=AUTO, deterministic=True)
        .apply(tf.data.experimental.ignore_errors())
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )

    val_options = tf.data.Options()
    val_options.experimental_deterministic = True

    val_ds = (
        tf.data.Dataset.from_tensor_slices((va_paths, va_ys))
        .with_options(val_options)
        .map(decode_with_label, num_parallel_calls=AUTO, deterministic=True)
        .apply(tf.data.experimental.ignore_errors())
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    base.trainable = False

    inp = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = inp
    x = tf.keras.applications.efficientnet.preprocess_input(x * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(len(class_names), activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )

    model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)




## === cell 10
num_test = len(IMAGE_PATHS)
steps = (num_test + BATCH_SIZE - 1) // BATCH_SIZE
probs = model.predict(test_dataset, verbose=1, steps=steps)
temp_probs = probs
print("probs shape:", probs.shape)




## === cell 11
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

thr = np.array([0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float32)
thr2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

p5 = temp_probs[:, :5].astype(np.float32, copy=False)
m1 = p5 > thr[None, :]
m2 = p5 > thr2[None, :]
count2 = m2.sum(axis=1)

has_complex = m1[:, 2]
add_complex = (count2 >= 2) & (~has_complex)

idx_to_label = np.array([name[i] for i in range(5)], dtype=object)

pred_string = []
for i in range(p5.shape[0]):
    labels = idx_to_label[m1[i]].tolist()
    if add_complex[i]:
        labels.append("complex")
    if not labels:
        labels = [name[5]]
    pred_string.append(" ".join(labels))

print("example preds:", pred_string[:5])




## === cell 12
sub = pd.read_csv(SAMPLE_SUB)
sample_images = sub["image"].tolist()
if len(sample_images) == len(pred_string) and set(sample_images) == set(test_files):
    images_for_sub = sample_images
    pred_map = dict(zip(test_files, pred_string))
    labels_for_sub = [pred_map[i] for i in images_for_sub]
else:
    images_for_sub = test_files
    labels_for_sub = pred_string

df = pd.DataFrame({"image": images_for_sub, "labels": labels_for_sub})
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)




## === cell 13
assert df.shape[0] > 0
assert list(df.columns) == ["image", "labels"]
assert df["labels"].isna().sum() == 0
assert df["image"].isna().sum() == 0
print("submission.csv ready")
