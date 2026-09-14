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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(
        False
    )  # keep deterministic numerics; XLA can change FP slightly
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)




## === cell 1
import pathlib




## === cell 2
@tf.function
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.AREA)
    image.set_shape((image_size[0], image_size[1], 3))
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 64
IMAGE_SIZE = (512, 512)




## === cell 4
source = "../input/plant-pathology-2021-fgvc8/test_images"
if not os.path.isdir(source):
    alt = "/kaggle/input/plant-pathology-2021-fgvcvc8/test_images"
    if os.path.isdir(alt):
        source = alt
    else:
        alt2 = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
        if os.path.isdir(alt2):
            source = alt2

p = pathlib.Path(source)
paths = []
for suf in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    paths.extend([str(x) for x in p.glob(suf)])

paths = sorted(set(paths))
test_files = [os.path.basename(x) for x in paths]
IMAGE_PATHS = paths

print("Test images:", len(IMAGE_PATHS))
print("First test image:", test_files[0] if test_files else None)




## === cell 5
IMAGE_PATHS[:5]




## === cell 6
AUTO = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.autotune_buffers = True
try:
    options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(options)
    .map(
        lambda x: decode_image(x, label=None, image_size=IMAGE_SIZE),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
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
MODEL_PATH = "../input/3smresnet50/3SMResNet50.h5"
if not os.path.exists(MODEL_PATH):
    abs_variant = "/kaggle/input/3smresnet50/3SMResNet50.h5"
    if os.path.exists(abs_variant):
        MODEL_PATH = abs_variant

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"
if not os.path.exists(train_csv_path):
    train_csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
if not os.path.isdir(train_img_dir):
    train_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"

class_names = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]


def labels_to_vector(label_str):
    parts = str(label_str).split()
    vec = np.zeros((len(class_names),), dtype=np.float32)
    for i, c in enumerate(class_names):
        if c in parts:
            vec[i] = 1.0
    return vec


if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False, custom_objects={"FixedDropout": FixedDropout}
    )
    print("Loaded external model:", MODEL_PATH)
else:
    print(
        "External model not found; training fallback model from train.csv:",
        train_csv_path,
    )

    train_df = pd.read_csv(train_csv_path)
    train_paths = [
        os.path.join(train_img_dir, img) for img in train_df["image"].tolist()
    ]
    y = np.stack([labels_to_vector(s) for s in train_df["labels"].tolist()], axis=0)

    idx = np.arange(len(train_paths))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_paths = np.array(train_paths, dtype=object)[tr_idx].tolist()
    va_paths = np.array(train_paths, dtype=object)[va_idx].tolist()
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((tr_paths, y_tr))
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .map(
            lambda pth, lab: decode_image(pth, lab, image_size=IMAGE_SIZE),
            num_parallel_calls=AUTO,
            deterministic=True,
        )
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((va_paths, y_va))
        .map(
            lambda pth, lab: decode_image(pth, lab, image_size=IMAGE_SIZE),
            num_parallel_calls=AUTO,
            deterministic=True,
        )
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)




## === cell 9
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs
print("probs shape:", np.shape(probs))




## === cell 10
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.25, 1: 0.35, 2: 0.25, 3: 0.35, 4: 0.35}

thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)  # (5,)
above = temp_probs[:, :5] > thr[None, :]  # (N,5)
counts = above.sum(axis=1)

cls = [name[i] for i in range(5)]
complex_idx = 2

pred_string = []
for row_above, cnt in zip(above, counts):
    parts = [cls[i] for i, flag in enumerate(row_above) if flag]

    if cnt >= 2:
        if not row_above[complex_idx]:
            parts.append("complex")

    pred_string.append(name[5] if not parts else " ".join(parts))

len(pred_string), pred_string[:5]




## === cell 11
df = pd.DataFrame({"image": test_files, "labels": pred_string})
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
