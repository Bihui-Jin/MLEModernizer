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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
except Exception:
    pass

tf.config.threading.set_inter_op_parallelism_threads(min(2, os.cpu_count() or 2))
tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 8))

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 1
import pathlib




## === cell 2
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.AREA, antialias=False
    )
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32



## === cell 4
source = "../input/plant-pathology-2021-fgvc8/test_images"

image_files = [
    f
    for f in os.listdir(source)
    if re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$", f, re.IGNORECASE)
]
image_files = sorted(image_files)
IMAGE_PATHS = [os.path.join(source, f) for f in image_files]

print("Num test images found:", len(IMAGE_PATHS))
print("First 3:", image_files[:3])



## === cell 5
IMAGE_PATHS[:5]



## === cell 6
sample_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
if os.path.exists(sample_path):
    ss = pd.read_csv(sample_path)
    print("sample_submission rows:", len(ss))
    if len(ss) != len(image_files):
        print("WARNING: sample_submission size != discovered test images size.")



## === cell 7
AUTO = tf.data.experimental.AUTOTUNE

data_options = tf.data.Options()
data_options.experimental_deterministic = True
try:
    data_options.autotune.enabled = True
except Exception:
    pass
try:
    data_options.experimental_optimization.apply_default_optimizations = True
    data_options.experimental_optimization.map_fusion = True
    data_options.experimental_optimization.parallel_batch = True
except Exception:
    pass
try:
    data_options.threading.private_threadpool_size = max(
        4, min(16, os.cpu_count() or 4)
    )
except Exception:
    pass



## === cell 8
test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(data_options)
    .map(
        lambda x: decode_image(x, label=None, image_size=(512, 512)),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .cache(os.path.join(".", "cache_test_512"))
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 9
import tensorflow as tf
from tensorflow import keras




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
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"

train_df = pd.read_csv(train_csv_path)
print("train.csv shape:", train_df.shape)
print(train_df.head())

CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

for c in CLASSES:
    train_df[c] = (
        train_df["labels"]
        .astype(str)
        .str.contains(rf"(?:^| ){re.escape(c)}(?: |$)", regex=True)
        .astype(np.float32)
    )

train_files_set = set(os.listdir(train_img_dir))
train_df = train_df[train_df["image"].isin(train_files_set)].reset_index(drop=True)
train_df["filepath"] = train_df["image"].map(lambda x: os.path.join(train_img_dir, x))
print("train_df usable rows:", len(train_df))



## === cell 12
IMAGE_SIZE = (512, 512)

paths = train_df["filepath"].values
targets = train_df[CLASSES].to_numpy(dtype=np.float32)

idx = np.arange(len(paths))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_paths, va_paths = paths[tr_idx], paths[va_idx]
tr_y, va_y = targets[tr_idx], targets[va_idx]


def make_ds(x_paths, y=None, training=False, cache_name=None):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(x_paths).with_options(data_options)
        ds = ds.map(
            lambda p: decode_image(p, label=None, image_size=IMAGE_SIZE),
            num_parallel_calls=AUTO,
            deterministic=True,
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices((x_paths, y)).with_options(data_options)
        ds = ds.map(
            lambda p, t: decode_image(p, label=t, image_size=IMAGE_SIZE),
            num_parallel_calls=AUTO,
            deterministic=True,
        )

    if cache_name is not None:
        cache_path = os.path.join(".", cache_name)
        ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    return ds


train_ds = make_ds(tr_paths, tr_y, training=True, cache_name="cache_train_512")
val_ds = make_ds(va_paths, va_y, training=False, cache_name="cache_val_512")

print("train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("val batches:", tf.data.experimental.cardinality(val_ds).numpy())



## === cell 13
base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)
base.trainable = False  # keep fast/stable under 600s

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    jit_compile=True,
)

model.summary()

EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 14
probs = model.predict(test_dataset, verbose=1)
temp_probs = np.asarray(probs)
print("probs shape:", temp_probs.shape)
print("probs min/max:", float(temp_probs.min()), float(temp_probs.max()))



## === cell 15
name = np.array(CLASSES, dtype=object)
thr = 0.5

mask = temp_probs > thr  # (N, C)
has_any = mask.any(axis=1)
argmax_idx = np.argmax(temp_probs, axis=1)

pred_string = []
for i in range(mask.shape[0]):
    if has_any[i]:
        labs = name[mask[i]]
    else:
        labs = name[[int(argmax_idx[i])]]
    pred_string.append(" ".join(labs.tolist()))

print("Num predictions:", len(pred_string))
print(pred_string[:10])



## === cell 16
if len(image_files) != len(pred_string):
    raise ValueError(
        f"Mismatch: {len(image_files)} test images but {len(pred_string)} predictions. "
        "Check the dataset pipeline/model output."
    )

df = pd.DataFrame({"image": image_files, "labels": pred_string})
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
