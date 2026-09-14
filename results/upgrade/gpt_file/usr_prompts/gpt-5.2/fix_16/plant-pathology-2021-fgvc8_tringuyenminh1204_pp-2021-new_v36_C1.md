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
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
import pathlib




## === cell 2
@tf.function
def read_image_bytes(filename):
    return tf.io.read_file(filename)


@tf.function
def decode_image_bytes(bits, label=None, image_size=(512, 512)):
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image.set_shape([None, None, 3])
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1], float32
    if label is None:
        return image
    return image, label


@tf.function
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = read_image_bytes(filename)
    return decode_image_bytes(bits, label=label, image_size=image_size)




## === cell 3
BATCH_SIZE = 32
IMAGE_SIZE = (512, 512)



## === cell 4
source = "../input/plant-pathology-2021-fgvc8/test_images"
_valid_ext = (".jpg", ".jpeg", ".png")

_all = tf.io.gfile.glob(os.path.join(source, "*"))
IMAGE_PATHS = sorted({p for p in _all if os.path.splitext(p)[1].lower() in _valid_ext})



## === cell 5
IMAGE_PATHS[:5], len(IMAGE_PATHS)



## === cell 6
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print(train_df.shape, sample_sub.shape)
train_df.head()



## === cell 7
AUTO = tf.data.AUTOTUNE

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True
try:
    data_opts.experimental_optimization.apply_default_optimizations = True
    data_opts.experimental_optimization.map_and_batch_fusion = True
    data_opts.experimental_optimization.parallel_batch = True
except Exception:
    pass


def maybe_cache(ds, cache_in_memory=False, cache_path=None):
    if cache_in_memory:
        try:
            return ds.cache()
        except Exception:
            return ds
    if cache_path is not None:
        return ds.cache(cache_path)
    return ds


test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(data_opts)
    .map(read_image_bytes, num_parallel_calls=AUTO, deterministic=True)
    .map(
        lambda b: decode_image_bytes(b, image_size=IMAGE_SIZE),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 8
CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

labels_series = train_df["labels"].astype(str)

split_labels = labels_series.str.split()
pairs = split_labels.explode().dropna().to_frame("lab")
pairs["row"] = pairs.index.astype(np.int32)
pairs["col"] = pairs["lab"].map(class_to_idx).astype("Int64")
pairs = pairs.dropna(subset=["col"])
rows = pairs["row"].to_numpy(np.int32)
cols = pairs["col"].to_numpy(np.int32)

target = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
target[rows, cols] = 1.0

train_df["filepath"] = (
    TRAIN_DIR.rstrip("/") + "/" + train_df["image"].astype(str)
).values
train_df["target"] = list(target)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

X_tr = tr_df["filepath"].values
y_tr = np.stack(tr_df["target"].values)
X_va = va_df["filepath"].values
y_va = np.stack(va_df["target"].values)

print("Train/Val:", X_tr.shape, y_tr.shape, X_va.shape, y_va.shape)




## === cell 9
def decode_with_label_bytes(bits, label, image_size=IMAGE_SIZE):
    return decode_image_bytes(bits, label, image_size=image_size)


train_cache_path = "tfdata_train_cache"
val_cache_path = "tfdata_val_cache"

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    .with_options(data_opts)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(
        lambda x, y: (read_image_bytes(x), y),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .cache()  # cache bytes+labels in RAM; deterministic and much faster than disk cache of decoded tensors
    .map(
        lambda b, y: decode_with_label_bytes(b, y),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_va, y_va))
    .with_options(data_opts)
    .map(
        lambda x, y: (read_image_bytes(x), y),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .cache()  # cache bytes+labels in RAM
    .map(
        lambda b, y: decode_with_label_bytes(b, y),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)




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
inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = FixedDropout(0.2)(x)
outputs = layers.Dense(len(CLASSES), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    jit_compile=True,
)

model.summary()



## === cell 12
EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

probs = model.predict(test_dataset, verbose=1)
temp_probs = probs
temp_probs.shape



## === cell 13
probs[:2]



## === cell 14
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {0: 0.25, 1: 0.4, 2: 0.25, 3: 0.4, 4: 0.4}


def get_key(val):
    for key, value in name.items():
        if val == value:
            return key
    return "key doesn't exist"


thr = np.array([threshold[i] for i in range(5)], dtype=temp_probs.dtype)  # (5,)
over = temp_probs[:, :5] > thr  # (N,5) boolean

count = over.sum(axis=1)
has_complex = over[:, 2]
need_add_complex = (count >= 2) & (~has_complex)

labels_arr = np.array([name[i] for i in range(5)], dtype=object)

parts = np.where(over, labels_arr, "").astype(object)  # (N,5)
pred_string = parts[:, 0]
for j in range(1, parts.shape[1]):
    pred_string = np.char.add(np.char.add(pred_string, " "), parts[:, j])

pred_string = np.char.strip(np.char.replace(pred_string, "  ", " "))
pred_string = np.char.strip(
    np.char.replace(np.char.replace(pred_string, "  ", " "), "  ", " ")
)

pred_string = np.where(
    need_add_complex,
    np.where(pred_string == "", "complex", np.char.add(pred_string, " complex")),
    pred_string,
)

pred_string = np.where(pred_string == "", name[6], pred_string).astype(object)

pred_string[:10], len(pred_string)



## === cell 15
pred_string[:5]



## === cell 16
test_images = [os.path.basename(p) for p in IMAGE_PATHS]
assert len(test_images) == len(pred_string), (len(test_images), len(pred_string))

df = pd.DataFrame({"image": test_images, "labels": pred_string.tolist()})

df = df[["image", "labels"]]
df.to_csv("submission.csv", index=False)

print(df.shape)
print(df.head())
