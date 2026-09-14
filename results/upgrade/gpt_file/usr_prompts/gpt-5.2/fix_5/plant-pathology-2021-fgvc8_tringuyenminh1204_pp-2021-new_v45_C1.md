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

print("TensorFlow:", tf.__version__)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTO = tf.data.AUTOTUNE
tf.keras.backend.clear_session()



## === cell 1
import pathlib




## === cell 2
def _decode_any_image(bits):
    return tf.cond(
        tf.image.is_jpeg(bits),
        lambda: tf.io.decode_jpeg(bits, channels=3),
        lambda: tf.io.decode_image(bits, channels=3, expand_animations=False),
    )


def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = _decode_any_image(bits)
    image = tf.image.convert_image_dtype(image, tf.float32)  # cast/255 equivalent
    image = tf.image.resize(image, image_size)
    image.set_shape([image_size[0], image_size[1], 3])
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32



## === cell 4
source = "../input/plant-pathology-2021-fgvc8/test_images"

try:
    IMAGE_PATHS = sorted(
        [
            os.path.join(source, f)
            for f in os.listdir(source)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
except FileNotFoundError:
    IMAGE_PATHS = []



## === cell 5
IMAGE_PATHS[:5], len(IMAGE_PATHS)



## === cell 6
SAMPLE_SUB_PATH = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image"].tolist()
test_image_paths = [os.path.join(source, img) for img in test_images]

missing = [p for p in test_image_paths if not tf.io.gfile.exists(p)]
print("Test images:", len(test_image_paths), " Missing:", len(missing))



## === cell 7
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_image_paths)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 8
import tensorflow as tf
from tensorflow import keras




## === cell 9
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




## === cell 10
TRAIN_CSV_PATH = "../input/plant-pathology-2021-fgvc8/train.csv"
TRAIN_IMG_DIR = "../input/plant-pathology-2021-fgvc8/train_images"

train_df = pd.read_csv(TRAIN_CSV_PATH)

classes = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
class_to_idx = {c: i for i, c in enumerate(classes)}


def labels_to_vec_series(labels_series: pd.Series) -> np.ndarray:
    mats = []
    for c in classes:
        mats.append(
            labels_series.str.contains(rf"(^| ){re.escape(c)}( |$)", regex=True)
            .astype(np.float32)
            .to_numpy()
        )
    return np.stack(mats, axis=1)


train_df["filepath"] = train_df["image"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))
targets = labels_to_vec_series(train_df["labels"])
train_df["target"] = list(targets)

rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_size = int(0.1 * len(train_df))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

IMAGE_SIZE = (512, 512)


def decode_image_with_label(filename, label, image_size=IMAGE_SIZE):
    bits = tf.io.read_file(filename)
    image = _decode_any_image(bits)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image.set_shape([image_size[0], image_size[1], 3])
    return image, label


train_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (tr_df["filepath"].values, np.stack(tr_df["target"].values))
    )
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(decode_image_with_label, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (val_df["filepath"].values, np.stack(val_df["target"].values))
    )
    .map(decode_image_with_label, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(len(classes), activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

EPOCHS = 3
history = model.fit(
    train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=2
)



## === cell 11
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs



## === cell 12
probs.shape



## === cell 13
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01}
threshold2 = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01}


def get_key(val):
    for key, value in name.items():
        if val == value:
            return key
    return "key doesn't exist"


thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)
mask = temp_probs[:, :5] > thr[None, :]

class_names = np.array([name[i] for i in range(5)], dtype=object)

class_strs = np.where(mask, class_names[None, :], "")
joined = np.char.strip(
    np.char.replace(np.sum(class_strs.astype("U"), axis=1), "  ", " ")
)
for _ in range(3):
    joined = np.char.replace(joined, "  ", " ")
joined = np.char.strip(joined)

pred_string = np.where(mask.any(axis=1), joined.astype(object), name[6]).tolist()



## === cell 14
pred_string[:10], len(pred_string)



## === cell 15
IMAGE_PATHS[:5], len(IMAGE_PATHS)



## === cell 16
df = pd.DataFrame({"image": test_images, "labels": pred_string})
df.to_csv("submission.csv", index=False)
print(df.shape)
print(df.head())
