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
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

print("tf:", tf.__version__)
print("keras:", keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)  # let TF decide
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTO = tf.data.AUTOTUNE



## === cell 1
path = "/kaggle/input/plant-pathology-2021-fgvc8/"

train = pd.read_csv(os.path.join(path, "train.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

train_img_dir = os.path.join(path, "train_images")
test_img_dir = os.path.join(path, "test_images")

assert {"image", "labels"}.issubset(train.columns)
assert {"image", "labels"}.issubset(sub.columns)

train.head()



## === cell 2
train["filepath"] = train_img_dir + "/" + train["image"].values
sub["filepath"] = test_img_dir + "/" + sub["image"].values

for fp in train["filepath"].head(1):
    if not tf.io.gfile.exists(fp):
        raise FileNotFoundError(fp)
for fp in sub["filepath"].head(1):
    if not tf.io.gfile.exists(fp):
        raise FileNotFoundError(fp)

train.shape, sub.shape




## === cell 3
def first_token(label_str: str) -> str:
    return str(label_str).split(" ")[0].strip()


train["label_1"] = train["labels"].astype(str).str.split(" ").str[0].str.strip()

classes = sorted(train["label_1"].unique().tolist())
num_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

train["target"] = train["label_1"].map(class_to_idx).astype(np.int32)

num_classes, classes[:20]



## === cell 4
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ]
)
def decode_image_labeled(filename, label):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST", ratio=2)
    image = tf.image.resize_with_pad(
        image,
        IMG_SIZE[0],
        IMG_SIZE[1],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return image, label


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def decode_image_unlabeled(filename):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST", ratio=2)
    image = tf.image.resize_with_pad(
        image,
        IMG_SIZE[0],
        IMG_SIZE[1],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return image




## === cell 5
from sklearn.model_selection import train_test_split

trn_df, val_df = train_test_split(
    train[["filepath", "target"]],
    test_size=0.1,
    random_state=SEED,
    stratify=train["target"],
)

trn_df = trn_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

trn_df.shape, val_df.shape



## === cell 6
trn_files = trn_df["filepath"].values
val_files = val_df["filepath"].values
test_files = sub["filepath"].values

trn_targets = trn_df["target"].to_numpy(np.int32, copy=False)
val_targets = val_df["target"].to_numpy(np.int32, copy=False)


def _with_fast_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    options = tf.data.Options()
    options.experimental_deterministic = True
    return ds.with_options(options)


base_train = tf.data.Dataset.from_tensor_slices((trn_files, trn_targets))
base_train = _with_fast_options(base_train)
base_train = base_train.map(
    decode_image_labeled, num_parallel_calls=AUTO, deterministic=True
)
base_train = base_train.cache()  # in-memory
base_train = base_train.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
base_train = base_train.apply(tf.data.experimental.ignore_errors())
train_ds = base_train.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_targets))
val_ds = _with_fast_options(val_ds)
val_ds = val_ds.map(decode_image_labeled, num_parallel_calls=AUTO, deterministic=True)
val_ds = val_ds.cache()  # in-memory
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices(test_files)
test_ds = _with_fast_options(test_ds)
test_ds = test_ds.map(
    decode_image_unlabeled, num_parallel_calls=AUTO, deterministic=True
)
test_ds = test_ds.cache()  # in-memory
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)



## === cell 7
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
    steps_per_execution=32,
)

model.summary()



## === cell 8
EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 9
probs = model.predict(test_ds, verbose=1)
probs.shape



## === cell 10
pred_idx = probs.argmax(axis=1)

idx_to_class_arr = np.array([idx_to_class[i] for i in range(num_classes)], dtype=object)
pred_labels = np.take(idx_to_class_arr, pred_idx).astype(str)

assert len(pred_labels) == len(sub), (len(pred_labels), len(sub))

pred_labels[:10], len(pred_labels)



## === cell 11
submission = sub[["image"]].copy()
submission["labels"] = pred_labels

submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
