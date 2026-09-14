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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import tensorflow as tf
from tensorflow import keras

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

SEED = 42
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["TF_NUM_INTRAOP_THREADS"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ["TF_NUM_INTEROP_THREADS"])
    )
except Exception:
    pass



## === cell 1
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras



## === cell 2
sam_sub = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
sam_sub.head()



## === cell 3
train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"



## === cell 4
train = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 5
train = train.copy()
train["label_single"] = (
    train["labels"].astype(str).str.strip().str.split().str[0].fillna("healthy")
)

train["image_path"] = train_dir.rstrip("/") + "/" + train["image"].astype(str)
exists_mask = train["image_path"].map(os.path.isfile).to_numpy()
train = train.loc[exists_mask].reset_index(drop=True)

train[["image", "labels", "label_single"]].head()



## === cell 6
test_df = sam_sub[["image"]].copy()
test_df["image_path"] = test_dir.rstrip("/") + "/" + test_df["image"].astype(str)
missing = (~test_df["image_path"].map(os.path.isfile)).sum()
print("Test images missing on disk:", int(missing))
test_df.head()



## === cell 7
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
TEST_BATCH_SIZE = 32
EPOCHS = 2
VAL_SPLIT = 0.1

classes = np.sort(train["label_single"].unique())
class_indices = {c: i for i, c in enumerate(classes)}
num_classes = len(class_indices)
print("num_classes:", num_classes)
print("classes:", class_indices)

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train))
val_size = int(round(len(train) * VAL_SPLIT))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

train_df = train.iloc[trn_idx].reset_index(drop=True)
valid_df = train.iloc[val_idx].reset_index(drop=True)

print("train samples:", len(train_df), "valid samples:", len(valid_df))

train_df = train_df.copy()
valid_df = valid_df.copy()
train_df["label_id"] = train_df["label_single"].map(class_indices).astype(np.int32)
valid_df["label_id"] = valid_df["label_single"].map(class_indices).astype(np.int32)

AUTOTUNE = tf.data.AUTOTUNE

DS_OPTS = tf.data.Options()
DS_OPTS.experimental_deterministic = True
DS_OPTS.experimental_optimization.apply_default_optimizations = True
DS_OPTS.experimental_optimization.map_and_batch_fusion = True
DS_OPTS.experimental_optimization.parallel_batch = True

rotation_layer = keras.layers.RandomRotation(
    factor=20.0 / 180.0, fill_mode="nearest", seed=SEED
)
translate_layer = keras.layers.RandomTranslation(
    height_factor=0.1, width_factor=0.1, fill_mode="nearest", seed=SEED
)
flip_layer = keras.layers.RandomFlip(mode="horizontal", seed=SEED)


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return img


def _apply_keras_aug(img):
    img = flip_layer(img, training=True)
    img = translate_layer(img, training=True)
    img = rotation_layer(img, training=True)
    return img


def _preprocess(img):
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


CACHE_DIR = "/kaggle/working/tf_cache_pp2021"
os.makedirs(CACHE_DIR, exist_ok=True)


def make_train_ds(df):
    paths = df["image_path"].values
    y = df["label_id"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.with_options(DS_OPTS)

    shuffle_buf = min(len(df), 4096)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _decode_resize(path)
        img = _apply_keras_aug(img)
        img = _preprocess(img)
        label_oh = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, label_oh

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(os.path.join(CACHE_DIR, "train.cache"))
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds(df):
    paths = df["image_path"].values
    y = df["label_id"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.with_options(DS_OPTS)

    def _map_fn(path, label):
        img = _decode_resize(path)
        img = _preprocess(img)
        label_oh = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, label_oh

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(os.path.join(CACHE_DIR, "valid.cache"))
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(df):
    paths = df["image_path"].values
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(DS_OPTS)

    def _map_fn(path):
        img = _decode_resize(path)
        img = _preprocess(img)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(os.path.join(CACHE_DIR, "test.cache"))
    ds = ds.batch(TEST_BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_df)
valid_ds = make_valid_ds(valid_df)
test_ds = make_test_ds(test_df)



## === cell 8
pass



## === cell 9
from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))




## === cell 10
base = ResNet50(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
trained_model_sub = keras.Model(inputs, outputs)

trained_model_sub.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=[],
)

trained_model_sub.summary()



## === cell 11
trained_model_sub.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 12
y_pred = trained_model_sub.predict(
    test_ds,
    verbose=1,
)
print("y_pred shape:", y_pred.shape)



## === cell 13
predicted_class_indices = np.argmax(y_pred, axis=1)

labels = class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

print(predictions[:10], len(predictions))



## === cell 14
sub = pd.DataFrame({"image": test_df["image"].values, "labels": predictions})
sub = sub[["image", "labels"]]
sub.head()



## === cell 15
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("submission.csv path:", os.path.abspath("submission.csv"))
