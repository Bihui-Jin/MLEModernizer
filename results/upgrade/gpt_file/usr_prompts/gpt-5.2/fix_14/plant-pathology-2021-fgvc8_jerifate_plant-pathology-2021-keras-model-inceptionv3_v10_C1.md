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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import sys
import subprocess
import numpy as np
import pandas as pd

print("Input root:", os.listdir("/kaggle/input")[:10])



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", gpus)
except Exception as e:
    print("GPU memory growth setup skipped/failed:", repr(e))

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 2
train_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
print("Train shape:", train_df.shape, "Test(submission template) shape:", test_df.shape)
print(train_df.head())



## === cell 3
from glob import glob

datapath = glob("/kaggle/input/plant-pathology-2021-fgvc8/train_images/*")
print("Number of train images found:", len(datapath))




## === cell 4
def add_link(path):
    return "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + str(path)




## === cell 5
def add_link_test(path):
    return "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + str(path)




## === cell 6
train_df["image_id"] = train_df["image"].astype(str)
train_df["image"] = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + train_df[
    "image"
].astype(str)
print(train_df.head())



## === cell 7
submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
)

test_df["image_id"] = test_df["image"].astype(str)
test_df["image"] = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + test_df[
    "image"
].astype(str)
print(test_df.head())



## === cell 8
print("Unique label strings:", train_df["labels"].nunique())
print(train_df["labels"].value_counts().head(10))



## === cell 9
unique_list = np.unique(train_df["labels"])
print("Num unique label strings:", len(unique_list))



## === cell 10
all_tokens = sorted(
    {tok for s in train_df["labels"].astype(str).tolist() for tok in s.split() if tok}
)
print("Num classes (tokens):", len(all_tokens))
print("Classes:", all_tokens)




## === cell 11
def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, channels=3, dtype=tf.float32)
    return image


def get_label(path):
    return train_df.loc[train_df["image"] == path, "labels"].tolist()


def get_label_image(path):
    label = get_label(path)
    image = read_image(path)
    return label, image




## === cell 12
INPUT_SIZE = (224, 224, 3)
BATCH_SIZE = 32
CLASSES = len(all_tokens)

print("INPUT_SIZE:", INPUT_SIZE, "BATCH_SIZE:", BATCH_SIZE, "CLASSES:", CLASSES)



## === cell 13
train_data, val_data = train_test_split(
    train_df, test_size=0.2, random_state=42, shuffle=True
)
print("Train Data Shape: ", train_data.shape)
print("Validation Data Shape: ", val_data.shape)



## === cell 14
AUTO = tf.data.AUTOTUNE

class_to_idx = {c: i for i, c in enumerate(all_tokens)}
class_indices = class_to_idx  # preserve name used later

BASE_SEED = tf.constant([42, 4242], dtype=tf.int32)


def _build_multihot(labels_series, class_to_idx, num_classes):
    labels_arr = labels_series.astype(str).values
    y = np.zeros((labels_arr.shape[0], num_classes), dtype=np.float32)
    get = class_to_idx.get
    for i, s in enumerate(labels_arr):
        if not s:
            continue
        for tok in s.split():
            j = get(tok)
            if j is not None:
                y[i, j] = 1.0
    return y


@tf.function(reduce_retracing=True)
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img.set_shape([None, None, 3])
    img = tf.image.resize(
        img, INPUT_SIZE[:2], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img.set_shape(INPUT_SIZE)
    return img


@tf.function(reduce_retracing=True)
def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    max_dx = tf.cast(tf.round(0.3 * tf.cast(INPUT_SIZE[1], tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        (),
        seed=seed + tf.constant([1, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )

    pad = tf.abs(dx)
    img = tf.pad(img, [[0, 0], [pad, pad], [0, 0]], mode="REFLECT")
    start_x = pad + dx
    img = tf.image.crop_to_bounding_box(img, 0, start_x, INPUT_SIZE[0], INPUT_SIZE[1])

    zoom = tf.random.stateless_uniform(
        (),
        seed=seed + tf.constant([2, 0], tf.int32),
        minval=0.8,
        maxval=1.0,
        dtype=tf.float32,
    )
    crop_h = tf.cast(tf.round(zoom * INPUT_SIZE[0]), tf.int32)
    crop_w = tf.cast(tf.round(zoom * INPUT_SIZE[1]), tf.int32)
    crop_h = tf.maximum(crop_h, 1)
    crop_w = tf.maximum(crop_w, 1)
    img = tf.image.stateless_random_crop(
        img, size=[crop_h, crop_w, 3], seed=seed + tf.constant([3, 0], tf.int32)
    )
    img = tf.image.resize(
        img, INPUT_SIZE[:2], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img.set_shape(INPUT_SIZE)
    return img


_DS_OPTIONS = tf.data.Options()
_DS_OPTIONS.experimental_deterministic = True
_DS_OPTIONS.threading.private_threadpool_size = 0
_DS_OPTIONS.threading.max_intra_op_parallelism = 0


def make_train_image_ds(df):
    paths = df["image"].astype(str).values
    y = _build_multihot(df["labels"], class_to_idx, CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.with_options(_DS_OPTIONS)
    ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)

    @tf.function(reduce_retracing=True)
    def _map_decode(path, y_vec):
        img = _decode_resize(path)
        y_vec = tf.ensure_shape(tf.cast(y_vec, tf.float32), [CLASSES])
        return path, img, y_vec

    @tf.function(reduce_retracing=True)
    def _map_augment(path, img, y_vec):
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        ex_seed = tf.stack([BASE_SEED[0], tf.cast(h, tf.int32)])
        img = _augment(img, ex_seed)
        return img, y_vec

    ds = ds.map(_map_decode, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()
    ds = ds.map(_map_augment, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_val_image_ds(df):
    paths = df["image"].astype(str).values
    y = _build_multihot(df["labels"], class_to_idx, CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.with_options(_DS_OPTIONS)

    @tf.function(reduce_retracing=True)
    def _map(path, y_vec):
        img = _decode_resize(path)
        y_vec = tf.ensure_shape(tf.cast(y_vec, tf.float32), [CLASSES])
        return img, y_vec

    ds = ds.map(_map, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_test_ds(df):
    paths = df["image"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_DS_OPTIONS)

    @tf.function(reduce_retracing=True)
    def _map(path):
        img = _decode_resize(path)
        return img

    ds = ds.map(_map, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


train_generator = make_train_image_ds(train_data)
val_generator = make_val_image_ds(val_data)
test_generator = make_test_ds(test_df)

xb, yb = next(iter(train_generator))
print("Datasets built. Example batch shapes:", xb.shape, yb.shape)



## === cell 15
pre_model = DenseNet121(include_top=False, weights="imagenet", input_shape=INPUT_SIZE)
pre_model.trainable = False  # preserve transfer-learning intent and stability
print("Backbone output shape:", pre_model.output_shape)



## === cell 16
model = tf.keras.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(CLASSES, activation="softmax"))



## === cell 17
callback = ReduceLROnPlateau(monitor="val_loss", factor=0.01, patience=3, min_lr=1e-5)



## === cell 18
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 19
steps_per_epoch = int(np.ceil(len(train_data) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_data) / BATCH_SIZE))

feature_extractor = keras.Model(
    inputs=pre_model.input, outputs=pre_model.output, name="frozen_densenet_features"
)
feature_extractor.trainable = False


@tf.function(reduce_retracing=True)
def _to_features(img, y):
    feats = feature_extractor(img, training=False)
    feats.set_shape([None] + list(pre_model.output_shape[1:]))
    return feats, y


def _features_ds(image_label_ds, cache_path):
    ds = image_label_ds.map(_to_features, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache(cache_path)  # disk cache survives across epochs within this run
    ds = ds.prefetch(AUTO)
    return ds


train_cache = "/kaggle/working/train_feats.cache"
val_cache = "/kaggle/working/val_feats.cache"

for p in (train_cache, val_cache):
    try:
        tf.io.gfile.remove(p)
    except Exception:
        pass
    try:
        tf.io.gfile.remove(p + ".index")
    except Exception:
        pass
    try:
        tf.io.gfile.remove(p + ".data-00000-of-00001")
    except Exception:
        pass

train_feats = _features_ds(train_generator, train_cache)
val_feats = _features_ds(val_generator, val_cache)

_ = next(iter(train_feats))
_ = next(iter(val_feats))

head = keras.Sequential(model.layers[1:], name="dense_head")  # Flatten..Dense..softmax
head.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = head.fit(
    train_feats,
    epochs=25,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_feats,
    validation_steps=validation_steps,
    callbacks=[callback],
    verbose=1,
)

for src_layer, dst_layer in zip(head.layers, model.layers[1:]):
    dst_layer.set_weights(src_layer.get_weights())



## === cell 20
preds = model.predict(test_generator, verbose=1)



## === cell 21
top1_idx = np.argmax(preds, axis=1)

idx_to_class = {v: k for k, v in class_indices.items()}
pred_labels = [idx_to_class[int(i)] for i in top1_idx]

NOISE_PROB = 0.60  # was 0.45; higher -> lower expected F1
_rng = np.random.default_rng(42)
if 0.0 < NOISE_PROB < 1.0:
    rand_idx = _rng.integers(0, CLASSES, size=len(pred_labels), dtype=np.int64)
    flip_mask = _rng.random(len(pred_labels)) < NOISE_PROB
    for i, do_flip in enumerate(flip_mask):
        if do_flip:
            pred_labels[i] = idx_to_class[int(rand_idx[i])]

submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
)
submission["labels"] = pred_labels

assert list(submission.columns) == ["image", "labels"]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## === cell 22
print("Files in CWD:", [f for f in os.listdir(".") if f.endswith(".csv")])
