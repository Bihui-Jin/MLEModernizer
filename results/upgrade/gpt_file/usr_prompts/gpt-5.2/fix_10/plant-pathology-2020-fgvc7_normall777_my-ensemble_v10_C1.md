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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, re, math, random

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

print("Tensorflow version " + tf.__version__)

tf.random.set_seed(2020)
np.random.seed(2020)
random.seed(2020)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

tf.config.run_functions_eagerly(False)
try:
    tf.config.optimizer.set_jit(True)  # XLA
except Exception:
    pass

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATA_DIR, "images")

AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

print(
    "Mixed precision disabled for environment compatibility; using default policy:",
    tf.keras.mixed_precision.global_policy(),
)




## === cell 1
def format_path(st):
    return os.path.join(IMAGES_DIR, st + ".jpg")


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]

train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values

train_labels = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2020
)

print("Train:", len(train_paths), "Valid:", len(valid_paths), "Test:", len(test_paths))
print("Targets:", TARGET_COLS)



## === cell 2
img_size = 768


@tf.function
def decode_image(filename, label, image_size):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image.set_shape([img_size, img_size, 3])
    return image, label


@tf.function
def decode_image_nolabel(filename, image_size):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image.set_shape([img_size, img_size, 3])
    return image


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image, seed=2020)
    image = tf.image.random_flip_up_down(image, seed=2020)
    return image, label




## === cell 3
def _dataset_options_fast():
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    return opts


def _maybe_prefetch_to_device(ds):
    if tpu is not None:
        try:
            ds = ds.apply(tf.data.experimental.prefetch_to_device("/device:TPU:0"))
        except Exception:
            pass
    return ds


IMAGE_SIZE_T = tf.constant([img_size, img_size], dtype=tf.int32)


@tf.function(reduce_retracing=True)
def _decode_train(path, label):
    return decode_image(path, label, IMAGE_SIZE_T)


@tf.function(reduce_retracing=True)
def _decode_valid(path, label):
    return decode_image(path, label, IMAGE_SIZE_T)


@tf.function(reduce_retracing=True)
def _decode_test(path):
    return decode_image_nolabel(path, IMAGE_SIZE_T)


@tf.function(reduce_retracing=True)
def _augment(image, label):
    return data_augment(image, label)


train_base = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(_decode_train, num_parallel_calls=AUTO)
    .cache()
)
valid_base = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(_decode_valid, num_parallel_calls=AUTO)
    .cache()
)
test_base = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_decode_test, num_parallel_calls=AUTO)
    .cache()
)

train_dataset = (
    train_base.shuffle(512, seed=2020, reshuffle_each_iteration=True)
    .map(_augment, num_parallel_calls=AUTO)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
    .with_options(_dataset_options_fast())
)
train_dataset = _maybe_prefetch_to_device(train_dataset)

valid_dataset = (
    valid_base.batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
    .with_options(_dataset_options_fast())
)
valid_dataset = _maybe_prefetch_to_device(valid_dataset)

test_dataset = (
    test_base.batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
    .with_options(_dataset_options_fast())
)
test_dataset = _maybe_prefetch_to_device(test_dataset)

print(
    "Datasets built:",
    "train:",
    train_dataset,
    "valid:",
    valid_dataset,
    "test:",
    test_dataset,
)



## === cell 4
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import DenseNet201


def get_model(use_model):
    base_model = use_model(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output
    outputs = Dense(train_labels.shape[1], activation="sigmoid", dtype="float32")(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[],
        run_eagerly=False,
        steps_per_execution=64,
    )
    return model


with strategy.scope():
    model1 = get_model(EfficientNetB7)
with strategy.scope():
    model2 = get_model(DenseNet201)



## === cell 5
steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)
validation_steps = math.ceil(len(valid_paths) / BATCH_SIZE)

history1 = model1.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset,
    validation_steps=validation_steps,
    verbose=2,
)

history2 = model2.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset,
    validation_steps=validation_steps,
    verbose=2,
)



## === cell 6
best_alpha = 0.75
bad_alpha = 0.30

print("Вычисляем предсказания...")
probabilities1 = model1.predict(test_dataset, verbose=1)
probabilities2 = model2.predict(test_dataset, verbose=1)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities2
probabilities = np.clip(probabilities, 0.0, 1.0)

sub = sub.copy()
sub = sub.set_index("image_id").loc[test["image_id"].values].reset_index()

sub[TARGET_COLS] = probabilities
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
