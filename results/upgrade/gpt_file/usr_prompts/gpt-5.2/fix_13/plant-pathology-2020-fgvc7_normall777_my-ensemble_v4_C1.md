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
import os, math, re, random

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

print("Tensorflow version " + tf.__version__)

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("float32")
    print("Mixed precision policy forced to:", mixed_precision.global_policy())
except Exception as e:
    print("Mixed precision policy not set:", repr(e))



## === cell 1
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)

EPOCHS = 8
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATA_DIR, "images")

assert os.path.exists(DATA_DIR), f"DATA_DIR not found: {DATA_DIR}"
assert os.path.exists(IMAGES_DIR), f"IMAGES_DIR not found: {IMAGES_DIR}"




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_cols = [c for c in sub.columns if c != "image_id"]
assert "image_id" in sub.columns
for c in target_cols:
    assert c in train.columns, f"Missing target column in train.csv: {c}"

train_ids = train["image_id"].to_numpy(dtype=str)
test_ids = test["image_id"].to_numpy(dtype=str)
train_paths = np.char.add(np.char.add(IMAGES_DIR + "/", train_ids), ".jpg")
test_paths = np.char.add(np.char.add(IMAGES_DIR + "/", test_ids), ".jpg")

train_labels = train[target_cols].to_numpy(dtype=np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED, shuffle=True
)

assert len(train_paths) == len(train_labels)
assert len(valid_paths) == len(valid_labels)
assert train_labels.shape[1] == len(target_cols)

print("Train/valid sizes:", len(train_paths), len(valid_paths))
print("Targets:", target_cols)



## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) * (1.0 / 255.0)
    image.set_shape([image_size[0], image_size[1], 3])
    if label is None:
        return image
    return image, label


def _seed_from_filename(filename, base_seed=SEED):
    h = tf.strings.to_hash_bucket_fast(filename, 2**31 - 1)  # int64
    s0 = tf.cast(tf.bitwise.bitwise_xor(h, tf.cast(base_seed, tf.int64)), tf.int32)
    s1 = tf.cast(
        tf.bitwise.bitwise_xor(
            tf.bitwise.right_shift(h, 1), tf.cast(base_seed * 9973, tf.int64)
        ),
        tf.int32,
    )
    return tf.stack([s0, s1], axis=0)


def data_augment(image, label=None, seed=None):
    if seed is None:
        seed = tf.constant([SEED, SEED], dtype=tf.int32)
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    image = tf.image.stateless_random_flip_up_down(
        image, seed=seed + tf.constant([1, 1], tf.int32)
    )
    if label is None:
        return image
    return image, label


def _with_fast_dataset_options(ds, deterministic: bool = True):
    opts = tf.data.Options()
    opts.experimental_deterministic = deterministic
    try:
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    return ds.with_options(opts)




## === cell 5
def build_train_dataset(paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = _with_fast_dataset_options(ds, deterministic=True)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img, y = decode_image(p, y)
        if training:
            seed = _seed_from_filename(p)
            img, y = data_augment(img, y, seed=seed)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    return ds


def build_test_dataset(paths, cache_in_memory: bool = True):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = _with_fast_dataset_options(ds, deterministic=False)
    ds = ds.map(decode_image, num_parallel_calls=AUTO, deterministic=False)
    if cache_in_memory:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    return ds


train_dataset = build_train_dataset(train_paths, train_labels, training=True)
valid_dataset = build_train_dataset(valid_paths, valid_labels, training=False)
test_dataset = build_test_dataset(test_paths, cache_in_memory=True)



## === cell 6
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.densenet import DenseNet201


def get_model(use_model, n_classes: int):
    base_model = use_model(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output
    predictions = Dense(n_classes, activation="sigmoid", dtype="float32")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                curve="ROC", multi_label=True, num_labels=n_classes, name="auc"
            )
        ],
    )
    return model


n_classes = train_labels.shape[1]

with strategy.scope():
    model1 = get_model(EfficientNetB7, n_classes=n_classes)
with strategy.scope():
    model2 = get_model(DenseNet201, n_classes=n_classes)



## === cell 7
steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)
val_steps = math.ceil(len(valid_paths) / BATCH_SIZE)

print("Training model1...")
hist1 = model1.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)

print("Training model2...")
hist2 = model2.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 8
best_alpha = 0.52

test_steps = math.ceil(len(test_paths) / BATCH_SIZE)

print("Computing predictions...")
probabilities1 = model1.predict(test_dataset, steps=test_steps, verbose=1)
probabilities2 = model2.predict(test_dataset, steps=test_steps, verbose=1)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities2

probabilities = np.clip(probabilities, 0.0, 1.0)
assert probabilities.shape == (len(test), len(target_cols)), (
    probabilities.shape,
    len(test),
    len(target_cols),
)

sub_out = sub.copy()
sub_out["image_id"] = test["image_id"].values
sub_out.loc[:, target_cols] = probabilities

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())
