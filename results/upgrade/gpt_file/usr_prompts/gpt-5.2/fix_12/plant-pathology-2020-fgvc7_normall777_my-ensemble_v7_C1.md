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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re, math, random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

print("Tensorflow version " + tf.__version__)
SEED = 2020
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 1
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

INFER_BATCH_SIZE = max(BATCH_SIZE, 32 * strategy.num_replicas_in_sync)



## === cell 2
DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATA_DIR, "images")


def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


assert os.path.isdir(IMAGES_DIR), f"Images dir not found: {IMAGES_DIR}"



## === cell 3
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_cols = [c for c in sub.columns if c != "image_id"]
assert (
    "image_id" in train.columns
    and "image_id" in test.columns
    and "image_id" in sub.columns
)
assert len(target_cols) == 4, f"Expected 4 target columns, got {target_cols}"

train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values
train_labels = train[target_cols].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED, shuffle=True
)

print(
    "Train samples:",
    len(train_paths),
    "Valid samples:",
    len(valid_paths),
    "Test samples:",
    len(test_paths),
)



## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    return image, label


def data_augment(image, label=None, seed=SEED):
    s = tf.constant([seed, 0], dtype=tf.int32)
    image = tf.image.stateless_random_flip_left_right(image, seed=s)
    image = tf.image.stateless_random_flip_up_down(image, seed=s)
    if label is None:
        return image
    return image, label




## === cell 5
def with_fast_options(ds, deterministic=True):
    opts = tf.data.Options()
    opts.experimental_deterministic = deterministic
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    return ds.with_options(opts)


CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, f"train_img{img_size}_seed{SEED}.cache")
VALID_CACHE = os.path.join(CACHE_DIR, f"valid_img{img_size}.cache")
TEST_CACHE = os.path.join(CACHE_DIR, f"test_img{img_size}.cache")

INFERENCE_ONLY = True

if not INFERENCE_ONLY:
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .map(data_augment, num_parallel_calls=AUTO)
        .cache(TRAIN_CACHE)
        .repeat()
        .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )
    train_dataset = with_fast_options(train_dataset, deterministic=True)
else:
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )
    train_dataset = with_fast_options(train_dataset, deterministic=True)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
valid_dataset = with_fast_options(valid_dataset, deterministic=True)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache(TEST_CACHE)
    .batch(INFER_BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
test_dataset = with_fast_options(test_dataset, deterministic=False)

STEPS_PER_EPOCH = math.ceil(len(train_paths) / BATCH_SIZE)
VALIDATION_STEPS = math.ceil(len(valid_paths) / BATCH_SIZE)
TEST_STEPS = math.ceil(len(test_paths) / INFER_BATCH_SIZE)
print(
    "STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
    "VALIDATION_STEPS:",
    VALIDATION_STEPS,
    "TEST_STEPS:",
    TEST_STEPS,
)



## === cell 6
from tensorflow.keras.applications import DenseNet201, InceptionResNetV2, EfficientNetB7


def get_model(use_model, base_weights):
    base_model = use_model(
        weights=base_weights,
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output
    predictions = Dense(train_labels.shape[1], activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def safe_load_weights(model, weights_path):
    candidates = []
    if weights_path:
        candidates.append(weights_path)

        base = os.path.basename(weights_path)
        candidates.extend(
            [
                os.path.join("/kaggle/input", base),
                os.path.join("/kaggle/data", base),
                os.path.join("/kaggle/working", base),
            ]
        )

    for p in candidates:
        if p and os.path.exists(p):
            model.load_weights(p)
            print(f"Loaded weights: {p}")
            return True

    print(f"WARNING: weights not found, using base weights only: {weights_path}")
    return False




## === cell 7
with strategy.scope():
    model1 = get_model(EfficientNetB7, base_weights="imagenet")
    model2 = get_model(DenseNet201, base_weights="imagenet")
    model3 = get_model(InceptionResNetV2, base_weights="imagenet")

loaded1 = safe_load_weights(
    model1, "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
)
loaded2 = safe_load_weights(
    model2, "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
)
loaded3 = safe_load_weights(
    model3, "/kaggle/input/tf-zoo-models-on-tpu-inceptionresnetv2/my_model.h5"
)

if INFERENCE_ONLY and (not loaded1 or not loaded2 or not loaded3):
    missing = []
    if not loaded1:
        missing.append("EfficientNetB7 weights")
    if not loaded2:
        missing.append("DenseNet201 weights")
    if not loaded3:
        missing.append("InceptionResNetV2 weights")
    raise FileNotFoundError(
        "Required pretrained .h5 weights not found for inference-only run: "
        + ", ".join(missing)
        + ". Without them the notebook would train 3 models for 40 epochs and timeout."
    )

if not loaded1:
    print("Training model1 (EfficientNetB7)...")
    model1.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_dataset,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )

if not loaded2:
    print("Training model2 (DenseNet201)...")
    model2.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_dataset,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )

if not loaded3:
    print("Training model3 (InceptionResNetV2)...")
    model3.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_dataset,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )



## === cell 8
best_alpha = 0.37
bad_alpha = 0.30

print("Вычисляем предсказания...")

_one_batch = next(iter(test_dataset.take(1)))
_ = model1(_one_batch, training=False)
_ = model2(_one_batch, training=False)
_ = model3(_one_batch, training=False)

probabilities1 = model1.predict(test_dataset, steps=TEST_STEPS, verbose=1)
probabilities2 = model2.predict(test_dataset, steps=TEST_STEPS, verbose=1)
probabilities3 = model3.predict(test_dataset, steps=TEST_STEPS, verbose=1)

probabilities1 = probabilities1[: len(test_paths)]
probabilities2 = probabilities2[: len(test_paths)]
probabilities3 = probabilities3[: len(test_paths)]

probabilities = (
    best_alpha * probabilities1
    + (1.0 - best_alpha - bad_alpha) * probabilities2
    + bad_alpha * probabilities3
)

probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

sub = sub.copy()
sub[target_cols] = probabilities.astype(np.float32)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
print("Submission columns:", list(sub.columns))
print("Test rows:", len(test), "Submission rows:", len(sub))
assert sub.shape[0] == test.shape[0] == 183
assert list(sub.columns) == ["image_id"] + target_cols
assert os.path.exists("submission.csv")
assert "submission.csv".endswith(".csv")
