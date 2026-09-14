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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split

print("Tensorflow version " + tf.__version__)

SEED = 2020
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
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

tf.keras.backend.set_learning_phase(1)



## === cell 1
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU:", tpu.master())
except Exception:
    tpu = None

if tpu is not None:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync



## === cell 2
DATASET_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

train = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATASET_DIR, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]
assert set(TARGET_COLS).issubset(
    set(train.columns)
), "Train columns do not match submission targets."


def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


train_paths = np.array([format_path(x) for x in train["image_id"].values], dtype=object)
test_paths = np.array([format_path(x) for x in test["image_id"].values], dtype=object)

train_labels = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED, stratify=None
)

print(
    "Train size:",
    len(train_paths),
    "Valid size:",
    len(valid_paths),
    "Test size:",
    len(test_paths),
)
print("Targets:", TARGET_COLS)



## === cell 3
img_size = 768

IMG_SIZE_TUPLE = tf.constant([img_size, img_size], dtype=tf.int32)


@tf.function
def decode_image(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)  # equivalent to /255.0
    image = tf.image.resize(image, IMG_SIZE_TUPLE, antialias=True)
    image = tf.ensure_shape(image, [img_size, img_size, 3])
    if label is None:
        return image
    return image, label


@tf.function
def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    if label is None:
        return image
    return image, label




## === cell 4
options = tf.data.Options()
options.experimental_deterministic = True  # preserve stable behavior
try:
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass
try:
    options.experimental_slack = True
except Exception:
    pass

CACHE_DIR = "/kaggle/working/tf_cache_pp2020"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, "train_decoded.cache")
VALID_CACHE = os.path.join(CACHE_DIR, "valid_decoded.cache")
TEST_CACHE = os.path.join(CACHE_DIR, "test_decoded.cache")

train_base = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache(TRAIN_CACHE)
)

train_dataset = (
    train_base.shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .map(data_augment, num_parallel_calls=AUTO, deterministic=True)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache(VALID_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache(TEST_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 5
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.efficientnet import EfficientNetB7


def get_model(use_model, weights):
    base_model = use_model(
        weights=weights,
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )

    base_model.trainable = False

    x = base_model.output
    predictions = Dense(train_labels.shape[1], activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[],
        steps_per_execution=256,
        run_eagerly=False,
    )
    return model




## === cell 6
STEPS_PER_EPOCH = int(np.ceil(len(train_paths) / BATCH_SIZE))
VALID_STEPS = int(np.ceil(len(valid_paths) / BATCH_SIZE))
TEST_STEPS = int(np.ceil(len(test_paths) / BATCH_SIZE))

with strategy.scope():
    model1 = get_model(EfficientNetB7, weights="imagenet")

MODEL1_WEIGHTS_PATH = (
    "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
)
model1_weights_loaded = False
if tf.io.gfile.exists(MODEL1_WEIGHTS_PATH):
    try:
        model1.load_weights(MODEL1_WEIGHTS_PATH)
        model1_weights_loaded = True
        print("Loaded model1 weights from dataset.")
    except Exception as e:
        print(
            "Found model1 weights path but failed to load; will train. Error:", repr(e)
        )
else:
    print("Model1 custom weights not found; will train from ImageNet weights.")

with strategy.scope():
    model2 = get_model(DenseNet201, weights="imagenet")

MODEL2_WEIGHTS_PATH = (
    "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
)
model2_weights_loaded = False
if tf.io.gfile.exists(MODEL2_WEIGHTS_PATH):
    try:
        model2.load_weights(MODEL2_WEIGHTS_PATH)
        model2_weights_loaded = True
        print("Loaded model2 weights from dataset.")
    except Exception as e:
        print(
            "Found model2 weights path but failed to load; will train. Error:", repr(e)
        )
else:
    print("Model2 custom weights not found; will train from ImageNet weights.")

if not model1_weights_loaded:
    print("Training model1...")
    model1.fit(
        train_dataset,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        validation_steps=VALID_STEPS,
        verbose=1,
    )
else:
    print("Skipping model1 training because custom weights were loaded.")

if not model2_weights_loaded:
    print("Training model2...")
    model2.fit(
        train_dataset,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        validation_steps=VALID_STEPS,
        verbose=1,
    )
else:
    print("Skipping model2 training because custom weights were loaded.")

best_alpha = 0.60
print("Computing predictions...")

probabilities1 = model1.predict(test_dataset, steps=TEST_STEPS, verbose=1)
probabilities2 = model2.predict(test_dataset, steps=TEST_STEPS, verbose=1)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities2
probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)

assert probabilities.shape[0] == len(sub), "Prediction rows must match submission rows."
assert probabilities.shape[1] == len(TARGET_COLS), "Prediction cols must match targets."

sub[TARGET_COLS] = probabilities
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
