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
import os, math, re, random, sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_JAX_DISABLED", "1")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications import EfficientNetB7

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU:", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

DS_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_PATH = os.path.join(DS_PATH, "images")

if not os.path.isdir(IMAGES_PATH):
    DS_PATH = "/kaggle/data/plant-pathology-2020-fgvc7"
    IMAGES_PATH = os.path.join(DS_PATH, "images")

assert os.path.isdir(IMAGES_PATH), f"Images folder not found at: {IMAGES_PATH}"




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_PATH, f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(DS_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DS_PATH, "test.csv"))
sub = pd.read_csv(os.path.join(DS_PATH, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]
assert TARGET_COLS == [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], f"Unexpected target columns: {TARGET_COLS}"

train_paths = train["image_id"].apply(format_path).values
test_paths = test["image_id"].apply(format_path).values

train_labels = train[TARGET_COLS].values.astype(np.float32)

n = len(train_paths)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

valid_frac = 0.15
n_valid = int(round(n * valid_frac))
valid_idx = idx[:n_valid]
train_idx = idx[n_valid:]

valid_paths = train_paths[valid_idx]
valid_labels = train_labels[valid_idx]
train_paths = train_paths[train_idx]
train_labels = train_labels[train_idx]

assert len(train_paths) == len(train_labels)
assert len(valid_paths) == len(valid_labels)
assert train_labels.shape[1] == len(TARGET_COLS)



## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        return image, label


def data_augment(image, label=None, seed=2020):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)

    if label is None:
        return image
    else:
        return image, label




## === cell 5
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




## === cell 6
def build_model(use_model, weights):
    base_model = use_model(
        weights=weights,
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output

    predictions = Dense(train_labels.shape[1], activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer="nadam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(
                curve="ROC", multi_label=True, num_labels=train_labels.shape[1]
            )
        ],
    )
    return model


with strategy.scope():
    model1 = build_model(EfficientNetB7, weights="imagenet")

with strategy.scope():
    model2 = build_model(DenseNet201, weights="imagenet")

with strategy.scope():
    model3 = build_model(InceptionResNetV2, weights="imagenet")

w1 = "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
w2 = "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
w3 = "/kaggle/input/tf-zoo-models-on-tpu-inceptionresnetv2/my_model.h5"

for m, w in [(model1, w1), (model2, w2), (model3, w3)]:
    if os.path.exists(w):
        m.load_weights(w)
    else:
        print(f"WARNING: weights not found, using base weights only: {w}")



## === cell 7
best_alpha = 0.36
bad_alpha = 0.30

print("Computing predictions...")
probabilities1 = model1.predict(test_dataset, verbose=1)
probabilities2 = model2.predict(test_dataset, verbose=1)
probabilities3 = model3.predict(test_dataset, verbose=1)

probabilities = (
    best_alpha * probabilities1
    + bad_alpha * probabilities2
    + (1.0 - best_alpha - bad_alpha) * probabilities3
)

probabilities = np.asarray(probabilities, dtype=np.float32)
probabilities = np.clip(probabilities, 0.0, 1.0)

assert probabilities.shape == (
    len(test),
    len(TARGET_COLS),
), f"Pred shape mismatch: {probabilities.shape} vs {(len(test), len(TARGET_COLS))}"

submission = sub.copy()
submission["image_id"] = test["image_id"].values  # ensure exact ordering
submission[TARGET_COLS] = probabilities

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)
print(submission.head())
