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
import os, random, re, math

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB7

print("Tensorflow version " + tf.__version__)

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "constant_folding": True,
            "shape_optimization": True,
            "function_optimization": True,
        }
    )
except Exception:
    pass



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

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATA_DIR, "images")




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]
assert TARGET_COLS == [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], f"Unexpected target cols: {TARGET_COLS}"

train_paths = (IMAGES_DIR + "/" + train["image_id"].astype(str) + ".jpg").values
test_paths = (IMAGES_DIR + "/" + test["image_id"].astype(str) + ".jpg").values

train_labels = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED, stratify=None
)

print("Train:", len(train_paths), "Valid:", len(valid_paths), "Test:", len(test_paths))
print("Label shape:", train_labels.shape)



## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size, antialias=True)
    if label is None:
        return image
    return image, label


def data_augment_stateless(index, image, label=None):
    seed = tf.stack([tf.cast(SEED, tf.int64), tf.cast(index, tf.int64)], axis=0)
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    image = tf.image.stateless_random_flip_up_down(
        image, seed=seed + tf.constant([0, 1], tf.int64)
    )
    if label is None:
        return image
    return image, label




## === cell 5
det_opt = tf.data.Options()
det_opt.experimental_deterministic = True
try:
    det_opt.experimental_optimization.apply_default_optimizations = True
    det_opt.experimental_optimization.map_fusion = True
    det_opt.experimental_optimization.map_parallelization = True
    det_opt.experimental_optimization.parallel_batch = True
except Exception:
    pass

CACHE_DIR = "/kaggle/working/tf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, f"train_decoded_{img_size}.cache")
VALID_CACHE = os.path.join(CACHE_DIR, f"valid_decoded_{img_size}.cache")
TEST_CACHE = os.path.join(CACHE_DIR, f"test_decoded_{img_size}.cache")

train_decoded = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(det_opt)
    .map(decode_image, num_parallel_calls=AUTO)
    .apply(tf.data.experimental.ignore_errors())
    .cache(TRAIN_CACHE)
)

train_dataset = (
    train_decoded.repeat()
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .enumerate()
    .map(lambda i, xy: data_augment_stateless(i, xy[0], xy[1]), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(det_opt)
    .map(decode_image, num_parallel_calls=AUTO)
    .apply(tf.data.experimental.ignore_errors())
    .cache(VALID_CACHE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(det_opt)
    .map(decode_image, num_parallel_calls=AUTO)
    .apply(tf.data.experimental.ignore_errors())
    .cache(TEST_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)




## === cell 6
def get_model(use_model, weights="imagenet"):
    base_model = use_model(
        weights=weights,
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
        steps_per_execution=16,
    )
    return model




## === cell 7
w1 = "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
w2 = "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
w3 = "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"  # placeholder to keep variables defined
w3 = "/kaggle/input/tf-zoo-models-on-tpu-inceptionresnetv2/my_model.h5"

steps_per_epoch = int(math.ceil(len(train_paths) / BATCH_SIZE))
val_steps = int(math.ceil(len(valid_paths) / BATCH_SIZE))
test_steps = int(math.ceil(len(test_paths) / BATCH_SIZE))
fallback_epochs = 3

best_alpha = 0.37
bad_alpha = 0.30


def predict_fast(model, ds, steps):
    preds = model.predict(ds, steps=steps, verbose=0)
    return np.asarray(preds)


def run_one(model_fn, weight_path, train_if_missing=True):
    tf.keras.backend.clear_session()
    with strategy.scope():
        m = get_model(model_fn, weights="imagenet")
    ok = os.path.exists(weight_path)
    if ok:
        m.load_weights(weight_path)
        print("Loaded weights:", weight_path)
    else:
        print("Weights not found, using base weights:", weight_path)
        if train_if_missing:
            print(f"Training missing model for {fallback_epochs} epoch(s) ...")
            m.fit(
                train_dataset,
                validation_data=valid_dataset,
                epochs=fallback_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=val_steps,
                verbose=1,
            )
    preds = predict_fast(m, test_dataset, test_steps)
    return preds


print("Computing predictions sequentially...")
probabilities1 = run_one(EfficientNetB7, w1, train_if_missing=True)
probabilities2 = run_one(DenseNet201, w2, train_if_missing=True)
probabilities3 = run_one(InceptionResNetV2, w3, train_if_missing=True)

probabilities = (
    best_alpha * probabilities1
    + (1.0 - best_alpha - bad_alpha) * probabilities2
    + bad_alpha * probabilities3
)

probabilities = np.clip(probabilities, 1e-7, 1.0)
probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

sub[TARGET_COLS] = probabilities.astype(np.float32)
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
