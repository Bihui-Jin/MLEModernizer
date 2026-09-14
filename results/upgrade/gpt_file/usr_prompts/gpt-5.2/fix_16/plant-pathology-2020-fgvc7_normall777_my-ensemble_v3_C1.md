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

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import DenseNet201, EfficientNetB7

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
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
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




## === cell 2
CANDIDATE_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_data_dir(candidates):
    for d in candidates:
        if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
            os.path.join(d, "images")
        ):
            return d
        nested = os.path.join(d, "plant-pathology-2020-fgvc7")
        if os.path.exists(os.path.join(nested, "train.csv")) and os.path.exists(
            os.path.join(nested, "images")
        ):
            return nested
    return None


DATA_DIR = _find_data_dir(CANDIDATE_DIRS)

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate dataset directory containing train.csv and images/. "
        f"Tried: {CANDIDATE_DIRS} (including nested plant-pathology-2020-fgvc7/)"
    )

IMAGES_DIR = os.path.join(DATA_DIR, "images")
print("Using DATA_DIR:", DATA_DIR)
print("Using IMAGES_DIR:", IMAGES_DIR)


def format_path(st):
    return os.path.join(IMAGES_DIR, st + ".jpg")




## === cell 3
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]

train_paths = (IMAGES_DIR + os.sep + train["image_id"].values + ".jpg").astype(str)
test_paths = (IMAGES_DIR + os.sep + test["image_id"].values + ".jpg").astype(str)

train_labels = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.15,
    random_state=SEED,
    stratify=np.argmax(train_labels, axis=1),
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

if len(train_paths) and (not os.path.exists(train_paths[0])):
    raise FileNotFoundError(f"Training image file missing, e.g.: {train_paths[0]}")




## === cell 4
img_size = 768


@tf.function(reduce_retracing=True)
def _decode_image_only(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(
        image, (img_size, img_size), method=tf.image.ResizeMethod.BILINEAR
    )
    image = tf.cast(
        image, tf.float32
    )  # keep 0..255 float (model preprocess handles it)
    image.set_shape((img_size, img_size, 3))
    return image


@tf.function(reduce_retracing=True)
def _decode_image_and_label(filename, label):
    return _decode_image_only(filename), label


@tf.function(reduce_retracing=True)
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label




## === cell 5
data_opts = tf.data.Options()
data_opts.experimental_deterministic = False

CACHE_DIR = os.path.join("/kaggle/working", "tfdata_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

VALID_CACHE = os.path.join(CACHE_DIR, f"valid_cache_{img_size}.cache")
TEST_CACHE = os.path.join(CACHE_DIR, f"test_cache_{img_size}.cache")
TRAIN_CACHE = os.path.join(CACHE_DIR, f"train_cache_{img_size}.cache")


def _maybe_cache(ds, cache_path):
    return ds.cache(cache_path)


train_dataset_raw = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(data_opts)
    .map(_decode_image_and_label, num_parallel_calls=AUTO, deterministic=False)
    .apply(tf.data.experimental.ignore_errors())
)
train_dataset_raw = _maybe_cache(train_dataset_raw, TRAIN_CACHE)

train_dataset = (
    train_dataset_raw.shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .map(data_augment, num_parallel_calls=AUTO, deterministic=False)
    .batch(BATCH_SIZE, drop_remainder=True)
    .repeat()
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(data_opts)
    .map(_decode_image_and_label, num_parallel_calls=AUTO, deterministic=False)
    .apply(tf.data.experimental.ignore_errors())
)
valid_dataset = _maybe_cache(valid_dataset, VALID_CACHE)
valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(AUTO)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(data_opts)
    .map(_decode_image_only, num_parallel_calls=AUTO, deterministic=False)
    .apply(tf.data.experimental.ignore_errors())
)
test_dataset = _maybe_cache(test_dataset, TEST_CACHE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTO)

STEPS_PER_EPOCH = int(np.ceil(len(train_paths) / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(valid_paths) / BATCH_SIZE))

print(
    "BATCH_SIZE:",
    BATCH_SIZE,
    "STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
    "VALIDATION_STEPS:",
    VALIDATION_STEPS,
)




## === cell 6
def get_model(use_model):
    if use_model.__name__ == "EfficientNetB7":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    elif use_model.__name__ == "DenseNet201":
        preprocess = tf.keras.applications.densenet.preprocess_input
    else:
        preprocess = None

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = inputs
    if preprocess is not None:
        x = tf.keras.layers.Lambda(preprocess, name=f"{use_model.__name__}_preprocess")(
            x
        )

    base_model = use_model(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model(x)

    predictions = Dense(train_labels.shape[1], activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                curve="ROC",
                multi_label=True,
                num_labels=train_labels.shape[1],
                name="auc",
            ),
        ],
        steps_per_execution=max(1, min(32, STEPS_PER_EPOCH)),
    )
    return model


def _resolve_weight_path(p):
    if os.path.exists(p):
        return p
    basename = os.path.basename(p)

    quick_roots = [
        "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7",
        "/kaggle/input/tf-zoo-models-on-tpu-densenet201",
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/working",
        DATA_DIR,
        os.path.dirname(DATA_DIR),
    ]
    for root in quick_roots:
        cand = os.path.join(root, basename)
        if os.path.exists(cand):
            return cand
    return p


def maybe_load_weights_or_train(model, weights_path, model_name, train_epochs):
    """
    Preserve existing behavior: load weights if present; otherwise train the same model.
    """
    weights_path_resolved = _resolve_weight_path(weights_path)
    if os.path.exists(weights_path_resolved):
        model.load_weights(weights_path_resolved)
        print(f"Loaded {model_name} weights:", weights_path_resolved)
        return model, True

    print(
        f"[WARN] {model_name} weights not found at '{weights_path}' (resolved: '{weights_path_resolved}'). "
        f"Falling back to training for {train_epochs} epoch(s) to produce a valid submission."
    )
    model.fit(
        train_dataset,
        epochs=train_epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_dataset,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )
    return model, False


FALLBACK_EPOCHS = 3 if tpu is None else 5

with strategy.scope():
    model1 = get_model(EfficientNetB7)
w1 = "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
model1, loaded1 = maybe_load_weights_or_train(
    model1, w1, "model1(EfficientNetB7)", FALLBACK_EPOCHS
)

with strategy.scope():
    model2 = get_model(DenseNet201)
w2 = "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
model2, loaded2 = maybe_load_weights_or_train(
    model2, w2, "model2(DenseNet201)", FALLBACK_EPOCHS
)

print("Weights loaded:", {"model1": loaded1, "model2": loaded2})




## === cell 7
best_alpha = 0.52

print("Вычисляем предсказания...")

n_test = len(test_paths)
n_classes = train_labels.shape[1]

for xb in test_dataset.take(1):
    _ = model1(xb, training=False)
    _ = model2(xb, training=False)

probabilities1 = model1.predict(test_dataset, verbose=0)
probabilities2 = model2.predict(test_dataset, verbose=0)

probabilities1 = probabilities1.astype(np.float32, copy=False)
probabilities2 = probabilities2.astype(np.float32, copy=False)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities2
probabilities = np.clip(probabilities, 0.0, 1.0)

sub_out = sub.copy()
sub_out.loc[:, TARGET_COLS] = probabilities.astype(np.float32)

sub_path = "submission.csv"
sub_out.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_out.head())
