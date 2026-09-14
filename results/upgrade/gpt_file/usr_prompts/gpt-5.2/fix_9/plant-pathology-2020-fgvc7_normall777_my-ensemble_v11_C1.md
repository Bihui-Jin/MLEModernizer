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
import os, random

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import DenseNet201

print("Tensorflow version " + tf.__version__)

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.get_logger().setLevel("ERROR")

try:
    ncpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(8, ncpu))
    tf.config.threading.set_inter_op_parallelism_threads(min(8, ncpu))
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




## === cell 1
AUTO = tf.data.AUTOTUNE
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

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"

EPOCHS = 5
BATCH_SIZE = 8 * strategy.num_replicas_in_sync




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(DATA_DIR, "images", f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_cols = [c for c in sub.columns if c != "image_id"]

train_paths = (DATA_DIR + "/images/" + train["image_id"].astype(str) + ".jpg").values
test_paths = (DATA_DIR + "/images/" + test["image_id"].astype(str) + ".jpg").values

train_labels = train[target_cols].values.astype(np.float32)

stratify_y = np.argmax(train_labels, axis=1)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.15,
    random_state=SEED,
    stratify=stratify_y,
)

print("Train/Valid sizes:", len(train_paths), len(valid_paths))
print("Targets:", target_cols)




## === cell 4
img_size = 512


@tf.function
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    return image, label


@tf.function
def data_augment(image, label=None):
    seed = tf.stack(
        [tf.constant(SEED, tf.int32), tf.shape(image)[0] + tf.shape(image)[1]]
    )
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    image = tf.image.stateless_random_flip_up_down(image, seed=seed + 1)
    if label is None:
        return image
    return image, label




## === cell 5
CACHE_DIR = "/kaggle/working/tf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, f"train_{img_size}.cache")
VALID_CACHE = os.path.join(CACHE_DIR, f"valid_{img_size}.cache")
TEST_CACHE = os.path.join(CACHE_DIR, f"test_{img_size}.cache")

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.map_and_batch_fusion = True

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache(TRAIN_CACHE)
    .map(data_augment, num_parallel_calls=AUTO)
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache(VALID_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache(TEST_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
val_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))

print(
    "BATCH_SIZE:",
    BATCH_SIZE,
    "steps_per_epoch:",
    steps_per_epoch,
    "val_steps:",
    val_steps,
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

    predictions = Dense(train_labels.shape[1], activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
        steps_per_execution=max(1, min(64, steps_per_epoch)),
    )
    return model


def maybe_load_weights(model, weight_path):
    if weight_path and tf.io.gfile.exists(weight_path):
        model.load_weights(weight_path)
        print(f"Loaded weights: {weight_path}")
        return True
    print(f"Pretrained .h5 weights not found (will train): {weight_path}")
    return False




## === cell 7
with strategy.scope():
    model1 = get_model(EfficientNetB7, weights="imagenet")
with strategy.scope():
    model2 = get_model(DenseNet201, weights="imagenet")

w1 = "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
w2 = "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"

loaded1 = maybe_load_weights(model1, w1)
loaded2 = maybe_load_weights(model2, w2)

if (not loaded1) or (not loaded2):
    print(
        "One or more pretrained weight files missing; training models for EPOCHS=",
        EPOCHS,
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, verbose=1
        )
    ]

    if not loaded1:
        print("Training EfficientNetB7...")
        model1.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_dataset,
            validation_steps=val_steps,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )

    if not loaded2:
        print("Training DenseNet201...")
        model2.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_dataset,
            validation_steps=val_steps,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )




## === cell 8
best_alpha = 0.90
print("Вычисляем предсказания...")

probabilities1 = model1.predict(test_dataset, verbose=1)
probabilities2 = model2.predict(test_dataset, verbose=1)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities2
probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)

sub = sub[["image_id"] + target_cols].copy()
sub[target_cols] = probabilities

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
