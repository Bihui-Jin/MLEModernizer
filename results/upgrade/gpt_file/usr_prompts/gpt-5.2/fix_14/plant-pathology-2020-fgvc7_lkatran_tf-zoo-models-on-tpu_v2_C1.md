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

3.8

# 3. Installed packages



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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "1")

import math, re, random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from matplotlib import pyplot as plt

print("TF version:", tf.__version__)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTO = tf.data.experimental.AUTOTUNE

tpu = None
strategy = tf.distribute.get_strategy()
print("REPLICAS:", strategy.num_replicas_in_sync)

EPOCHS = 12

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATA_DIR, "images")




## === cell 1
def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, image_id + ".jpg")


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

assert (
    "image_id" in train.columns
    and "image_id" in test.columns
    and "image_id" in sub.columns
)
for c in TARGET_COLS:
    assert c in train.columns and c in sub.columns

train_paths_all = (IMAGES_DIR + "/" + train["image_id"].astype(str) + ".jpg").values
train_labels_all = train[TARGET_COLS].values.astype(np.float32)

test_paths = (IMAGES_DIR + "/" + test["image_id"].astype(str) + ".jpg").values

print("Train:", train.shape, "Test:", test.shape, "Sub:", sub.shape)
print("Example train path:", train_paths_all[0])




## === cell 2
rng = np.random.RandomState(SEED)
idx = np.arange(len(train_paths_all))
rng.shuffle(idx)

val_frac = 0.15
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_paths = train_paths_all[trn_idx]
train_labels = train_labels_all[trn_idx]
valid_paths = train_paths_all[val_idx]
valid_labels = train_labels_all[val_idx]

print("Train split:", train_paths.shape, train_labels.shape)
print("Valid split:", valid_paths.shape, valid_labels.shape)




## === cell 3
nb_classes = 4

img_size = 384

DECODE_CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(DECODE_CACHE_DIR, exist_ok=True)
TRAIN_DECODE_CACHE = os.path.join(DECODE_CACHE_DIR, f"train_decode_{img_size}.cache")
VALID_DECODE_CACHE = os.path.join(DECODE_CACHE_DIR, f"valid_decode_{img_size}.cache")
TEST_DECODE_CACHE = os.path.join(DECODE_CACHE_DIR, f"test_decode_{img_size}.cache")

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True


@tf.function
def _decode_resize_only(path, image_size=(img_size, img_size)):
    bits = tf.io.read_file(path)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.AREA)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([img_size, img_size, 3])
    return image


@tf.function
def _augment_stateless(image, label, seed_pair):
    image = tf.image.stateless_random_flip_left_right(image, seed=seed_pair)
    image = tf.image.stateless_random_flip_up_down(
        image, seed=seed_pair + tf.constant([0, 1], tf.int32)
    )
    return image, label


@tf.function
def _decode_resize_from_path_noaug(path, label=None, image_size=(img_size, img_size)):
    image = _decode_resize_only(path, image_size=image_size)
    if label is None:
        return image
    return image, label


train_paths_tf = tf.constant(train_paths)
train_labels_tf = tf.constant(train_labels)
valid_paths_tf = tf.constant(valid_paths)
valid_labels_tf = tf.constant(valid_labels)
test_paths_tf = tf.constant(test_paths)

train_decoded = (
    tf.data.Dataset.from_tensor_slices((train_paths_tf, train_labels_tf))
    .with_options(options)
    .map(
        lambda p, y: (_decode_resize_only(p), y),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .cache(TRAIN_DECODE_CACHE)
)

train_dataset = (
    train_decoded.enumerate()  # provides a stable per-element index each iteration of the repeated dataset
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .map(
        lambda i, xy: _augment_stateless(
            xy[0],
            xy[1],
            tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)]),
        ),
        num_parallel_calls=AUTO,
        deterministic=False,  # order doesn't matter; seeds ensure deterministic augmentation per element index
    )
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths_tf, valid_labels_tf))
    .with_options(options)
    .map(
        lambda p, y: _decode_resize_from_path_noaug(p, y),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .cache(VALID_DECODE_CACHE)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    .apply(tf.data.experimental.ignore_errors())
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths_tf)
    .with_options(options)
    .map(
        lambda p: _decode_resize_from_path_noaug(p, None),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .cache(TEST_DECODE_CACHE)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    .apply(tf.data.experimental.ignore_errors())
)

print(
    "Datasets built:",
    "train batch spec:",
    train_dataset.element_spec,
    "valid batch spec:",
    valid_dataset.element_spec,
    "test batch spec:",
    test_dataset.element_spec,
)




## === cell 4
LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 15
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = 0.8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        ) + LR_MIN
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng_epochs = list(range(EPOCHS))
y = [lrfn(x) for x in rng_epochs]
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))




## === cell 5
def get_model():
    base_model = tf.keras.applications.EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)


with strategy.scope():
    model = get_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
        steps_per_execution=32,
    )

model.summary()




## === cell 6
steps_per_epoch = max(1, int(train_labels.shape[0] // BATCH_SIZE))
validation_steps = max(1, int(np.ceil(valid_labels.shape[0] / BATCH_SIZE)))

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    callbacks=[lr_callback],
    validation_data=valid_dataset,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 7
def display_training_curves(training, validation, title, subplot):
    """
    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
    """
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10), facecolor="#F0F0F0")
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor("#F8F8F8")
    ax.plot(training)
    if validation is not None:
        ax.plot(validation)
        ax.legend(["train", "valid."])
    else:
        ax.legend(["train"])
    ax.set_title("model " + title)
    ax.set_ylabel(title)
    ax.set_xlabel("epoch")


display_training_curves(
    history.history.get("loss", []),
    history.history.get("val_loss", None),
    "loss",
    211,
)
display_training_curves(
    history.history.get("categorical_accuracy", []),
    history.history.get("val_categorical_accuracy", None),
    "accuracy",
    212,
)
plt.show()




## === cell 8
name_model = "efficientnet.h5"
model.save(name_model)
print("Saved model to:", name_model)




## === cell 9
probs = model.predict(test_dataset, verbose=1)
probs = np.asarray(probs)

if probs.shape[0] != len(test):
    probs = probs[: len(test)]

assert probs.shape[0] == len(test), (probs.shape, len(test))
assert probs.shape[1] == len(TARGET_COLS), (probs.shape, len(TARGET_COLS))

submission = pd.DataFrame({"image_id": test["image_id"].values})
for i, c in enumerate(TARGET_COLS):
    submission[c] = probs[:, i]

submission = submission[["image_id"] + TARGET_COLS]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
