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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ.pop("TF_NUM_INTRAOP_THREADS", None)
os.environ.pop("TF_NUM_INTEROP_THREADS", None)

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow.keras.backend as K

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__)



## === cell 1
SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", e)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Could not enable op determinism:", e)

try:
    pass
except Exception as e:
    print("Threading config skipped:", e)



## === cell 2
AUTO = tf.data.AUTOTUNE

tpu = None
strategy = tf.distribute.get_strategy()
print("REPLICAS:", strategy.num_replicas_in_sync)



## === cell 3
path = "../input/plant-pathology-2020-fgvc7/"



## === cell 4
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
sub = pd.read_csv(path + "sample_submission.csv")

IMG_DIR = path + "images/"

train_paths = (IMG_DIR + train.image_id.values + ".jpg").astype(str)
test_paths = (IMG_DIR + test.image_id.values + ".jpg").astype(str)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
train_labels = train[target_cols].values.astype(np.float32)

print("Train:", train.shape, "Test:", test.shape, "Sub:", sub.shape)
print("Targets:", target_cols)



## === cell 5
nb_classes = 4
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
img_size = 896
EPOCHS = 40

print("BATCH_SIZE:", BATCH_SIZE, "img_size:", img_size, "EPOCHS:", EPOCHS)




## === cell 6
def _decode_resize(filename):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _augment(image):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    return image


def decode_image_train(filename, label):
    image = _decode_resize(filename)
    image = _augment(image)
    return image, label


def decode_image_test(filename):
    image = _decode_resize(filename)
    return image




## === cell 7
tfdata_opts = tf.data.Options()
tfdata_opts.experimental_deterministic = True

try:
    tfdata_opts.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    tfdata_opts.experimental_optimization.map_fusion = True
except Exception:
    pass
try:
    tfdata_opts.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    tfdata_opts.experimental_optimization.parallel_batch = True
except Exception:
    pass

cache_dir = "/kaggle/working/tf_cache_pp2020"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(cache_dir, f"train_{img_size}.cache")
test_cache_path = os.path.join(cache_dir, f"test_{img_size}.cache")

decoded_train = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(tfdata_opts)
    .map(lambda f, y: (_decode_resize(f), y), num_parallel_calls=AUTO)
    .cache(train_cache_path)
)

if not (
    os.path.exists(train_cache_path) or os.path.exists(train_cache_path + ".index")
):
    _ = list(decoded_train.take(1))  # triggers cache file creation header
    for _ in decoded_train:
        pass

train_dataset = (
    decoded_train.shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .map(lambda x, y: (_augment(x), y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=True)
    .repeat()
    .prefetch(AUTO)
)

decoded_test = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(tfdata_opts)
    .map(decode_image_test, num_parallel_calls=AUTO)
    .cache(test_cache_path)
)

if not (os.path.exists(test_cache_path) or os.path.exists(test_cache_path + ".index")):
    _ = list(decoded_test.take(1))
    for _ in decoded_test:
        pass

test_dataset = decoded_test.batch(BATCH_SIZE).prefetch(AUTO)



## === cell 8
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


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)




## === cell 9
def get_model():
    pretrained_model = tf.keras.applications.EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    pretrained_model.trainable = True

    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    return model


with strategy.scope():
    model = get_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=128,
    )

model.summary()



## === cell 10
steps_per_epoch = train_labels.shape[0] // BATCH_SIZE

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    callbacks=[lr_callback],
    epochs=EPOCHS,
    verbose=2,
)



## === cell 11
probs = model.predict(test_dataset, verbose=1)
probs = probs[: len(test)]
print("probs shape:", probs.shape)

sub = sub.copy()
sub[target_cols] = probs.astype(np.float32)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
