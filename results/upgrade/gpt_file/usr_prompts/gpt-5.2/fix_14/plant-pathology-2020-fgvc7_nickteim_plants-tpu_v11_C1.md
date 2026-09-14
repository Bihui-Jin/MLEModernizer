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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import os, math, random
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

print("tf:", tf.__version__)
print("keras:", tf.keras.__version__)

SEED = 2020
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.mixed_precision.set_global_policy("float32")
    print("Mixed precision disabled: using float32 for determinism/perf stability")
except Exception as e:
    print("Mixed precision policy set failed (continuing):", repr(e))

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT enable failed (continuing):", repr(e))

try:
    tf.config.experimental.enable_op_determinism(True)
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




## === cell 2
DATA_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/"
IMG_DIR = os.path.join(DATA_PATH, "images")

train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

train_paths = (IMG_DIR + "/" + train["image_id"].astype(str) + ".jpg").to_numpy()
test_paths = (IMG_DIR + "/" + test["image_id"].astype(str) + ".jpg").to_numpy()

train_labels = train[TARGET_COLS].values.astype(np.float32)

print(train.shape, test.shape, sub.shape)
print("Example path exists:", train_paths[0], os.path.exists(train_paths[0]))




## === cell 3
print("Skipping debug image read for runtime.")




## === cell 4
nb_classes = 4
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
img_size = 768
EPOCHS = 40

IMG_SIZE = int(img_size)




## === cell 5
@tf.function(reduce_retracing=True)
def decode_resized(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return image


@tf.function(reduce_retracing=True)
def aug_from_image(image, filename, label, seed0=SEED):
    h = tf.strings.to_hash_bucket_fast(filename, 2**31 - 1)
    seed = tf.stack([tf.cast(seed0, tf.int32), tf.cast(h, tf.int32)], axis=0)

    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    image = tf.image.stateless_random_flip_up_down(
        image, seed=seed + tf.constant([0, 1], tf.int32)
    )
    image = tf.image.adjust_saturation(image, 2.0)
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return image, label


@tf.function(reduce_retracing=True)
def decode_and_aug(filename, label, seed0=SEED):
    image = decode_resized(filename)
    return aug_from_image(image, filename, label, seed0=seed0)


@tf.function(reduce_retracing=True)
def decode_only(filename):
    return decode_resized(filename)




## === cell 6
ds_options = tf.data.Options()
ds_options.experimental_deterministic = True
try:
    ds_options.experimental_optimization.map_and_batch_fusion = True
    ds_options.experimental_optimization.parallel_batch = True
    ds_options.experimental_optimization.autotune_buffers = True
    ds_options.experimental_optimization.apply_default_optimizations = True
    ds_options.experimental_optimization.map_parallelization = True
    ds_options.experimental_optimization.experimental_slack = True
except Exception:
    pass

try:
    ds_options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
except Exception:
    pass

try:
    ds_options.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
    ds_options.threading.max_intra_op_parallelism = 1
except Exception:
    pass

train_files = tf.data.Dataset.from_tensor_slices(train_paths).with_options(ds_options)
train_labels_ds = tf.data.Dataset.from_tensor_slices(train_labels).with_options(
    ds_options
)

decoded_train = (
    tf.data.Dataset.zip((train_files, train_labels_ds))
    .map(lambda fn, lab: (fn, decode_resized(fn), lab), num_parallel_calls=AUTO)
    .cache()  # in-memory cache of (filename, decoded_image, label)
)

train_dataset = (
    decoded_train.shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .map(
        lambda fn, img, lab: aug_from_image(img, fn, lab, seed0=SEED),
        num_parallel_calls=AUTO,
    )
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

test_base = tf.data.Dataset.from_tensor_slices(test_paths).with_options(ds_options)
test_dataset = (
    test_base.map(decode_only, num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)





## === cell 7
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

rng = list(range(EPOCHS))
y = [lrfn(x) for x in rng]
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))




## === cell 8
def get_model():
    pretrained_model = tf.keras.applications.EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    pretrained_model.trainable = True

    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.Dense(nb_classes, activation="softmax", dtype="float32"),
        ]
    )
    return model


with strategy.scope():
    model = get_model()

steps_per_epoch = train_labels.shape[0] // BATCH_SIZE
spe = max(1, min(steps_per_epoch, 2048))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=spe,
)
model.summary()




## === cell 9
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    callbacks=[lr_callback],
    epochs=16,
    verbose=1,
)

probs = model.predict(test_dataset, verbose=1)
probs = probs[: len(sub)]

sub[TARGET_COLS] = probs
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
