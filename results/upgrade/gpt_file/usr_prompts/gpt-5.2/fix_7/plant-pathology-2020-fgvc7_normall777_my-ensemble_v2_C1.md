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
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import EfficientNetB7

print("Tensorflow version " + tf.__version__)

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.optimizer.set_jit(True)
tf.config.run_functions_eagerly(False)
try:
    tf.keras.backend.set_learning_phase(0)
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

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATA_DIR, "images")

assert os.path.isdir(IMAGES_DIR), f"Images directory not found: {IMAGES_DIR}"




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]

train_paths = train["image_id"].apply(format_path).values
test_paths = test["image_id"].apply(format_path).values

train_labels = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED, stratify=None
)

print("Train/Valid sizes:", len(train_paths), len(valid_paths))
print("Targets:", TARGET_COLS)




## === cell 4
img_size = 768


@tf.function
def _decode_only(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (img_size, img_size))
    if label is None:
        return image
    return image, label


@tf.function
def _augment_stateless(image, label, seed_pair):
    image = tf.image.stateless_random_flip_left_right(image, seed=seed_pair)
    seed_pair2 = tf.stack(
        [
            seed_pair[0] ^ tf.constant(0x1234, tf.int32),
            seed_pair[1] ^ tf.constant(0x5678, tf.int32),
        ]
    )
    image = tf.image.stateless_random_flip_up_down(image, seed=seed_pair2)
    return image, label




## === cell 5
options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
    options.threading.max_intra_op_parallelism = 1
except Exception:
    pass

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(lambda f, y: _decode_only(f, y), num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
).with_options(options)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_decode_only, num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
).with_options(options)

train_base = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_base = train_base.shuffle(512, seed=SEED, reshuffle_each_iteration=True).repeat()
train_base = train_base.enumerate()  # deterministic per-element index


def _train_map(i, xy):
    f = xy[0]
    y = xy[1]
    image, y = _decode_only(f, y)
    seed_pair = tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)])
    image, y = _augment_stateless(image, y, seed_pair)
    return image, y


train_dataset = (
    train_base.map(_train_map, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
).with_options(options)

try:
    if not tpu:
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            train_dataset = train_dataset.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTO)
            )
            valid_dataset = valid_dataset.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTO)
            )
            test_dataset = test_dataset.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTO)
            )
except Exception:
    pass

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
valid_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))
test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))

print(
    "steps_per_epoch:",
    steps_per_epoch,
    "valid_steps:",
    valid_steps,
    "test_steps:",
    test_steps,
)




## === cell 6
def get_model(use_model):
    base_model = use_model(
        weights="imagenet",
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
        run_eagerly=False,
        jit_compile=True,
    )
    return model


def _find_weight_file(preferred_path: str) -> str:
    if not preferred_path:
        return preferred_path

    if os.path.exists(preferred_path):
        return preferred_path

    fname = os.path.basename(preferred_path)

    candidate_dirs = [
        os.path.dirname(preferred_path),
        "/kaggle/input",
        "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7",
        "/kaggle/input/tf-zoo-models-on-tpu-densenet201",
        "/kaggle/input/plant-pathology-2020-fgvc7",
    ]

    for d in candidate_dirs:
        if not d:
            continue
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p

    return preferred_path


def try_load_weights(model, weight_path):
    weight_path = _find_weight_file(weight_path)
    if weight_path and os.path.exists(weight_path):
        model.load_weights(weight_path)
        print(f"Loaded weights: {weight_path}")
        return True
    print(f"Weights not found, will train from scratch/fine-tune: {weight_path}")
    return False




## === cell 7
with strategy.scope():
    model1 = get_model(EfficientNetB7)

w1 = "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
loaded1 = try_load_weights(model1, w1)




## === cell 8
with strategy.scope():
    model2 = get_model(DenseNet201)

w2 = "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
loaded2 = try_load_weights(model2, w2)




## === cell 9
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, verbose=1
    ),
]

if not loaded1:
    print("Training model1...")
    model1.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=valid_steps,
        callbacks=callbacks,
        verbose=1,
    )

if not loaded2:
    print("Training model2...")
    model2.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=valid_steps,
        callbacks=callbacks,
        verbose=1,
    )




## === cell 10
best_alpha = 0.52
print("Вычисляем предсказания...")

probabilities1 = model1.predict(test_dataset, steps=test_steps, verbose=1)
probabilities2 = model2.predict(test_dataset, steps=test_steps, verbose=1)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities2
probabilities = np.clip(probabilities, 0.0, 1.0)

assert probabilities.shape[0] == sub.shape[0], (probabilities.shape, sub.shape)

sub[TARGET_COLS] = probabilities
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
