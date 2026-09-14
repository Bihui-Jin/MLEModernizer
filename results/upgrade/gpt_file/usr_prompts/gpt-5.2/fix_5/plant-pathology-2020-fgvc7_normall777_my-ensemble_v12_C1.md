# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.9711393062766024

# 6. Current score

0.46085

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.46085) has done: 'I fix the TensorFlow/protobuf import crash by removing the environment override that forces the pure-Python protobuf implementation (it’s incompatible with the Kaggle TF/protobuf build). Then I make the tf.data options code compatible across TF versions by not setting the now-removed `autotune_buffers` flag, which currently stops execution before datasets are created (causing the later `test_dataset` NameError). Finally, I keep the model/training/prediction logic intact and ensure the script always writes a valid `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os, re, math, random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import DenseNet201, InceptionResNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB7

print("Tensorflow version " + tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 2
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




## === cell 3
def format_path(st):
    return os.path.join(IMAGES_DIR, st + ".jpg")




## === cell 4
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]

train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values

train_labels = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED
)

print("Train/valid sizes:", len(train_paths), len(valid_paths))
print("Targets:", TARGET_COLS)



## === cell 5
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)

    shape = tf.image.extract_jpeg_shape(bits)
    crop_window = tf.stack([0, 0, shape[0], shape[1]])

    image = tf.image.decode_and_crop_jpeg(
        bits, crop_window=crop_window, channels=3, dct_method="INTEGER_FAST"
    )
    image = tf.image.resize(image, image_size)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.ensure_shape(image, [img_size, img_size, 3])

    if label is None:
        return image
    return image, label


def data_augment(image, label=None, seed=SEED):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    if label is None:
        return image
    return image, label




## === cell 6
def with_fast_options(ds, deterministic=True):
    opts = tf.data.Options()
    opts.experimental_deterministic = deterministic
    try:
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    return ds.with_options(opts)


train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
train_dataset = with_fast_options(train_dataset, deterministic=False)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTO)
)
valid_dataset = with_fast_options(valid_dataset, deterministic=True)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTO)
)
test_dataset = with_fast_options(test_dataset, deterministic=True)




## === cell 7
def get_model(use_model, weights):
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
    )
    return model


def safe_load_weights(model, path):
    if path and tf.io.gfile.exists(path):
        model.load_weights(path)
        print(f"Loaded weights: {path}")
        return True
    print(f"Weight file not found, using built-in weights only: {path}")
    return False




## === cell 8
with strategy.scope():
    model1 = get_model(EfficientNetB7, weights="imagenet")
safe_load_weights(
    model1, "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
)

with strategy.scope():
    model2 = get_model(DenseNet201, weights="imagenet")
safe_load_weights(
    model2, "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
)

with strategy.scope():
    model3 = get_model(InceptionResNetV2, weights="imagenet")
safe_load_weights(
    model3, "/kaggle/input/tf-zoo-models-on-tpu-inceptionresnetv2/my_model.h5"
)



## === cell 9
best_alpha = 0.50
bad_alpha = 0.20

print("Вычисляем предсказания...")

_ = next(iter(test_dataset.take(1)))

probabilities1 = model1.predict(test_dataset, verbose=1)
probabilities2 = model2.predict(test_dataset, verbose=1)
probabilities3 = model3.predict(test_dataset, verbose=1)

probabilities = (
    best_alpha * probabilities1
    + (1.0 - best_alpha - bad_alpha) * probabilities2
    + bad_alpha * probabilities3
)

probabilities = np.clip(probabilities, 0.0, 1.0)

assert probabilities.shape[0] == len(sub), (probabilities.shape, len(sub))
assert probabilities.shape[1] == len(TARGET_COLS), (
    probabilities.shape,
    len(TARGET_COLS),
)

sub[TARGET_COLS] = probabilities
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
