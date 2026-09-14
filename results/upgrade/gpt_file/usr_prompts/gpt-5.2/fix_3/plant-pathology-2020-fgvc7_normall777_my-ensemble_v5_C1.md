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

0.9722616009777888

# 6. Current score

0.47009

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.47842) has done: 'I remove the notebook-only `pip install` and `%matplotlib inline` lines that break scripted execution, and fix the protobuf/TensorFlow import issue by avoiding the extra `efficientnet` pip package and using `tf.keras.applications.EfficientNetB7` instead. I also eliminate the unauthenticated `KaggleDatasets().get_gcs_path(...)` dependency and build image paths directly from the local `/kaggle/input/plant-pathology-2020-fgvc7/images` folder, which unblocks dataset creation. Finally, I keep your core ensemble/prediction logic intact, but make the model output activation/loss consistent with the competition’s multi-label targets (sigmoid + binary_crossentropy) so predictions are properly calibrated for mean ROC AUC, and ensure a valid `submission.csv` is written with the required columns.'
- What this solution (achieved 0.47009) has done: 'I fix the TensorFlow/protobuf crash occurring at import time by forcing TensorFlow to use the pure-Python protobuf implementation before `tensorflow` is imported (this is the standard workaround for the `MessageFactory.GetPrototype` error in Kaggle images). I keep your model/ensemble logic intact, but also add the correct preprocessing functions for each backbone (EfficientNet/DenseNet/InceptionResNetV2) so the ImageNet weights are used as intended, which should materially improve ROC AUC toward the target without changing the architecture. Finally, I ensure the submission columns exactly match `sample_submission.csv` and that a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import DenseNet201, InceptionResNetV2
from tensorflow.keras.applications import (
    EfficientNetB7,
)  # use built-in TF EfficientNet to avoid protobuf issues

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as effnet_preprocess,
)
from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess,
)
from tensorflow.keras.applications.inception_resnet_v2 import (
    preprocess_input as irv2_preprocess,
)

print("Tensorflow version " + tf.__version__)

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

INPUT_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

assert os.path.isdir(IMAGES_DIR), f"Images directory not found: {IMAGES_DIR}"




## === cell 2
def format_path(st):
    return os.path.join(IMAGES_DIR, st + ".jpg")




## === cell 3
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

target_cols = [c for c in train.columns if c != "image_id"]
assert target_cols == [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], f"Unexpected target columns: {target_cols}"

assert (
    list(sub.columns) == ["image_id"] + target_cols
), f"Unexpected submission columns: {list(sub.columns)}"

train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values

train_labels = train[target_cols].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED
)

assert len(train_paths) == len(train_labels)
assert len(valid_paths) == len(valid_labels)
assert os.path.isfile(
    train_paths[0]
), f"Example train image not found: {train_paths[0]}"
assert os.path.isfile(test_paths[0]), f"Example test image not found: {test_paths[0]}"



## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        return image, label


def data_augment(image, label=None, seed=SEED):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)

    if label is None:
        return image
    else:
        return image, label


def preprocess_for(model_name, image, label=None):
    if model_name == "efficientnetb7":
        x = effnet_preprocess(image)
    elif model_name == "densenet201":
        x = densenet_preprocess(image)
    elif model_name == "inceptionresnetv2":
        x = irv2_preprocess(image)
    else:
        x = image
    if label is None:
        return x
    return x, label




## === cell 5
def make_dataset(paths, labels=None, training=False, model_name="efficientnetb7"):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: decode_image(p, None), num_parallel_calls=AUTO)
        ds = ds.map(
            lambda img: preprocess_for(model_name, img, None), num_parallel_calls=AUTO
        )
        ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_image, num_parallel_calls=AUTO)

    if training:
        ds = ds.map(data_augment, num_parallel_calls=AUTO)

    ds = ds.map(
        lambda img, y: preprocess_for(model_name, img, y), num_parallel_calls=AUTO
    )

    if training:
        ds = ds.repeat().shuffle(512, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE)
    if not training:
        ds = ds.cache()
    ds = ds.prefetch(AUTO)
    return ds




## === cell 6
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
    model.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["AUC"])
    return model




## === cell 7
with strategy.scope():
    model1 = get_model(EfficientNetB7, weights="imagenet")
try:
    model1.load_weights(
        "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
    )
    print("Loaded weights for model1.")
except Exception as e:
    print(f"Could not load model1 weights, using imagenet init. Reason: {e}")

with strategy.scope():
    model2 = get_model(DenseNet201, weights="imagenet")
try:
    model2.load_weights(
        "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
    )
    print("Loaded weights for model2.")
except Exception as e:
    print(f"Could not load model2 weights, using imagenet init. Reason: {e}")

with strategy.scope():
    model3 = get_model(InceptionResNetV2, weights="imagenet")
try:
    model3.load_weights(
        "/kaggle/input/tf-zoo-models-on-tpu-inceptionresnetv2/my_model.h5"
    )
    print("Loaded weights for model3.")
except Exception as e:
    print(f"Could not load model3 weights, using imagenet init. Reason: {e}")



## === cell 8
best_alpha = 0.36
bad_alpha = 0.30

print("Вычисляем предсказания...")

test_ds_1 = make_dataset(
    test_paths, labels=None, training=False, model_name="efficientnetb7"
)
test_ds_2 = make_dataset(
    test_paths, labels=None, training=False, model_name="densenet201"
)
test_ds_3 = make_dataset(
    test_paths, labels=None, training=False, model_name="inceptionresnetv2"
)

probabilities1 = model1.predict(test_ds_1, verbose=1)
probabilities2 = model2.predict(test_ds_2, verbose=1)
probabilities3 = model3.predict(test_ds_3, verbose=1)

probabilities = (
    best_alpha * probabilities1
    + bad_alpha * probabilities2
    + (1 - best_alpha - bad_alpha) * probabilities3
)

probabilities = np.clip(probabilities, 0.0, 1.0)

sub = sub.copy()
sub["image_id"] = test["image_id"].values
sub[target_cols] = probabilities.astype(np.float32)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert os.path.isfile("submission.csv") and os.path.getsize("submission.csv") > 0
