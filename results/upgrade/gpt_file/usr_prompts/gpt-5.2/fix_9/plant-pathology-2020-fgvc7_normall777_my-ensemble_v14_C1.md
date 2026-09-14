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

0.9737660402378432

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.48697) has done: 'I remove the two notebook-only / environment-specific failure points: the `efficientnet` pip install (not needed here) and the unauthenticated `KaggleDatasets().get_gcs_path(...)` call, replacing them with the standard `/kaggle/input/...` path so image loading works. I also fix the initial import crash caused by protobuf incompatibility by avoiding the external `efficientnet` package entirely and using `tf.keras.applications.EfficientNetB7` (same model family) while keeping the ensemble/prediction logic intact. To ensure the script always runs end-to-end, I load pretrained weights only if the referenced Kaggle dataset files exist; otherwise it fall back to the base ImageNet weights (score may be lower but it produce a valid submission). Finally, I make the submission columns exactly match `sample_submission.csv` (`healthy, multiple_diseases, rust, scab`) and write `submission.csv`.'
- What this solution (achieved 0.49977) has done: 'I first fix the immediate TensorFlow import crash caused by a protobuf incompatibility by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`. Then I correct a label/activation mismatch: your targets are multi-label one-hot columns, so the model should use `sigmoid` outputs with `binary_crossentropy` instead of `softmax` + categorical crossentropy (this is the main reason the score is very low). Finally, I keep the rest of the pipeline intact and ensure the submission columns match `sample_submission.csv`, writing a valid `submission.csv`.'
- What this solution (achieved 0.49977) has done: 'The TensorFlow import crash is happening before any modeling due to an incompatible protobuf runtime; the most reliable Kaggle-safe fix is to force-install a protobuf version that TensorFlow in this environment supports, then import TensorFlow normally. After that, the rest of your pipeline can run end-to-end unchanged (same models, sigmoid + BCE, same ensemble alpha, same paths), producing a valid `submission.csv`. I’m also adding a tiny safety fallback to locate the dataset path if Kaggle mounts it under `/kaggle/input/plant-pathology-2020-fgvc7` vs the nested folder, without changing I/O semantics. No score-tuning changes are made beyond unblocking TensorFlow so the same intended logic actually executes.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import random
import numpy as np
import pandas as pd

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT enabled")
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)

EPOCHS = 6
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

CANDIDATE_BASE_PATHS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7",
]
BASE_PATH = None
for p in CANDIDATE_BASE_PATHS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "images")
    ):
        BASE_PATH = p
        break
if BASE_PATH is None:
    BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
print("BASE_PATH:", BASE_PATH)




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(BASE_PATH, "images", f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]

train_paths = (
    os.path.join(BASE_PATH, "images") + os.sep + train["image_id"].astype(str) + ".jpg"
).values
test_paths = (
    os.path.join(BASE_PATH, "images") + os.sep + test["image_id"].astype(str) + ".jpg"
).values
train_labels = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED, stratify=None
)

print("Train/valid sizes:", len(train_paths), len(valid_paths))
print("Targets:", TARGET_COLS)



## === cell 4
img_size = 512


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image.set_shape([image_size[0], image_size[1], 3])
    if label is None:
        return image
    return image, label


def data_augment_stateless(image, label, index):
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(index, tf.int32)], axis=0)
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    image = tf.image.stateless_random_flip_up_down(
        image, seed=seed + tf.constant([0, 1], tf.int32)
    )
    return image, label




## === cell 5
from tensorflow.keras.applications import EfficientNetB7, InceptionResNetV2
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as eff_preprocess,
)
from tensorflow.keras.applications.inception_resnet_v2 import (
    preprocess_input as inc_preprocess,
)


def add_preprocess(preprocess_fn):
    def _pp(image, label=None):
        image = preprocess_fn(image)
        if label is None:
            return image
        return image, label

    return _pp


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.autotune_buffers = True
options.experimental_optimization.map_parallelization = True

base_valid = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
)

base_test = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
)

base_train = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(options)
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .enumerate()
    .map(
        lambda i, xy: (decode_image(xy[0], xy[1])[0], xy[1], i), num_parallel_calls=AUTO
    )
    .map(data_augment_stateless, num_parallel_calls=AUTO)
    .repeat()
)

train_dataset_eff = (
    base_train.map(add_preprocess(eff_preprocess), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False, num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

valid_dataset_eff = (
    base_valid.map(add_preprocess(eff_preprocess), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_dataset_eff = (
    base_test.map(add_preprocess(eff_preprocess), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

train_dataset_inc = (
    base_train.map(add_preprocess(inc_preprocess), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False, num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

valid_dataset_inc = (
    base_valid.map(add_preprocess(inc_preprocess), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_dataset_inc = (
    base_test.map(add_preprocess(inc_preprocess), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3791118517.py in <cell line: 0>()
     23 options.experimental_deterministic = True
     24 options.experimental_optimization.apply_default_optimizations = True
---> 25 options.experimental_optimization.autotune_buffers = True
     26 options.experimental_optimization.map_parallelization = True
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 6
def build_model(base_ctor, weights, num_classes: int):
    base_model = base_ctor(
        weights=weights,
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output
    preds = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=preds)
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
        jit_compile=True,
    )
    return model


NUM_CLASSES = train_labels.shape[1]



## === cell 7
EF_B7_WEIGHTS = "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
INCRES_WEIGHTS = "/kaggle/input/tf-zoo-models-on-tpu/InceptionResNetV2.h5"

with strategy.scope():
    model1 = build_model(EfficientNetB7, weights="imagenet", num_classes=NUM_CLASSES)
if os.path.exists(EF_B7_WEIGHTS):
    model1.load_weights(EF_B7_WEIGHTS)
    print("Loaded EfficientNetB7 weights:", EF_B7_WEIGHTS)
else:
    print(
        "EfficientNetB7 weights not found; will fine-tune on competition data:",
        EF_B7_WEIGHTS,
    )

with strategy.scope():
    model3 = build_model(InceptionResNetV2, weights="imagenet", num_classes=NUM_CLASSES)
if os.path.exists(INCRES_WEIGHTS):
    model3.load_weights(INCRES_WEIGHTS)
    print("Loaded InceptionResNetV2 weights:", INCRES_WEIGHTS)
else:
    print(
        "InceptionResNetV2 weights not found; will fine-tune on competition data:",
        INCRES_WEIGHTS,
    )

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
val_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps, "EPOCHS:", EPOCHS)

model1.get_layer(index=1).trainable = False  # base model is layer 1 in this build
model1.compile(
    optimizer="nadam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
    jit_compile=True,
)
model1.fit(
    train_dataset_eff,
    validation_data=valid_dataset_eff,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)

model3.get_layer(index=1).trainable = False
model3.compile(
    optimizer="nadam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
    jit_compile=True,
)
model3.fit(
    train_dataset_inc,
    validation_data=valid_dataset_inc,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)


def unfreeze_top_layers(model, n_layers_to_unfreeze=40):
    backbone = model.layers[1]
    backbone.trainable = True
    for l in backbone.layers[:-n_layers_to_unfreeze]:
        l.trainable = False
    for l in backbone.layers[-n_layers_to_unfreeze:]:
        l.trainable = True


unfreeze_top_layers(model1, n_layers_to_unfreeze=30)
model1.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
    jit_compile=True,
)
model1.fit(
    train_dataset_eff,
    validation_data=valid_dataset_eff,
    epochs=EPOCHS - 2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)

unfreeze_top_layers(model3, n_layers_to_unfreeze=30)
model3.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
    jit_compile=True,
)
model3.fit(
    train_dataset_inc,
    validation_data=valid_dataset_inc,
    epochs=EPOCHS - 2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3422335799.py in <cell line: 0>()
     36 )
     37 model1.fit(
---> 38     train_dataset_eff,
     39     validation_data=valid_dataset_eff,
     40     epochs=2,

NameError: name 'train_dataset_eff' is not defined

## === cell 8
best_alpha = 0.52

print("Computing predictions...")
test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))
probabilities1 = model1.predict(test_dataset_eff, steps=test_steps, verbose=1)
probabilities3 = model3.predict(test_dataset_inc, steps=test_steps, verbose=1)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities3

probabilities = np.asarray(probabilities, dtype=np.float32)
if probabilities.shape[1] != len(TARGET_COLS):
    raise ValueError(
        f"Prediction has shape {probabilities.shape}, expected (*, {len(TARGET_COLS)})"
    )
probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

sub[TARGET_COLS] = probabilities
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1059381372.py in <cell line: 0>()
      3 print("Computing predictions...")
      4 test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))
----> 5 probabilities1 = model1.predict(test_dataset_eff, steps=test_steps, verbose=1)
      6 probabilities3 = model3.predict(test_dataset_inc, steps=test_steps, verbose=1)
      7 

NameError: name 'test_dataset_eff' is not defined

## === cell 9
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image_id"] + TARGET_COLS
assert len(check) == len(test)
print("Submission looks valid.")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/42327585.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 check = pd.read_csv("submission.csv")
      3 assert list(check.columns) == ["image_id"] + TARGET_COLS
      4 assert len(check) == len(test)
      5 print("Submission looks valid.")

AssertionError:
