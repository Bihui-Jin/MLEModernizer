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

0.9695887397601226

# 6. Current score

0.46943

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.45069) has done: 'I remove the `efficientnet` pip install (it triggers a protobuf incompatibility in this environment) and instead use the built-in `tf.keras.applications.EfficientNetB7` so imports work reliably. I also replace the unauthenticated `KaggleDatasets().get_gcs_path(...)` call with a simple local `/kaggle/input/...` path fallback, which fixes all downstream `GCS_DS_PATH`/`train_paths`/dataset/model NameErrors. Finally, I keep your ensemble and pretrained-weight loading logic intact, only making it robust to missing weight files and ensuring the submission columns exactly match `sample_submission.csv` and writes `submission.csv`.'
- What this solution (achieved 0.46943) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation before TensorFlow is imported. Then I make the dataset path resolution robust to either `/kaggle/input/plant-pathology-2020-fgvc7` or `/kaggle/input` layouts, so CSV/image reads don’t silently point to missing files. Finally, I correct the model head/loss for this multi-label competition (sigmoid + binary_crossentropy) and use the proper preprocessing functions for each backbone; this keeps the same overall approach (pretrained CNNs + ensemble) but should substantially improve ROC AUC toward the target while still producing a valid `submission.csv` with the exact sample columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import math, re, random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

print("TensorFlow:", tf.__version__)
SEED = 2020
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



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

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

CANDIDATE_DATA_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
]
DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "images")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"

IMAGES_DIR = os.path.join(DATA_DIR, "images")
print("DATA_DIR:", DATA_DIR)
print("IMAGES_DIR exists:", os.path.exists(IMAGES_DIR))




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET_COLS = list(sub.columns[1:])  # ['healthy','multiple_diseases','rust','scab']

train_paths = train["image_id"].apply(format_path).values
test_paths = test["image_id"].apply(format_path).values

train_labels = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED, stratify=None
)

print("Train:", len(train_paths), "Valid:", len(valid_paths), "Test:", len(test_paths))
print("Targets:", TARGET_COLS)

if len(train_paths) == 0 or len(test_paths) == 0:
    raise RuntimeError("No image paths found. Check DATA_DIR/IMAGES_DIR.")
if not os.path.exists(train_paths[0]):
    raise FileNotFoundError(f"Example train image not found: {train_paths[0]}")
if not os.path.exists(test_paths[0]):
    raise FileNotFoundError(f"Example test image not found: {test_paths[0]}")



## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    if label is None:
        return image
    return image, label


def data_augment(image, label=None, seed=SEED):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    if label is None:
        return image
    return image, label




## === cell 5
from tensorflow.keras.applications import DenseNet201, InceptionResNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess,
)
from tensorflow.keras.applications.inception_resnet_v2 import (
    preprocess_input as irv2_preprocess,
)
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as effnet_preprocess,
)


def make_preprocess_fn(preprocess_fn):
    def _pp(image, label=None):
        image = preprocess_fn(image)
        if label is None:
            return image
        return image, label

    return _pp




## === cell 6
train_dataset_base = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset_base = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset_base = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)

train_dataset_eff = train_dataset_base.map(
    make_preprocess_fn(effnet_preprocess), num_parallel_calls=AUTO
)
valid_dataset_eff = valid_dataset_base.map(
    make_preprocess_fn(effnet_preprocess), num_parallel_calls=AUTO
)
test_dataset_eff = test_dataset_base.map(
    make_preprocess_fn(effnet_preprocess), num_parallel_calls=AUTO
)

train_dataset_den = train_dataset_base.map(
    make_preprocess_fn(densenet_preprocess), num_parallel_calls=AUTO
)
valid_dataset_den = valid_dataset_base.map(
    make_preprocess_fn(densenet_preprocess), num_parallel_calls=AUTO
)
test_dataset_den = test_dataset_base.map(
    make_preprocess_fn(densenet_preprocess), num_parallel_calls=AUTO
)

train_dataset_irv = train_dataset_base.map(
    make_preprocess_fn(irv2_preprocess), num_parallel_calls=AUTO
)
valid_dataset_irv = valid_dataset_base.map(
    make_preprocess_fn(irv2_preprocess), num_parallel_calls=AUTO
)
test_dataset_irv = test_dataset_base.map(
    make_preprocess_fn(irv2_preprocess), num_parallel_calls=AUTO
)




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
        metrics=[],
    )
    return model




## === cell 8
WEIGHT_1 = "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
WEIGHT_2 = "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
WEIGHT_3 = "/kaggle/input/tf-zoo-models-on-tpu-inceptionresnetv2/my_model.h5"

with strategy.scope():
    model1 = get_model(EfficientNetB7, weights="imagenet")
with strategy.scope():
    model2 = get_model(DenseNet201, weights="imagenet")
with strategy.scope():
    model3 = get_model(InceptionResNetV2, weights="imagenet")


def try_load_weights(model, path):
    if tf.io.gfile.exists(path):
        model.load_weights(path)
        print(f"Loaded weights: {path}")
        return True
    print(f"Warning: weights not found, using ImageNet init: {path}")
    return False


_ = try_load_weights(model1, WEIGHT_1)
_ = try_load_weights(model2, WEIGHT_2)
_ = try_load_weights(model3, WEIGHT_3)



## === cell 9
best_alpha = 0.34
bad_alpha = 0.33

print("Computing predictions...")
probabilities1 = model1.predict(test_dataset_eff, verbose=1)
probabilities2 = model2.predict(test_dataset_den, verbose=1)
probabilities3 = model3.predict(test_dataset_irv, verbose=1)

probabilities = (
    best_alpha * probabilities1
    + (1.0 - best_alpha - bad_alpha) * probabilities2
    + bad_alpha * probabilities3
)

probabilities = np.asarray(probabilities, dtype=np.float32)
if probabilities.shape[1] != len(TARGET_COLS):
    raise ValueError(
        f"Pred shape {probabilities.shape} does not match target cols {len(TARGET_COLS)}"
    )

probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)

sub = sub.copy()
sub_ids = sub["image_id"].values
test_ids = test["image_id"].values
if not np.array_equal(sub_ids, test_ids):
    id_to_idx = {img_id: i for i, img_id in enumerate(test_ids)}
    order = [id_to_idx[i] for i in sub_ids]
    probabilities = probabilities[order]

sub.loc[:, TARGET_COLS] = probabilities
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
