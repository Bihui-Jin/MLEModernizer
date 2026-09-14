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

0.9682332315790864

# 6. Current score

0.45039

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.45898) has done: 'I remove the KaggleDatasets GCS dependency (it fails with “Unauthenticated”) and instead build image paths from the already-mounted `/kaggle/input/plant-pathology-2020-fgvc7/images` folder, which unblocks all downstream variables (`GCS_DS_PATH`, `train_paths`, datasets, models). I also fix the protobuf-related crash by avoiding the external `efficientnet` pip install and using `tf.keras.applications` EfficientNet as a fallback while still loading the provided weights when possible. To keep core logic intact, the ensemble, image size, decoding/augmentation, model heads, and weighted averaging remain the same; only compatibility/robustness fixes are added. Finally, the script always write a valid `submission.csv` with the exact sample submission columns and row alignment to `test.csv`.'
- What this solution (achieved 0.45305) has done: 'I fix the crash happening before the first cell completes by removing the `sklearn` dependency (not guaranteed in your environment) and implementing a small deterministic train/valid split with NumPy instead. I also address the protobuf `MessageFactory.GetPrototype` import crash by avoiding the third‑party `efficientnet` package entirely and using `tf.keras.applications.EfficientNetB7` only, which is compatible in Kaggle’s TF runtime. Finally, I keep the exact model heads/ensemble logic intact, but make the label loss robust to one-hot targets by explicitly using `CategoricalCrossentropy` and ensure the submission columns/order exactly match `sample_submission.csv`.'
- What this solution (achieved 0.45305) has done: 'The crash happens immediately on import due to an incompatibility between TensorFlow and the `protobuf` runtime in this environment (the `MessageFactory.GetPrototype` attribute error). The most minimal, Kaggle-safe fix is to force Python protobuf implementation before importing TensorFlow, which avoids the C++ protobuf path that triggers this error. After unblocking execution, your code run end-to-end and write a correctly formatted `submission.csv`. I’m not changing any model/training logic here (you’re currently doing pure inference with pretrained weights), so this should be score-neutral aside from restoring the ability to generate a submission.'
- What this solution (achieved 0.45305) has done: 'I fix the TensorFlow import crash caused by the protobuf C++ runtime by forcing the pure-Python protobuf implementation and restarting the protobuf import state before importing TensorFlow. I also make the environment settings more robust by additionally disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is the most common Kaggle-safe workaround for `MessageFactory.GetPrototype` errors. No model/training/inference logic is changed, so this should be score-neutral except that it restores end-to-end execution and produces a valid `submission.csv`. The script still read from the same dataset paths and write `submission.csv` in the required column order.'
- What this solution (achieved 0.45039) has done: 'The timeout is dominated by building three very large backbones at 768×768 and then forcing all test images into one big tensor via a Python loop/`tf.concat`, which adds extra decode work and memory pressure before prediction. To preserve identical inference semantics while cutting runtime, the optimized version removes the eager materialization of `test_images_tensor` and instead streams the test `tf.data` pipeline directly into `multi_model.predict`, keeping the same batching and deterministic options. It also caches the decoded/resized test dataset (safe because it has no augmentation) so images are decoded exactly once across the ensemble forward pass. No model architecture, weights, alpha blending, or evaluation semantics are changed.'

# 9. Code solution

## === cell 0
import os, math, re, random, sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_JAX_DISABLED", "1")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications import EfficientNetB7

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as e:
    print("WARNING: could not enable op determinism:", e)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("WARNING: could not enable XLA JIT:", e)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("WARNING: could not set threading:", e)



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
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

PRED_BATCH_SIZE = max(32 * strategy.num_replicas_in_sync, BATCH_SIZE)

DS_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_PATH = os.path.join(DS_PATH, "images")

if not os.path.isdir(IMAGES_PATH):
    DS_PATH = "/kaggle/data/plant-pathology-2020-fgvc7"
    IMAGES_PATH = os.path.join(DS_PATH, "images")

assert os.path.isdir(IMAGES_PATH), f"Images folder not found at: {IMAGES_PATH}"




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_PATH, f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(DS_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DS_PATH, "test.csv"))
sub = pd.read_csv(os.path.join(DS_PATH, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]
assert TARGET_COLS == [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], f"Unexpected target columns: {TARGET_COLS}"

train_paths = train["image_id"].apply(format_path).values
test_paths = test["image_id"].apply(format_path).values

train_labels = train[TARGET_COLS].values.astype(np.float32)

n = len(train_paths)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

valid_frac = 0.15
n_valid = int(round(n * valid_frac))
valid_idx = idx[:n_valid]
train_idx = idx[n_valid:]

valid_paths = train_paths[valid_idx]
valid_labels = train_labels[valid_idx]
train_paths = train_paths[train_idx]
train_labels = train_labels[train_idx]

assert len(train_paths) == len(train_labels)
assert len(valid_paths) == len(valid_labels)
assert train_labels.shape[1] == len(TARGET_COLS)



## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # equivalent to cast/255.0
    image = tf.image.resize(image, image_size, antialias=False)
    if label is None:
        return image
    else:
        return image, label


def data_augment(image, label=None, seed=2020):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    if label is None:
        return image
    else:
        return image, label




## === cell 5
opts_det = tf.data.Options()
opts_det.experimental_deterministic = True
opts_det.experimental_optimization.apply_default_optimizations = True
opts_det.experimental_optimization.map_parallelization = True
opts_det.experimental_optimization.map_and_batch_fusion = True

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(opts_det)
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(opts_det)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_images = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(opts_det)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(PRED_BATCH_SIZE, drop_remainder=False)
    .cache()  # safe: no augmentation; ensures images are decoded/resized only once
    .prefetch(AUTO)
)




## === cell 6
def build_model(use_model, weights):
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
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(
                curve="ROC", multi_label=True, num_labels=train_labels.shape[1]
            )
        ],
    )
    return model


with strategy.scope():
    model1 = build_model(EfficientNetB7, weights="imagenet")

with strategy.scope():
    model2 = build_model(DenseNet201, weights="imagenet")

with strategy.scope():
    model3 = build_model(InceptionResNetV2, weights="imagenet")

w1 = "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
w2 = "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
w3 = "/kaggle/input/tf-zoo-models-on-tpu-inceptionresnetv2/my_model.h5"

for m, w in [(model1, w1), (model2, w2), (model3, w3)]:
    if os.path.exists(w):
        m.load_weights(w)
    else:
        print(f"WARNING: weights not found, using base weights only: {w}")



## === cell 7
best_alpha = 0.36
bad_alpha = 0.30

print(
    "Computing predictions (streaming cached test dataset; single multi-output forward pass)..."
)

with strategy.scope():
    ens_input = tf.keras.Input(shape=(img_size, img_size, 3), name="ens_input")
    out1 = model1(ens_input, training=False)
    out2 = model2(ens_input, training=False)
    out3 = model3(ens_input, training=False)
    multi_model = tf.keras.Model(inputs=ens_input, outputs=[out1, out2, out3])

probabilities1, probabilities2, probabilities3 = multi_model.predict(
    test_images, verbose=1
)

probabilities = (
    best_alpha * probabilities1
    + bad_alpha * probabilities2
    + (1.0 - best_alpha - bad_alpha) * probabilities3
)

probabilities = np.asarray(probabilities, dtype=np.float32)
probabilities = np.clip(probabilities, 0.0, 1.0)

assert probabilities.shape == (
    len(test),
    len(TARGET_COLS),
), f"Pred shape mismatch: {probabilities.shape} vs {(len(test), len(TARGET_COLS))}"

submission = sub.copy()
submission["image_id"] = test["image_id"].values  # ensure exact ordering
submission[TARGET_COLS] = probabilities

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)
print(submission.head())
