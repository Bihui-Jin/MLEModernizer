# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
seaborn==0.12.2
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8700513750377757

# 6. Current score

0.77728

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75411) has done: 'I fix the environment-breaking import error by removing unused visualization imports that trigger the protobuf/TF incompatibility, while keeping the TF/Keras inference logic intact. Since the referenced pretrained model file doesn’t exist in your input folders, I replace that step with a minimal EfficientNetB3-based classifier trained on the provided `train.csv` images (same general EfficientNet family and Keras workflow), then use it to predict the test set. I also correct the path handling to use the actual dataset location under `/kaggle/input/...` and build the submission by following `sample_submission.csv` order to guarantee alignment. Finally, I ensure a valid `submission.csv` is written end-to-end.'
- What this solution (achieved 0.75673) has done: 'The immediate blocker is the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known incompatibility between TensorFlow and protobuf 6.x in this environment; fixing it requires pinning protobuf to a TF-compatible version (runtime-safe pip install) before importing TensorFlow. After that, I keep your EfficientNetB3 training/inference pipeline intact, but add the missing caching and basic augmentation to the `tf.data` pipeline (same resize/softmax/cross-entropy semantics) to legitimately improve accuracy toward the target without changing the model architecture or training loop structure. I also ensure the EfficientNet-specific preprocess import matches the model family (`efficientnet`) to avoid subtle preprocessing mismatches. Finally, the script always write a valid `submission.csv` with `image_id,label` in `sample_submission.csv` order.'
- What this solution (achieved 0.65396) has done: 'Your current gap to the target is about 0.1133 (0.75673 → 0.87005), so we should make small, legitimate accuracy-improving changes without changing the core model/training structure. I keep EfficientNetB3, the same two-stage “freeze then unfreeze” training approach, and the same loss/metric, but fix a key preprocessing mismatch by switching to the correct EfficientNetB3 preprocessing function. I also add label smoothing (still categorical cross-entropy semantics) and class weighting (to address class imbalance common in this dataset) which typically yields a modest, stable accuracy lift. Finally, I ensure dataset caching happens after preprocessing (and keep deterministic seeding) so training is stable and still finishes within the time budget.'
- What this solution (achieved 0.65396) has done: 'Your current score (0.65396) is far below the target (0.87005), so we should make small, legitimate fixes that plausibly raise accuracy without changing the overall EfficientNetB3 + (freeze → unfreeze) fine-tuning approach. The biggest issue in your code is a preprocessing bug: you compute `x` twice and effectively ignore the EfficientNetB3 preprocessing, which can heavily hurt accuracy; I fix it to use exactly the correct `keras.applications.efficientnet.preprocess_input` once. I also change `.cache()` placement so you cache decoded/resized images but still apply random augmentation fresh each epoch (right now you cache *after* augmentation, freezing the randomness and reducing effective augmentation). Finally, I keep everything else (model, loss, epochs, class_weight, submission alignment) the same so changes are minimal and focused on improving score.'
- What this solution (achieved 0.65396) has done: 'I fix the tf.data pipeline bug that currently feeds dummy/invalid images into training (your `train_ds` is created from a cached dataset but then you map a function that reads from an empty path, effectively breaking learning and crushing accuracy). I keep the same EfficientNetB3 model, the same freeze→unfreeze fine-tuning structure, the same loss/metrics, and the same augmentation operations, but apply augmentation correctly on the cached decoded images and then preprocess with the matching EfficientNetB3 preprocess function. I also ensure both train/val datasets return `(preprocessed_image, one_hot_label)` consistently so the model sees the intended inputs. These are minimal, correctness-focused changes that should legitimately increase accuracy toward your 0.870 target while preserving your overall approach and still writing a valid `submission.csv`.'
- What this solution (achieved 0.60987) has done: 'Your current score is far below the target, so the most likely “minimal but meaningful” lift is to fix training/inference consistency and a fine-tuning correctness issue without changing your EfficientNetB3 + (freeze→unfreeze) approach. I keep the same model and loss, but (1) ensure the base model runs in training mode during the unfreeze stage (BatchNorm updates matter a lot for EfficientNet fine-tuning), (2) use the exact same preprocessing path for train/val/test, and (3) add a simple, metric-consistent test-time augmentation (horizontal flip) averaged over two passes to boost accuracy without altering the core training loop. These are small changes that typically improve accuracy materially on Cassava while staying within your constraints and runtime. The submission writing stays identical (`submission.csv`, `image_id,label`, sample order).'
- What this solution (achieved 0.77055) has done: 'Your score gap to the target is large (0.60987 → 0.87005), so we should focus on small, correctness-preserving improvements that typically lift accuracy without changing the overall EfficientNetB3 + freeze→unfreeze approach. The biggest minimal win here is to avoid rebuilding a second model for fine-tuning (which can desync optimizer state and layer objects) and instead continue fine-tuning the same model after unfreezing, while also freezing BatchNorm layers to stabilize training. I also add a small, standard “center crop after resize” step to reduce background noise (common in Cassava) while keeping the same input size/architecture. Submission writing and sample-order alignment remain unchanged, and the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.75037) has done: 'Your current score (0.77055) is well below the target (0.87005), so the smallest likely uplift without changing your core EfficientNetB3 + freeze→unfreeze pipeline is to fix the fine-tuning BatchNorm behavior and slightly strengthen test-time augmentation. Concretely: during the unfreeze stage we should call the base model with `training=True` (while still keeping BN layers frozen as you already do) so non-BN layers follow training behavior and fine-tuning is consistent; this is a common, minimal correctness fix that can lift accuracy. Then, extend your existing TTA from 2 to 4 passes (original + hflip + vflip + hvflip), which usually gives a modest, stable boost for classification without changing training. Everything else (data paths, preprocessing, model, loss, epochs, submission alignment/format) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.77765) has done: 'To move your accuracy upward toward the 0.87005 target with minimal disruption, I’m keeping the same EfficientNetB3 + (freeze → unfreeze) training flow, loss, augmentations, and submission alignment. The main change is a correctness/performance fix: during fine-tuning you currently rebuild a new model, which discards the already-trained classifier head and optimizer state; instead we continue training the same model after unfreezing (this typically yields an immediate, legitimate accuracy lift). I also adjust the fine-tuning call to use `base(inputs, training=True)` while keeping BatchNorm layers frozen, matching common EfficientNet fine-tuning practice. Everything else (paths, preprocessing, TTA, CSV writing) remains the same and it still produces `submission.csv`.'
- What this solution (achieved 0.42115) has done: 'Your current score (0.77765) is below the target (0.87005), so we should make small, legitimate accuracy lifts without changing the EfficientNetB3 + freeze→unfreeze structure or the loss/metric semantics. The most impactful minimal fix here is to ensure EfficientNet sees inputs in the range its preprocessing expects by decoding images directly to float32 in [0,1] (currently you cast 0–255 to float32, which hurts the pretrained backbone after `preprocess_input`). I apply that same decode fix consistently to train/val/test, and I also stop applying EfficientNet preprocessing twice in the TTA branches by moving flips to happen before preprocessing (still the same TTA idea, just correct ordering). Everything else (architecture, epochs, class weighting, label smoothing, submission alignment/format) remains intact, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.42115) has done: 'Your current score is far below the target (gap ≈ -0.449), so the smallest changes most likely to improve accuracy are to fix the fine-tuning graph so it truly continues from the already-trained head (instead of re-wiring layers in a way that can subtly break training/inference behavior), and to ensure the fine-tuning stage uses the same base-model call path as the initial stage while keeping BatchNorm frozen. Concretely, we keep the same EfficientNetB3, the same freeze→unfreeze training approach, the same loss/metrics, augmentations, and TTA, but we stop rebuilding a new Keras `Model` for fine-tuning and instead continue training the existing `model` after unfreezing (this is a common correctness fix that typically yields a noticeable uplift). We also explicitly call the model with `training=False` during prediction to avoid any stochastic behavior from Dropout across TTA passes. Submission writing remains identical and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.77728) has done: 'Your current score (0.42115) is far below the target (0.87005), so the most likely “minimal-change” path is to fix two correctness issues that can severely depress accuracy without changing your model, loss, or training flow. First, you are feeding `keras.applications.efficientnet.preprocess_input` images in `[0,1]`, but that preprocessing expects raw `[0,255]` float inputs; I switch decoding to float32 in `[0,255]` consistently for train/val/test while keeping everything else the same. Second, your extra center-crop after resizing is effectively padding/cropping in a way that can distort content; I replace it with a true center-crop-from-slightly-larger-resize (still the same “center crop” idea, just correctly implemented). These two fixes usually restore pretrained EfficientNet behavior and should move accuracy substantially upward toward the target while preserving your architecture, training approach, and submission format.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )


_ensure_protobuf_compatible()

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

for p in [TRAIN_CSV, SAMPLE_SUB, TRAIN_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected path: {p}")

print("TensorFlow:", tf.__version__)
print("Train CSV:", TRAIN_CSV)
print("Sample submission:", SAMPLE_SUB)



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
ss = pd.read_csv(SAMPLE_SUB)

num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

IMG_SIZE = 300  # keep consistent with the original 300x300 resize intent
BATCH_SIZE = 32
EPOCHS = 3  # unchanged

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label"]
)


def make_paths_labels(df, img_dir):
    paths = df["image_id"].apply(lambda x: os.path.join(img_dir, x)).values
    labels = df["label"].values.astype("int32")
    return paths, labels


train_paths, train_labels = make_paths_labels(train_df, TRAIN_IMG_DIR)
val_paths, val_labels = make_paths_labels(val_df, TRAIN_IMG_DIR)

AUTOTUNE = tf.data.AUTOTUNE


def _center_crop_after_resize(img, crop_frac=0.92):
    scale = 1.0 / float(crop_frac)
    new_size = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * scale), tf.int32)
    new_size = tf.maximum(IMG_SIZE, new_size)
    img = tf.image.resize(img, [new_size, new_size], method="bilinear")
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)
    return img


def decode_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8 [0,255]
    img = tf.cast(img, tf.float32)  # float32 [0,255] for efficientnet.preprocess_input
    img = _center_crop_after_resize(img, crop_frac=0.92)
    y = tf.one_hot(label, depth=5)
    return img, y


def augment(img, y):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.10)
    img = tf.image.random_contrast(img, lower=0.90, upper=1.10)
    return img, y


def preprocess(img, y):
    img = keras.applications.efficientnet.preprocess_input(img)
    return img, y


class_counts = train_df["label"].value_counts().sort_index().values.astype("float32")
class_weight = {
    i: float(class_counts.sum() / (len(class_counts) * class_counts[i]))
    for i in range(5)
}
print("Class counts:", class_counts.tolist())
print("Class weights:", class_weight)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE).cache()
train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE).cache()
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

base = keras.applications.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)

loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss=loss_fn,
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    class_weight=class_weight,
)

base.trainable = True
for layer in base.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds, validation_data=val_ds, epochs=1, verbose=1, class_weight=class_weight
)



## === cell 2
test_paths = ss["image_id"].apply(lambda x: os.path.join(TEST_IMG_DIR, x)).values


def decode_resize_only_raw(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8 [0,255]
    img = tf.cast(img, tf.float32)  # float32 [0,255]
    img = _center_crop_after_resize(img, crop_frac=0.92)
    return img


def preprocess_only(img):
    return keras.applications.efficientnet.preprocess_input(img)


def hflip(img):
    return tf.image.flip_left_right(img)


def vflip(img):
    return tf.image.flip_up_down(img)


def hvflip(img):
    return tf.image.flip_up_down(tf.image.flip_left_right(img))


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(decode_resize_only_raw, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(preprocess_only, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
probs_1 = model.predict(test_ds, verbose=1)

test_ds_h = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds_h = test_ds_h.map(decode_resize_only_raw, num_parallel_calls=AUTOTUNE)
test_ds_h = test_ds_h.map(hflip, num_parallel_calls=AUTOTUNE)
test_ds_h = test_ds_h.map(preprocess_only, num_parallel_calls=AUTOTUNE)
test_ds_h = test_ds_h.batch(BATCH_SIZE).prefetch(AUTOTUNE)
probs_2 = model.predict(test_ds_h, verbose=1)

test_ds_v = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds_v = test_ds_v.map(decode_resize_only_raw, num_parallel_calls=AUTOTUNE)
test_ds_v = test_ds_v.map(vflip, num_parallel_calls=AUTOTUNE)
test_ds_v = test_ds_v.map(preprocess_only, num_parallel_calls=AUTOTUNE)
test_ds_v = test_ds_v.batch(BATCH_SIZE).prefetch(AUTOTUNE)
probs_3 = model.predict(test_ds_v, verbose=1)

test_ds_hv = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds_hv = test_ds_hv.map(decode_resize_only_raw, num_parallel_calls=AUTOTUNE)
test_ds_hv = test_ds_hv.map(hvflip, num_parallel_calls=AUTOTUNE)
test_ds_hv = test_ds_hv.map(preprocess_only, num_parallel_calls=AUTOTUNE)
test_ds_hv = test_ds_hv.batch(BATCH_SIZE).prefetch(AUTOTUNE)
probs_4 = model.predict(test_ds_hv, verbose=1)

probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0
preds = np.argmax(probs, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 3
my_submission
