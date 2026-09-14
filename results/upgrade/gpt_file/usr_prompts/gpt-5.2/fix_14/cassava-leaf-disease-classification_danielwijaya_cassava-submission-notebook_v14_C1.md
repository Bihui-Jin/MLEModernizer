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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8927168328800241

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10987) has done: 'The runtime fails early due to incompatible/unused imports (`kaggle_datasets`, custom EfficientNet/keras_applications) and missing external pretrained model files, which prevents `model_15` etc. from being defined. To make the notebook run end-to-end and still follow the same “TFRecord → EfficientNet → predict → submission” semantics, I replace the unavailable pretrained-model loading with a single built-in `tf.keras.applications.EfficientNetB0` model and keep the same inference/prediction flow. I also fix TFRecord decoding to avoid forcing a wrong fixed shape (which can break) by resizing after decode, and I correct `dataset.with_options(...)` (must be assigned) for determinism. Finally, I ensure the generated `submission.csv` matches `sample_submission.csv` ordering and has correct `image_id,label` columns.'
- What this solution (achieved 0.05531) has done: 'We need to fix the TensorFlow import crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`), which is a protobuf/TensorFlow incompatibility that prevents the entire pipeline from running. The safest minimal fix in this Kaggle environment is to force the Python protobuf implementation before importing TensorFlow and restart the TF import cleanly, keeping the rest of your TFRecord → EfficientNet → predict → submission flow unchanged. I also add a tiny defensive check to ensure TFRecord filenames are found and keep submission generation identical (same columns/order as `sample_submission.csv`). These changes are score-neutral by themselves, but they unblock execution so you can produce a valid submission and then iterate toward the target score.'
- What this solution (achieved 0.05531) has done: 'I fix the TensorFlow import crash by forcing a protobuf version compatible with TF 2.18 (protobuf<6) via a pip install at runtime before importing TensorFlow; this is the minimal, standard Kaggle-side workaround for the `MessageFactory.GetPrototype` error. Then I fix TFRecord parsing for the test set by accepting both possible ID keys (`"id"` and `"image_id"`) so we don’t crash when the TFRecords use a different feature name than expected. Finally, I ensure we always produce a valid `submission.csv` with exactly the `image_id,label` columns in the same order as `sample_submission.csv`, keeping the rest of your EfficientNetB0 inference flow unchanged.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, and the biggest likely cause (without changing your core “TFRecord → EfficientNetB0 → predict → submission” pipeline) is that you’re doing inference with an untrained ImageNet-headed classifier (random Dense head), plus you’re skipping the EfficientNet-specific `preprocess_input`, both of which collapse accuracy. I keep the same model architecture and inference flow, but (1) load weights for your exact architecture by briefly training only the top Dense layer on the provided TFRecords (base frozen), and (2) apply the correct EfficientNetB0 preprocessing while keeping image size and batching intact. This is a minimal change that should move accuracy sharply upward toward your target without altering the fundamental approach. The submission writing stays identical (same columns and sample_submission ordering) and still produces `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so we should improve accuracy while keeping your same TFRecord → EfficientNetB0 → train-head-only → predict → submission flow. The biggest low-risk gain is to (1) train the head for a few more epochs (still only the final Dense layer, base frozen) and (2) add a small validation split from TFRecords to ensure training is actually learning (without changing the model or loss). I also fix a subtle dataset issue: your `to_float32` mapping is incorrect for `(image, image_id)` test batches (it treats `image_id` as a label), which can silently corrupt the dataset pipeline and predictions. Finally, I keep submission creation identical and still align to `sample_submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.8927), so we need a real accuracy boost while keeping your same TFRecord → EfficientNetB0 → “train head only” → predict → submission pipeline. The biggest likely issue is that you’re freezing the EfficientNet base but still leaving BatchNorm layers in training mode during `fit`, which often destabilizes transfer learning and yields very poor accuracy; we keep the base frozen but force BN layers to run in inference mode. Next, your validation split is currently the first `take()` chunk of a shuffled stream, which can vary and be less representative; we instead split deterministically by shuffling once and using a fixed `take/skip` to make training more stable. Finally, we add label smoothing in the same loss (still SparseCategoricalCrossentropy) to improve generalization without changing the model or training approach.'
- What this solution (achieved 0.05531) has done: 'I fix the immediate crash by removing the unsupported `label_smoothing` argument (it’s not available for `SparseCategoricalCrossentropy` in this TF build) while keeping the same loss type and head-only training setup. To recover some of the intended regularization benefit without changing the model/training loop, I instead apply label smoothing via a minimal one-hot conversion inside the dataset pipeline and switch the loss to standard `CategoricalCrossentropy` (same softmax, same optimization objective). I also make the dataset parsing slightly safer by ensuring the label is always in `[0,4]` and by explicitly casting to the right dtypes. Submission generation stays identical and still write a valid `submission.csv` with `image_id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.8927), so we should improve true model accuracy without changing the core “EfficientNetB0 base + softmax head, TFRecords input, train head then predict” approach. The most likely issue is that the EfficientNet base is being kept in training mode during `fit`, which makes its BatchNorm layers update statistics even when `trainable=False`, often collapsing transfer learning performance; we force `training=False` for the base during the forward pass while still only training the final Dense head. To avoid subtle label/ID TFRecord mismatches hurting training, we also read labels more robustly by preferring `label` then falling back to `target` (while keeping the same semantics), and we keep preprocessing identical. These are minimal, execution-safe changes that typically move accuracy sharply upward toward your target while preserving architecture and the train-head-only loop.'
- What this solution (achieved 0.05531) has done: 'Your score gap to the target is large (0.05531 → 0.8927), and the most likely cause (without changing your core TFRecords→EfficientNetB0→train-head→predict pipeline) is that the training labels are being corrupted by the TFRecord parsing fallback: many TFRecords store `target` as the true label and leave `label` at its default, but your code currently prefers `label` whenever it’s non-negative (including 0), which silently flips most labels to class 0. I make the label selection robust by using `target` when it exists and only falling back to `label` otherwise, keeping the same preprocessing, model, loss, and training loop. I also ensure `raw_all` uses `.repeat()` so training doesn’t prematurely stop because the dataset is finite (this keeps your epoch count/semantics intact but ensures the optimizer actually sees enough steps). Submission writing stays identical and still aligns to `sample_submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so we should increase true accuracy without changing your core TFRecords→EfficientNetB0→train-head→predict→submission pipeline. The biggest likely blocker is that the EfficientNet base is never actually used during training/inference because the wrapper’s `call()` bypasses Keras’ graph and incorrectly feeds raw images directly into the base (without the base’s own preprocessing path), which can silently break learning; we keep the same architecture but rebuild it as a standard functional model where the base is called with `training=False` (BatchNorm frozen) and the head is trained as before. Next, we fix training-layer freezing so we freeze only the EfficientNet base and train the head layers (Dropout + Dense) as intended (your current loop freezes almost everything and is brittle). Finally, we keep submission generation identical but make ID extraction deterministic and safe by reading IDs directly from the ordered dataset once.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        pass



## === cell 1
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<6"])

import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
import re
import random

print("TensorFlow:", tf.__version__)
print("Protobuf:", __import__("google.protobuf").protobuf.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



## === cell 2
NUM_CLASSES = 5
IMAGE_SIZE = [512, 512]


def build_model(img_size=(512, 512), num_classes=5):
    """
    Change (score-improving, same core logic):
    - Keep the same EfficientNetB0(base)+Dropout+Dense-softmax architecture,
      but implement it as a standard functional model so the base is actually used
      correctly during both fit() and predict().
    - Force base forward pass with training=False to keep BatchNorm in inference mode
      while still allowing head training (transfer learning stability).
    """
    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3), name="image")

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=None, pooling="avg"
    )
    base.trainable = False  # base frozen, head trained (same approach as before)

    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="effnetb0_head")
    return model, base


model, base_model = build_model(tuple(IMAGE_SIZE), NUM_CLASSES)



## === cell 3
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"

test_df = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
print(test_df.head())

AUTOTUNE = tf.data.AUTOTUNE
GCS_PATH = DATA_ROOT
BATCH_SIZE = 128  # preserves original intent (16*8)


def dataset_sizes(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))


TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")
if len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(
        f"No TFRecords found under: {GCS_PATH}/test_tfrecords/ld_test*.tfrec"
    )

TRAIN_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/train_tfrecords/ld_train*.tfrec")
if len(TRAIN_FILENAMES) == 0:
    raise FileNotFoundError(
        f"No TFRecords found under: {GCS_PATH}/train_tfrecords/ld_train*.tfrec"
    )

NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)
NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
print("Found test TFRecords:", len(TEST_FILENAMES), "NUM_TEST_IMAGES:", NUM_TEST_IMAGES)
print(
    "Found train TFRecords:",
    len(TRAIN_FILENAMES),
    "NUM_TRAIN_IMAGES:",
    NUM_TRAIN_IMAGES,
)




## === cell 4
def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def read_tfrecord(example, labeled):
    if labeled:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        }
    else:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
        }

    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    img = decode_img(example["image"])

    if labeled:
        target = tf.cast(example["target"], tf.int32)
        label = tf.cast(example["label"], tf.int32)
        y = tf.where(target >= 0, target, label)
        y = tf.clip_by_value(y, 0, NUM_CLASSES - 1)
        return img, y
    else:
        image_id = tf.where(
            tf.strings.length(example["id"]) > 0, example["id"], example["image_id"]
        )
        return img, image_id


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    if not ordered:
        options.experimental_deterministic = False  # speed
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return ds


def get_test_data(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_train_data(ordered=False):
    ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=ordered)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 5
print("Training a minimal top-classifier head on TFRecords (base frozen)...")

base_model.trainable = False
for layer in model.layers:
    if layer.name in ("dropout", "pred"):
        layer.trainable = True

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

LABEL_SMOOTHING = 0.05


def to_smoothed_onehot(image, label):
    label = tf.cast(label, tf.int32)
    onehot = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    if LABEL_SMOOTHING and LABEL_SMOOTHING > 0.0:
        onehot = onehot * (1.0 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / NUM_CLASSES)
    return image, onehot


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
)

EPOCHS = 6  # unchanged

raw_all = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=True)
raw_all = raw_all.shuffle(
    NUM_TRAIN_IMAGES, seed=SEED, reshuffle_each_iteration=False
).repeat()

val_take = max(1024, int(0.10 * NUM_TRAIN_IMAGES))
val_ds = raw_all.take(val_take).map(to_smoothed_onehot, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = raw_all.skip(val_take)
train_ds = train_ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(to_smoothed_onehot, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

steps_per_epoch = (NUM_TRAIN_IMAGES - val_take) // BATCH_SIZE
validation_steps = val_take // BATCH_SIZE

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)
print("Finished head training.")
print(
    "Final train acc:",
    float(history.history["acc"][-1]),
    "Final val acc:",
    float(history.history["val_acc"][-1]),
)




## === cell 6
def test_to_float32(image, image_id):
    return tf.cast(image, tf.float32), image_id


test_ds = get_test_data(ordered=True)
test_ds = test_ds.map(test_to_float32, num_parallel_calls=AUTOTUNE)

print("Computing predictions...")
test_images_ds = test_ds.map(lambda image, image_id: image, num_parallel_calls=AUTOTUNE)

probabilities = model.predict(test_images_ds, verbose=1)
predictions = np.argmax(probabilities, axis=-1).astype(int)

if len(predictions) != NUM_TEST_IMAGES:
    raise RuntimeError(
        f"Prediction count mismatch: got {len(predictions)} vs expected {NUM_TEST_IMAGES}. "
        "Check TFRecord parsing/batching."
    )

print("Predictions shape:", predictions.shape, "Unique labels:", np.unique(predictions))



## === cell 7
print("Generating submission.csv file...")

test_ids_ds = (
    get_test_data(ordered=True)
    .map(lambda image, image_id: image_id, num_parallel_calls=AUTOTUNE)
    .unbatch()
)
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy()

test_ids = np.array(
    [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray, np.bytes_)) else str(x)
        for x in test_ids
    ],
    dtype=object,
)

assert len(test_ids) == len(predictions), (len(test_ids), len(predictions))

sub = pd.DataFrame({"image_id": test_ids, "label": predictions})

sub = test_df[["image_id"]].merge(sub, on="image_id", how="left")
sub["label"] = sub["label"].fillna(0).astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())
