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

0.8794197642792384

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the incompatible `kaggle_datasets` import (it triggers a protobuf error in this environment) and eliminate the missing external EfficientNet/keras_applications dependency. Since the referenced pre-trained `.h5` file isn’t available, I keep the same overall TFRecords→CNN→argmax pipeline but instantiate a small Keras CNN and train it briefly on `train_tfrecords` so `model_19` exists and predictions can be generated. I also fix the TFRecords parsing/decoding bug (the TFRecords contain raw JPEG bytes with variable size, so forcing a fixed `[512,512]` reshape is incorrect) by resizing after decode. Finally, I ensure the submission is written as a valid `submission.csv` with exactly `image_id,label` aligned to the ordered test TFRecord stream.'
- What this solution (achieved 0.61099) has done: 'I fix the crash happening at `import tensorflow as tf`, which is due to an incompatibility between TensorFlow 2.18 and the installed `protobuf==6.x` (the `MessageFactory.GetPrototype` AttributeError). The minimal, Kaggle-safe fix is to pin protobuf to a TF-compatible 4.x version at runtime before importing TensorFlow. After that, I keep your TFRecords→CNN→argmax pipeline intact, but make a small, metric-aligned improvement by using a lightweight ImageNet-pretrained backbone (EfficientNetB0 from `tf.keras.applications`) while keeping the same training loop structure; this should move accuracy substantially toward the target. Finally, I keep the ordered test TFRecord reading and submission writing logic, ensuring the CSV is valid and aligned.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.8794), so we should improve accuracy with the smallest changes that keep your TFRecords→EfficientNetB0→softmax pipeline intact. The biggest likely issue is that you train only 2 epochs with the backbone fully frozen at a large 512×512 input, which tends to underfit; we keep the same model but (1) switch to the canonical EfficientNetB0 resolution (224×224) to let you train longer within time, (2) add a proper train/valid split with `model.fit(..., validation_data=...)` to stabilize training, and (3) do a minimal fine-tuning phase by unfreezing only the top of the backbone at a lower learning rate. These are standard, metric-aligned changes that preserve architecture and loss while materially improving accuracy toward your target. Submission writing/ordering stays the same.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.61099) is well below the target (0.8794), so we should improve accuracy with minimal, metric-aligned changes while keeping the same TFRecords → EfficientNetB0 → softmax training pipeline. The biggest gap is likely coming from limited data augmentation and training configuration rather than the model definition, so I add lightweight, standard augmentations (flip/rotate/zoom/contrast) applied only on the training dataset. I also add label smoothing in the same cross-entropy loss to improve generalization (still the same loss family and evaluation semantics) and make the TFRecord file order deterministic to avoid accidental distribution shifts. Submission generation stays identical and still writes a valid `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.61099) has done: 'I fix the runtime error by replacing the unsupported `label_smoothing` argument in `SparseCategoricalCrossentropy` with the equivalent `CategoricalCrossentropy(label_smoothing=...)` while keeping the same training pipeline and semantics. To do that safely, I one-hot encode labels inside the TFRecord parsing function so the model still trains on the same data and outputs the same 5-class softmax. I also make the test `predict` call use the same batched dataset (images only) to avoid any accidental cardinality surprises, but keep the ordered TFRecord reading and submission alignment intact. These changes are directly tied to the crash and should restore the intended regularization (label smoothing), improving accuracy toward your target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_ROOT = "/kaggle/input"
print("Listing input root:", DATA_ROOT)
try:
    print("Top-level entries:", sorted(os.listdir(DATA_ROOT))[:50])
except Exception as e:
    print("Could not list input root:", repr(e))



## === cell 1
import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            print(
                f"WARNING: protobuf version {pb_ver} detected (major>=5). "
                "If TF import fails, pin protobuf<5 in the environment. Skipping pip install for speed."
            )
    except Exception as e:
        print("WARNING: Could not check protobuf version due to:", repr(e))


_ensure_compatible_protobuf()



## === cell 2
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
from sklearn.model_selection import train_test_split
import re
import random
import math

SEED = 42

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TensorFlow:", tf.__version__)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_TFRECORDS_GLOB = os.path.join(DATA_DIR, "train_tfrecords", "ld_train*.tfrec")
TEST_TFRECORDS_GLOB = os.path.join(DATA_DIR, "test_tfrecords", "ld_test*.tfrec")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(SAMPLE_SUB)
print("train_df:", train_df.shape, "test_df:", test_df.shape)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64

IMAGE_SIZE = (224, 224)

NUM_CLASSES = 5


def dataset_sizes(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))


TRAIN_FILENAMES = sorted(tf.io.gfile.glob(TRAIN_TFRECORDS_GLOB))
TEST_FILENAMES = sorted(tf.io.gfile.glob(TEST_TFRECORDS_GLOB))

NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

print(
    "TFRecords:",
    len(TRAIN_FILENAMES),
    "train files;",
    len(TEST_FILENAMES),
    "test files",
)
print("NUM_TRAIN_IMAGES:", NUM_TRAIN_IMAGES, "NUM_TEST_IMAGES:", NUM_TEST_IMAGES)




## === cell 4
def decode_img(img_bytes):
    img = tf.io.decode_and_crop_jpeg(img_bytes, crop_window=[0, 0, 0, 0], channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)  # keep [0,255]
    return img


def read_tfrecord(example, labeled):
    if labeled:
        tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = decode_img(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        label = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
        return img, label
    else:
        image_id = example["image_name"]
        return img, image_id


_TFDATA_OPTS_FAST = tf.data.Options()
_TFDATA_OPTS_FAST.experimental_optimization.map_and_batch_fusion = True
_TFDATA_OPTS_FAST.experimental_optimization.parallel_batch = True
_TFDATA_OPTS_FAST.experimental_optimization.autotune_buffers = True
_TFDATA_OPTS_FAST.experimental_optimization.autotune_cpu_budget = True
_TFDATA_OPTS_FAST.experimental_optimization.autotune_ram_budget = True


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.with_options(_TFDATA_OPTS_FAST)

    if not ordered:
        try:
            ds = ds.with_options(tf.data.Options())
            ds.options().experimental_deterministic = False
        except Exception:
            pass

    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


def split_filenames(filenames, val_ratio=0.1, seed=SEED):
    filenames = list(filenames)
    filenames = sorted(filenames)  # deterministic
    tr, va = train_test_split(filenames, test_size=val_ratio, random_state=seed)
    return tr, va


TRAIN_FILES_SPLIT, VAL_FILES_SPLIT = split_filenames(
    TRAIN_FILENAMES, val_ratio=0.1, seed=SEED
)
NUM_TRAIN_IMAGES_SPLIT = dataset_sizes(TRAIN_FILES_SPLIT)
NUM_VAL_IMAGES_SPLIT = dataset_sizes(VAL_FILES_SPLIT)

print(
    "Split TFRecords -> train files:",
    len(TRAIN_FILES_SPLIT),
    "val files:",
    len(VAL_FILES_SPLIT),
)
print(
    "Split sizes -> train images:",
    NUM_TRAIN_IMAGES_SPLIT,
    "val images:",
    NUM_VAL_IMAGES_SPLIT,
)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.05, seed=SEED),
        tf.keras.layers.RandomZoom(0.10, seed=SEED),
        tf.keras.layers.RandomContrast(0.10, seed=SEED),
    ],
    name="data_augmentation",
)


def _augment_train(image, label):
    image = data_augmentation(image, training=True)
    return image, label


def get_train_data():
    ds = load_dataset(TRAIN_FILES_SPLIT, labeled=True, ordered=False)

    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_augment_train, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_val_data():
    ds = load_dataset(VAL_FILES_SPLIT, labeled=True, ordered=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_data(ordered=False, cache=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    if cache:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input)(inputs)
backbone = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
)
backbone.trainable = False
x = backbone(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model_19 = tf.keras.Model(inputs, outputs)

model_19.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)

EPOCHS_STAGE1 = 6
steps_per_epoch = math.ceil(NUM_TRAIN_IMAGES_SPLIT / BATCH_SIZE)
val_steps = math.ceil(NUM_VAL_IMAGES_SPLIT / BATCH_SIZE)

train_ds = get_train_data()
val_ds = get_val_data()

history = model_19.fit(
    train_ds,
    epochs=EPOCHS_STAGE1,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    verbose=1,
)

backbone.trainable = True
for layer in backbone.layers[:-30]:
    layer.trainable = False

model_19.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)

EPOCHS_STAGE2 = 2
history_ft = model_19.fit(
    train_ds,
    epochs=EPOCHS_STAGE2,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2534070030.py in <cell line: 0>()
     35 _TFDATA_OPTS_FAST.experimental_optimization.map_and_batch_fusion = True
     36 _TFDATA_OPTS_FAST.experimental_optimization.parallel_batch = True
---> 37 _TFDATA_OPTS_FAST.experimental_optimization.autotune_buffers = True
     38 _TFDATA_OPTS_FAST.experimental_optimization.autotune_cpu_budget = True
     39 _TFDATA_OPTS_FAST.experimental_optimization.autotune_ram_budget = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 5
test_df = pd.read_csv(SAMPLE_SUB)
print(test_df.head())

GCS_PATH = DATA_DIR
CLASSES = ["0", "1", "2", "3", "4"]




## === cell 6
def to_float32(image, label_or_id):
    return tf.cast(image, tf.float32), label_or_id


test_ds = get_test_data(ordered=True, cache=False).map(
    to_float32, num_parallel_calls=AUTOTUNE
)

print("Computing predictions + collecting ids (single pass)...")

img_in = tf.keras.Input(shape=(*IMAGE_SIZE, 3), dtype=tf.float32, name="img")
id_in = tf.keras.Input(shape=(), dtype=tf.string, name="image_id")
probs_out = model_19(img_in, training=False)
pred_model = tf.keras.Model([img_in, id_in], [probs_out, id_in])

probabilities, test_ids = pred_model.predict(test_ds, verbose=1)
predictions = np.argmax(probabilities, axis=-1).astype(np.int32)

test_ids = np.asarray(test_ids)
test_ids = np.array(
    [
        s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
        for s in test_ids
    ],
    dtype=object,
)

if len(test_ids) != len(predictions):
    raise ValueError(
        f"Length mismatch: test_ids={len(test_ids)} vs predictions={len(predictions)}"
    )

print("predictions shape:", predictions.shape, "unique:", np.unique(predictions))

print("Generating submission.csv file...")
sub = pd.DataFrame({"image_id": test_ids, "label": predictions})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1721917259.py in <cell line: 0>()
      5 # --- Timeout fix: avoid a second full pass over the test dataset to collect ids.
      6 # We predict on (image, id) in a single pass and collect both outputs from the same iterator.
----> 7 test_ds = get_test_data(ordered=True, cache=False).map(
      8     to_float32, num_parallel_calls=AUTOTUNE
      9 )

NameError: name 'get_test_data' is not defined
