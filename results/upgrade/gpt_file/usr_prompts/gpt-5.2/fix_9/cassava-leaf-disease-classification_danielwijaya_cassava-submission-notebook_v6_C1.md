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

0.8800241764883651

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.179) has done: 'I remove the failing dependencies on `kaggle_datasets`, `keras_applications`, and the missing external pretrained `.h5` file, since they prevent the notebook from running in this environment. I keep your TFRecord input pipeline and prediction-to-submission logic intact, but replace the missing model load with a minimal EfficientNetB5-based classifier built in-place (same general architecture family) so predictions can be produced end-to-end. I also fix a small tf.data bug (`dataset.with_options` wasn’t assigned) and make ID extraction deterministic/aligned with predictions. Finally, I always write a valid `submission.csv` with columns `image_id,label` and exactly the sample_submission order.'
- What this solution (achieved 0.76756) has done: 'We fix the immediate runtime failure caused by an incompatibility between TensorFlow 2.18 and protobuf 6.x (the `MessageFactory.GetPrototype` error) by pinning protobuf to the TensorFlow-compatible 4.x range at runtime before importing TensorFlow. Then we keep your existing TFRecord input pipeline and submission-writing logic intact, but add a minimal (and score-improving) training step on `train_tfrecords` using the same EfficientNetB5 model (frozen backbone, same head/loss) so predictions aren’t essentially random. Finally, we ensure deterministic ordering for IDs/predictions and always emit a valid `submission.csv` with the exact `image_id,label` columns in sample_submission order.'
- What this solution (achieved 0.79148) has done: 'The timeout is dominated by (1) `EfficientNetB5` training at 512×512 for 5 epochs and (2) expensive TFRecord decoding/resizing with suboptimal tf.data ordering. To keep the exact same model/loss/training loop semantics, the main speedups are: enable XLA JIT for the model, make the input pipeline more efficient (cache-before-batch/prefetch, avoid extra passes, add `ignore_errors`, and set non-determinism only where allowed), and eliminate redundant dataset materialization in prediction/id collection. These changes don’t alter the algorithm (same data, same labels, same augmentation = none, same epochs/steps, same architecture), but reduce overhead and improve throughput so the run fits under 600 seconds on typical Kaggle hardware.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd


def _ensure_protobuf_compatible():
    """
    Bugfix: TF 2.18 can be incompatible with protobuf 5/6 in some Kaggle images.
    We pin protobuf to <5 before importing TF if needed.
    """
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version: {pb_ver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf>=3.20.3,<5"]
        )
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


_ensure_protobuf_compatible()

import re
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 1
from tensorflow.keras import layers



## === cell 2
IMAGE_SIZE = [512, 512]
NUM_CLASSES = 5


def build_model(image_size=(512, 512, 3), num_classes=5):
    inputs = tf.keras.Input(shape=image_size)
    inputs_f32 = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs_f32 * 255.0)
    backbone = tf.keras.applications.EfficientNetB5(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    backbone.trainable = False
    x = backbone.output
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


model_18 = build_model(image_size=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)

model_18.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3729635439.py in <cell line: 0>()
     23 
     24 
---> 25 model_18 = build_model(image_size=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
     26 
     27 model_18.compile(

/tmp/ipykernel_55/3729635439.py in build_model(image_size, num_classes)
      7     # Change (score): force float32 before preprocess_input to avoid dtype quirks and
      8     # keep numerics consistent; does not change model semantics beyond negligible FP diffs.
----> 9     inputs_f32 = tf.cast(inputs, tf.float32)
     10     x = tf.keras.applications.efficientnet.preprocess_input(inputs_f32 * 255.0)
     11     backbone = tf.keras.applications.EfficientNetB5(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 3
GCS_PATH = "/kaggle/input/cassava-leaf-disease-classification"
test_df = pd.read_csv(f"{GCS_PATH}/sample_submission.csv")
print(test_df.head())

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

CLASSES = ["1", "2", "3", "4", "5"]  # kept from original code


def dataset_sizes(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")
TEST_FILENAMES = sorted(TEST_FILENAMES)  # deterministic file order
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

print("Found TFRecords:", len(TEST_FILENAMES), "NUM_TEST_IMAGES:", NUM_TEST_IMAGES)




## === cell 4
def to_float32(image, label):
    return tf.cast(image, tf.float32), label


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, tf.float32)  # == cast/255.0
    img = tf.image.resize(
        img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return img


def read_tfrecord(example, labeled):
    if labeled:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    img = decode_img(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return img, label
    else:
        idNum = example["image_name"]
        return img, idNum


def load_dataset(filenames, labeled=True, ordered=False):
    """
    Speed: keep the same decoding/feature extraction logic, but:
      - set deterministic=False only when 'ordered' is False (training)
      - add ignore_errors() to skip rare corrupt records without stalling
      - keep parallel reads and parallel map
    This preserves semantics for valid records and improves throughput.
    """
    options = tf.data.Options()
    if not ordered:
        options.experimental_deterministic = False  # speed for training only

    try:
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.ignore_errors()
    return dataset


def get_test_data(ordered=False):
    dataset = load_dataset(filenames=TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


TRAIN_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/train_tfrecords/ld_train*.tfrec")
TRAIN_FILENAMES = sorted(TRAIN_FILENAMES)
NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
print(
    "Found TRAIN TFRecords:",
    len(TRAIN_FILENAMES),
    "NUM_TRAIN_IMAGES:",
    NUM_TRAIN_IMAGES,
)


def get_train_data(filenames, ordered=False, shuffle_buffer=2048, cache_path=None):
    ds = load_dataset(filenames=filenames, labeled=True, ordered=ordered)
    ds = ds.map(to_float32, num_parallel_calls=AUTOTUNE)

    if cache_path is not None:
        ds = ds.cache(cache_path)

    ds = ds.shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_valid_data(filenames, ordered=True, cache_path=None):
    ds = load_dataset(filenames=filenames, labeled=True, ordered=ordered)
    ds = ds.map(to_float32, num_parallel_calls=AUTOTUNE)

    if cache_path is not None:
        ds = ds.cache(cache_path)
    else:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


val_count = max(1, int(0.1 * len(TRAIN_FILENAMES)))
train_files = TRAIN_FILENAMES[:-val_count]
val_files = TRAIN_FILENAMES[-val_count:]

train_cache = "/kaggle/working/train_cache.tfdata"
val_cache = "/kaggle/working/val_cache.tfdata"

for p in (train_cache, val_cache):
    try:
        if tf.io.gfile.exists(p):
            tf.io.gfile.remove(p)
    except Exception:
        pass

train_ds = get_train_data(train_files, ordered=False, cache_path=train_cache)
val_ds = get_valid_data(val_files, ordered=True, cache_path=val_cache)

EPOCHS = 5

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
    )
]

train_steps = int(np.ceil((dataset_sizes(train_files)) / BATCH_SIZE))
val_steps = int(np.ceil((dataset_sizes(val_files)) / BATCH_SIZE))

print("Training (frozen backbone)...")
model_18.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)

backbone = None
for layer in model_18.layers:
    if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
        backbone = layer
        break
if backbone is None:
    for layer in model_18.layers:
        if "efficientnetb5" in layer.name.lower():
            backbone = layer
            break

if backbone is not None:
    backbone.trainable = True
    for l in backbone.layers[:-40]:
        l.trainable = False

model_18.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

FINE_TUNE_EPOCHS = (
    1  # minimal extra training to move score upward without large runtime increase
)

print("Training (fine-tuning tail of backbone)...")
model_18.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    verbose=1,
    callbacks=callbacks,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2252427709.py in <cell line: 0>()
    135 
    136 print("Training (frozen backbone)...")
--> 137 model_18.fit(
    138     train_ds,
    139     validation_data=val_ds,

NameError: name 'model_18' is not defined

## === cell 5
test_ds = get_test_data(ordered=True).map(to_float32)

print("Computing predictions...")

test_images_ds = test_ds.map(
    lambda image, idnum: image, num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)
probabilities = model_18.predict(test_images_ds, verbose=1)
predictions = np.argmax(probabilities, axis=-1).astype(int)
print("Predictions shape:", predictions.shape, "Unique labels:", np.unique(predictions))

print("Generating submission.csv file...")

test_ids = np.concatenate(
    [id_batch.numpy().astype("U") for _, id_batch in test_ds], axis=0
)

if len(test_ids) != len(predictions):
    print(
        "WARNING: test_ids/predictions length mismatch. Using sample_submission image_id order."
    )
    test_ids = test_df["image_id"].values.astype("U")

sub = pd.DataFrame({"image_id": test_ids, "label": predictions[: len(test_ids)]})
sub = test_df[["image_id"]].merge(sub, on="image_id", how="left")
sub["label"] = sub["label"].fillna(0).astype(int)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(sub.head())
print("Wrote:", out_path, "rows:", len(sub))

with open(out_path, "r") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1336106532.py in <cell line: 0>()
      6     lambda image, idnum: image, num_parallel_calls=AUTOTUNE
      7 ).prefetch(AUTOTUNE)
----> 8 probabilities = model_18.predict(test_images_ds, verbose=1)
      9 predictions = np.argmax(probabilities, axis=-1).astype(int)
     10 print("Predictions shape:", predictions.shape, "Unique labels:", np.unique(predictions))

NameError: name 'model_18' is not defined
