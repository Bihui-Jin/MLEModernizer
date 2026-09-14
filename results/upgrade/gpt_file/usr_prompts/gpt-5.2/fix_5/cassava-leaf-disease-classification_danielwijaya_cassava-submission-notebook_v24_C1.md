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

0.8812330009066183

# 6. Current score

0.45889

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.08296) has done: 'I remove the failing legacy `kaggle_datasets/efficientnet.keras/keras_applications` imports that crash under the current TensorFlow/Protobuf versions, and replace the missing external pretrained `.h5` load with an in-notebook EfficientNet model trained on the provided `train_tfrecords`. I keep the same overall TFRecord input pipeline style (decode → batch → prefetch) and produce predictions with `argmax` to match the accuracy metric. Finally, I ensure `submission.csv` is written with exactly `image_id,label` and that prediction order matches the ordered test dataset IDs.'
- What this solution (achieved 0.49514) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation before TensorFlow is imported (this addresses the `MessageFactory.GetPrototype` error). Then I fix the OOM by enabling GPU memory growth (if a GPU exists) and by reducing the batch size (this is the smallest change that preserves the same model/training loop semantics but makes it runnable). Finally, I correct the train/val split logic so validation is taken from a deterministic ordered dataset rather than from a shuffled stream, which should substantially improve accuracy toward your target without changing the core architecture or loss/metric, and I keep the submission writing logic intact.'
- What this solution (achieved 0.45553) has done: 'We fix the protobuf crash by setting both the implementation and version environment variables *before* any TensorFlow/protobuf-related imports, which is the known workaround for the `MessageFactory.GetPrototype` issue in this environment. We also make TFRecord file discovery deterministic by sorting the globbed filenames, improving the stability and correctness of the train/val split and test prediction ordering without changing the model/training core logic. Finally, we make the submission `image_id` extraction robust by decoding bytes to strings safely, ensuring the CSV is always valid and aligned.'
- What this solution (achieved 0.45889) has done: 'We fix the protobuf/TensorFlow incompatibility that’s currently crashing the notebook by ensuring the pure-Python protobuf implementation is used *before any protobuf module is imported*, and by force-reloading `google.protobuf` if it was already loaded. This is a runtime-stability fix only and won’t change your model/training logic. Then we keep the existing TFRecord pipeline and EfficientNetB4 training exactly as-is, but make sure test IDs/predictions are extracted deterministically and safely decoded so a valid `submission.csv` is always produced. No architecture, loss, or training-loop changes are made.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import re
import random
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth:", gpus)
    except Exception as e:
        print("Could not set memory growth:", repr(e))
else:
    print("No GPU found, using CPU.")

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
GCS_PATH = INPUT_DIR  # kept variable name from original notebook pattern

AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 16

IMAGE_SIZE = [512, 512]
NUM_CLASSES = 5

TRAIN_FILENAMES = sorted(
    tf.io.gfile.glob(os.path.join(GCS_PATH, "train_tfrecords/ld_train*.tfrec"))
)
TEST_FILENAMES = sorted(
    tf.io.gfile.glob(os.path.join(GCS_PATH, "test_tfrecords/ld_test*.tfrec"))
)


def dataset_sizes(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))


NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

print("Train tfrecords:", len(TRAIN_FILENAMES), "images:", NUM_TRAIN_IMAGES)
print("Test tfrecords:", len(TEST_FILENAMES), "images:", NUM_TEST_IMAGES)

test_df = pd.read_csv(SAMPLE_SUB)
print("Sample submission head:\n", test_df.head())
print("Sample submission rows:", len(test_df))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def to_float32(image, label):
    return tf.cast(image, tf.float32), label


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [*IMAGE_SIZE, 3])
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
        image_id = example["image_name"]
        return img, image_id


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(opts)
    ds = ds.map(
        lambda x: read_tfrecord(x, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return ds


VAL_FRAC = 0.1
NUM_VAL = int(NUM_TRAIN_IMAGES * VAL_FRAC)
NUM_TRN = NUM_TRAIN_IMAGES - NUM_VAL


def get_train_val_data():
    ds_all = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=True)

    val_ds = ds_all.take(NUM_VAL)
    trn_ds = ds_all.skip(NUM_VAL)

    trn_ds = trn_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    trn_ds = trn_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return trn_ds, val_ds


def get_test_data(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


trn_ds, val_ds = get_train_val_data()

train_steps = int(np.ceil(NUM_TRN / BATCH_SIZE))
val_steps = int(np.ceil(NUM_VAL / BATCH_SIZE))

print("NUM_TRN:", NUM_TRN, "NUM_VAL:", NUM_VAL)
print("train_steps:", train_steps, "val_steps:", val_steps)



## === cell 2
inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs,
    pooling="avg",
)
x = tf.keras.layers.Dropout(0.2)(base.output)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model_25 = tf.keras.Model(inputs, outputs)

model_25.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

base.trainable = False
EPOCHS_WARMUP = 2
history1 = model_25.fit(
    trn_ds,
    validation_data=val_ds,
    epochs=EPOCHS_WARMUP,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=2,
)

base.trainable = True
model_25.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
EPOCHS_FINETUNE = 2
history2 = model_25.fit(
    trn_ds,
    validation_data=val_ds,
    epochs=EPOCHS_WARMUP + EPOCHS_FINETUNE,
    initial_epoch=EPOCHS_WARMUP,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=2,
)



## === cell 3
test_ds = get_test_data(ordered=True)
test_ds = test_ds.map(to_float32, num_parallel_calls=AUTOTUNE)

print("Computing predictions...")
test_images_ds = test_ds.map(lambda image, idnum: image, num_parallel_calls=AUTOTUNE)
prob8 = model_25.predict(test_images_ds, verbose=1)
probabilities = prob8
predictions = np.argmax(probabilities, axis=-1).astype(int)
print("Predictions shape:", predictions.shape, "unique:", np.unique(predictions))



## === cell 4
print("Generating submission.csv file...")

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids_raw = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy()

if len(test_ids_raw) == 0:
    raise RuntimeError("No test IDs were read from TFRecords; cannot write submission.")

if isinstance(test_ids_raw[0], (bytes, np.bytes_)):
    test_ids = np.array([x.decode("utf-8") for x in test_ids_raw], dtype="U")
else:
    test_ids = test_ids_raw.astype("U")

if len(test_ids) != len(predictions):
    raise ValueError(
        f"ID/pred length mismatch: ids={len(test_ids)} preds={len(predictions)}"
    )

sub = pd.DataFrame({"image_id": test_ids, "label": predictions})
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
