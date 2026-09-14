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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.8563009972801451

# 6. Current score

0.43199

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.43199) has done: 'The timeout is dominated by end-to-end fine-tuning EfficientNetB4 at 256px with JPEG decode/resize done on-the-fly each epoch. To keep the same model/training loop semantics but cut redundant work, the main speedups are (1) switching to the provided TFRecords (much faster input pipeline than many small JPEG reads), (2) caching the *already decoded+resized+preprocessed* training/validation datasets (with shuffle after cache for training so randomness is preserved), and (3) enabling XLA JIT compilation (often a large training throughput gain without changing the model). These changes preserve the architecture, loss, optimizer, epochs, label handling, and evaluation semantics while removing avoidable I/O/CPU bottlenecks.'
- What this solution (achieved 0.43199) has done: 'I fix the TensorFlow import crash caused by forcing the pure-Python protobuf backend (incompatible with TF 2.18 + protobuf 6), by removing that environment override. Then I fix the TFRecord parsing error by reading `image_name` as a normal `tf.string` (no `decode_raw`/unicode ops), so the train/valid split filter works and the model trains successfully. Finally, I keep the same model/training setup but ensure the pipeline completes and writes a correctly formatted `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd



## === cell 1
import tensorflow as tf
from tensorflow import keras

SEED = 42
keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
TRAIN_TFREC_DIR = f"{DATA_DIR}/train_tfrecords"
TEST_TFREC_DIR = f"{DATA_DIR}/test_tfrecords"

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
ss = pd.read_csv(SAMPLE_SUB)

num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 3  # keep runtime within limits; enough to produce meaningful predictions

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)


def _tfrecord_files_in(dir_path, prefix):
    files = tf.io.gfile.glob(os.path.join(dir_path, f"{prefix}*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecords found in {dir_path} with prefix {prefix}"
        )
    return files


TRAIN_TFRECS = _tfrecord_files_in(TRAIN_TFREC_DIR, "ld_train")
TEST_TFRECS = _tfrecord_files_in(TEST_TFREC_DIR, "ld_test")

FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def decode_image_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet.preprocess_input(img)
    return img


AUTOTUNE = tf.data.AUTOTUNE

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True
try:
    data_opts.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass

tr_names = tf.constant(tr_df["image_id"].values.astype(str))
va_names = tf.constant(va_df["image_id"].values.astype(str))

tr_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tr_names, tf.ones_like(tr_names, dtype=tf.int32)
    ),
    default_value=0,
)
va_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        va_names, tf.ones_like(va_names, dtype=tf.int32)
    ),
    default_value=0,
)


def _with_name_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURES)
    name = ex["image_name"]
    img = decode_image_bytes(ex["image"])
    img = tf.image.random_flip_left_right(img, seed=SEED)
    label = tf.one_hot(tf.cast(ex["target"], tf.int32), depth=5)
    return name, img, label


def _with_name_valid(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURES)
    name = ex["image_name"]
    img = decode_image_bytes(ex["image"])
    label = tf.one_hot(tf.cast(ex["target"], tf.int32), depth=5)
    return name, img, label


train_ds = (
    tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=AUTOTUNE)
    .with_options(data_opts)
    .map(_with_name_train, num_parallel_calls=AUTOTUNE)
    .filter(lambda name, img, y: tf.equal(tr_table.lookup(name), 1))
    .map(lambda name, img, y: (img, y), num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=AUTOTUNE)
    .with_options(data_opts)
    .map(_with_name_valid, num_parallel_calls=AUTOTUNE)
    .filter(lambda name, img, y: tf.equal(va_table.lookup(name), 1))
    .map(lambda name, img, y: (img, y), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base = keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs,
    pooling="avg",
)
base.trainable = True

x = keras.layers.Dropout(0.3)(base.output)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 3
def parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURES)
    img = decode_image_bytes(ex["image"])
    return img


test_ds = (
    tf.data.TFRecordDataset(TEST_TFRECS, num_parallel_reads=AUTOTUNE)
    .with_options(data_opts)
    .map(parse_test_example, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

assert len(preds) == len(
    ss
), f"Pred length {len(preds)} != sample_submission length {len(ss)}"

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 4
try:
    from IPython.display import display

    display(my_submission.head())
except Exception:
    print(my_submission.head())

print("Unique labels:", np.unique(my_submission["label"].values))
assert my_submission.shape[0] == ss.shape[0]
assert list(my_submission.columns) == ["image_id", "label"]
assert my_submission["image_id"].iloc[0].endswith(".jpg")
assert my_submission["label"].between(0, 4).all()
