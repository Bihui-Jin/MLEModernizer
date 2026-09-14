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

0.6500453309156845

# 6. Current score

0.78326

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.8083) has done: 'I fix the environment crash caused by a protobuf incompatibility by pinning protobuf to a TensorFlow-compatible version at runtime before importing TensorFlow. Then I remove the dependency on a missing external `.h5` model file by training the same kind of image classifier directly from the provided TFRecords (keeping the same general TFRecord parsing + Keras training/inference flow). Finally, I make submission generation robust by collecting predictions for every test image, ordering them to match `sample_submission.csv`, and writing a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.8012) has done: 'I fix the crash before TensorFlow imports by removing the protobuf runtime pinning (it’s now causing incompatibility with TF 2.18) and instead rely on the Kaggle environment’s preinstalled protobuf. Then I correct a data loading bug where a Python list of TFRecord filenames was passed into `tf.io.gfile.glob`, which can silently break dataset creation; this should restore proper training/inference behavior without changing the model architecture or training loop. Finally, I keep submission generation the same but make ordering robust by defaulting missing IDs (shouldn’t happen) to a safe class so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.78326) has done: 'I remove the protobuf runtime pinning/restart logic because TensorFlow 2.18 in your environment already works with protobuf 6.x, and the forced downgrade/re-exec is the most likely reason you get “Not yielded” (kernel restart/timeout before writing the CSV). Then I make one minimal training change that doesn’t alter your architecture or pipeline: unfreeze the last ResNet50 stage (conv5) so the model can adapt slightly to cassava images within the same 2-epoch budget, which should improve accuracy toward your target. Finally, I keep your submission logic but add a small safeguard to ensure every `image_id` is unique and ordered exactly like `sample_submission.csv`, so the generated `submission.csv` is always valid.'
- What this solution (achieved 0.78326) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf C++ implementation by forcing the safe-Python protobuf runtime before importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error). I keep your model architecture, TFRecord parsing, training loop, and submission logic the same, only adding this compatibility shim plus a small safety fallback if TensorFlow still fails to import. Since your current score (0.78326) is already above the target (0.6500) and within the ±10% tolerance band, I not make any score-improving changes—just ensure the notebook runs end-to-end and reliably writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception as e:
    for m in list(sys.modules.keys()):
        if m.startswith("tensorflow") or m.startswith("google.protobuf"):
            sys.modules.pop(m, None)
    import tensorflow as tf  # noqa: F401

print("Python:", sys.version.split()[0])
print("TensorFlow:", tf.__version__)
try:
    import google.protobuf

    print("protobuf:", google.protobuf.__version__)
except Exception as e:
    print("protobuf: <unavailable>", e)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
ROOT_DIR = "/kaggle/input/"
DATA_DIR = ROOT_DIR + "cassava-leaf-disease-classification/"

AUTOTUNE = tf.data.AUTOTUNE

HEIGHT, WIDTH = 240, 240
BATCH_SIZE = 32
NUM_CLASSES = 5

TRAIN_TFRECS = DATA_DIR + "train_tfrecords/ld_train*.tfrec"
TEST_TFRECS = DATA_DIR + "test_tfrecords/ld_test*.tfrec"

train_csv_path = DATA_DIR + "train.csv"
sample_sub_path = DATA_DIR + "sample_submission.csv"

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print("Train rows:", len(train_df), " Sample submission rows:", len(sample_sub))




## === cell 2
def _parse_function(example, feature_description):
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(parsed_example["image"], channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (HEIGHT, WIDTH))
    image = tf.keras.applications.resnet50.preprocess_input(image)

    if "target" in feature_description:
        target = tf.cast(parsed_example["target"], tf.int32)
        return image, target

    return image, parsed_example["image_name"]


def load_data(path_or_files, batch_size=32, train=True, shuffle=False):
    if isinstance(path_or_files, (list, tuple)):
        filenames = list(path_or_files)
    else:
        filenames = tf.io.gfile.glob(path_or_files)

    if not filenames:
        raise RuntimeError(f"No TFRecord files found for: {path_or_files}")

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)

    if train:
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }

    dataset = dataset.map(
        lambda x: _parse_function(x, feature_description), num_parallel_calls=AUTOTUNE
    )
    if shuffle:
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True, seed=SEED)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset




## === cell 3
def build_model():
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(HEIGHT, WIDTH, 3),
        pooling="avg",
    )

    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers:
        if layer.name.startswith("conv5_"):
            layer.trainable = True

    x = tf.keras.layers.Dropout(0.3)(base.output)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)
    return model


train_files = sorted(tf.io.gfile.glob(TRAIN_TFRECS))
if len(train_files) < 2:
    raise RuntimeError(
        "Expected multiple train TFRecord shards but found: %d" % len(train_files)
    )

val_files = train_files[-2:]  # small val on last shards
tr_files = train_files[:-2]

train_ds = load_data(tr_files, batch_size=BATCH_SIZE, train=True, shuffle=True)
val_ds = load_data(val_files, batch_size=BATCH_SIZE, train=True, shuffle=False)

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)



## === cell 4
EPOCHS = 2

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 5
test_set = load_data(TEST_TFRECS, batch_size=32, train=False, shuffle=False)

test_IDs = []
pred_labels = []

for images, names in test_set:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1).astype(int)

    names = [n.decode("utf-8") for n in names.numpy().tolist()]
    test_IDs.extend(names)
    pred_labels.extend(preds.tolist())

pred_map = {}
for k, v in zip(test_IDs, pred_labels):
    pred_map[k] = int(v)

ordered_ids = sample_sub["image_id"].tolist()
ordered_preds = [int(pred_map.get(iid, 0)) for iid in ordered_ids]

submission_df = pd.DataFrame({"image_id": ordered_ids, "label": ordered_preds})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with rows:", len(submission_df))
print(
    "Missing IDs filled with class 0:", sum(iid not in pred_map for iid in ordered_ids)
)
print("Unique predicted labels:", submission_df["label"].value_counts().to_dict())
