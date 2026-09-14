# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os, json, warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dropout,
    Input,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

general_path = "../input/cassava-leaf-disease-classification/"
assert os.path.exists(general_path), f"Path not found: {general_path}"

try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

print(
    "Found files:",
    sorted(
        [
            f
            for f in os.listdir(general_path)
            if f.endswith(".csv") or f.endswith(".json")
        ]
    ),
)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
with open(os.path.join(general_path, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k): v for k, v in map_classes.items()}
print(json.dumps(map_classes, indent=4))




## === cell 2
train_img_dir = os.path.join(general_path, "train_images")
input_files = os.listdir(train_img_dir)
print(f"Number of train images: {len(input_files)}")




## === cell 3
print("Skipping image shape scan (not required for training/inference).")




## === cell 4
df_train = pd.read_csv(os.path.join(general_path, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()




## === cell 5
print("Skipping dataset visualization plots (not required for training/inference).")




## === cell 6
for cls in range(5):
    tmp_df = df_train[df_train["label"] == cls]
    print(f"Total train images for class {cls}: {tmp_df.shape[0]}")




## === cell 7
print(
    "Skipping albumentations augmentation demos (not required for training/inference here)."
)




## === cell 8
img_width, img_height = 224, 224
num_classes = 5
BATCH_SIZE = 64
EPOCHS = 5

train_tfrec_dir = os.path.join(general_path, "train_tfrecords")
assert os.path.exists(train_tfrec_dir), f"TFRecord dir not found: {train_tfrec_dir}"

train_files = sorted(
    [
        os.path.join(train_tfrec_dir, f)
        for f in os.listdir(train_tfrec_dir)
        if f.endswith(".tfrec")
    ]
)
assert len(train_files) > 0, "No train TFRecord files found."

split_index = int(round(0.8 * len(train_files)))
train_tfrec_files = train_files[:split_index]
valid_tfrec_files = train_files[split_index:]
assert len(train_tfrec_files) > 0 and len(valid_tfrec_files) > 0, "Bad TFRecord split."

AUTO = tf.data.AUTOTUNE

_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def _parse_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, [img_height, img_width], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0  # identical to rescale=1/255
    label = tf.cast(ex["target"], tf.int32)
    label = tf.one_hot(label, num_classes)
    return img, label


def _make_ds(files, training: bool):
    opts = tf.data.Options()
    opts.experimental_deterministic = not training

    try:
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO).with_options(opts)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_parse_example, num_parallel_calls=AUTO, deterministic=(not training))
    if not training:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


train_ds = _make_ds(train_tfrec_files, training=True)
valid_ds = _make_ds(valid_tfrec_files, training=False)

TRAIN_EXAMPLES = 1338 * len(train_tfrec_files)
VALID_EXAMPLES = 1338 * len(valid_tfrec_files)
TRAIN_STEPS_PER_EPOCH = int(np.ceil(TRAIN_EXAMPLES / BATCH_SIZE))
VALID_STEPS = int(np.ceil(VALID_EXAMPLES / BATCH_SIZE))

print(
    f"TFRecords: {len(train_tfrec_files)} train shards, {len(valid_tfrec_files)} val shards"
)
print(
    f"Using steps_per_epoch={TRAIN_STEPS_PER_EPOCH}, validation_steps={VALID_STEPS} (computed from shard sizes)"
)




## === cell 9
x, y = next(iter(train_ds))
print(
    "Example batch shapes:",
    x.shape,
    y.shape,
    "label argmax:",
    int(tf.argmax(y[0]).numpy()),
)




## === cell 10
model = Sequential(
    [
        Input(shape=(img_width, img_height, 3)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()




## === cell 11
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
    validation_steps=VALID_STEPS,
    verbose=1,
)




## === cell 12
ss = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))
test_tfrec_dir = os.path.join(general_path, "test_tfrecords")
assert os.path.exists(test_tfrec_dir), f"Test TFRecord dir not found: {test_tfrec_dir}"

test_files = sorted(
    [
        os.path.join(test_tfrec_dir, f)
        for f in os.listdir(test_tfrec_dir)
        if f.endswith(".tfrec")
    ]
)
assert len(test_files) > 0, "No test TFRecord files found."

_TEST_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def _parse_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURE_DESCRIPTION)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, [img_height, img_width], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    name = ex["image_name"]
    return img, name


test_opts = tf.data.Options()
test_opts.experimental_deterministic = True
try:
    test_opts.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    test_opts.experimental_optimization.parallel_batch = True
except Exception:
    pass

test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTO).with_options(
    test_opts
)
test_ds = test_ds.map(_parse_test, num_parallel_calls=AUTO, deterministic=True)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

all_names = []
all_preds = []
for batch_imgs, batch_names in test_ds:
    batch_probs = model(batch_imgs, training=False).numpy()
    all_preds.append(np.argmax(batch_probs, axis=1).astype(np.int64))
    all_names.append(batch_names.numpy())

pred_labels = np.concatenate(all_preds, axis=0)
all_names = np.concatenate(all_names, axis=0).astype("U")  # decode bytes to str

pred_map = dict(zip(all_names.tolist(), pred_labels.tolist()))
preds = [int(pred_map[iid]) for iid in ss["image_id"].values]

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
my_submission.head()




## === cell 13
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert sub.shape[0] == ss.shape[0]
assert sub["label"].between(0, 4).all()
sub.head()
