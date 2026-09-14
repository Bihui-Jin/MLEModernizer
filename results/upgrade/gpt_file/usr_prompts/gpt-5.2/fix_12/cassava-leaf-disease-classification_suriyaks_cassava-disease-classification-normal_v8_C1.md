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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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
tf_keras==2.18.0

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
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

tf.keras.utils.set_random_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

train = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")



## === cell 1
path = "/kaggle/input/cassava-leaf-disease-classification/"
_ = path  # keep variable to preserve notebook structure without extra I/O



## === cell 2
_ = train.head()



## === cell 3
_ = (train.shape, train.dtypes.to_dict())



## === cell 4
_ = train["label"].unique()



## === cell 5
train_path = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"



## === cell 6
file = open(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
json_data = json.load(file)
file.close()



## === cell 7
train["label"] = train["label"].astype(np.int32)

label_map = {int(k): v for k, v in json_data.items()}
_ = train.head()



## === cell 8
size = 512
bat_size = 4
split = 0.33
epoch = 5

AUTOTUNE = tf.data.AUTOTUNE

_resizer = keras.layers.Resizing(
    size, size, interpolation="nearest", crop_to_aspect_ratio=False, name="resize_nn"
)
_rescaler = keras.layers.Rescaling(1.0 / 255.0, name="rescale_255")


@tf.function
def _augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = _resizer(img)
    img = _rescaler(img)
    return img


def _parse_train_example(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TRAIN)
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.one_hot(tf.cast(ex["target"], tf.int32), depth=5, dtype=tf.float32)
    return img, label


def _parse_test_example(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TEST)
    img = _decode_resize_from_bytes(ex["image"])
    return img, ex["image_name"]


train_tfrecord_dir = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
train_tfrecord_files = sorted(
    [
        os.path.join(train_tfrecord_dir, f)
        for f in os.listdir(train_tfrecord_dir)
        if f.endswith(".tfrec")
    ]
)

_val_mod = max(int(round(1.0 / split)), 2)


def _make_train_val_from_tfrecords(files, training: bool):
    ds = tf.data.Dataset.from_tensor_slices(files)

    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=min(16, len(files)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.enumerate()

    if training:
        ds = ds.filter(lambda i, _: tf.not_equal(tf.math.floormod(i, _val_mod), 0))
    else:
        ds = ds.filter(lambda i, _: tf.equal(tf.math.floormod(i, _val_mod), 0))

    ds = ds.map(lambda _, x: x, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    opts = tf.data.Options()
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    try:
        opts.experimental_slack = True
    except Exception:
        pass
    ds = ds.with_options(opts)

    if training:
        ds = ds.shuffle(buffer_size=8192, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda x, y: (_augment(x), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )
    else:
        ds = ds.cache("/kaggle/working/val_cache")

    ds = ds.batch(bat_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_generator = _make_train_val_from_tfrecords(train_tfrecord_files, training=True)
validation_generator = _make_train_val_from_tfrecords(
    train_tfrecord_files, training=False
)

class_names = ["0", "1", "2", "3", "4"]
class_indices = {c: i for i, c in enumerate(class_names)}



## === cell 9
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)



## === cell 10
from tensorflow.keras.applications import EfficientNetB7

backbone = EfficientNetB7(
    include_top=False, weights="imagenet", input_shape=(512, 512, 3)
)
backbone.trainable = False

inputs = keras.Input(shape=(512, 512, 3))
x = backbone(inputs, training=False)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

model.summary()



## === cell 11
history = model.fit(
    train_generator,
    epochs=epoch,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[callback],
)



## === cell 12
_ = None



## === cell 13
ss = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

inv_class_indices = {v: int(k) for k, v in class_indices.items()}

test_files = [
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test00-1338.tfrec",
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test01-1338.tfrec",
]

test_ds_full = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE)
test_ds_full = test_ds_full.map(
    _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds_full = test_ds_full.cache("/kaggle/working/test_cache")
test_ds_full = test_ds_full.batch(bat_size, drop_remainder=False).prefetch(AUTOTUNE)

probs = model.predict(test_ds_full.map(lambda x, _: x), verbose=0)

names_list = []
for _, batch_names in test_ds_full:
    names_list.append(batch_names.numpy())
all_names = np.concatenate(names_list, axis=0).astype("U")

cls_idx = np.argmax(probs, axis=1).astype(int)
preds = np.asarray([inv_class_indices[i] for i in cls_idx.tolist()], dtype=np.int32)

pred_df = pd.DataFrame({"image_id": all_names, "label": preds})
my_submission = ss[["image_id"]].merge(pred_df, on="image_id", how="left")
my_submission["label"] = my_submission["label"].fillna(0).astype(np.int32)

my_submission.to_csv("submission.csv", index=False)
_ = my_submission.head()



## === cell 14
model.save("model.keras")
