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
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/"
TEST_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/"



## === cell 2
import json



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
import tensorflow as tf

input_files = tf.io.gfile.listdir(TRAIN_DIR)
print(f"Number of train images: {len(input_files)}")



## === cell 6
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32
PRE_TRAINED_MODEL = (
    "../input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.hdf5"
)



## === cell 7
AUGMENTATIONS_TRAIN = None
AUGMENTATIONS_TEST = None



## === cell 8
from tensorflow import keras

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 9
pass



## === cell 10
test_filenames = sorted(tf.io.gfile.listdir(TEST_DIR))
test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
test_samples



## === cell 11
_AUTOTUNE = tf.data.AUTOTUNE


def _tfrecord_files(dir_path, prefix):
    if not tf.io.gfile.exists(dir_path):
        return []
    files = tf.io.gfile.glob(os.path.join(dir_path, f"{prefix}*.tfrec"))
    return sorted(files)


@tf.function
def _decode_resize_norm_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.LANCZOS3
    )
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (IMG_HEIGHT, IMG_WIDTH, 3))
    return img


def _build_test_dataset_from_jpegs(image_ids, batch_size):
    image_ids = tf.convert_to_tensor(np.asarray(image_ids), dtype=tf.string)
    base_dir = tf.constant(TEST_DIR, dtype=tf.string)

    def _load_and_preprocess(image_id):
        path = tf.strings.join([base_dir, image_id], separator="/")
        img_bytes = tf.io.read_file(path)
        return _decode_resize_norm_from_bytes(img_bytes)

    ds = tf.data.Dataset.from_tensor_slices(image_ids)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=_AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(_AUTOTUNE)
    return ds


def _build_test_dataset_from_tfrecords(tfrecord_files, batch_size):
    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_example(example_proto):
        ex = tf.io.parse_single_example(example_proto, feature_spec)
        img = _decode_resize_norm_from_bytes(ex["image"])
        return img

    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=_AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_parse_example, num_parallel_calls=_AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(_AUTOTUNE)
    return ds


PRED_BATCH_SIZE = max(batch_size, 64)

test_tfrec_files = _tfrecord_files(TEST_TFREC_DIR, "ld_test")
if len(test_tfrec_files) > 0:
    test_ds = _build_test_dataset_from_tfrecords(
        test_tfrec_files, batch_size=PRED_BATCH_SIZE
    )
else:
    test_ds = _build_test_dataset_from_jpegs(
        test_df["image_id"].values, batch_size=PRED_BATCH_SIZE
    )



## === cell 12
from tensorflow.keras.models import load_model

model = None

_pretrained_candidates = [
    PRE_TRAINED_MODEL,
    "/kaggle/input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.hdf5",
    "/kaggle/input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.h5",
    "/kaggle/input/inceptionresnetv1/Cassava_Best_InceptionResNet_Model_V01.keras",
]
_pretrained_path = next((p for p in _pretrained_candidates if os.path.exists(p)), None)

if _pretrained_path is not None:
    model = load_model(_pretrained_path, compile=False)
    print("Loaded pretrained model:", _pretrained_path)
else:
    print("Pretrained model not found at:", PRE_TRAINED_MODEL)
    print(
        "Falling back to training an ImageNet-initialized InceptionResNetV2 classifier on train set."
    )



## === cell 13
if model is None:
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    rng = np.random.RandomState(42)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_tfrec_files = _tfrecord_files(TRAIN_TFREC_DIR, "ld_train")

    if len(train_tfrec_files) == 0:
        tr_image_ids = tr_df["image_id"].values
        tr_labels = tr_df["label"].values.astype(np.int64)
        va_image_ids = va_df["image_id"].values
        va_labels = va_df["label"].values.astype(np.int64)

        base_dir_train = tf.constant(TRAIN_DIR, dtype=tf.string)

        def _decode_resize_norm(image_id):
            path = tf.strings.join([base_dir_train, image_id], separator="/")
            img_bytes = tf.io.read_file(path)
            return _decode_resize_norm_from_bytes(img_bytes)

        def _train_map(image_id, label):
            img = _decode_resize_norm(image_id)
            return img, label

        def _val_map(image_id, label):
            img = _decode_resize_norm(image_id)
            return img, label

        train_ds = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(tr_image_ids, dtype=tf.string), tr_labels)
        )
        val_ds = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(va_image_ids, dtype=tf.string), va_labels)
        )

        opts = tf.data.Options()
        opts.experimental_deterministic = True
        train_ds = train_ds.with_options(opts)
        val_ds = val_ds.with_options(opts)

        train_ds = (
            train_ds.map(_train_map, num_parallel_calls=_AUTOTUNE)
            .cache()
            .batch(batch_size, drop_remainder=False)
            .prefetch(_AUTOTUNE)
        )
        val_ds = (
            val_ds.map(_val_map, num_parallel_calls=_AUTOTUNE)
            .cache()
            .batch(batch_size, drop_remainder=False)
            .prefetch(_AUTOTUNE)
        )
    else:
        tr_set = set(tr_df["image_id"].tolist())
        va_set = set(va_df["image_id"].tolist())

        feature_spec = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }

        def _parse_train(example_proto):
            ex = tf.io.parse_single_example(example_proto, feature_spec)
            img = _decode_resize_norm_from_bytes(ex["image"])
            label = ex["target"]
            return ex["image_name"], img, label

        def _in_tr(name, img, label):
            return tf.numpy_function(
                lambda x: x.decode("utf-8") in tr_set, [name], Tout=tf.bool
            )

        def _in_va(name, img, label):
            return tf.numpy_function(
                lambda x: x.decode("utf-8") in va_set, [name], Tout=tf.bool
            )

        def _drop_name(name, img, label):
            return img, label

        raw = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=_AUTOTUNE)

        opts = tf.data.Options()
        opts.experimental_deterministic = True
        raw = raw.with_options(opts)

        parsed = raw.map(_parse_train, num_parallel_calls=_AUTOTUNE)

        train_ds = (
            parsed.filter(_in_tr)
            .map(_drop_name, num_parallel_calls=_AUTOTUNE)
            .cache()
            .batch(batch_size, drop_remainder=False)
            .prefetch(_AUTOTUNE)
        )
        val_ds = (
            parsed.filter(_in_va)
            .map(_drop_name, num_parallel_calls=_AUTOTUNE)
            .cache()
            .batch(batch_size, drop_remainder=False)
            .prefetch(_AUTOTUNE)
        )

    base = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling="avg",
    )
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.applications.inception_resnet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_steps = int(np.ceil(len(tr_df) / batch_size))
    val_steps = int(np.ceil(len(va_df) / batch_size))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        verbose=1,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
    )



## === cell 14
pred_probs = model.predict(
    test_ds,
    verbose=1,
)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)



## === cell 15
sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sub = pd.read_csv(sub_path)

pred_df = pd.DataFrame({"image_id": test_df["image_id"].values, "label": pred_labels})
sub = sub.drop(columns=["label"]).merge(pred_df, on="image_id", how="left")

if sub["label"].isna().any():
    fill_label = int(pd.Series(pred_labels).mode().iloc[0])
    sub["label"] = sub["label"].fillna(fill_label).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 16
assert os.path.exists("submission.csv"), "submission.csv was not created"
_check = pd.read_csv("submission.csv")
assert list(_check.columns) == ["image_id", "label"], f"Bad columns: {_check.columns}"
assert len(_check) == len(
    pd.read_csv(sub_path)
), "Row count mismatch vs sample_submission"
assert _check["label"].notna().all(), "Submission has NaN labels"
print("Submission looks valid.")
