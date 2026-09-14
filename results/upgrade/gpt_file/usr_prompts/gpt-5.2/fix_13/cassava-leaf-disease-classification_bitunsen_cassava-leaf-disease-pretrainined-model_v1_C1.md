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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import json

import tensorflow as tf
from tensorflow import keras

print("TF version:", tf.__version__)



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 6
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 32

PRE_TRAINED_MODEL_CANDIDATES = [
    "../input/xceptionv1/Cassava_Best_Model_XceptionV3_V01.hdf5",
    "/kaggle/input/xceptionv1/Cassava_Best_Model_XceptionV3_V01.hdf5",
]
PRE_TRAINED_MODEL = next(
    (p for p in PRE_TRAINED_MODEL_CANDIDATES if os.path.exists(p)), None
)
print("PRE_TRAINED_MODEL:", PRE_TRAINED_MODEL)



## === cell 7
pass



## === cell 8
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_df = sample_sub[["image_id"]].copy()
test_samples = test_df.shape[0]
test_samples




## === cell 9
def make_test_dataset(image_ids, batch_size):
    image_ids = tf.convert_to_tensor(image_ids, dtype=tf.string)
    base = tf.constant(TEST_DIR, dtype=tf.string)

    def _load_and_preprocess(image_id):
        path = tf.strings.join([base, image_id])
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # RGB
        img = tf.image.resize(
            img,
            [IMG_HEIGHT, IMG_WIDTH],
            method=tf.image.ResizeMethod.AREA,
            antialias=False,
        )
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = tf.data.Dataset.from_tensor_slices(image_ids)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = (
        ds.cache()
    )  # correctness-preserving for test; avoids repeated decode/resize overhead
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_dataset(test_df["image_id"].values, batch_size=batch_size)



## === cell 10
import random
from keras.models import load_model


def build_fallback_model(img_h=IMG_HEIGHT, img_w=IMG_WIDTH, n_classes=5):
    base = keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(img_h, img_w, 3),
        pooling="avg",
    )
    inputs = keras.Input(shape=(img_h, img_w, 3))
    x = keras.applications.xception.preprocess_input(inputs * 255.0)  # inputs are 0..1
    x = base(x, training=False)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass
try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass


def make_train_valid_datasets(
    tr_image_ids, tr_labels, va_image_ids, va_labels, batch_size
):
    tr_image_ids = tf.convert_to_tensor(tr_image_ids, dtype=tf.string)
    va_image_ids = tf.convert_to_tensor(va_image_ids, dtype=tf.string)

    tr_labels = tf.convert_to_tensor(tr_labels, dtype=tf.int64)
    va_labels = tf.convert_to_tensor(va_labels, dtype=tf.int64)

    base = tf.constant(TRAIN_DIR, dtype=tf.string)

    def _decode_resize_from_id(image_id):
        path = tf.strings.join([base, image_id])
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(
            img,
            [IMG_HEIGHT, IMG_WIDTH],
            method=tf.image.ResizeMethod.AREA,
            antialias=False,
        )
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def _augment_stateless(img, idx):
        r0 = tf.random.stateless_uniform(
            [], seed=[SEED, tf.cast(idx, tf.int32)], minval=0.0, maxval=1.0
        )
        img = tf.cond(r0 < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

        r1 = tf.random.stateless_uniform(
            [], seed=[SEED + 1, tf.cast(idx, tf.int32)], minval=0.0, maxval=1.0
        )

        def _cb():
            alpha = tf.random.stateless_uniform(
                [], seed=[SEED + 2, tf.cast(idx, tf.int32)], minval=0.8, maxval=1.2
            )
            beta = tf.random.stateless_uniform(
                [], seed=[SEED + 3, tf.cast(idx, tf.int32)], minval=-0.2, maxval=0.2
            )
            x = tf.clip_by_value(img * alpha + beta, 0.0, 1.0)
            return x

        img = tf.cond(r1 < 0.5, _cb, lambda: img)

        r2 = tf.random.stateless_uniform(
            [], seed=[SEED + 4, tf.cast(idx, tf.int32)], minval=0.0, maxval=1.0
        )

        def _rot():
            angle_deg = tf.random.stateless_uniform(
                [], seed=[SEED + 5, tf.cast(idx, tf.int32)], minval=-15.0, maxval=15.0
            )
            angle = angle_deg * (np.pi / 180.0)
            try:
                return tf.image.rotate(
                    img,
                    angles=angle,
                    interpolation="BILINEAR",
                    fill_mode="CONSTANT",
                    fill_value=0.0,
                )
            except Exception:
                return img

        img = tf.cond(r2 < 0.2, _rot, lambda: img)
        return img

    def _train_map(idx, image_id, y):
        img = _decode_resize_from_id(image_id)
        r = tf.random.stateless_uniform(
            [], seed=[SEED + 6, tf.cast(idx, tf.int32)], minval=0.0, maxval=1.0
        )
        img = tf.cond(r > 0.5, lambda: _augment_stateless(img, idx), lambda: img)
        return img, y

    def _valid_map(image_id, y):
        img = _decode_resize_from_id(image_id)
        return img, y

    tr_ds = tf.data.Dataset.from_tensor_slices((tr_image_ids, tr_labels)).enumerate()
    va_ds = tf.data.Dataset.from_tensor_slices((va_image_ids, va_labels))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    tr_ds = tr_ds.with_options(opts)
    va_ds = va_ds.with_options(opts)

    tr_ds = tr_ds.shuffle(
        buffer_size=min(8192, int(tr_labels.shape[0])),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    tr_ds = tr_ds.map(
        lambda idx, xy: _train_map(idx, xy[0], xy[1]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    tr_ds = tr_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    va_ds = va_ds.map(_valid_map, num_parallel_calls=tf.data.AUTOTUNE)
    va_ds = va_ds.cache()  # correctness-preserving; validation is deterministic/static
    va_ds = va_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return tr_ds, va_ds


model = None
if PRE_TRAINED_MODEL is not None and os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL)
else:
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(train_df))
    split = int(0.9 * len(train_df))
    tr_idx = perm[:split]
    va_idx = perm[split:]

    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_ds, valid_ds = make_train_valid_datasets(
        tr_df["image_id"].values,
        tr_df["label"].values,
        va_df["image_id"].values,
        va_df["label"].values,
        batch_size=batch_size,
    )

    model = build_fallback_model()
    model.fit(train_ds, validation_data=valid_ds, epochs=1, verbose=1)

model.summary()



## === cell 11
predict = model.predict(
    test_ds,
    steps=int(np.ceil(test_samples / batch_size)),
    verbose=1,
)
predict.shape



## === cell 12
test_pred_labels = np.argmax(predict, axis=1).astype(int)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": test_pred_labels}
)
submission.to_csv("submission.csv", index=False)
submission.head(3)



## === cell 13
check = pd.read_csv("submission.csv")
print(check.shape)
print(check.columns.tolist())
check.head(3)
