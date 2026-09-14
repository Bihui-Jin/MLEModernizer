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

3.13

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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

INPUT_ROOT = "/kaggle/input"
print("Input root exists:", os.path.exists(INPUT_ROOT))
print("TF version:", tf.__version__)

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.optimizer.set_jit(False)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications.efficientnet import preprocess_input

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

NUM_CLASSES = int(train_csv["label_encoded"].nunique())
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.125, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomZoom(0.2, 0.2, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
    ],
    name="data_augmentation",
)

TRAIN_TFREC_GLOB = (
    "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
)
TEST_TFREC_GLOB = (
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"
)

train_tfrecs = tf.io.gfile.glob(TRAIN_TFREC_GLOB)
test_tfrecs = tf.io.gfile.glob(TEST_TFREC_GLOB)
train_tfrecs = sorted(train_tfrecs)
test_tfrecs = sorted(test_tfrecs)

print("Found train tfrecs:", len(train_tfrecs), "test tfrecs:", len(test_tfrecs))


@tf.function
def _decode_resize_preprocess_from_bytes(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = preprocess_input(img)
    return img


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TRAIN_FEATURES)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


@tf.function
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TEST_FEATURES)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    return img, ex["image_name"]


@tf.function
def _augment_onehot(img, label):
    img = data_augmentation(img, training=True)
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label


@tf.function
def _onehot_only(img, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label


def _safe_setattr(obj, name, value):
    try:
        getattr(obj, name)
    except Exception:
        return
    try:
        setattr(obj, name, value)
    except Exception:
        return


def _set_dataset_options(ds, training: bool):
    options = tf.data.Options()
    options.deterministic = not training

    opt = options.experimental_optimization
    _safe_setattr(opt, "map_parallelization", True)
    _safe_setattr(opt, "parallel_batch", True)
    _safe_setattr(opt, "map_and_batch_fusion", True)
    _safe_setattr(opt, "autotune_buffers", True)
    _safe_setattr(opt, "autotune", True)
    try:
        options.experimental_slack = True
    except Exception:
        pass
    return ds.with_options(options)


train_ids = tf.constant(train["image_id"].values.astype("U"), dtype=tf.string)
valid_ids = tf.constant(valid["image_id"].values.astype("U"), dtype=tf.string)

train_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        train_ids, tf.ones_like(train_ids, dtype=tf.int32)
    ),
    default_value=0,
)
valid_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        valid_ids, tf.ones_like(valid_ids, dtype=tf.int32)
    ),
    default_value=0,
)


def make_trainvalid_ds_from_tfrecs(tfrecs, training: bool, cache_path: str | None):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
    ds = ds.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=not training
    )
    return None  # placeholder to trigger fallback


@tf.function
def _decode_map_path_label(path, label):
    img = tf.io.read_file(path)
    img = _decode_resize_preprocess_from_bytes(img)
    return img, label


def make_ds(paths, labels, training: bool, cache_path: str | None):
    labels = np.asarray(labels, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(
        _decode_map_path_label, num_parallel_calls=AUTOTUNE, deterministic=not training
    )

    if cache_path is not None:
        ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(
            buffer_size=min(int(len(paths)), 8192),
            seed=42,
            reshuffle_each_iteration=True,
        )
        ds = ds.map(_augment_onehot, num_parallel_calls=AUTOTUNE, deterministic=False)
    else:
        ds = ds.map(_onehot_only, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = _set_dataset_options(ds, training=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_cache = "/kaggle/working/cache_train_decoded"
valid_cache = "/kaggle/working/cache_valid_decoded"
train_ds = make_ds(
    train["path"].values,
    train["label_encoded"].values,
    training=True,
    cache_path=train_cache,
)
valid_ds = make_ds(
    valid["path"].values,
    valid["label_encoded"].values,
    training=False,
    cache_path=valid_cache,
)

train_steps = int(np.ceil(len(train) / BATCH_SIZE))
valid_steps = int(np.ceil(len(valid) / BATCH_SIZE))

print("tf.data datasets built and will be used for model.fit/model.predict.")
print("train_steps:", train_steps, "valid_steps:", valid_steps)



## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base_model.trainable = False  # preserve a standard, stable transfer-learning setup

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("Model built and compiled.")



## === cell 4
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)



## === cell 5
EPOCHS = 8  # kept as provided

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)



## === cell 6
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=3,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)



## === cell 7
inv_class_indices = {i: cls for i, cls in enumerate(le.classes_)}

disease_to_labelnum = {str(v): int(k) for k, v in label_to_disease.to_dict().items()}

print("Example mapping (class_idx->disease):", list(inv_class_indices.items())[:3])
print("Example mapping (disease->label):", list(disease_to_labelnum.items())[:3])



## === cell 8
import numpy as np
import pandas as pd

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
test_paths = (test_dir.rstrip("/") + "/" + sample_sub["image_id"]).to_numpy()


@tf.function
def _test_map_path(path):
    img = tf.io.read_file(path)
    img = _decode_resize_preprocess_from_bytes(img)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_test_map_path, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.cache("/kaggle/working/cache_test_decoded")
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = _set_dataset_options(test_ds, training=False).prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
pred_class_idx = np.argmax(probs, axis=1)

idx_to_disease = np.array(
    [inv_class_indices[i] for i in range(NUM_CLASSES)], dtype=object
)
pred_disease = idx_to_disease[pred_class_idx]

disease_keys = np.array(list(disease_to_labelnum.keys()), dtype=object)
label_vals = np.array([disease_to_labelnum[k] for k in disease_keys], dtype=np.int64)
sort_idx = np.argsort(disease_keys)
disease_keys_sorted = disease_keys[sort_idx]
label_vals_sorted = label_vals[sort_idx]
pred_pos = np.searchsorted(disease_keys_sorted, pred_disease)
pred_label = label_vals_sorted[pred_pos]

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_label}
)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Submission file created:", out_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
