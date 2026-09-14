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

# 5. Target score

0.8118766999093382

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

CANDIDATE_DATA_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
DATA_DIR = next(
    (p for p in CANDIDATE_DATA_DIRS if os.path.exists(p)), CANDIDATE_DATA_DIRS[0]
)

TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

print("Using DATA_DIR:", DATA_DIR)
print("TF version:", tf.__version__)

CACHE_DIR = "/kaggle/working/tf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

IMG_SIZE = (512, 512)
NUM_CLASSES = 5

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(factor=10.0 / 360.0, seed=SEED),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05, seed=SEED),
        layers.RandomZoom(
            height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=SEED
        ),
        layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="augmentation",
)


def build_model():
    base = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False  # keep identical training approach (no finetuning)

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = data_augmentation(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


my_model = build_model()
my_model.summary()



## === cell 2
if not os.path.exists(TRAIN_CSV_PATH):
    raise FileNotFoundError(f"train.csv not found at: {TRAIN_CSV_PATH}")

df_train = pd.read_csv(TRAIN_CSV_PATH)

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df_train,
    test_size=0.1,
    random_state=SEED,
    stratify=df_train["label"],
)

_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
try:
    _DATASET_OPTIONS.experimental_optimization.map_parallelization = True
    _DATASET_OPTIONS.experimental_optimization.parallel_batch = True
except Exception:
    pass


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)  # rescaling happens in-model
    return img


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    y = tf.cast(ex["target"], tf.int32)
    return img, y


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    image_name = ex["image_name"]
    return img, image_name


def _list_tfrecs(tfrecord_dir, pattern="*.tfrec"):
    files = sorted(glob.glob(os.path.join(tfrecord_dir, pattern)))
    return files


def _tfrecord_index_paths(tfrec_files):
    idxs = []
    for f in tfrec_files:
        idx = f + ".idx"
        idxs.append(idx if tf.io.gfile.exists(idx) else "")
    if all(x == "" for x in idxs):
        return None
    return idxs


def make_train_val_datasets(batch_size=16):
    train_tfrecs = (
        _list_tfrecs(TRAIN_TFREC_DIR) if os.path.isdir(TRAIN_TFREC_DIR) else []
    )
    if len(train_tfrecs) == 0:
        if not os.path.isdir(TRAIN_IMG_DIR):
            raise FileNotFoundError(f"train_images dir not found at: {TRAIN_IMG_DIR}")
        df_train_local = df_train.copy()
        df_train_local["path"] = (
            TRAIN_IMG_DIR.rstrip("/") + "/" + df_train_local["image_id"].astype(str)
        )

        tr_df, va_df = train_test_split(
            df_train_local,
            test_size=0.1,
            random_state=SEED,
            stratify=df_train_local["label"],
        )

        x_train = tr_df["path"].to_numpy()
        y_train = tr_df["label"].to_numpy(dtype=np.int32)
        x_val = va_df["path"].to_numpy()
        y_val = va_df["label"].to_numpy(dtype=np.int32)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).with_options(
            _DATASET_OPTIONS
        )
        train_ds = train_ds.shuffle(
            buffer_size=len(x_train), seed=SEED, reshuffle_each_iteration=True
        )
        train_ds = train_ds.map(
            lambda p, y: (_decode_and_resize(p), y), num_parallel_calls=AUTOTUNE
        )

        train_ds = train_ds.cache()
        train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).with_options(
            _DATASET_OPTIONS
        )
        val_ds = val_ds.map(
            lambda p, y: (_decode_and_resize(p), y), num_parallel_calls=AUTOTUNE
        )
        val_ds = val_ds.cache()
        val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return train_ds, val_ds

    train_ids = train_df["image_id"].astype(str).to_numpy()
    train_labels = train_df["label"].to_numpy(dtype=np.int32)

    tr_ids, va_ids, tr_y, va_y = train_test_split(
        train_ids, train_labels, test_size=0.1, random_state=SEED, stratify=train_labels
    )

    tr_set = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(tr_ids), tf.ones([len(tr_ids)], dtype=tf.int32)
        ),
        default_value=tf.constant(0, tf.int32),
    )
    va_set = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(va_ids), tf.ones([len(va_ids)], dtype=tf.int32)
        ),
        default_value=tf.constant(0, tf.int32),
    )

    raw = tf.data.TFRecordDataset(
        train_tfrecs,
        num_parallel_reads=AUTOTUNE,
        index_path=_tfrecord_index_paths(train_tfrecs),
    ).with_options(_DATASET_OPTIONS)

    _NAME_TARGET_FEATURES = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_name_target_and_imagebytes(example_proto):
        ex = tf.io.parse_single_example(example_proto, _NAME_TARGET_FEATURES)
        return ex["image"], tf.cast(ex["target"], tf.int32), ex["image_name"]

    def _decode_resize_from_bytes(image_bytes, y):
        img = tf.image.decode_jpeg(image_bytes, channels=3)
        img = tf.image.resize(
            img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img = tf.cast(img, tf.float32)
        return img, y

    parsed = raw.map(_parse_name_target_and_imagebytes, num_parallel_calls=AUTOTUNE)

    def _is_train(image_bytes, y, image_name):
        return tf.equal(tr_set.lookup(image_name), 1)

    def _is_val(image_bytes, y, image_name):
        return tf.equal(va_set.lookup(image_name), 1)

    train_ds = (
        parsed.filter(_is_train)
        .map(lambda image_bytes, y, name: (image_bytes, y), num_parallel_calls=AUTOTUNE)
        .map(
            lambda image_bytes, y: _decode_resize_from_bytes(image_bytes, y),
            num_parallel_calls=AUTOTUNE,
        )
    )
    val_ds = (
        parsed.filter(_is_val)
        .map(lambda image_bytes, y, name: (image_bytes, y), num_parallel_calls=AUTOTUNE)
        .map(
            lambda image_bytes, y: _decode_resize_from_bytes(image_bytes, y),
            num_parallel_calls=AUTOTUNE,
        )
    )

    train_ds = train_ds.shuffle(
        buffer_size=len(tr_ids), seed=SEED, reshuffle_each_iteration=True
    )

    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return train_ds, val_ds


BATCH_SIZE = 16
train_ds, val_ds = make_train_val_datasets(batch_size=BATCH_SIZE)

EPOCHS = 3 if not DEBUG else 1
history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/985591575.py in <cell line: 0>()
    222 
    223 BATCH_SIZE = 16
--> 224 train_ds, val_ds = make_train_val_datasets(batch_size=BATCH_SIZE)
    225 
    226 EPOCHS = 3 if not DEBUG else 1

/tmp/ipykernel_11/985591575.py in make_train_val_datasets(batch_size)
    158     )
    159 
--> 160     raw = tf.data.TFRecordDataset(
    161         train_tfrecs,
    162         num_parallel_reads=AUTOTUNE,

TypeError: TFRecordDatasetV2.__init__() got an unexpected keyword argument 'index_path'

## === cell 3
def make_test_dataset(batch_size=64):
    test_tfrecs = _list_tfrecs(TEST_TFREC_DIR) if os.path.isdir(TEST_TFREC_DIR) else []
    if len(test_tfrecs) > 0:
        raw = tf.data.TFRecordDataset(
            test_tfrecs,
            num_parallel_reads=AUTOTUNE,
            index_path=_tfrecord_index_paths(test_tfrecs),
        ).with_options(_DATASET_OPTIONS)
        ds = raw.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
        ds = ds.cache()
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return ds, None

    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    if len(test_images) == 0:
        test_images = sorted(
            glob.glob(os.path.join(TEST_IMG_DIR, "**", "*.jpg"), recursive=True)
        )
    if len(test_images) == 0:
        raise FileNotFoundError(f"No .jpg images found under: {TEST_IMG_DIR}")

    df_test_local = pd.DataFrame({"path": test_images})
    x_test = df_test_local["path"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(x_test).with_options(_DATASET_OPTIONS)
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds, df_test_local


test_ds, df_test_fallback = make_test_dataset(batch_size=64)

spec = test_ds.element_spec
yields_name = isinstance(spec, (tuple, list)) and len(spec) == 2

if yields_name:
    pred_chunks = []
    name_chunks = []
    for batch_imgs, batch_names in test_ds:
        preds = my_model(batch_imgs, training=False)
        pred_chunks.append(preds.numpy())
        name_chunks.append(batch_names.numpy())
    pred_test = np.concatenate(pred_chunks, axis=0)
    test_names = np.concatenate(name_chunks, axis=0).astype(str)
    df_test = pd.DataFrame({"image_id": test_names})
else:
    pred_test = my_model.predict(test_ds, verbose=1)
    df_test = df_test_fallback.copy()
    df_test["image_id"] = df_test["path"].str.rsplit("/", n=1).str[-1]

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["label"] = pred_test_labels

if os.path.exists(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    final_csv = sample_sub[["image_id"]].merge(
        final_submission[["image_id", "label"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    if final_csv["label"].isna().any():
        final_csv["label"] = final_csv["label"].fillna(0).astype(int)
else:
    final_csv = final_submission[["image_id", "label"]]

final_csv.to_csv("submission.csv", index=False)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2664905332.py in <cell line: 0>()
     29 
     30 
---> 31 test_ds, df_test_fallback = make_test_dataset(batch_size=64)
     32 
     33 spec = test_ds.element_spec

/tmp/ipykernel_11/2664905332.py in make_test_dataset(batch_size)
      2     test_tfrecs = _list_tfrecs(TEST_TFREC_DIR) if os.path.isdir(TEST_TFREC_DIR) else []
      3     if len(test_tfrecs) > 0:
----> 4         raw = tf.data.TFRecordDataset(
      5             test_tfrecs,
      6             num_parallel_reads=AUTOTUNE,

TypeError: TFRecordDatasetV2.__init__() got an unexpected keyword argument 'index_path'

## === cell 4
print("submission.csv written:", os.path.abspath("submission.csv"))
print("Rows:", len(final_csv), "Columns:", list(final_csv.columns))
print(final_csv.head())
print(final_csv["label"].value_counts().sort_index())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1104500619.py in <cell line: 0>()
      1 print("submission.csv written:", os.path.abspath("submission.csv"))
----> 2 print("Rows:", len(final_csv), "Columns:", list(final_csv.columns))
      3 print(final_csv.head())
      4 print(final_csv["label"].value_counts().sort_index())

NameError: name 'final_csv' is not defined
