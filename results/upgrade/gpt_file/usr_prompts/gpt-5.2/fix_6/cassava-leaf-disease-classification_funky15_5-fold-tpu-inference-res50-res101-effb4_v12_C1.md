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
import os, glob, math, re, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

print("TensorFlow version:", tf.__version__)

SEED = 42
keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print("Determinism: enabled")
except Exception as e:
    print("Determinism: not enabled (continuing). Reason:", repr(e))




## === cell 1
IMAGE_SIZE = 224  # reduced from 512 to fit within Kaggle CPU/GPU memory/time reliably
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 3

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

train_paths = (TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)).to_numpy()
train_labels = train_df["label"].to_numpy(dtype=np.int32)

print("Train rows:", len(train_df), "Test rows:", len(sample_df))




## === cell 2
from sklearn.model_selection import train_test_split

tr_paths, va_paths, tr_y, va_y = train_test_split(
    train_paths,
    train_labels,
    test_size=0.1,
    random_state=SEED,
    stratify=train_labels,
)

TFREC_FEATURES_L = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
TFREC_FEATURES_U = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _list_tfrecords(dir_path, pattern="*.tfrec"):
    if not tf.io.gfile.exists(dir_path):
        return []
    files = tf.io.gfile.glob(os.path.join(dir_path, pattern))
    files.sort()
    return files


train_tfrec_files = _list_tfrecords(TRAIN_TFREC_DIR, "*.tfrec")
test_tfrec_files = _list_tfrecords(TEST_TFREC_DIR, "*.tfrec")

tr_set = set(os.path.basename(p) for p in tr_paths.tolist())
va_set = set(os.path.basename(p) for p in va_paths.tolist())


@tf.function
def _decode_and_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_brightness(img, 0.1, seed=SEED)
    return img


@tf.function
def _parse_labeled_example(example_proto, training):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES_L)
    img = _decode_and_resize_from_bytes(ex["image"])
    if training:
        img = _augment(img)
    y = tf.one_hot(tf.cast(ex["target"], tf.int32), NUM_CLASSES)
    return img, y


@tf.function
def _parse_unlabeled_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES_U)
    img = _decode_and_resize_from_bytes(ex["image"])
    return img, ex["image_name"]


def _base_ds_options(deterministic=True):
    options = tf.data.Options()
    options.experimental_deterministic = bool(deterministic)
    try:
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass
    try:
        options.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    return options


def make_ds_from_jpegs(paths, labels=None, training=False):
    options = _base_ds_options(deterministic=True)

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)

        @tf.function
        def _decode_path(p):
            img_bytes = tf.io.read_file(p)
            return _decode_and_resize_from_bytes(img_bytes)

        ds = ds.map(
            _decode_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(options)
        if training:
            ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

        @tf.function
        def _decode_path_label(p, y):
            img_bytes = tf.io.read_file(p)
            img = _decode_and_resize_from_bytes(img_bytes)
            if training:
                img = _augment(img)
            return img, tf.one_hot(y, NUM_CLASSES)

        ds = ds.map(
            _decode_path_label, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_ds_from_tfrecords_labeled(tfrec_files, training=False):
    options = _base_ds_options(deterministic=True)

    files_ds = tf.data.Dataset.from_tensor_slices(tfrec_files).with_options(options)
    if training:
        files_ds = files_ds.shuffle(
            len(tfrec_files), seed=SEED, reshuffle_each_iteration=True
        )

    ds = files_ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=tf.data.AUTOTUNE),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    training_tf = tf.constant(bool(training))
    ds = ds.map(
        lambda x: _parse_labeled_example(x, training_tf),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_ds_from_tfrecords_unlabeled(tfrec_files):
    options = _base_ds_options(deterministic=True)

    files_ds = tf.data.Dataset.from_tensor_slices(tfrec_files).with_options(options)
    ds = files_ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=tf.data.AUTOTUNE),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(
        _parse_unlabeled_example,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


use_tfrecords = (len(train_tfrec_files) > 0) and (len(test_tfrec_files) > 0)
print(
    "TFRecords present:",
    use_tfrecords,
    "| train tfrecs:",
    len(train_tfrec_files),
    "| test tfrecs:",
    len(test_tfrec_files),
)

if use_tfrecords:
    try:
        tr_set_tf = tf.constant(list(tr_set))
        va_set_tf = tf.constant(list(va_set))
        tr_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tr_set_tf, tf.ones_like(tr_set_tf, dtype=tf.int64)
            ),
            default_value=0,
        )
        va_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                va_set_tf, tf.ones_like(va_set_tf, dtype=tf.int64)
            ),
            default_value=0,
        )

        def make_train_val_from_tfrecs(tfrec_files, training):
            options = _base_ds_options(deterministic=True)

            files_ds = tf.data.Dataset.from_tensor_slices(tfrec_files).with_options(
                options
            )
            if training:
                files_ds = files_ds.shuffle(
                    len(tfrec_files), seed=SEED, reshuffle_each_iteration=True
                )

            ds = files_ds.interleave(
                lambda f: tf.data.TFRecordDataset(
                    f, num_parallel_reads=tf.data.AUTOTUNE
                ),
                cycle_length=tf.data.AUTOTUNE,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )

            def _parse_name_and_example(x):
                ex = tf.io.parse_single_example(
                    x,
                    {
                        **TFREC_FEATURES_L,
                        "image_name": tf.io.FixedLenFeature([], tf.string),
                    },
                )
                return ex["image_name"], ex["image"], ex["target"]

            ds = ds.map(
                _parse_name_and_example,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )

            if training:
                ds = ds.filter(lambda name, img, target: tr_lookup.lookup(name) > 0)
            else:
                ds = ds.filter(lambda name, img, target: va_lookup.lookup(name) > 0)

            def _finalize(name, img_bytes, target):
                img = _decode_and_resize_from_bytes(img_bytes)
                if training:
                    img = _augment(img)
                y = tf.one_hot(tf.cast(target, tf.int32), NUM_CLASSES)
                return img, y

            ds = ds.map(
                _finalize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
            )

            if training:
                ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

            ds = ds.batch(BATCH_SIZE, drop_remainder=False)

            if not training:
                ds = ds.cache()

            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = make_train_val_from_tfrecs(train_tfrec_files, training=True)
        val_ds = make_train_val_from_tfrecs(train_tfrec_files, training=False)

        _ = next(iter(train_ds.take(1)))
        _ = next(iter(val_ds.take(1)))
        print("TFRecords pipeline: OK")
    except Exception as e:
        print("TFRecords pipeline failed, falling back to JPEGs. Reason:", repr(e))
        use_tfrecords = False

if not use_tfrecords:
    train_ds = make_ds_from_jpegs(tr_paths, tr_y, training=True)
    val_ds = make_ds_from_jpegs(va_paths, va_y, training=False)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    print("JPEG pipeline: OK")




## === cell 3
base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling=None,
)
base.trainable = False  # quick and stable; avoids long runtime

inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.applications.resnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 4
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=1, verbose=1
    ),
]

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2
)




## === cell 5
def get_preds_model_list(image_dir, model_obj_list):
    img_ids = sample_df["image_id"].to_numpy()

    if use_tfrecords and (image_dir == TEST_IMG_DIR):
        test_ds_named = make_ds_from_tfrecords_unlabeled(test_tfrec_files)

        preds_all = None
        names_chunks = []

        for batch_imgs, batch_names in test_ds_named:
            p_sum = None
            for mod in model_obj_list:
                p = mod(batch_imgs, training=False).numpy()
                p_sum = p if p_sum is None else (p_sum + p)
            p_avg = p_sum / float(len(model_obj_list))

            preds_all = (
                p_avg
                if preds_all is None
                else np.concatenate([preds_all, p_avg], axis=0)
            )
            names_chunks.append(batch_names.numpy())

        names_all = np.concatenate(names_chunks, axis=0)
        labels = preds_all.argmax(axis=1).astype(np.int32)

        if names_all.dtype.kind in ("S", "O"):
            names_str = np.array(
                [
                    n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                    for n in names_all.tolist()
                ]
            )
        else:
            names_str = names_all.astype(str)

        pred_map = dict(zip(names_str.tolist(), labels.tolist()))
        out_labels = np.fromiter(
            (pred_map.get(i, 0) for i in img_ids.tolist()),
            dtype=np.int32,
            count=len(img_ids),
        )

        return pd.DataFrame(
            {"image_id": img_ids.tolist(), "label": out_labels.astype(int)}
        )
    else:
        paths = (image_dir + "/" + img_ids.astype(str)).astype(str)
        test_ds = make_ds_from_jpegs(paths, labels=None, training=False)

        preds = None
        for mod in model_obj_list:
            p = mod.predict(test_ds, verbose=0)
            preds = p if preds is None else (preds + p)
        preds = preds / float(len(model_obj_list))

        labels = preds.argmax(axis=1).astype(int)
        return pd.DataFrame({"image_id": img_ids.tolist(), "label": labels})


def get_preds(image_dir, model_obj):
    return get_preds_model_list(image_dir, [model_obj])




## === cell 6
model0 = model
mod_lst = [model0]

predict_df = get_preds_model_list(TEST_IMG_DIR, mod_lst)

predict_df = sample_df[["image_id"]].merge(predict_df, on="image_id", how="left")
predict_df["label"] = predict_df["label"].fillna(0).astype(int)

predict_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predict_df.shape)
print(predict_df.head())




## === cell 7
try:
    from IPython.display import display

    display(predict_df.head(10))
except Exception:
    print(predict_df.head(10))
