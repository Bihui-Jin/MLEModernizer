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

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



## === cell 1
IMAGE_SIZE = 224  # keep identical; already reduced for runtime
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
def _parse_labeled_example_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES_L)
    img = _decode_and_resize_from_bytes(ex["image"])
    img = _augment(img)
    y = tf.one_hot(tf.cast(ex["target"], tf.int32), NUM_CLASSES)
    return img, y


@tf.function
def _parse_labeled_example_eval(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES_L)
    img = _decode_and_resize_from_bytes(ex["image"])
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
    try:
        options.autotune.enabled = True
        options.autotune.cpu_budget = 0  # let TF decide
    except Exception:
        pass
    try:
        options.experimental_slack = True
    except Exception:
        pass
    return options


use_tfrecords = (len(train_tfrec_files) > 0) and (len(test_tfrec_files) > 0)
print(
    "TFRecords present:",
    use_tfrecords,
    "| train tfrecs:",
    len(train_tfrec_files),
    "| test tfrecs:",
    len(test_tfrec_files),
    "| using tfrecords for train/val:",
    use_tfrecords,
)

_TFREC_BUFFER_SIZE = 32 * 1024 * 1024  # 32MB


def _maybe_prefetch_to_device(ds):
    return ds


def make_ds_from_jpegs(paths, labels=None, training=False, cache=False):
    options = _base_ds_options(deterministic=not training)

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)

        @tf.function
        def _decode_path(p):
            img_bytes = tf.io.read_file(p)
            return _decode_and_resize_from_bytes(img_bytes)

        ds = ds.map(
            _decode_path,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not training,
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
            _decode_path_label,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not training,
        )

    if cache:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def make_ds_from_tfrecords_unlabeled(tfrec_files):
    options = _base_ds_options(deterministic=True)
    ds = tf.data.TFRecordDataset(
        tfrec_files,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=_TFREC_BUFFER_SIZE,
    ).with_options(options)
    ds = ds.map(
        _parse_unlabeled_example,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def make_ds_from_tfrecords_labeled(tfrec_files, training=False, cache=False):
    options = _base_ds_options(deterministic=not training)
    ds = tf.data.TFRecordDataset(
        tfrec_files,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=_TFREC_BUFFER_SIZE,
    ).with_options(options)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            _parse_labeled_example_train,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
    else:
        ds = ds.map(
            _parse_labeled_example_eval,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

    if cache:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


if use_tfrecords:
    RECORDS_PER_TFREC = 1338  # competition constant (ld_train**-1338.tfrec)
    n_total = int(len(train_df))
    n_val = int(round(0.1 * n_total))
    n_train = n_total - n_val

    total_files = len(train_tfrec_files)
    n_val_files = int(math.ceil(n_val / float(RECORDS_PER_TFREC)))
    n_val_files = max(1, min(total_files - 1, n_val_files))
    tr_files = train_tfrec_files[: total_files - n_val_files]
    va_files = train_tfrec_files[total_files - n_val_files :]

    tr_count = len(tr_files) * RECORDS_PER_TFREC
    va_count = len(va_files) * RECORDS_PER_TFREC

    train_ds = make_ds_from_tfrecords_labeled(tr_files, training=True, cache=False)
    val_ds = make_ds_from_tfrecords_labeled(va_files, training=False, cache=True)

    steps_per_epoch = int(math.ceil(tr_count / float(BATCH_SIZE)))
    validation_steps = int(math.ceil(va_count / float(BATCH_SIZE)))

    print(
        "TFRecord pipeline (file-based shard split): OK | n_total:",
        n_total,
        "| requested n_train:",
        n_train,
        "| requested n_val:",
        n_val,
        "| train files:",
        len(tr_files),
        "| val files:",
        len(va_files),
        "| implied tr_count:",
        tr_count,
        "| implied va_count:",
        va_count,
    )
else:
    train_ds = make_ds_from_jpegs(tr_paths, tr_y, training=True, cache=False)
    val_ds = make_ds_from_jpegs(va_paths, va_y, training=False, cache=True)

    steps_per_epoch = int(math.ceil(len(tr_paths) / float(BATCH_SIZE)))
    validation_steps = int(math.ceil(len(va_paths) / float(BATCH_SIZE)))

    print("JPEG pipeline: OK")



## === cell 3
base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling=None,
)
base.trainable = False  # keep identical (runtime-friendly)

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
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)




## === cell 5
def get_preds_model_list(image_dir, model_obj_list):
    img_ids = sample_df["image_id"].to_numpy()

    if use_tfrecords and (image_dir == TEST_IMG_DIR):
        test_ds_named = make_ds_from_tfrecords_unlabeled(test_tfrec_files)

        n = len(img_ids)
        preds_all = np.empty((n, NUM_CLASSES), dtype=np.float32)
        names_all = np.empty((n,), dtype=object)

        offset = 0
        for batch_img, batch_name in test_ds_named:
            batch_preds = None
            for mod in model_obj_list:
                p = mod.predict_on_batch(batch_img)
                batch_preds = p if batch_preds is None else (batch_preds + p)
            batch_preds = batch_preds / float(len(model_obj_list))

            bsz = int(batch_preds.shape[0])
            preds_all[offset : offset + bsz] = batch_preds

            bn = batch_name.numpy()
            if isinstance(bn, np.ndarray) and bn.dtype.kind in ("S", "O"):
                try:
                    bn = np.char.decode(bn.astype("S"), "utf-8")
                except Exception:
                    bn = np.array(
                        [
                            (
                                x.decode("utf-8")
                                if isinstance(x, (bytes, bytearray))
                                else str(x)
                            )
                            for x in bn
                        ],
                        dtype=object,
                    )
            else:
                bn = bn.astype(str)
            names_all[offset : offset + bsz] = bn
            offset += bsz

        name_to_idx = {name: i for i, name in enumerate(names_all.tolist())}

        labels = np.zeros((n,), dtype=np.int32)
        for i, img_id in enumerate(img_ids.tolist()):
            j = name_to_idx.get(img_id, None)
            if j is not None:
                labels[i] = int(np.argmax(preds_all[j]))

        return pd.DataFrame({"image_id": img_ids.tolist(), "label": labels.astype(int)})

    else:
        paths = (image_dir + "/" + img_ids.astype(str)).astype(str)
        test_ds = make_ds_from_jpegs(
            paths, labels=None, training=False, cache=(len(model_obj_list) > 1)
        )

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
