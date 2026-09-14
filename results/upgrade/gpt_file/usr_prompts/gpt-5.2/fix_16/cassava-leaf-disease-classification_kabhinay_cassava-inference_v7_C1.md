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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
os.environ.setdefault("TF_GPU_THREAD_COUNT", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())

SEED = 1337
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_TFRECORD_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFRECORD_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

for p in [
    TRAIN_CSV,
    SAMPLE_SUB,
    TRAIN_DIR,
    TEST_DIR,
    TRAIN_TFRECORD_DIR,
    TEST_TFRECORD_DIR,
]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

assert set(["image_id", "label"]).issubset(train_df.columns)
assert set(["image_id", "label"]).issubset(sample_df.columns)

num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)
print(train_df.head())

train_df = train_df.copy()
train_df["label"] = train_df["label"].astype(str)

val_frac = 0.1
train_parts = []
val_parts = []
for lbl, g in train_df.groupby("label", sort=False):
    g = g.sample(frac=1.0, random_state=SEED)
    n_val = max(1, int(round(len(g) * val_frac)))
    val_parts.append(g.iloc[:n_val])
    train_parts.append(g.iloc[n_val:])

train_df_split = (
    pd.concat(train_parts, axis=0)
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)
val_df_split = pd.concat(val_parts, axis=0).reset_index(drop=True)

print("Train split:", len(train_df_split), "Val split:", len(val_df_split))




## === cell 2
IMG_SIZE = 448
BATCH_SIZE = 16  # unchanged

preprocess = tf.keras.applications.efficientnet.preprocess_input
AUTOTUNE = tf.data.AUTOTUNE

labels_sorted = sorted(train_df_split["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(labels_sorted)}
print("Class indices:", class_to_index)

train_tfrec_files = sorted(
    [
        os.path.join(TRAIN_TFRECORD_DIR, f)
        for f in os.listdir(TRAIN_TFRECORD_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(TEST_TFRECORD_DIR, f)
        for f in os.listdir(TEST_TFRECORD_DIR)
        if f.endswith(".tfrec")
    ]
)
if not train_tfrec_files:
    raise RuntimeError("No train TFRecord files found.")
if not test_tfrec_files:
    raise RuntimeError("No test TFRecord files found.")

augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.055, fill_mode="reflect", seed=SEED),
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=SEED),
        tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", seed=SEED),
    ],
    name="augmenter",
)


@tf.function
def _decode_resize_preprocess_from_jpeg_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    img = preprocess(img)
    return img


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_and_preprocess(serialized):
    ex = tf.io.parse_single_example(serialized, _TRAIN_FEATURES)
    img = _decode_resize_preprocess_from_jpeg_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
    image_name = ex["image_name"]
    return img, y, image_name


@tf.function
def _parse_test_and_preprocess(serialized):
    ex = tf.io.parse_single_example(serialized, _TEST_FEATURES)
    img = _decode_resize_preprocess_from_jpeg_bytes(ex["image"])
    return img


def _safe_set(obj, name, value):
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False
    return False


def _dataset_options_deterministic():
    opt = tf.data.Options()
    _safe_set(opt, "deterministic", True)

    eo = getattr(opt, "experimental_optimization", None)
    if eo is not None:
        _safe_set(eo, "apply_default_optimizations", True)
        _safe_set(eo, "autotune", True)
        _safe_set(eo, "map_fusion", True)
        _safe_set(eo, "map_parallelization", True)
        _safe_set(eo, "parallel_batch", True)
        _safe_set(eo, "filter_fusion", True)
        _safe_set(eo, "inject_prefetch", True)
        _safe_set(eo, "autotune_buffers", True)

    th = getattr(opt, "threading", None)
    if th is not None:
        _safe_set(th, "private_threadpool_size", 0)
        _safe_set(th, "max_intra_op_parallelism", 0)

    return opt


def _select_tfrecord_shards_by_image_ids(all_files, image_ids):
    name_to_path = {os.path.basename(p): p for p in all_files}

    idx = train_df.reset_index()[["index", "image_id"]]
    needed = idx[idx["image_id"].isin(set(image_ids))]["index"].to_numpy(dtype=np.int32)

    shard_ids = np.unique(needed // 1338)
    selected = []
    for sid in shard_ids.tolist():
        fname = f"ld_train{sid:02d}-1338.tfrec"
        p = name_to_path.get(fname)
        if p is not None:
            selected.append(p)

    return selected if selected else list(all_files)


def _tfrecord_dataset_from_files(files):
    ds = tf.data.TFRecordDataset(
        files, num_parallel_reads=AUTOTUNE, compression_type=""
    )
    return ds


def _make_split_lookup(df):
    keys = df["image_id"].astype(str).values
    vals = np.ones(len(keys), dtype=np.int32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(keys, dtype=tf.string),
            tf.constant(vals, dtype=tf.int32),
        ),
        default_value=tf.constant(0, dtype=tf.int32),
    )


_train_lookup = _make_split_lookup(train_df_split)
_val_lookup = _make_split_lookup(val_df_split)


def _filter_by_lookup(lookup: tf.lookup.StaticHashTable):
    @tf.function
    def _fn(img, y, image_name):
        is_missing = tf.equal(tf.strings.length(image_name), 0)
        in_split = tf.equal(lookup.lookup(image_name), tf.constant(1, tf.int32))
        return tf.logical_or(is_missing, in_split)

    return _fn


def make_train_ds_from_tfrecords_filtered(tfrec_files, lookup, cache_path):
    ds = _tfrecord_dataset_from_files(tfrec_files)
    ds = ds.with_options(_dataset_options_deterministic())
    ds = ds.cache(cache_path + "_serialized")  # cache raw records
    ds = ds.map(
        _parse_train_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.filter(_filter_by_lookup(lookup))
    ds = ds.shuffle(buffer_size=4096, seed=SEED, reshuffle_each_iteration=True)

    @tf.function
    def _drop_name(img, y, image_name):
        return img, y

    ds = ds.map(_drop_name, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds_from_tfrecords_filtered(tfrec_files, lookup, cache_path):
    ds = _tfrecord_dataset_from_files(tfrec_files)
    ds = ds.with_options(_dataset_options_deterministic())
    ds = ds.cache(cache_path + "_serialized")  # cache raw records
    ds = ds.map(
        _parse_train_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.filter(_filter_by_lookup(lookup))

    @tf.function
    def _drop_name(img, y, image_name):
        return img, y

    ds = ds.map(_drop_name, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_tfrec_files_train = _select_tfrecord_shards_by_image_ids(
    train_tfrec_files, train_df_split["image_id"].tolist()
)
train_tfrec_files_val = _select_tfrecord_shards_by_image_ids(
    train_tfrec_files, val_df_split["image_id"].tolist()
)

print(
    "Train tfrec files (train split):",
    len(train_tfrec_files_train),
    train_tfrec_files_train,
)
print(
    "Train tfrec files (val split):", len(train_tfrec_files_val), train_tfrec_files_val
)

train_cache = "/kaggle/working/train_cache"
val_cache = "/kaggle/working/val_cache"
train_gen = make_train_ds_from_tfrecords_filtered(
    train_tfrec_files_train, _train_lookup, train_cache
)
val_gen = make_val_ds_from_tfrecords_filtered(
    train_tfrec_files_val, _val_lookup, val_cache
)




## === cell 3
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = augmenter(inp)
x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
model_v4 = tf.keras.Model(inp, out, name="cassava_efficientnetb0")

model_v4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model_v4.summary()




## === cell 4
EPOCHS = 5

history = model_v4.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model_v4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model_v4.fit(
    train_gen,
    validation_data=val_gen,
    epochs=1,
    verbose=1,
)




## === cell 5
test_df = sample_df[["image_id"]].copy()


def make_test_ds_from_tfrecords(tfrec_files, batch_size=32):
    ds = tf.data.TFRecordDataset(
        tfrec_files, num_parallel_reads=AUTOTUNE, compression_type=""
    )
    ds = ds.with_options(_dataset_options_deterministic())
    ds = ds.cache("/kaggle/working/test_cache_serialized")
    ds = ds.map(
        _parse_test_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_gen = make_test_ds_from_tfrecords(test_tfrec_files, batch_size=32)

pred_v4 = model_v4.predict(test_gen, verbose=1)
predicted_class_indices_v4 = np.argmax(pred_v4, axis=1).astype(int)

sub = sample_df.copy()
if len(predicted_class_indices_v4) != len(sub):
    raise RuntimeError(
        f"Prediction rows ({len(predicted_class_indices_v4)}) != sample rows ({len(sub)})"
    )

sub["label"] = predicted_class_indices_v4

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Rows:", len(sub), "Cols:", sub.columns.tolist())
print("Label distribution:", sub["label"].value_counts().to_dict())
