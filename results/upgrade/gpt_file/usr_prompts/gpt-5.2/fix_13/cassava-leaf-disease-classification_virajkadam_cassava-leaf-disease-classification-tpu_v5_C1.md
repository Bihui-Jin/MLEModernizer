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

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
train_dir = os.path.join(BASE, "train_images")
test_dir = os.path.join(BASE, "test_images")
train_csv_path = os.path.join(BASE, "train.csv")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")
train_tfrecord_dir = os.path.join(BASE, "train_tfrecords")
test_tfrecord_dir = os.path.join(BASE, "test_tfrecords")

assert os.path.exists(train_dir), f"Missing train_dir: {train_dir}"
assert os.path.exists(test_dir), f"Missing test_dir: {test_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv: {sample_sub_path}"
assert os.path.exists(
    train_tfrecord_dir
), f"Missing train_tfrecords: {train_tfrecord_dir}"
assert os.path.exists(test_tfrecord_dir), f"Missing test_tfrecords: {test_tfrecord_dir}"

train = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print(train.shape, sample_sub.shape)
print(train.head())
print(sample_sub.head())




## === cell 2
IMG_SIZE = (512, 512)
BATCH_SIZE = 8  # preserve
NUM_CLASSES = 5
AUTOTUNE = tf.data.AUTOTUNE

paths = (train_dir + "/" + train["image_id"].astype(str)).to_numpy()
labels = train["label"].to_numpy(np.int32)

n = len(paths)
val_count = int(round(n * 0.1))  # preserve behavior
train_count = n - val_count

rng = np.random.RandomState(SEED)
idx = np.arange(n)
rng.shuffle(idx)

train_idx = idx[:train_count]
val_idx = idx[train_count:]

train_paths = paths[train_idx]
train_labels = labels[train_idx]
val_paths = paths[val_idx]
val_labels = labels[val_idx]


@tf.function
def _decode_resize(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


@tf.function
def _parse_tfrec_train_xy(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    lbl = tf.cast(ex["label"], tf.int32)
    tgt = tf.cast(ex["target"], tf.int32)
    lbl = tf.where(lbl >= 0, lbl, tgt)
    return img, lbl


@tf.function
def _parse_tfrec_test_xid(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    image_id = ex["image_id"]
    image_id = tf.cond(
        tf.size(image_id) > 0, lambda: image_id, lambda: ex["image_name"]
    )
    return img, image_id


def _make_image_ds(paths_np, labels_np=None, training=False):
    if labels_np is None:
        ds = tf.data.Dataset.from_tensor_slices(paths_np)
        ds = ds.map(lambda p: _decode_resize(p, None), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths_np, labels_np))
    ds = ds.map(lambda p, y: _decode_resize(p, y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(
            buffer_size=min(8192, len(paths_np)),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    else:
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_tfrec_train_val_ds(train_tfrec_dir, training=True):
    tfrec_files = sorted(
        [
            os.path.join(train_tfrec_dir, f)
            for f in os.listdir(train_tfrec_dir)
            if f.endswith(".tfrec")
        ]
    )
    assert len(tfrec_files) > 0, "No train tfrecords found"

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass

    cycle_len = min(8, max(1, len(tfrec_files)))
    ds = tf.data.TFRecordDataset(
        tfrec_files,
        num_parallel_reads=cycle_len,
    ).with_options(opts)

    ds = ds.map(_parse_tfrec_train_xy, num_parallel_calls=AUTOTUNE)

    ds = ds.shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=False)

    val_ds_local = ds.take(val_count)
    train_ds_local = ds.skip(val_count)

    cache_root = "/kaggle/working/tfdata_cache"
    os.makedirs(cache_root, exist_ok=True)
    train_cache_path = os.path.join(
        cache_root, f"train_cache_sz{IMG_SIZE[0]}_b{BATCH_SIZE}_seed{SEED}"
    )
    val_cache_path = os.path.join(
        cache_root, f"val_cache_sz{IMG_SIZE[0]}_b{BATCH_SIZE}_seed{SEED}"
    )

    train_ds_local = train_ds_local.cache(train_cache_path)
    val_ds_local = val_ds_local.cache(val_cache_path)

    if training:
        train_ds_local = train_ds_local.shuffle(
            buffer_size=8192, seed=SEED, reshuffle_each_iteration=True
        )
        train_ds_local = train_ds_local.batch(BATCH_SIZE, drop_remainder=True)
    else:
        train_ds_local = train_ds_local.batch(BATCH_SIZE, drop_remainder=False)

    val_ds_local = val_ds_local.batch(BATCH_SIZE, drop_remainder=True)

    train_ds_local = train_ds_local.prefetch(AUTOTUNE)
    val_ds_local = val_ds_local.prefetch(AUTOTUNE)
    return train_ds_local, val_ds_local


train_ds, val_ds = _make_tfrec_train_val_ds(train_tfrecord_dir, training=True)

backbone = tf.keras.applications.EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Rescaling(1.0 / 255.0)(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = backbone(x, training=False)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 3
backbone.trainable = False
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=2,
)

backbone.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    verbose=2,
)




## === cell 4
def sample_df(sample_size=50, seed=SEED):
    df = train.sample(sample_size, random_state=seed).reset_index(drop=True)
    return df


dfs = sample_df(sample_size=20)
y_true = dfs["label"].to_numpy(np.int32)

sample_paths = (train_dir + "/" + dfs["image_id"].astype(str)).to_numpy()

sample_ds = tf.data.Dataset.from_tensor_slices(sample_paths).map(
    lambda p: _decode_resize(p, None), num_parallel_calls=AUTOTUNE
)
sample_ds = sample_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

probs = model.predict(sample_ds, verbose=0)
preds = probs.argmax(axis=1).astype(int)

sample_test = pd.DataFrame(
    {
        "Prediction": preds.tolist(),
        "Actual": y_true.tolist(),
        "image_id": dfs["image_id"],
    }
)
print(sample_test.head(10))
print("Sample accuracy:", (sample_test["Prediction"] == sample_test["Actual"]).mean())




## === cell 5
test_image_ids = sample_sub["image_id"].astype(str).tolist()

test_tfrec_files = sorted(
    [
        os.path.join(test_tfrecord_dir, f)
        for f in os.listdir(test_tfrecord_dir)
        if f.endswith(".tfrec")
    ]
)
assert len(test_tfrec_files) > 0, "No test tfrecords found"

opts = tf.data.Options()
opts.experimental_deterministic = True
try:
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.autotune_buffers = True
except Exception:
    pass

cycle_len = min(8, max(1, len(test_tfrec_files)))
test_ds = tf.data.TFRecordDataset(
    test_tfrec_files, num_parallel_reads=cycle_len
).with_options(opts)
test_ds = test_ds.map(_parse_tfrec_test_xid, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

all_ids = []
all_preds = []
for batch_imgs, batch_ids in test_ds:
    batch_probs = model(batch_imgs, training=False)
    batch_preds = tf.argmax(batch_probs, axis=1, output_type=tf.int32)
    all_ids.append(batch_ids)
    all_preds.append(batch_preds)

all_ids = tf.concat(all_ids, axis=0).numpy()
all_preds = tf.concat(all_preds, axis=0).numpy().astype(np.int32)

if all_ids.dtype.kind in ("S", "O"):
    all_ids = np.array(
        [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in all_ids
        ],
        dtype="U",
    )
else:
    all_ids = all_ids.astype("U")

preds_by_id = dict(zip(all_ids.tolist(), all_preds.tolist()))

missing = [iid for iid in test_image_ids if iid not in preds_by_id]
if len(missing) > 0:
    print(
        f"Warning: {len(missing)} ids missing from TFRecords; falling back to default class 0 for those."
    )
predictions = [preds_by_id.get(iid, 0) for iid in test_image_ids]

submission = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
