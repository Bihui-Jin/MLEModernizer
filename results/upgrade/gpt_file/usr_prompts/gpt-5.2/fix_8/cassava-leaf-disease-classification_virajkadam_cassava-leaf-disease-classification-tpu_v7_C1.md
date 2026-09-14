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
import random
import numpy as np
import pandas as pd
import tensorflow as tf

keras = tf.keras

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

print("TensorFlow:", tf.__version__)
print("tf.keras:", tf.keras.__name__)




## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_dir = os.path.join(BASE_DIR, "train_images")
test_dir = os.path.join(BASE_DIR, "test_images")

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"

train = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print(train.head())
print(sample_sub.head())
print("Train size:", len(train), " Test size:", len(sample_sub))




## === cell 2
NUM_CLASSES = 5
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # keep conservative for memory
AUTOTUNE = tf.data.AUTOTUNE

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")
try:
    tf.config.optimizer.set_jit(True)  # XLA (same math, faster graph execution)
except Exception:
    pass


def build_model(backbone_name: str, input_shape=(512, 512, 3), num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    if backbone_name == "inceptionresnetv2":
        backbone = keras.applications.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone_name == "efficientnetv2b0":
        backbone = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError("Unknown backbone")

    backbone.trainable = True  # preserve original intent (fine-tuning)
    x = backbone(x, training=True)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs, outputs)
    return model


model1 = build_model(
    "inceptionresnetv2", input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES
)
model2 = build_model(
    "efficientnetv2b0", input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES
)

opt1 = keras.optimizers.Adam(learning_rate=1e-4)
opt2 = keras.optimizers.Adam(learning_rate=1e-4)

model1.compile(
    optimizer=opt1,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)
model2.compile(
    optimizer=opt2,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

print("Model1 params:", model1.count_params())
print("Model2 params:", model2.count_params())




## === cell 3
def decode_image_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.ensure_shape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _build_paths_np(base_dir, image_ids):
    image_ids = image_ids.astype(str).astype("U")
    return np.char.add(base_dir + "/", image_ids)


def _make_tfdata_options(deterministic=True):
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    try:
        options.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    try:
        options.experimental_optimization.map_and_batch_fusion = True
    except Exception:
        pass
    return options


TFDATA_OPTS_DETERMINISTIC = _make_tfdata_options(deterministic=True)

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def make_train_ds(df, training=True, cache_name="train"):
    paths_np = _build_paths_np(train_dir, df["image_id"].values)
    labels_np = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths_np, labels_np))

    def _map_decode(p, y):
        x = decode_image_from_path(p)
        return x, y

    def _map_aug(x, y):
        if training:
            x = tf.image.random_flip_left_right(x, seed=SEED)
        return x, y

    if training:
        ds = ds.shuffle(min(len(df), 2048), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_map_decode, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache(os.path.join(CACHE_DIR, f"{cache_name}.cache"))

    ds = ds.map(_map_aug, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.with_options(TFDATA_OPTS_DETERMINISTIC)

    ds = ds.batch(BATCH_SIZE, drop_remainder=training).prefetch(AUTOTUNE)
    return ds


train_shuffled = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_shuffled) * val_frac)
val_df = train_shuffled.iloc[:val_size].copy()
trn_df = train_shuffled.iloc[val_size:].copy()

train_ds = make_train_ds(trn_df, training=True, cache_name="train_ds")
val_ds = make_train_ds(val_df, training=False, cache_name="val_ds")

print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())




## === cell 4
EPOCHS = 2  # preserve original training plan

history1 = model1.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
history2 = model2.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## === cell 5
def sample_df(sample_size=50):
    df = train.sample(sample_size, random_state=SEED).reset_index(drop=True)
    return df


dfs = sample_df(sample_size=50)

sample_paths = _build_paths_np(train_dir, dfs["image_id"].values)
sample_labels = dfs["label"].values.astype(np.int32)

sample_ds = tf.data.Dataset.from_tensor_slices(sample_paths)


def _map_sample(p):
    return decode_image_from_path(p)


sample_ds = (
    sample_ds.map(_map_sample, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache(os.path.join(CACHE_DIR, "sample_ds.cache"))
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(TFDATA_OPTS_DETERMINISTIC)
)

probs1_s = model1.predict(sample_ds, verbose=0)
probs2_s = model2.predict(sample_ds, verbose=0)
probs_s = 0.5 * probs1_s + 0.5 * probs2_s
preds = np.argmax(probs_s, axis=1).astype(int).tolist()

y_true = sample_labels
acc = (np.array(preds) == y_true).mean()
print("Sample accuracy (50 imgs):", acc)




## === cell 6
sample_test = pd.DataFrame({"Prediction": preds, "Actual": y_true})
print(sample_test.head(30))




## === cell 7
PRED_BATCH_SIZE = 32

test_paths = _build_paths_np(test_dir, sample_sub["image_id"].values)
test_ds_pred = tf.data.Dataset.from_tensor_slices(test_paths)


def _map_test(p):
    return decode_image_from_path(p)


test_ds_pred = (
    test_ds_pred.map(_map_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache(os.path.join(CACHE_DIR, "test_ds.cache"))
    .batch(PRED_BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(TFDATA_OPTS_DETERMINISTIC)
)

print("Test batches:", tf.data.experimental.cardinality(test_ds_pred).numpy())

probs1 = model1.predict(test_ds_pred, verbose=1)
probs2 = model2.predict(test_ds_pred, verbose=1)

probs = 0.5 * probs1 + 0.5 * probs2
predictions = np.argmax(probs, axis=1).astype(int)

print("Predictions shape:", predictions.shape, "Unique labels:", np.unique(predictions))

submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": predictions}
)
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

print(submission.head())
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(submission))
