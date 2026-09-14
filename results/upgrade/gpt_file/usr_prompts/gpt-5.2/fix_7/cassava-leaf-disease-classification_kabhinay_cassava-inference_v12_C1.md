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
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
    print("Determinism enabled.")
except Exception as e:
    print("Could not enable determinism:", repr(e))

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
def resolve_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


BASE_DIR = resolve_path(
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset directory."
    )

TRAIN_CSV = resolve_path(
    os.path.join(BASE_DIR, "train.csv"),
    "/kaggle/input/train.csv",
    "../input/train.csv",
)
SAMPLE_SUB = resolve_path(
    os.path.join(BASE_DIR, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "../input/sample_submission.csv",
)
TRAIN_IMG_DIR = resolve_path(
    os.path.join(BASE_DIR, "train_images"),
    "/kaggle/input/train_images",
    "../input/train_images",
)
TEST_IMG_DIR = resolve_path(
    os.path.join(BASE_DIR, "test_images"),
    "/kaggle/input/test_images",
    "../input/test_images",
)

for req, p in [
    ("TRAIN_CSV", TRAIN_CSV),
    ("SAMPLE_SUB", SAMPLE_SUB),
    ("TRAIN_IMG_DIR", TRAIN_IMG_DIR),
    ("TEST_IMG_DIR", TEST_IMG_DIR),
]:
    if p is None:
        raise FileNotFoundError(f"Missing required path: {req}")

print("Using BASE_DIR:", BASE_DIR)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError("train.csv must contain columns: image_id, label")
if not {"image_id", "label"}.issubset(sample_df.columns):
    raise ValueError("sample_submission.csv must contain columns: image_id, label")

num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)




## === cell 2
IMG_SIZE = (448, 448)
BATCH_SIZE = 16
EPOCHS = 2  # keep identical

train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].copy()
trn_df = train_df.iloc[val_size:].copy()

IMG_H, IMG_W = IMG_SIZE


def _decode_only_uint8(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    return img


def _resize_and_norm(img_uint8):
    img = tf.image.resize(
        img_uint8, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img, idx):
    img = tf.image.stateless_random_flip_left_right(img, seed=tf.stack([42, idx]))

    angle = tf.random.stateless_uniform(
        [], seed=tf.stack([43, idx]), minval=-10.0, maxval=10.0
    )
    angle = angle * (np.pi / 180.0)

    tx = tf.random.stateless_uniform(
        [], seed=tf.stack([44, idx]), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_W, tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=tf.stack([45, idx]), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_H, tf.float32)

    scale = tf.random.stateless_uniform(
        [], seed=tf.stack([46, idx]), minval=0.9, maxval=1.1
    )

    cx = (tf.cast(IMG_W, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_H, tf.float32) - 1.0) / 2.0
    cos_a = tf.cos(angle) / scale
    sin_a = tf.sin(angle) / scale

    a0 = cos_a
    a1 = -sin_a
    a2 = (1.0 - cos_a) * cx + sin_a * cy - tx
    b0 = sin_a
    b1 = cos_a
    b2 = (1.0 - cos_a) * cy - sin_a * cx - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=[IMG_H, IMG_W],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)
    return img


def _build_paths_tensor(img_dir, image_ids):
    return tf.strings.join(
        [tf.constant(img_dir + os.sep), tf.constant(image_ids, dtype=tf.string)]
    )


def make_dataset(df, training: bool):
    image_ids = df["image_id"].astype(str).values
    paths = _build_paths_tensor(TRAIN_IMG_DIR, image_ids)
    labels = tf.constant(df["label"].astype(np.int32).values)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(lambda p, y: (_decode_only_uint8(p), y), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()

    if training:
        shuffle_buf = min(len(df), 4096)
        ds = ds.shuffle(buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=True)

    ds = ds.enumerate()  # (idx, (img_uint8, label))

    def _postprocess(idx, data):
        img_uint8, label = data
        img = _resize_and_norm(img_uint8)
        if training:
            img = _augment(img, tf.cast(idx, tf.int32))
        return img, label

    ds = ds.map(_postprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(trn_df, training=True)
val_ds = make_dataset(val_df, training=False)

inputs = tf.keras.Input(
    shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32, name="image"
)
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model_v1 = tf.keras.Model(inputs=inputs, outputs=outputs, name="cassava_baseline_cnn")

model_v1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model_v1.summary()

history = model_v1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 3
test_df = sample_df[["image_id"]].copy()

test_paths = _build_paths_tensor(TEST_IMG_DIR, test_df["image_id"].astype(str).values)


def make_test_dataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_decode_only_uint8, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.map(_resize_and_norm, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(32, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_dataset(test_paths)
print("Test samples:", len(test_df))

pred_v1 = model_v1.predict(test_ds, verbose=1)
pred_v1 = np.asarray(pred_v1)
print("Pred shape:", pred_v1.shape)




## === cell 4
predicted_class_indices_v1 = np.argmax(pred_v1, axis=1).astype(int)

results_v1 = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices_v1}
)

results_v1 = sample_df[["image_id"]].merge(results_v1, on="image_id", how="left")
if results_v1["label"].isna().any():
    missing = results_v1.loc[results_v1["label"].isna(), "image_id"].head(5).tolist()
    raise RuntimeError(
        f"Some test image_ids were not predicted (showing up to 5): {missing}. "
        "Check dataset paths/filenames alignment."
    )
results_v1["label"] = results_v1["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
results_v1.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(results_v1.head())
print("Submission shape:", results_v1.shape)
