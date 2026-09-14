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

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")

print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))




## === cell 1
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf

MODEL_PATH = "../input/mdpa56/initialweightInceptionResnet4.h5"

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        0
    )  # let TF pick a good default
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass


def _build_fallback_model(num_classes: int = 5):
    base = tf.keras.applications.InceptionResNetV2(
        include_top=False, weights="imagenet", input_shape=(299, 299, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model


model = None
load_error = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Loaded model from:", MODEL_PATH)
    except Exception as e:
        load_error = e
        model = None

if model is None:
    print("WARNING: Could not load provided .h5 model. Using fallback model instead.")
    if load_error is not None:
        print("Load error:", repr(load_error))
    model = _build_fallback_model(num_classes=5)

try:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    H = input_shape[1] or 299
    W = input_shape[2] or 299
except Exception:
    H, W = 299, 299

print("Model input size:", (H, W))




## === cell 2
import numpy as np
import pandas as pd
from PIL import Image  # kept (may be unused now but preserved)

from sklearn.model_selection import train_test_split
import json

print("Imports OK")




## === cell 3
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)

train["class"] = train["label"].apply(lambda x: classes[str(x)])
print(train.head())




## === cell 4
train["path"] = train["image_id"].apply(
    lambda x: os.path.join(INPUT_DIR, "train_images", str(x))
)
train_df, val_df = train_test_split(
    train, test_size=0.05, random_state=100, stratify=train["label"].values
)
print("Train/val sizes:", len(train_df), len(val_df))




## === cell 5
tf.keras.utils.set_random_seed(123)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

batch_size = 12

train_df = train_df.copy()
val_df = val_df.copy()
train_df["label"] = train_df["label"].astype(str)
val_df["label"] = val_df["label"].astype(str)

class_names = sorted(train_df["label"].unique().tolist())
num_classes = len(class_names)
label_to_index = {c: i for i, c in enumerate(class_names)}

train_paths = train_df["path"].to_numpy()
val_paths = val_df["path"].to_numpy()
train_labels_idx = train_df["label"].map(label_to_index).to_numpy(dtype=np.int32)
val_labels_idx = val_df["label"].map(label_to_index).to_numpy(dtype=np.int32)

train_labels_oh = tf.one_hot(train_labels_idx, depth=num_classes, dtype=tf.float32)
val_labels_oh = tf.one_hot(val_labels_idx, depth=num_classes, dtype=tf.float32)


@tf.function
def _decode_resize_preprocess(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (H, W), method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    return img


AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True

TRAIN_CACHE = os.path.join(OUTPUT_DIR, "cache_train_incv2_{}_{}.cache".format(H, W))
VAL_CACHE = os.path.join(OUTPUT_DIR, "cache_val_incv2_{}_{}.cache".format(H, W))


def _make_ds(paths_np, labels_oh_tensor, training: bool):
    paths = tf.constant(paths_np)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_oh_tensor))

    if training:
        shuffle_buf = int(min(len(paths_np), 4096))
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=123, reshuffle_each_iteration=True
        )

    @tf.function
    def _load_example(path, y):
        img_bytes = tf.io.read_file(path)
        img = _decode_resize_preprocess(img_bytes)
        return img, y

    ds = ds.map(_load_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    return ds


train_ds = _make_ds(train_paths, train_labels_oh, training=True)
val_ds = _make_ds(val_paths, val_labels_oh, training=False)

train_ds = train_ds.cache(TRAIN_CACHE)
val_ds = val_ds.cache(VAL_CACHE)

train_ds = (
    train_ds.batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)
val_ds = (
    val_ds.batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)

print("tf.data train/val pipelines ready:", num_classes, "classes")




## === cell 6
os.makedirs("./checkpoints", exist_ok=True)
print("checkpoints dir ready")




## === cell 7
if getattr(model, "optimizer", None) is None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
else:
    try:
        model.compile(
            optimizer=model.optimizer,
            loss=(
                model.loss
                if getattr(model, "loss", None) is not None
                else "categorical_crossentropy"
            ),
            metrics=["accuracy"],
        )
    except Exception:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

ckpt_path = "./checkpoints/best.weights.h5"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
]

EPOCHS = 1

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists(ckpt_path):
    try:
        model.load_weights(ckpt_path)
        print("Loaded best weights from:", ckpt_path)
    except Exception as e:
        print("WARNING: could not load checkpoint weights:", repr(e))




## === cell 8
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

TEST_DIR = os.path.join(INPUT_DIR, "test_images")
print("Num test images in sample:", len(test_image_ids))
print("Example:", test_image_ids[:3])




## === cell 9
def _build_test_dataset(image_ids, batch_size: int):
    paths_np = np.fromiter(
        (os.path.join(TEST_DIR, x) for x in image_ids),
        dtype=object,
        count=len(image_ids),
    )
    paths = tf.constant(paths_np.tolist())

    @tf.function
    def _load_and_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = _decode_resize_preprocess(img_bytes)
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        _load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )

    TEST_CACHE = os.path.join(OUTPUT_DIR, "cache_test_incv2_{}_{}.cache".format(H, W))
    ds = ds.cache(TEST_CACHE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)
    return ds


bs = 32
test_ds = _build_test_dataset(test_image_ids, batch_size=bs)

probs = model.predict(test_ds, verbose=0)
predictions = probs.argmax(axis=1).astype(int).tolist()

print("Predictions generated:", len(predictions))

submission = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
