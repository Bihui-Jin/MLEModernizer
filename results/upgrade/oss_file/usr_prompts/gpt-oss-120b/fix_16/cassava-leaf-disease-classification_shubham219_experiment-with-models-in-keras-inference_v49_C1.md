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
import os, random, glob
import pandas as pd, numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

try:
    import tensorflow_addons as tfa

    TFA_AVAILABLE = True
except Exception:
    TFA_AVAILABLE = False

SEED = 42
BATCH_SIZE = 64
IMG_SIZE = (224, 224)  # EfficientNetB0 default size
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

BASE_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
train_df["label"] = train_df["label"].astype(str)
train_df["filepath"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x)
)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=SEED,
)

label_names = sorted(train_df["label"].unique())
label_num_classes = len(label_names)
label_to_index = {name: idx for idx, name in enumerate(label_names)}
train_split["label_idx"] = train_split["label"].map(label_to_index)
val_split["label_idx"] = val_split["label"].map(label_to_index)




## === cell 1
def decode_image(path):
    """Read JPEG file and resize to target size (no augmentation)."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image


def augment_image(image):
    """Apply the same random augmentations as the original pipeline."""
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    if TFA_AVAILABLE:
        angle = tf.random.uniform([], -20.0, 20.0, seed=SEED) * (3.14159265 / 180.0)
        image = tfa.image.rotate(image, angle, fill_mode="nearest")
    zoom = tf.random.uniform([], 0.8, 1.2, seed=SEED)
    new_size = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * zoom), tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0], IMG_SIZE[1])
    return image


def preprocess_image(image):
    """EfficientNet‑B0 preprocessing (same as original)."""
    return tf.keras.applications.efficientnet.preprocess_input(image)


def decode_and_preprocess(path, label=None, training=False):
    """Fallback kept for test pipeline; unchanged logic."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)

    if training:
        image = tf.image.random_flip_left_right(image, seed=SEED)
        image = tf.image.random_flip_up_down(image, seed=SEED)
        if TFA_AVAILABLE:
            angle = tf.random.uniform([], -20.0, 20.0, seed=SEED) * (3.14159265 / 180.0)
            image = tfa.image.rotate(image, angle, fill_mode="nearest")
        zoom = tf.random.uniform([], 0.8, 1.2, seed=SEED)
        new_size = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * zoom), tf.int32)
        image = tf.image.resize(image, new_size)
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0], IMG_SIZE[1])

    image = tf.keras.applications.efficientnet.preprocess_input(image)
    if label is None:
        return image
    else:
        label_onehot = tf.one_hot(label, depth=label_num_classes)
        return image, label_onehot




## === cell 2
import tempfile

train_cache_file = os.path.join(tempfile.gettempdir(), "train_ds_cache")
val_cache_file = os.path.join(tempfile.gettempdir(), "val_ds_cache")

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_split["filepath"].values, train_split["label_idx"].values)
)

train_ds = train_ds.shuffle(
    buffer_size=len(train_split), seed=SEED, reshuffle_each_iteration=False
)

train_ds = train_ds.map(
    lambda p, l: (decode_image(p), l),
    num_parallel_calls=AUTOTUNE,
    deterministic=False,
)
train_ds = train_ds.cache(train_cache_file)  # cache decoded images
train_ds = (
    train_ds.map(
        lambda img, l: (augment_image(img), tf.one_hot(l, depth=label_num_classes)),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    .map(
        lambda img, l: (preprocess_image(img), l),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_split["filepath"].values, val_split["label_idx"].values)
)

val_ds = val_ds.map(
    lambda p, l: (decode_image(p), tf.one_hot(l, depth=label_num_classes)),
    num_parallel_calls=AUTOTUNE,
    deterministic=False,
)
val_ds = val_ds.cache(val_cache_file)
val_ds = (
    val_ds.map(
        lambda img, l: (preprocess_image(img), l),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)




## === cell 3
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
)
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(5, activation="softmax")(x)  # 5 classes
model = Model(inputs=base_model.input, outputs=output)

base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)




## === cell 4
INITIAL_EPOCHS = 8
model.fit(train_ds, epochs=INITIAL_EPOCHS, validation_data=val_ds, verbose=2)

base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.8)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

FINE_TUNE_EPOCHS = 5
model.fit(train_ds, epochs=FINE_TUNE_EPOCHS, validation_data=val_ds, verbose=2)




## === cell 5
test_images = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
test_df = pd.DataFrame({"filepath": test_images})

test_ds = tf.data.Dataset.from_tensor_slices(test_df["filepath"].values)
test_ds = test_ds.map(
    lambda p: decode_and_preprocess(p, training=False),
    num_parallel_calls=AUTOTUNE,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

preds = model.predict(test_ds, verbose=2)
pred_labels = np.argmax(preds, axis=1)

submission = pd.DataFrame(
    {
        "image_id": test_df["filepath"].apply(lambda x: os.path.basename(x)),
        "label": pred_labels,
    }
)
submission.to_csv("submission.csv", index=False)
