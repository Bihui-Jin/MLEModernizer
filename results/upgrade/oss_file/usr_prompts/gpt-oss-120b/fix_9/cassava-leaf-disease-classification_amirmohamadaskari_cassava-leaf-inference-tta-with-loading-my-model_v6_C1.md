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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

tf.config.optimizer.set_jit(True)

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import random




## === cell 1
def get_strategy():
    """
    Detects and returns the best TensorFlow distribution strategy:
    - TPUStrategy for TPU(s)
    - MirroredStrategy for GPU(s)
    - Default strategy for CPU
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect TPU
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Using TPU strategy:", type(strategy).__name__)
    except (ValueError, tf.errors.NotFoundError):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            print("Using GPU strategy:", type(strategy).__name__)
        else:
            strategy = tf.distribute.get_strategy()
            print("No TPU/GPU found. Using default strategy:", type(strategy).__name__)
    print("REPLICAS:", strategy.num_replicas_in_sync)
    return strategy


strategy = get_strategy()




## === cell 2
AUTO = tf.data.AUTOTUNE
IMAGE_SIZE = (512, 512)
BATCH_SIZE_PER_REPLICA = 32
NUM_CLASSES = 5
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
print(f"Global Batch size: {BATCH_SIZE}")




## === cell 3
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
MODEL_DIR = (
    "/kaggle/input/cassava-leaf-model/tensorflow2/default/1/final_model_cassava.keras"
)




## === cell 4
from tensorflow.keras import mixed_precision


def seed_everthing(SEED=28):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    print(f"Global seed set to {SEED} 🌱")


seed_everthing()

mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision policy set to:", mixed_precision.global_policy())




## === cell 5
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

train_image_paths = (
    train_df["image_id"]
    .apply(lambda x: os.path.join(DATA_DIR, "train_images", x))
    .values
)
train_labels = train_df["label"].values.astype(np.int32)


def _load_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    label_onehot = tf.one_hot(label, NUM_CLASSES)
    return image, label_onehot


train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_ds = train_ds.map(
    _load_preprocess,
    num_parallel_calls=AUTO,
    deterministic=False,  # Allows nondeterministic order for speed
)

train_ds = train_ds.cache(filename="/tmp/train_cache")
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTO)

base = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,)
)
x = GlobalAveragePooling2D()(base.output)
output = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base.input, outputs=output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_ds, epochs=2, verbose=2)




## === cell 6
def decode_test_example(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)

    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)

    return image, example["image_name"]




## === cell 7
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess(image, label):
    image = preprocess_input(image)
    return image, label




## === cell 8
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(40 / 360),  # Randomly rotate images
        tf.keras.layers.RandomTranslation(
            0.2, 0.2
        ),  # Randomly shift images horizontally and vertically
        tf.keras.layers.RandomZoom(0.2, 0.2),  # Randomly zoom into images
        tf.keras.layers.RandomFlip("horizontal"),  # Randomly flip images horizontally
        tf.keras.layers.RandomFlip("vertical"),  # Randomly flip images vertically
    ],
    name="data_augmentation",
)




## === cell 9
TEST_DIR = os.path.join(DATA_DIR, "test_tfrecords")
test_files = tf.io.gfile.glob(os.path.join(TEST_DIR, "*.tfrec"))
test_dataset = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTO)
test_dataset = (
    test_dataset.map(decode_test_example, num_parallel_calls=AUTO, deterministic=False)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = test_dataset.cache(filename="/tmp/test_cache")

print("Test dataset created Successfully !")




## === cell 10
def apply_test_augmentation(image):
    image = tf.expand_dims(image, 0)
    image = data_augmentation(image)
    image = tf.squeeze(image, 0)
    return image




## === cell 11
def tta_predict_image(model, image, num_tta):
    augmented_images = tf.stack(
        [apply_test_augmentation(image) for _ in range(num_tta)], axis=0
    )
    preprocessed_images = preprocess_input(augmented_images)
    predictions = model(preprocessed_images, training=False)
    mean_predictions = tf.reduce_mean(predictions, axis=0)
    return mean_predictions




## === cell 12
@tf.function  # graph‑mode execution speeds up the heavy TTA loop
def tta_predict_batch(model, images, num_tta):
    """
    images: Tensor of shape (B, H, W, C) – raw uint8 images
    Returns: Tensor (B, NUM_CLASSES) of averaged predictions.
    """
    repeated = tf.repeat(images, repeats=num_tta, axis=0)
    augmented = data_augmentation(repeated)
    preprocessed = preprocess_input(augmented)
    preds = model(preprocessed, training=False)  # (B*num_tta, NUM_CLASSES)
    B = tf.shape(images)[0]
    preds = tf.reshape(preds, (B, num_tta, NUM_CLASSES))
    mean_preds = tf.reduce_mean(preds, axis=1)  # (B, NUM_CLASSES)
    return mean_preds


tta_num_augmentations = 10  # Number of augmented images to create per test image
tta_predictions = []
tta_image_names = []

for images, ids in test_dataset:
    batch_mean_preds = tta_predict_batch(model, images, tta_num_augmentations).numpy()
    tta_predictions.append(batch_mean_preds)
    tta_image_names.extend([i.numpy().decode("utf-8") for i in ids])

tta_predictions = np.concatenate(
    tta_predictions, axis=0
)  # shape: (num_images, NUM_CLASSES)




## === cell 13
final_tta_labels = np.argmax(tta_predictions, axis=1)
pred_labels = final_tta_labels.astype(int)




## === cell 14
submission_df = pd.DataFrame({"image_id": tta_image_names, "label": pred_labels})

submission_df.to_csv("submission.csv", index=False)
print("Submission file created successfully!")
