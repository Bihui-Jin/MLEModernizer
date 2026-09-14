# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8856149894227864

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The fixes remove the unavailable kaggle_datasets import, handle missing pretrained model files by falling back to a simple dummy model that predicts the most frequent class, replace the image‑loading logic (cv2 is not available) with a generator that yields zero‑filled arrays, and correct the generator loop so it stops correctly. This eliminates the runtime errors, ensures a valid submission.csv is written, and provides a baseline prediction (all‑majority class) so the notebook runs end‑to‑end.'
- What this solution (achieved 0.61099) has done: 'The script was timing out because it tried to train fallback EfficientNet‑B0 models when the pretrained files were missing, which is far too costly. I replaced that fallback with a lightweight `DummyModel` that always predicts the majority class, keeping the same interface. I also simplified the model‑loading logic to use the dummy instantly, eliminating the expensive training loop while preserving deterministic behavior. No other logic or I/O paths were altered.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter

print("Tensorflow version " + tf.__version__)

BASE_PATH = "../input/cassava-leaf-disease-classification"
if not os.path.isdir(BASE_PATH):
    BASE_PATH = "."

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUBMIT = os.path.join(BASE_PATH, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
majority_label = Counter(train_df["label"]).most_common(1)[0][0]




## === cell 1
class DummyModel:
    def __init__(self, predicted_class, num_classes=5):
        self.predicted_class = predicted_class
        self.num_classes = num_classes

    def predict(self, batches, verbose=0):
        preds = []
        for batch in batches:
            batch_size = batch.shape[0]
            prob = np.zeros((batch_size, self.num_classes), dtype=np.float32)
            prob[np.arange(batch_size), self.predicted_class] = 1.0
            preds.append(prob)
        return np.concatenate(preds, axis=0)


def build_finetune_model(input_shape=(224, 224, 3), num_classes=5):
    """EfficientNet‑B0 (ImageNet) + classification head."""
    base = tf.keras.applications.EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base.trainable = False  # freeze backbone
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def tf_dataset_from_df(df, img_dir, batch_size=32, shuffle=False):
    """Create a tf.data.Dataset yielding (image, label) pairs."""

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        return img, label

    paths = tf.constant(df["image_id"].apply(lambda x: os.path.join(img_dir, x)).values)
    labels = tf.constant(df["label"].values, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train_fallback_model():
    """Train a tiny EfficientNet‑B0 model – retained for API compatibility."""
    train_ds = tf_dataset_from_df(train_df, TRAIN_IMG_DIR, batch_size=32, shuffle=True)
    model = build_finetune_model()
    model.fit(train_ds, epochs=3, verbose=0)
    return model


def load_model_or_dummy(path, fallback_label):
    """
    Load a pretrained Keras model if it exists; otherwise train a lightweight
    EfficientNet‑B0 fine‑tuned for a few epochs (fallback) and return it.
    """
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            print(f"Could not load model at {path}: {e}")
    print(f"Training fallback model for missing {os.path.basename(path)}")
    return train_fallback_model()


dense201 = load_model_or_dummy(
    "../input/train-model-cassava/densenet201.h5", majority_label
)
inception = load_model_or_dummy(
    "../input/train-model-cassava/inceptionv3.h5", majority_label
)
efficient_net = load_model_or_dummy(
    "../input/train-model-cassava/efficient_netb3.h5", majority_label
)




## === cell 2
def generator(filepath, image_ids, batch_size=32):
    """Yield real image batches (numpy) resized to 224×224."""
    total = len(image_ids)
    idx = 0
    while idx < total:
        cur_batch = min(batch_size, total - idx)
        batch_paths = [
            os.path.join(filepath, img_id)
            for img_id in image_ids[idx : idx + cur_batch]
        ]
        batch_imgs = []
        for p in batch_paths:
            img_raw = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img_raw, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            batch_imgs.append(img.numpy())
        batch_arr = np.stack(batch_imgs, axis=0)
        idx += cur_batch
        yield batch_arr




## === cell 3
submission = pd.read_csv(SAMPLE_SUBMIT)




## === cell 4
def vote_in_ensemble(v1, v2, v3):
    if v1 == v2:
        return v1
    if v2 == v3:
        return v2
    if v1 == v3:
        return v3
    return v1




## === cell 5
def predict_for_pretrained(model):
    """Generate predictions for the test set using the given model."""
    preds = []
    for batch in generator(TEST_IMG_DIR, np.sort(submission.image_id.values)):
        batch_pred = (
            model.predict(batch, verbose=0)
            if not isinstance(model, DummyModel)
            else model.predict([batch])
        )
        preds.append(np.argmax(batch_pred, axis=-1))
    return np.concatenate(preds, axis=0)


dense_preds = predict_for_pretrained(dense201)
inception_preds = predict_for_pretrained(inception)
efficient_net_preds = predict_for_pretrained(efficient_net)

result = []
for idx in range(len(dense_preds)):
    result.append(
        vote_in_ensemble(
            dense_preds[idx], inception_preds[idx], efficient_net_preds[idx]
        )
    )



## === cell 6
submission["label"] = result
submission.to_csv("submission.csv", index=False)
