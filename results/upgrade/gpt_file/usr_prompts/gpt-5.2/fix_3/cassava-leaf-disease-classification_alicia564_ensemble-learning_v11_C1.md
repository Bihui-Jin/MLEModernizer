# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.8928679359323058

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by per-image Python overhead in the test inference loop: each image is loaded/decoded in Python and then each of the 3 models is called separately for every single image (≈ 2,676 × 3 forward passes with high dispatch overhead). I keep the same ensemble logic and weights, but batch the test images via a `tf.data` pipeline and run each model on batches, which is mathematically identical to per-image inference and preserves predictions (up to negligible FP order effects). I also cache the SavedModel directory discovery to avoid repeated `os.walk`, and ensure deterministic behavior via fixed seeds and TF determinism settings. All training/data-generator cells are kept intact (not used for submission), but the test prediction cell is rewritten to eliminate the Python loop bottleneck.'

# 9. Code solution

## === cell 0
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

print("TF version:", tf.__version__)

os.environ.setdefault("PYTHONHASHSEED", "42")
try:
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

datagen_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

train_generator = datagen_aug.flow_from_dataframe(
    dataframe=train,
    x_col="path",
    y_col="disease",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
)

datagen_valid = ImageDataGenerator(preprocessing_function=preprocess_input)

valid_generator = datagen_valid.flow_from_dataframe(
    dataframe=valid,
    x_col="path",
    y_col="disease",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)




## === cell 2
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices(
    (train["path"].values, train["label_encoded"].values)
)
valid_ds = tf.data.Dataset.from_tensor_slices(
    (valid["path"].values, valid["label_encoded"].values)
)

train_ds = (
    train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
valid_ds = (
    valid_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)




## === cell 3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)




## === cell 4
from tensorflow.keras.layers import Input, TFSMLayer
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

_SAVEDMODEL_DIR_CACHE = {}


def find_savedmodel_dir(root_path: str) -> str | None:
    """Return the first directory under root_path containing saved_model.pb, else None."""
    if not root_path:
        return None
    if root_path in _SAVEDMODEL_DIR_CACHE:
        return _SAVEDMODEL_DIR_CACHE[root_path]
    if not os.path.exists(root_path):
        _SAVEDMODEL_DIR_CACHE[root_path] = None
        return None
    if os.path.isfile(os.path.join(root_path, "saved_model.pb")):
        _SAVEDMODEL_DIR_CACHE[root_path] = root_path
        return root_path
    for dirpath, dirnames, filenames in os.walk(root_path):
        if "saved_model.pb" in filenames:
            _SAVEDMODEL_DIR_CACHE[root_path] = dirpath
            return dirpath
    _SAVEDMODEL_DIR_CACHE[root_path] = None
    return None


def build_tfsm_model(
    savedmodel_root: str, input_shape=(224, 224, 3), endpoint="serving_default"
):
    sm_dir = find_savedmodel_dir(savedmodel_root)
    if sm_dir is None:
        return None, None
    layer = TFSMLayer(sm_dir, call_endpoint=endpoint)
    inp = Input(shape=input_shape)
    out = layer(inp)
    return Model(inputs=inp, outputs=out), sm_dir


def standardize_model_output(pred):
    if isinstance(pred, dict):
        pred = list(pred.values())[0]
    if hasattr(pred, "numpy"):
        pred = pred.numpy()
    return pred


cropnet_model_root = "/kaggle/input/cropnet_model_unfrozen"
old_densenet_model_root = "/kaggle/input/old_densenet"
efficientnet_model_root = "/kaggle/input/efficientnet_model"

cropnet_model, cropnet_sm_dir = build_tfsm_model(cropnet_model_root)
old_densenet_model, densenet_sm_dir = build_tfsm_model(old_densenet_model_root)
efficientnet_model, effnet_sm_dir = build_tfsm_model(efficientnet_model_root)

print("cropnet SavedModel dir:", cropnet_sm_dir)
print("densenet SavedModel dir:", densenet_sm_dir)
print("efficientnet SavedModel dir:", effnet_sm_dir)

fallback_model = None
if cropnet_model is None or old_densenet_model is None or efficientnet_model is None:
    base = EfficientNetB0(
        include_top=True, weights="imagenet", input_shape=(224, 224, 3)
    )
    fallback_model = base
    print(
        "One or more SavedModels not found; using EfficientNetB0(ImageNet) as fallback where needed."
    )




## === cell 5
import numpy as np
import pandas as pd

image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
img_size = (224, 224)

test_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
test_paths = [os.path.join(image_dir, f) for f in test_files]

cropnet_weight = 0.5
densenet_weight = 0.3
efficientnet_weight = 0.2

BATCH_SIZE = 64  # safe throughput improvement; does not change math, only batching.


def _decode_resize_scale(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_decode_resize_scale, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)


def ensure_5_classes_np(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[0] == 1:
        p = p[0]
    if p.shape[-1] != 5:
        p = p[..., :5]
    return p.astype(np.float32, copy=False)


def model_predict_batches(model, dataset) -> np.ndarray:
    outs = []
    for batch in dataset:
        pred = model(batch, training=False)
        pred = standardize_model_output(pred)
        pred = ensure_5_classes_np(pred)
        outs.append(pred)
    return np.concatenate(outs, axis=0)


cropnet_used = cropnet_model if cropnet_model is not None else fallback_model
densenet_used = old_densenet_model if old_densenet_model is not None else fallback_model
effnet_used = efficientnet_model if efficientnet_model is not None else fallback_model

cropnet_preds = model_predict_batches(cropnet_used, test_ds)
densenet_preds = model_predict_batches(densenet_used, test_ds)
effnet_preds = model_predict_batches(effnet_used, test_ds)

avg_pred = (
    cropnet_weight * cropnet_preds
    + densenet_weight * densenet_preds
    + efficientnet_weight * effnet_preds
)
predictions = np.argmax(avg_pred, axis=1).astype(int)

submission_df = pd.DataFrame({"image_id": test_files, "label": predictions})

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(0).astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Submission file created:", out_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
