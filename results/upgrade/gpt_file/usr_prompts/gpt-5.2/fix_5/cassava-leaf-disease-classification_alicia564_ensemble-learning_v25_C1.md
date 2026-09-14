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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
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




## === cell 2
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label




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
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def _is_saved_model_dir(d):
    return os.path.isdir(d) and (
        os.path.exists(os.path.join(d, "saved_model.pb"))
        or os.path.exists(os.path.join(d, "saved_model.pbtxt"))
    )


_SAVED_MODEL_DIR_CACHE = {}


def find_saved_model_dir(search_root):
    """
    Returns the first directory under search_root that looks like a TensorFlow SavedModel.
    """
    if search_root in _SAVED_MODEL_DIR_CACHE:
        return _SAVED_MODEL_DIR_CACHE[search_root]

    if not os.path.exists(search_root):
        _SAVED_MODEL_DIR_CACHE[search_root] = None
        return None
    if _is_saved_model_dir(search_root):
        _SAVED_MODEL_DIR_CACHE[search_root] = search_root
        return search_root

    for root, dirs, files in os.walk(search_root):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            _SAVED_MODEL_DIR_CACHE[search_root] = root
            return root

    _SAVED_MODEL_DIR_CACHE[search_root] = None
    return None


def safe_load_tfsm_layer(possible_roots, call_endpoint="serving_default"):
    """
    Try multiple roots (dataset folders) and return a Keras Model wrapper if found.
    Otherwise return None.
    """
    try:
        from tensorflow.keras.layers import TFSMLayer
    except Exception as e:
        print("TFSMLayer not available in this environment:", e)
        return None

    for r in possible_roots:
        sm_dir = find_saved_model_dir(r)
        if sm_dir is None:
            continue
        try:
            layer = TFSMLayer(sm_dir, call_endpoint=call_endpoint)
            input_layer = Input(shape=(224, 224, 3))
            output_layer = layer(input_layer)
            model = Model(inputs=input_layer, outputs=output_layer)
            print(f"Loaded SavedModel from: {sm_dir}")
            return model
        except Exception as e:
            print(
                f"Failed to load SavedModel from candidate root '{r}' (found '{sm_dir}'):",
                repr(e),
            )
            continue
    return None


def to_numpy_logits(pred):
    """
    TFSMLayer may return a dict of outputs; normalize to a numpy array.
    """
    if isinstance(pred, dict):
        pred = list(pred.values())[0]
    if hasattr(pred, "numpy"):
        pred = pred.numpy()
    return pred




## === cell 5
cropnet_candidates = [
    "/kaggle/input/cropnet_from_kaggle",
    "/kaggle/input/cropnet-from-kaggle",
]
densenet_candidates = [
    "/kaggle/input/old_densenet",
    "/kaggle/input/old-densenet",
]
efficientnet_candidates = [
    "/kaggle/input/old_efficient_net",
    "/kaggle/input/old-efficient-net",
]

cropnet_model = safe_load_tfsm_layer(cropnet_candidates)
old_densenet_model = safe_load_tfsm_layer(densenet_candidates)
old_efficientnet_model = safe_load_tfsm_layer(efficientnet_candidates)

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def build_fallback_classifier():
    base = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    x = GlobalAveragePooling2D()(base.output)
    out = Dense(5, activation="softmax")(x)
    m = Model(inputs=base.input, outputs=out)
    return m


fallback_model = build_fallback_classifier()

if cropnet_model is None:
    cropnet_model = fallback_model
    print("Using fallback model for cropnet_model.")
if old_densenet_model is None:
    old_densenet_model = fallback_model
    print("Using fallback model for old_densenet_model.")
if old_efficientnet_model is None:
    old_efficientnet_model = fallback_model
    print("Using fallback model for old_efficientnet_model.")



## === cell 6
using_fallback = (
    (cropnet_model is fallback_model)
    or (old_densenet_model is fallback_model)
    or (old_efficientnet_model is fallback_model)
)
print("Using fallback for at least one ensemble member:", using_fallback)

if using_fallback:
    from tensorflow.keras.optimizers import Adam

    df = train_csv.copy()
    df["label_int"] = df["label"].astype(int)

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    val_frac = 0.1
    n_val = int(len(df) * val_frac)
    val_df = df.iloc[:n_val].reset_index(drop=True)
    trn_df = df.iloc[n_val:].reset_index(drop=True)

    trn_paths = trn_df["path"].astype(str).values
    trn_labels = trn_df["label_int"].values.astype(np.int32)
    val_paths = val_df["path"].astype(str).values
    val_labels = val_df["label_int"].values.astype(np.int32)

    BATCH_TRAIN = 32  # keep identical

    train_cache_path = "/kaggle/working/tf_cache_train_cassava"
    try:
        tf.io.gfile.rmtree(train_cache_path)
    except Exception:
        pass

    trn_ds = tf.data.Dataset.from_tensor_slices((trn_paths, trn_labels))
    trn_ds = (
        trn_ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache(train_cache_path)
        .batch(BATCH_TRAIN, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = (
        val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(BATCH_TRAIN, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    fallback_model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = fallback_model.fit(
        trn_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
        verbose=2,
    )



## === cell 7
import numpy as np
import pandas as pd

cropnet_weight = 0.7
densenet_weight = 0.3
efficientnet_weight = 0.0

img_size = (224, 224)

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

BATCH_SIZE = 64  # keep identical

image_ids = sample_sub["image_id"].astype(str).values
paths = np.array([os.path.join(test_image_dir, fn) for fn in image_ids], dtype=object)


def _load_image_only(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img


test_ds = tf.data.Dataset.from_tensor_slices(paths)
test_ds = (
    test_ds.map(_load_image_only, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)


@tf.function(reduce_retracing=True)
def _predict_batch(batch_imgs):
    p1 = cropnet_model(batch_imgs, training=False)
    p2 = old_densenet_model(batch_imgs, training=False)
    p3 = old_efficientnet_model(batch_imgs, training=False)
    if isinstance(p1, dict):
        p1 = list(p1.values())[0]
    if isinstance(p2, dict):
        p2 = list(p2.values())[0]
    if isinstance(p3, dict):
        p3 = list(p3.values())[0]
    avg = cropnet_weight * p1 + densenet_weight * p2 + efficientnet_weight * p3
    return tf.argmax(avg, axis=-1, output_type=tf.int64)


all_preds = []
for batch_imgs in test_ds:
    batch_classes = _predict_batch(batch_imgs).numpy()
    all_preds.append(batch_classes)

predictions = np.concatenate(all_preds, axis=0).tolist()

submission_df = pd.DataFrame({"image_id": image_ids.tolist(), "label": predictions})
submission_df["image_id"] = submission_df["image_id"].astype(str)
submission_df["label"] = submission_df["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print(f"Submission file created: {out_path}")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
