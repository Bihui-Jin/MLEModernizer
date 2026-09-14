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

import numpy as np
import pandas as pd
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())

SEED = 42
tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
np.random.seed(SEED)

tf.config.experimental.enable_op_determinism(True)
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
LABEL_MAP = f"{DATA_DIR}/label_num_to_disease_map.json"

label_to_disease = pd.read_json(LABEL_MAP, typ="series")
train_csv = pd.read_csv(TRAIN_CSV)

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_csv["image_id"]

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["disease"].astype(str))

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)

class_names = sorted(train["disease"].unique().tolist())
class_indices = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_indices)
print("Num classes (from class_indices):", num_classes)
print("Class indices:", class_indices)




## === cell 2
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

disease_lookup = tf.keras.layers.StringLookup(
    vocabulary=class_names, mask_token=None, num_oov_indices=0
)


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=45.0 / 360.0, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=SEED,
        ),
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED),
    ],
    name="augmenter",
)

try:
    import tensorflow.keras.backend as K  # noqa: F401

    def _apply_shear(img, level=0.2):
        shape = tf.shape(img)
        return img

    HAS_SHEAR = False  # set False because robust on-graph shear without addons is not guaranteed
except Exception:
    HAS_SHEAR = False


def make_dataset(df, training: bool):
    paths = df["path"].values.astype(str)
    labels = df["disease"].values.astype(str)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 0
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, disease_str):
        img = _read_decode_resize(path)
        img = preprocess_input(img)  # identical preprocessing function as original
        y = disease_lookup(disease_str)  # int64 class id
        y = tf.one_hot(tf.cast(y, tf.int32), depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    if training:
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    if not training:
        ds = ds.cache()

    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train, training=True)
valid_ds = make_dataset(valid, training=False)

print("tf.data pipelines ready:", train_ds, valid_ds)




## === cell 3
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False  # minimal, stable baseline

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)

val_loss, val_acc = model.evaluate(valid_ds, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")




## === cell 4
from tensorflow.keras.callbacks import Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


print("Callbacks already configured in training cell.")




## === cell 5
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model as KModel

cropnet_model = None
model_path = "/kaggle/input/cropnet_feature_extraction_200/tensorflow2/default/1/kaggle/working/model_feature_extraction_tf"

try:
    if os.path.exists(model_path):
        from tensorflow.keras.layers import TFSMLayer

        layer = TFSMLayer(model_path, call_endpoint="serving_default")
        input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        output_layer = layer(input_layer)
        cropnet_model = KModel(inputs=input_layer, outputs=output_layer)
        print("Loaded CropNet TFSMLayer model.")
    else:
        print(
            "CropNet SavedModel path not found; using trained EfficientNet classifier for inference."
        )
except Exception as e:
    print(
        "Failed to load CropNet model; using trained EfficientNet classifier for inference. Error:",
        repr(e),
    )

inference_model = cropnet_model if cropnet_model is not None else model




## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample_sub["image_id"].tolist()

test_df = pd.DataFrame(
    {
        "image_id": test_image_ids,
        "path": [TEST_IMG_DIR.rstrip("/") + "/" + x for x in test_image_ids],
    }
)

test_paths = test_df["path"].values.astype(str)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _map_test(path):
    img = _read_decode_resize(path)
    img = preprocess_input(img)
    return img


test_options = tf.data.Options()
test_options.experimental_deterministic = True
test_ds = test_ds.with_options(test_options)

test_ds = (
    test_ds.map(_map_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .apply(tf.data.experimental.ignore_errors())
    .prefetch(AUTOTUNE)
)

steps = int(np.ceil(len(test_df) / BATCH_SIZE))
pred = inference_model.predict(test_ds, verbose=1, steps=steps)

if isinstance(pred, dict):
    first_key = list(pred.keys())[0]
    pred_np = np.asarray(pred[first_key])
else:
    pred_np = np.asarray(pred)

if pred_np.ndim == 1:
    cls = int(np.argmax(pred_np))
    pred_class_indices = np.full((len(test_df),), cls, dtype=int)
else:
    pred_class_indices = np.argmax(pred_np, axis=1).astype(int)

idx_to_disease_arr = np.array([None] * num_classes, dtype=object)
for name, idx in class_indices.items():
    idx_to_disease_arr[idx] = name

disease_to_labelnum = {v: int(k) for k, v in label_to_disease.items()}
labelnum_by_idx = np.array(
    [disease_to_labelnum.get(d, 0) for d in idx_to_disease_arr], dtype=int
)

pred_labelnums = labelnum_by_idx[pred_class_indices]

submission_df = pd.DataFrame({"image_id": test_image_ids, "label": pred_labelnums})
submission_df["image_id"] = submission_df["image_id"].astype(str)
submission_df["label"] = submission_df["label"].astype(int)
submission_df = submission_df[["image_id", "label"]]

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print("Unique predicted labels:", submission_df["label"].value_counts().to_dict())
