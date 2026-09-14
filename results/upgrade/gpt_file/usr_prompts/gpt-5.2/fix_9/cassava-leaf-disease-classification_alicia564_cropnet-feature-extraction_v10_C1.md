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

0.8916591115140526

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

train_csv["disease"] = train_csv["label"].map(label_to_disease).astype(str)
train_csv["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_csv["image_id"].astype(str)

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["disease"].astype(str))

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


def _read_decode_resize_uint8(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.cast(img, tf.uint8)
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

    HAS_SHEAR = False
except Exception:
    HAS_SHEAR = False


def make_dataset(df, training: bool, cache_name: str):
    paths = df["path"].values.astype(str)
    labels = df["disease"].values.astype(str)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.autotune_buffers = True
    ds = ds.with_options(options)

    if training:
        shuffle_buf = min(len(df), 4096)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

    def _decode_only(path, disease_str):
        img_u8 = _read_decode_resize_uint8(path)
        return img_u8, disease_str

    ds = ds.map(_decode_only, num_parallel_calls=AUTOTUNE)

    cache_path = f"/kaggle/working/tfdata_cache_{cache_name}_u8"
    ds = ds.cache(cache_path)

    def _preprocess_and_label(img_u8, disease_str):
        img = tf.cast(img_u8, tf.float32)
        img = preprocess_input(img)  # identical preprocessing
        y = disease_lookup(disease_str)
        y = tf.one_hot(tf.cast(y, tf.int32), depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_preprocess_and_label, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    if training:
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train, training=True, cache_name="train")
valid_ds = make_dataset(valid, training=False, cache_name="valid")

print("tf.data pipelines ready:", train_ds, valid_ds)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3086056254.py in <cell line: 0>()
    105 
    106 
--> 107 train_ds = make_dataset(train, training=True, cache_name="train")
    108 valid_ds = make_dataset(valid, training=False, cache_name="valid")
    109 

/tmp/ipykernel_11/3086056254.py in make_dataset(df, training, cache_name)
     62     options.experimental_optimization.map_and_batch_fusion = True
     63     options.experimental_optimization.parallel_batch = True
---> 64     options.experimental_optimization.autotune_buffers = True
     65     ds = ds.with_options(options)
     66 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

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
    steps_per_execution=8,
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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2434736392.py in <cell line: 0>()
     26 
     27 history = model.fit(
---> 28     train_ds,
     29     validation_data=valid_ds,
     30     epochs=10,

NameError: name 'train_ds' is not defined

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
test_image_ids = sample_sub["image_id"].astype(str).tolist()

test_df = pd.DataFrame(
    {
        "image_id": test_image_ids,
        "path": [TEST_IMG_DIR.rstrip("/") + "/" + x for x in test_image_ids],
    }
)

test_paths = test_df["path"].values.astype(str)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _map_test_decode_and_preprocess(path):
    img_u8 = _read_decode_resize_uint8(path)
    img = tf.cast(img_u8, tf.float32)
    img = preprocess_input(img)
    return img


test_options = tf.data.Options()
test_options.experimental_deterministic = True
test_options.experimental_optimization.map_and_batch_fusion = True
test_options.experimental_optimization.parallel_batch = True
test_options.experimental_optimization.autotune_buffers = True
test_ds = test_ds.with_options(test_options)


def _decode_only_test(path):
    return _read_decode_resize_uint8(path)


test_ds = (
    test_ds.map(_decode_only_test, num_parallel_calls=AUTOTUNE)
    .cache("/kaggle/working/tfdata_cache_test_u8")
    .map(
        lambda img_u8: preprocess_input(tf.cast(img_u8, tf.float32)),
        num_parallel_calls=AUTOTUNE,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

pred = inference_model.predict(test_ds, verbose=1)

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

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2919849799.py in <cell line: 0>()
     24 test_options.experimental_optimization.map_and_batch_fusion = True
     25 test_options.experimental_optimization.parallel_batch = True
---> 26 test_options.experimental_optimization.autotune_buffers = True
     27 test_ds = test_ds.with_options(test_options)
     28 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.
