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

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.8851616802659413

# 6. Current score

0.63976

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.64387) has done: 'I fix the runtime failure that prevents the model from loading by switching protobuf to the pure-Python implementation *before* importing TensorFlow/tf_keras/keras, which addresses the `MessageFactory.GetPrototype` error in this environment. I also make the model path robust by falling back to running an in-notebook small CNN if the external `../input/my-model/...` file is not present, so the notebook always produces a valid `submission.csv`. To nudge accuracy upward (since you currently have no score at all), the fallback model train on the provided `train_images` with a simple stratified split and standard normalization, while preserving the original “predict then argmax” inference semantics. Finally, I ensure the submission format matches `sample_submission.csv` exactly and that the output file has a `.csv` suffix.'
- What this solution (achieved 0.63976) has done: 'I fix the protobuf/TensorFlow import ordering bug that’s still triggering `MessageFactory.GetPrototype` by moving the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting to the very first cell, before any TensorFlow/tf_keras (or anything that may transitively load protobuf) is imported. I also switch `BASE_DIR` to the correct Kaggle dataset mount (`/kaggle/input/...`) so the notebook can reliably find the images/CSVs in this environment. To improve the score toward your target with minimal core-logic disruption, I keep the same small CNN and training loop but train a bit longer (still a simple fit loop) and add lightweight in-model augmentation layers (doesn’t change inference semantics: still `predict -> argmax`). The script still always write a valid `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"



## === cell 1
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k): v for k, v in map_classes.items()}

print(json.dumps(map_classes, indent=4))



## === cell 2
import pandas as pd
import cv2



## === cell 3
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 4
img_shapes = {}
for image_name in os.listdir(os.path.join(BASE_DIR, "train_images"))[:300]:
    image = cv2.imread(os.path.join(BASE_DIR, "train_images", image_name))
    if image is None:
        continue
    img_shapes[image.shape] = img_shapes.get(image.shape, 0) + 1

print(img_shapes)



## === cell 5
df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()



## === cell 6
df_train["image_id"] = df_train["image_id"].astype("str")
df_train["label"] = df_train["label"].astype("str")



## === cell 7
df_train["label"].value_counts()



## === cell 8
import numpy as np
import tensorflow as tf
import tf_keras

try:
    import keras  # Keras 3.x (optional)
except Exception as e:
    keras = None
    print(
        "Warning: keras (Keras 3) import failed; will rely on tf_keras. Error:", repr(e)
    )

print("tf version:", tf.__version__)
print("tf_keras version:", getattr(tf_keras, "__version__", "unknown"))

MODEL_PATH = "../input/my-model/Cassava_best_model.h5"
final_model = None
load_errors = []

if os.path.exists(MODEL_PATH):
    try:
        final_model = tf_keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        load_errors.append(("tf_keras.models.load_model", repr(e)))

    if final_model is None and keras is not None:
        try:
            final_model = keras.saving.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            load_errors.append(("keras.saving.load_model", repr(e)))

    if final_model is None:
        print(
            "Warning: Model could not be loaded from external path; will train fallback model.\n"
            + "\n".join([f"- {src}: {err}" for src, err in load_errors])
        )
else:
    print("External model not found at:", MODEL_PATH, "-> will train fallback model.")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
try:
    if final_model is not None:
        try:
            final_model.summary()
        except Exception as e:
            print("Warning: final_model.summary() failed (non-fatal). Error:", repr(e))
except Exception:
    pass


def _infer_hw_from_model(m):
    for attr in ("input_shape", "inputs"):
        try:
            val = getattr(m, attr, None)
        except Exception:
            val = None

        if attr == "input_shape" and isinstance(val, tuple) and len(val) == 4:
            _, h, w, c = val
            return h, w, c

        if attr == "inputs" and val is not None:
            try:
                t = val[0] if isinstance(val, (list, tuple)) else val
                shp = tuple(getattr(t, "shape", ()))
                if len(shp) == 4:
                    _, h, w, c = shp
                    return h, w, c
            except Exception:
                pass

    return None, None, None


H, W, C = (None, None, None)
if final_model is not None:
    H, W, C = _infer_hw_from_model(final_model)

if H is None or W is None:
    H, W = 224, 224  # keep original fallback size
if C not in (1, 3, None):
    print(f"Warning: unexpected channel dimension C={C}; will feed RGB (3 channels).")
    C = 3

print("Using inference input size H,W,C =", H, W, C)



## === cell 10
from sklearn.model_selection import train_test_split

if final_model is None:
    SEED = 42
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train_dir = os.path.join(BASE_DIR, "train_images")
    df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["label"].astype(int)

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=SEED,
        stratify=df["label"],
    )

    def _make_ds(frame, training):
        paths = [os.path.join(train_dir, x) for x in frame["image_id"].tolist()]
        labels = frame["label"].to_numpy(dtype=np.int32)

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _load(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [int(H), int(W)], method="bilinear")
            img = tf.cast(img, tf.float32) / 255.0
            return img, label

        if training:
            ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
        return ds

    ds_train = _make_ds(train_df, training=True)
    ds_val = _make_ds(val_df, training=False)

    inputs = tf_keras.layers.Input(shape=(int(H), int(W), 3))
    x = tf_keras.layers.RandomFlip("horizontal")(inputs)
    x = tf_keras.layers.RandomRotation(0.05)(x)
    x = tf_keras.layers.RandomZoom(0.1)(x)

    x = tf_keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf_keras.layers.MaxPooling2D()(x)
    x = tf_keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf_keras.layers.MaxPooling2D()(x)
    x = tf_keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf_keras.layers.GlobalAveragePooling2D()(x)
    x = tf_keras.layers.Dropout(0.2)(x)
    outputs = tf_keras.layers.Dense(5, activation="softmax")(x)

    final_model = tf_keras.Model(inputs, outputs)

    final_model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf_keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    final_model.fit(ds_train, validation_data=ds_val, epochs=12, verbose=1)



## === cell 11
from PIL import Image

TEST_DIR = os.path.join(BASE_DIR, "test_images")

sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
test_images = sub["image_id"].astype(str).tolist()

size = (int(W), int(H))  # PIL uses (width, height)
batch_size = 32

all_preds = []
for start in range(0, len(test_images), batch_size):
    batch_files = test_images[start : start + batch_size]
    batch = np.zeros((len(batch_files), int(H), int(W), 3), dtype=np.float32)

    for i, fname in enumerate(batch_files):
        img_path = os.path.join(TEST_DIR, fname)
        img = Image.open(img_path).convert("RGB").resize(size)
        batch[i] = np.asarray(img, dtype=np.float32) / 255.0

    probs = final_model.predict(batch, verbose=0)

    probs = np.asarray(probs)
    if probs.ndim == 1:
        pred = probs.astype(int)
    else:
        pred = np.argmax(probs, axis=-1)

    all_preds.append(pred)

predictions = np.concatenate(all_preds).astype(int).tolist()

print("n_test_images:", len(test_images))
print("n_predictions:", len(predictions))



## === cell 12
if len(predictions) != len(test_images):
    raise ValueError(
        f"Prediction length mismatch: {len(predictions)} vs {len(test_images)}"
    )

submission = pd.DataFrame({"image_id": test_images, "label": predictions})
submission["image_id"] = submission["image_id"].astype(str)
submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
