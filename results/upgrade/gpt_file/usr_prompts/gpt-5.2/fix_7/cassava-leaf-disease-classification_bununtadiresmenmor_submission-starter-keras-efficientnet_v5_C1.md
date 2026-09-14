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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.637503777576307

# 6. Current score

0.77504

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.77504) has done: 'I fix the environment-breaking import error by removing nonessential imports that trigger the protobuf incompatibility, and I ensure the notebook uses the correct Kaggle dataset paths you actually have. Because the referenced pre-trained `.h5` file doesn’t exist, I add a minimal fallback that trains a small EfficientNet model on `train.csv` + `train_images/` and then predicts on `test_images/`, preserving the image-classification approach and producing a valid `submission.csv`. I also make the inference loop efficient and deterministic enough to finish within the time limit, and I guarantee the submission columns and ordering match `sample_submission.csv`.'
- What this solution (achieved 0.77504) has done: 'I fix the runtime crash happening at import time (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`, which is the standard workaround for this protobuf 6.x incompatibility. I also renumber your cells to start at `## === cell 1` so the script matches the required format, while keeping your model/training/inference logic identical. Finally, I keep the submission-writing code unchanged except for adding a tiny safety check to ensure predictions align with `sample_submission.csv` ordering and produce a valid `submission.csv`.'
- What this solution (achieved 0.77504) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x `MessageFactory.GetPrototype` incompatibility by forcing the pure-Python protobuf implementation *before* any TensorFlow-related imports, and by defensively importing `google.protobuf` first so the env var is honored. I also keep the existing model/training/inference logic intact, only adding a small safety fallback to use `sample_submission.csv` from either base path and ensuring consistent ordering/alignment. Since your current score (0.77504) is already well above the target (0.6375), I not make any modeling changes that could further increase score; the goal here is correctness and a valid `submission.csv` output.'
- What this solution (achieved 0.77541) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf environment variables before *any* protobuf-related import and by avoiding an explicit `google.protobuf` import that can lock in the wrong implementation. I also add a safe fallback to force the pure-Python protobuf at runtime if TensorFlow still fails to import, so the notebook always runs end-to-end in this environment. Since your current score (0.77504) is already well above the target (0.6375), I not change the model/training/inference logic to avoid unnecessary score shifts. Finally, I renumber cells to start at 1 and keep submission writing unchanged while ensuring it always produces `submission.csv`.'
- What this solution (achieved 0.77504) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf environment variables before any protobuf/TensorFlow import and by retrying the TensorFlow import after clearing already-loaded protobuf/tensorflow modules (the current try/except doesn’t catch the right failure mode). I keep the model/training/inference logic the same so performance stays in the same regime (and since your current score is already above the target, we avoid score-changing tweaks). I also renumber the cells to start at 1 (your current script starts at cell 0) while keeping code order identical, and ensure `submission.csv` is always written with the required columns and length.'
- What this solution (achieved 0.77504) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *before any TensorFlow/protobuf-related import occurs*, and by importing TensorFlow directly (no retry logic that can re-import after protobuf is already initialized). This change is purely to unblock execution and is intended to be score-neutral (it doesn’t alter the model/training/inference logic). I also renumber the cells to start at `## === cell 1` as required, keeping code order and behavior the same. Finally, I keep the submission writing intact and ensure it always produces a valid `submission.csv` with the required columns and row order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import random
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(SEED)

BASE1 = "/kaggle/input/cassava-leaf-disease-classification"
BASE2 = "/kaggle/data/cassava-leaf-disease-classification"
BASE = BASE1 if os.path.exists(BASE1) else BASE2

TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir at {TEST_IMG_DIR}"

print("Using BASE:", BASE)
print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_PATH = "/kaggle/input/fork-of-starter-keras-efficientnet/firstTry.h5"

model = None
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Loaded pretrained model:", MODEL_PATH)
else:
    print("Pretrained model not found at:", MODEL_PATH)
    print("Falling back to training a compact EfficientNet model from train_images.")



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
ss = pd.read_csv(SAMPLE_SUB)

NUM_CLASSES = int(train_df["label"].nunique())
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"

IMG_SIZE = 300  # keep consistent with original resize to (300,300)
BATCH_SIZE = 16
EPOCHS = 2  # small to fit 600s; without a provided pretrained model, this is necessary


def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img


def make_train_ds(df, training=True):
    paths = tf.constant([os.path.join(TRAIN_IMG_DIR, x) for x in df["image_id"].values])
    labels = tf.constant(df["label"].values, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img = decode_image(p)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        y = tf.one_hot(y, NUM_CLASSES)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def make_pred_ds(image_ids):
    paths = tf.constant([os.path.join(TEST_IMG_DIR, x) for x in image_ids])
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(p):
        img = decode_image(p)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


val_frac = 0.1
train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(train_df_shuf) * val_frac)
val_df = train_df_shuf.iloc[:n_val].reset_index(drop=True)
trn_df = train_df_shuf.iloc[n_val:].reset_index(drop=True)

train_ds = make_train_ds(trn_df, training=True)
val_ds = make_train_ds(val_df, training=False)

print("Train/val sizes:", len(trn_df), len(val_df))



## === cell 3
if model is None:
    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    base.trainable = False  # fast, stable baseline

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=2,
    )



## === cell 4
test_image_ids = ss["image_id"].values
test_ds = make_pred_ds(test_image_ids)

probs = model.predict(test_ds, verbose=0)
preds = np.argmax(probs, axis=1).astype(int)

assert len(preds) == len(
    test_image_ids
), "Prediction count mismatch vs sample_submission"

submission = pd.DataFrame({"image_id": test_image_ids, "label": preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert os.path.exists("submission.csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(ss)



## === cell 5
submission
