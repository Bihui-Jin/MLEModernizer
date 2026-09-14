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

0.8822907222725899

# 6. Current score

0.65172

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.65172) has done: 'I fix two execution blockers so the notebook runs end-to-end and writes `submission.csv`: (1) the TensorFlow import crash caused by an incompatible protobuf runtime, and (2) the NumPy string concatenation error when building `test_paths`. To keep core modeling logic unchanged, I only add a safe environment pin (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) before importing TensorFlow and switch path construction to a robust `os.path.join` list/array. These changes are score-neutral (they don’t change training/inference semantics) but ensure you actually get a valid submission file. I also keep the existing optional pretrained-model loading and the fallback training path intact.'
- What this solution (achieved 0.65172) has done: 'The crash happens before any training/inference because the installed TensorFlow/protobuf combination is incompatible, producing `MessageFactory.GetPrototype` errors; the safest minimal fix in Kaggle is to force the pure-Python protobuf implementation and disable the C++ fast path *before any TensorFlow import*. I also add a small compatibility fallback that sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is commonly required in these environments, without changing your model/training logic. Everything else (data loading, model architecture, training loop, inference pipeline, and submission formatting) is kept the same so behavior is unchanged except that it now runs end-to-end and writes `submission.csv`. With TensorFlow importing correctly again, your pretrained-model loading (if present) and/or fallback training execute normally, which should move your score up toward the target versus a broken/forced-degraded run.'
- What this solution (achieved 0.65172) has done: 'We fix the TensorFlow import crash by setting the protobuf environment variables **before any other imports** and by additionally disabling the C++ protobuf fast-path, which addresses the `MessageFactory.GetPrototype` AttributeError in Kaggle’s TF/protobuf combo. This is an execution blocker; once TF imports, the rest of your pipeline (optional pretrained model load, otherwise the same fallback CNN training and inference) can run unchanged. I also keep your existing robust `os.path.join` path building and ensure the submission is written as `submission.csv` with the required `image_id,label` columns. No model/training logic is altered beyond what’s needed to make it run end-to-end reliably.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX", "1")

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

ROOT_DIR = "../input/cassava-leaf-disease-classification/"
print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("ROOT_DIR files:", os.listdir(ROOT_DIR)[:10])



## === cell 1
import tensorflow as tf
from PIL import Image  # kept for compatibility with original logic

tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE
print("TensorFlow:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images")
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")

assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isfile(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

print("Num test images:", len(os.listdir(TEST_DIR)))
print("Num train images:", len(os.listdir(TRAIN_DIR)))



## === cell 3
CANDIDATE_MODEL_PATHS = [
    "../input/cassava-leaf-disease-first-look-and-training/best_model.hdf5",
    "../input/cassava-leaf-disease-first-look-and-training/best_model.h5",
    "../input/best_model.hdf5",
    "../input/best_model.h5",
]

new_model = None
for p in CANDIDATE_MODEL_PATHS:
    if os.path.exists(p):
        print("Found model file:", p)
        new_model = tf.keras.models.load_model(p, compile=False)
        break

if new_model is None:
    print(
        "No pretrained model found. Training a small fallback model from train_images/train.csv..."
    )

    df = pd.read_csv(TRAIN_CSV)
    num_classes = int(df["label"].nunique())
    assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

    IMG_SIZE = 300
    size = (IMG_SIZE, IMG_SIZE)

    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    y_all = df["label"].values.astype(np.int64)
    idx_train, idx_val = train_test_split(
        idx, test_size=0.1, random_state=SEED, stratify=y_all
    )

    train_paths = (TRAIN_DIR + "/" + df.loc[idx_train, "image_id"].values).astype(str)
    val_paths = (TRAIN_DIR + "/" + df.loc[idx_val, "image_id"].values).astype(str)
    y_train = y_all[idx_train]
    y_val = y_all[idx_val]

    def _decode_resize_norm(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(
            img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    BATCH_SIZE = 32

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, y_train))
    train_ds = train_ds.shuffle(
        buffer_size=len(train_paths), seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(_decode_resize_norm, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, y_val))
    val_ds = val_ds.map(_decode_resize_norm, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    new_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )

    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    new_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        verbose=2,
    )



## === cell 4
new_model.summary()



## === cell 5
if "IMG_SIZE" not in globals():
    IMG_SIZE = 300
size = (IMG_SIZE, IMG_SIZE)
print("Inference IMG_SIZE:", IMG_SIZE)



## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
test_images = sample_sub["image_id"].tolist()

missing = [fn for fn in test_images if not os.path.exists(os.path.join(TEST_DIR, fn))]
print("Missing test files:", len(missing))
assert (
    len(missing) == 0
), "Some test images referenced in sample_submission are missing on disk."



## === cell 7
test_paths = np.array([os.path.join(TEST_DIR, fn) for fn in test_images], dtype=str)


def _decode_resize_norm_infer(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


INFER_BATCH_SIZE = 64

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_decode_resize_norm_infer, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(INFER_BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

probs = new_model.predict(test_ds, verbose=0)
preds = probs.argmax(axis=1).astype(int).tolist()

print("Num preds:", len(preds), "Num test_images:", len(test_images))



## === cell 8
preds[:20], pd.Series(preds).value_counts().sort_index()



## === cell 9
assert len(test_images) == len(preds), "Prediction count mismatch with test images."

sub = pd.DataFrame({"image_id": test_images, "label": preds})
print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
