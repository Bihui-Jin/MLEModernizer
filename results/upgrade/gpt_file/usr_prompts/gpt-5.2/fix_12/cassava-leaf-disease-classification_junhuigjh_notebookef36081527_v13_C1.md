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

0.7893623451193714

# 6. Current score

0.67564

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.64462) has done: 'I fix the initial TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation before importing TensorFlow (this unblocks everything). Next, I remove the hard dependency on an external uploaded model (`/kaggle/input/abc/...`) by training the same simple Keras model architecture inside the notebook using the provided `train_tfrecords`, then run inference on `test_tfrecords`. Finally, I keep the existing TFRecord decode → PIL resize/normalize → `model.predict` → `argmax` core inference semantics, and ensure the output is merged onto `sample_submission.csv` and written as `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.64163) has done: 'I fix the TensorFlow import crash by pinning protobuf to the pure-Python implementation *and* forcing the classic API, which resolves the `MessageFactory.GetPrototype` error in newer protobuf runtimes. Then I remove the PIL-based per-image preprocessing/predict loop (which is slow and slightly inconsistent with training) and instead run batched inference directly from the decoded TFRecord dataset using the same `_decode_and_resize` pipeline as training. Finally, I keep the same model architecture and training loop, but make inference deterministic, faster, and correctly aligned to `sample_submission.csv`, writing a valid `submission.csv`.'
- What this solution (achieved 0.10762) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime and disabling the C++ implementation before TensorFlow is imported, and by setting `TF_USE_LEGACY_KERAS=1` for compatibility in newer Kaggle images. I also make the dataset root detection robust to your provided folder layout (including `/kaggle/data/...` and `/kaggle/input/.../cassava-leaf-disease-classification`). To nudge accuracy upward toward the target without changing the model architecture or training loop, I add lightweight, standard training-time augmentation (flip/brightness/contrast) and label smoothing in the same loss (still sparse categorical cross-entropy semantics). Finally, I keep batched TFRecord inference and ensure the submission is aligned to `sample_submission.csv` and written as `submission.csv`.'
- What this solution (achieved 0.64312) has done: 'I fix the two hard runtime issues that prevent training/inference: (1) the TensorFlow import crash caused by protobuf API mismatch by also forcing the pure-Python protobuf backend via `google.protobuf.internal.api_implementation`, and (2) the `label_smoothing` argument error by switching to a loss implementation that supports label smoothing while keeping sparse labels and the same cross-entropy semantics. These changes are minimal and keep the same model architecture, TFRecord pipeline, and training loop intact. After that, the script train, run batched TFRecord inference, and write a correctly formatted `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.64088) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow-related imports, and by additionally setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus a safe monkeypatch for the missing `MessageFactory.GetPrototype` method that some TF/protobuf combinations still call. This is a runtime-only compatibility patch and does not change your model, data pipeline, or training/inference logic. After TF imports cleanly, the rest of your training on `train_tfrecords`, batched inference on `test_tfrecords`, and submission merge/write run end-to-end and produce `submission.csv`. No score-tuning changes are introduced beyond unblocking execution and keeping determinism the same.'
- What this solution (achieved 0.6917) has done: 'The timeout is dominated by forcing the pure-Python protobuf implementation (very slow for TFRecords) and by building tf.data pipelines that don’t cache decoded/resized images, causing repeated expensive JPEG decode/resize across epochs and across the second “full” training phase. I switch protobuf back to the default (C++/upb) implementation, add safe in-memory caching of the parsed (but not augmented) images/labels so decode/resize happens once, and ensure the training/val/full datasets reuse the same cached base dataset. I also eliminate per-batch `model2.predict()` calls (which incur overhead) by calling the model directly in inference, and vectorize filename decoding. These changes keep the same model, losses, epochs, augmentations, and data split semantics while cutting constant-factor overhead enough to fit within 600 seconds.'
- What this solution (achieved 0.69432) has done: 'I fix the runtime crash in the very first cell by applying a small protobuf compatibility patch *before* importing TensorFlow, which addresses the `MessageFactory.GetPrototype` AttributeError seen in newer protobuf runtimes. I keep your model, tf.data pipelines, training loop, augmentation, and label-smoothing semantics unchanged, only adjusting imports/environment to ensure TensorFlow loads reliably under Python 3.13. I also keep the same dataset root discovery and submission merge logic, ensuring `submission.csv` is written with the correct columns and row count. These changes are score-neutral but unblock end-to-end execution so you can iterate toward the target score.'
- What this solution (achieved 0.6932) has done: 'I fix the TensorFlow import crash by strengthening the protobuf compatibility patch so `MessageFactory.GetPrototype` exists both on the class and any already-instantiated objects that TensorFlow/protobuf might use. This is a runtime-only change that does not alter your model, data pipeline, or training/inference semantics, but unblocks the notebook under the Python 3.13 + newer protobuf environment. I also keep all paths and submission-writing logic identical, ensuring `submission.csv` is produced with the required columns and row count. No score-tuning changes are introduced beyond making the existing approach run reliably end-to-end.'
- What this solution (achieved 0.67564) has done: 'Your current score (0.6932) is below the target (0.78936), so we should make a small, metric-aligned improvement without changing the model architecture or overall training/inference approach. The biggest score drag in your script is that you apply random augmentation to the *final full-data training phase* (`ds_full`), which injects label noise and can reduce accuracy; we keep augmentation for the train split but disable it for the final fit so the model better matches clean test images. We also make the image resize use the common “crop or pad to square then resize” approach (still 224×224 and same normalization) to reduce distortion vs. plain resize, which often improves accuracy for leaf datasets. Finally, we keep everything else (splits, epochs, loss, label smoothing, inference, submission merge) identical and deterministic.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

try:
    from google.protobuf import message_factory as _message_factory

    def _ensure_getprototype_on_messagefactory():
        MF = _message_factory.MessageFactory

        if not hasattr(MF, "GetPrototype") and hasattr(MF, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            try:
                MF.GetPrototype = _GetPrototype
            except Exception:
                pass

        try:
            _default_factory = getattr(_message_factory, "_DEFAULT_FACTORY", None)
            if _default_factory is not None and not hasattr(
                _default_factory, "GetPrototype"
            ):
                if hasattr(_default_factory, "GetMessageClass"):
                    _default_factory.GetPrototype = (
                        lambda descriptor, _f=_default_factory: _f.GetMessageClass(
                            descriptor
                        )
                    )
        except Exception:
            pass

        if not hasattr(MF, "GetPrototype"):

            def _GetPrototype_stub(self, descriptor):
                raise AttributeError(
                    "MessageFactory.GetPrototype is unavailable in this protobuf runtime."
                )

            try:
                MF.GetPrototype = _GetPrototype_stub
            except Exception:
                pass

    _ensure_getprototype_on_messagefactory()
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not locate dataset root. Tried: {DATA_ROOT_CANDIDATES}"
    )

TRAIN_TFRECORDS_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFRECORDS_DIR = os.path.join(DATA_ROOT, "test_tfrecords")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

if not os.path.isdir(TRAIN_TFRECORDS_DIR):
    raise FileNotFoundError(f"Missing train_tfrecords at: {TRAIN_TFRECORDS_DIR}")
if not os.path.isdir(TEST_TFRECORDS_DIR):
    raise FileNotFoundError(f"Missing test_tfrecords at: {TEST_TFRECORDS_DIR}")
if not os.path.exists(TRAIN_CSV_PATH):
    raise FileNotFoundError(f"Missing train.csv at: {TRAIN_CSV_PATH}")
if not os.path.exists(SAMPLE_SUB_PATH):
    raise FileNotFoundError(f"Missing sample_submission.csv at: {SAMPLE_SUB_PATH}")

feature_description_train = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
feature_description_test = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

print("Using DATA_ROOT:", DATA_ROOT)
print("TF version:", tf.__version__)



## === cell 1
train_files = sorted(
    [
        os.path.join(TRAIN_TFRECORDS_DIR, f)
        for f in os.listdir(TRAIN_TFRECORDS_DIR)
        if (f.endswith(".tfrec") or f.endswith(".tfrecord") or f.startswith("ld_train"))
    ]
)
test_files = sorted(
    [
        os.path.join(TEST_TFRECORDS_DIR, f)
        for f in os.listdir(TEST_TFRECORDS_DIR)
        if (f.endswith(".tfrec") or f.endswith(".tfrecord") or f.startswith("ld_test"))
    ]
)

if len(train_files) == 0:
    raise FileNotFoundError(f"No TFRecord files found in {TRAIN_TFRECORDS_DIR}")
if len(test_files) == 0:
    raise FileNotFoundError(f"No TFRecord files found in {TEST_TFRECORDS_DIR}")

IMG_SIZE = 224
NUM_CLASSES = 5
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize_with_crop_or_pad(
        img,
        tf.maximum(tf.shape(img)[0], tf.shape(img)[1]),
        tf.maximum(tf.shape(img)[0], tf.shape(img)[1]),
    )
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_brightness(img, max_delta=0.08, seed=SEED)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1, seed=SEED)
    return tf.clip_by_value(img, 0.0, 1.0)


def parse_train_example(raw_record):
    ex = tf.io.parse_single_example(raw_record, feature_description_train)
    img = _decode_and_resize(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return img, y


def parse_test_example(raw_record):
    ex = tf.io.parse_single_example(raw_record, feature_description_test)
    img = _decode_and_resize(ex["image"])
    image_name = ex["image_name"]
    return img, image_name


ds_all_base = tf.data.TFRecordDataset(train_files, num_parallel_reads=AUTOTUNE).map(
    parse_train_example, num_parallel_calls=AUTOTUNE
)
ds_all_base = ds_all_base.cache()
ds_all = ds_all_base.shuffle(8192, seed=SEED, reshuffle_each_iteration=False)

ds_train = (
    ds_all.shard(num_shards=10, index=0)
    .concatenate(ds_all.shard(num_shards=10, index=1))
    .concatenate(ds_all.shard(num_shards=10, index=2))
    .concatenate(ds_all.shard(num_shards=10, index=3))
    .concatenate(ds_all.shard(num_shards=10, index=4))
    .concatenate(ds_all.shard(num_shards=10, index=5))
    .concatenate(ds_all.shard(num_shards=10, index=6))
    .concatenate(ds_all.shard(num_shards=10, index=7))
    .concatenate(ds_all.shard(num_shards=10, index=8))
)
ds_val = ds_all.shard(num_shards=10, index=9)

ds_train = ds_train.map(lambda x, y: (_augment(x), y), num_parallel_calls=AUTOTUNE)

ds_train = (
    ds_train.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
ds_val = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model2 = tf.keras.Model(inputs, outputs)

LABEL_SMOOTHING = 0.05


def to_one_hot(img, y):
    y_oh = tf.one_hot(tf.cast(y, tf.int32), depth=NUM_CLASSES, dtype=tf.float32)
    return img, y_oh


ds_train_oh = ds_train.map(to_one_hot, num_parallel_calls=AUTOTUNE)
ds_val_oh = ds_val.map(to_one_hot, num_parallel_calls=AUTOTUNE)

model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"],
)

EPOCHS = 8
history = model2.fit(ds_train_oh, validation_data=ds_val_oh, epochs=EPOCHS, verbose=2)

ds_full = ds_all_base.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds_full_oh = ds_full.map(to_one_hot, num_parallel_calls=AUTOTUNE)

FINAL_EPOCHS = 3
_ = model2.fit(ds_full_oh, epochs=FINAL_EPOCHS, verbose=2)



## === cell 2
test_ds = (
    tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE)
    .map(parse_test_example, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

image_ids = []
prediction = []

for batch_imgs, batch_names in test_ds:
    probs = model2(batch_imgs, training=False).numpy()
    preds = np.argmax(probs, axis=1).astype(int)

    names = batch_names.numpy().astype("U")  # vectorized decode to unicode
    image_ids.extend(names.tolist())
    prediction.extend(preds.tolist())

if len(image_ids) != len(prediction):
    raise RuntimeError(
        f"Mismatch: got {len(image_ids)} ids but {len(prediction)} predictions"
    )

print("Predictions:", len(prediction), "Unique IDs:", len(set(image_ids)))



## === cell 3
sample = pd.read_csv(SAMPLE_SUB_PATH)

pred_df = pd.DataFrame({"image_id": image_ids, "label": prediction})
pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first")

submission = sample[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():
    fill_val = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    submission["label"] = submission["label"].fillna(fill_val).astype(int)
else:
    submission["label"] = submission["label"].astype(int)

submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(submission.head())
print("Rows:", len(submission), "Expected:", len(sample))
if len(submission) != len(sample):
    raise RuntimeError("Submission row count does not match sample_submission.csv")
