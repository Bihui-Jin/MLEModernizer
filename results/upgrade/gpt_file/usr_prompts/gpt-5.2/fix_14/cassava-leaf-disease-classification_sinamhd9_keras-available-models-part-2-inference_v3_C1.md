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

0.8398

# 6. Current score

0.57025

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.63528) has done: 'I fix the environment-breaking import issue by using `tf_keras` (the Kaggle-installed TensorFlow-Keras) instead of standalone `keras`, which avoids the protobuf `MessageFactory.GetPrototype` crash. Since the referenced pre-trained model file path doesn’t exist in your environment, I replace that with a minimal, fast-to-run baseline that legitimately trains on the provided `train.csv` + `train_images` and then predicts on `test_images`. I also fix the submission construction by using `sample_submission.csv` to guarantee correct `image_id` ordering and matching lengths, then write a valid `submission.csv`. The core objective (5-class image classification with accuracy metric) is preserved, and the pipeline run end-to-end within the time limit.'
- What this solution (achieved 0.05531) has done: 'The timeout is dominated by Python/PIL image decoding and resizing inside a pure-Python generator for every batch/epoch, plus repeated `df.loc` lookups and per-image file path joins. I keep the exact model and training loop semantics, but move the input pipeline to `tf.data` with parallel decode/resize, caching of filename/label arrays, deterministic shuffling, and prefetching to overlap CPU input work with training. I also switch prediction to a batched `tf.data` pipeline to avoid Python loops and repeated PIL overhead, while preserving identical preprocessing (224 resize, RGB, float32 / 255). These changes are equivalent in outputs (up to negligible FP differences) but dramatically reduce Python overhead and improve throughput within 600s.'
- What this solution (achieved 0.52354) has done: 'I fix the environment-breaking protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which is what triggers the `MessageFactory.GetPrototype` error when importing TensorFlow in this Kaggle image. Next, I fix the dataset path construction bug by ensuring `image_id` is a proper unicode NumPy array before using `np.char.add`, avoiding the dtype mismatch error so `train_ds`/`val_ds` are created. With those two runtime blockers removed, the rest of your pipeline (same model, loss, training loop, and submission-building via `sample_submission.csv`) can run end-to-end and produce a valid `submission.csv`. These changes are primarily correctness/stability fixes; they should also substantially improve score versus the broken run because the model actually train.'
- What this solution (achieved 0.55419) has done: 'I fix the root runtime blocker causing the protobuf `MessageFactory.GetPrototype` crash by importing TensorFlow explicitly and using `tf.keras` (which is compatible with this environment) instead of relying on `tf_keras.backend.tf`. Once TensorFlow imports cleanly, the downstream `NameError`s disappear because earlier cells execute and define `TRAIN_CSV`, `tr_df`, `num_classes`, etc. I also keep your exact data pipeline, model, training loop, and submission construction logic unchanged, only adjusting the imports and `tf` wiring needed to run end-to-end. The script write a valid `submission.csv` with the required columns and the same ordering as `sample_submission.csv`.'
- What this solution (achieved 0.53214) has done: 'The crash happens before any training because this Kaggle image has an incompatibility between the default `tensorflow` import path and the protobuf runtime (`MessageFactory.GetPrototype`). The minimal fix is to use the Kaggle-installed `tf_keras` package (TF-Keras 2.18) for model/optimizer/layers while keeping TensorFlow only for `tf.data`/image IO; this avoids the protobuf crash and preserves your exact model/training logic. I also make the TensorFlow import more robust (try `tensorflow.compat.v2`) and keep your data pipeline, splits, class weights, training loop, and submission formatting unchanged. This should run end-to-end and, since your current run never trains due to the import crash, it should materially increase accuracy toward the target.'
- What this solution (achieved 0.53812) has done: 'We need to fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) that happens before training starts, otherwise nothing runs and the submission can’t be generated. The most reliable minimal change in this Kaggle image is to force TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the C++ protobuf API mismatch that triggers this error. I also make the `DATA_DIR` fallback a bit more robust (without changing paths) and keep your model, tf.data pipeline, training loop, and submission formatting identical so the score should improve materially just by actually training successfully. The rest of the code remains the same to preserve evaluation semantics.'
- What this solution (achieved 0.54895) has done: 'The crash happens immediately on importing TensorFlow because forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` triggers a protobuf/TensorFlow incompatibility in this Kaggle image, so the fix is to remove that override and import TensorFlow normally (keeping `tf_keras` for Keras APIs). I keep your exact tf.data pipeline, model architecture, training loop, and submission creation intact, only making the import section robust and deterministic so training actually runs and a valid `submission.csv` is always written. This should materially increase the score from the current broken/crippled state toward the target because the model train and infer end-to-end without the protobuf crash. No changes are made to preprocessing semantics, augmentations, loss, optimizer, or epochs.'
- What this solution (achieved 0.53214) has done: 'We need to fix the immediate runtime crash occurring at TensorFlow import (`MessageFactory.GetPrototype`), which is a protobuf/TensorFlow binary mismatch that can be reliably avoided in Kaggle by forcing the pure-Python protobuf implementation *before* importing TensorFlow. I apply that environment fix in the first cell (and keep using `tf_keras` for Keras APIs as you already do), leaving your tf.data pipeline, model architecture, training loop, and submission formatting unchanged. This should both make the notebook run end-to-end and improve the score versus the current broken/crippled run because the model actually train and predict consistently. No performance/accuracy tuning beyond this stability fix is introduced.'
- What this solution (achieved 0.52055) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment override and importing TensorFlow normally (this is the root cause of the `MessageFactory.GetPrototype` error in this Kaggle image). I keep using `tf_keras` for Keras APIs as you already do, and keep the exact same model, tf.data pipeline, training loop, and submission formatting. This change is primarily stability/correctness, but it should also increase the score materially versus the current broken run because training/inference actually execute end-to-end. The script still write a valid `submission.csv` with the required columns and ordering from `sample_submission.csv`.'
- What this solution (achieved 0.52728) has done: 'I fix the TensorFlow/protobuf crash happening at import time by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this environment sometimes ships an incompatible C++ protobuf). This is a minimal stability change that unblocks training/inference end-to-end and ensures `submission.csv` is written. I keep your exact data pipeline, model, loss, optimizer, epochs, and submission-building logic unchanged to preserve evaluation semantics. No score-tuning changes beyond making the run actually execute reliably.'
- What this solution (achieved 0.57025) has done: 'I fix the immediate TensorFlow/protobuf import crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which is what triggers the `MessageFactory.GetPrototype` error in this Kaggle environment. Then I keep your exact tf.data pipeline, augmentations, model architecture, loss, optimizer, epochs, and submission construction unchanged so evaluation semantics stay the same. This should both make the notebook run end-to-end reliably and improve your score materially versus the current broken run (because it actually train and predict). I also add a tiny, score-neutral safety fallback to locate `DATA_DIR` if Kaggle mounts it under `/kaggle/data/...` in your environment, without changing the preferred path.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
import tf_keras as keras

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(DATA_DIR):
    alt = "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
    if os.path.exists(alt):
        DATA_DIR = alt
    else:
        alt2 = "/kaggle/data/cassava-leaf-disease-classification"
        if os.path.exists(alt2):
            DATA_DIR = alt2

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

print("Using DATA_DIR:", DATA_DIR)
print("Train csv exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)
print("Submission rows:", len(sub_df), "Columns:", sub_df.columns.tolist())

idx = np.arange(len(train_df))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train split:", len(tr_df), "Val split:", len(va_df))

label_counts = tr_df["label"].value_counts().sort_index()
class_weight = {
    int(k): float(len(tr_df) / (num_classes * v)) for k, v in label_counts.items()
}
print("Class weights:", class_weight)



## === cell 2
img_size = 224
batch_size = 32


def _decode_resize_normalize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment_tf(img):
    img = tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: tf.image.flip_left_right(img),
        lambda: img,
    )

    def _add_delta():
        delta = tf.random.uniform((), minval=-0.10, maxval=0.10, dtype=tf.float32)
        return tf.clip_by_value(img + delta, 0.0, 1.0)

    img = tf.cond(tf.random.uniform(()) < 0.5, _add_delta, lambda: img)
    return img


def make_dataset_from_df(df, img_dir, training: bool):
    image_ids = df["image_id"].astype(str).to_numpy(dtype=str)
    labels = df["label"].to_numpy(dtype=np.int32)

    base = img_dir.rstrip("/") + "/"
    paths = np.char.add(base, image_ids).astype(str)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _decode_resize_normalize(path)
        if training:
            img = _augment_tf(img)
        return img, label

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset_from_df(tr_df, TRAIN_IMG_DIR, training=True)
val_ds = make_dataset_from_df(va_df, TRAIN_IMG_DIR, training=False)

steps_per_epoch = int(np.ceil(len(tr_df) / batch_size))
val_steps = int(np.ceil(len(va_df) / batch_size))
steps_per_epoch, val_steps



## === cell 3
inputs = keras.Input(shape=(img_size, img_size, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 4
epochs = 6

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    epochs=epochs,
    verbose=1,
    class_weight=class_weight,
)



## === cell 5
test_image_ids = sub_df["image_id"].astype(str).tolist()
test_paths = [os.path.join(TEST_IMG_DIR, image_id) for image_id in test_image_ids]
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _map_test(path):
    img = _decode_resize_normalize(path)
    return img


test_ds = test_ds.map(_map_test, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=0)
y_preds = np.argmax(probs, axis=1).astype(int).tolist()

print("Preds:", len(y_preds), "Test:", len(test_image_ids))



## === cell 6
df_sub = pd.DataFrame({"image_id": test_image_ids, "label": y_preds})
print(df_sub.head())
print(df_sub.shape)
assert df_sub.shape[0] == sub_df.shape[0]
assert list(df_sub.columns) == ["image_id", "label"]

df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv, bytes:", os.path.getsize("submission.csv"))
