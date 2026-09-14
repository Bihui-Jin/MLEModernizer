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

2.7

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

0.8634028407373829

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.12257) has done: 'The changes keep the same Xception‑based architecture and training loop but speed up I/O and computation: the image size is reduced to Xception’s native 299 × 299 (still compatible with the model), the batch size is doubled to halve the number of steps per epoch, and multiprocessing is disabled in the generators to avoid the overhead of spawning processes in the limited environment. All other logic—including augmentation, early stopping, and prediction—remains unchanged.'

# 9. Code solution

## === cell 0
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

CPU_COUNT = os.cpu_count() or 1
MAX_THREADS = min(CPU_COUNT, 8)  # avoid oversubscription
tf.config.threading.set_intra_op_parallelism_threads(MAX_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(MAX_THREADS)

tf.compat.v1.disable_eager_execution()

tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
ROOT = "../input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(ROOT, "train.csv")
test_dir = os.path.join(ROOT, "test_images")

train_df = pd.read_csv(train_csv_path)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df["label"] = train_df["label"].astype(int)

val_split = 0.1
val_size = int(len(train_df) * val_split)
val_df = train_df.iloc[:val_size].copy()
train_df = train_df.iloc[val_size:].copy()

IMG_SIZE = (299, 299)
BATCH_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE


def _load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.int32)


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


train_paths = [
    os.path.join(ROOT, "train_images", fname) for fname in train_df["image_id"]
]
train_labels = train_df["label"].values

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(_load_image, num_parallel_calls=AUTOTUNE, deterministic=False)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=True)
train_ds = train_ds.map(_augment, num_parallel_calls=AUTOTUNE, deterministic=False)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_paths = [os.path.join(ROOT, "train_images", fname) for fname in val_df["image_id"]]
val_labels = val_df["label"].values
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(
    _load_image, num_parallel_calls=AUTOTUNE, deterministic=False
).cache()
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

options = tf.data.Options()
options.experimental_deterministic = False
options.experimental_optimization.map_parallelization = True
train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)




## === cell 2
base_model = tf.keras.applications.Xception(
    weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False  # freeze base initially

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(5, activation="softmax"),
    ]
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

es = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True, mode="max"
)

model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[es],
    verbose=1,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2833968785.py in <cell line: 0>()
     23 )
     24 
---> 25 model.fit(
     26     train_ds,
     27     epochs=10,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/tf_dataset_adapter.py in num_batches(self)
     66         if callable(cardinality):
     67             # `dataset.cardinality` is normally expected to be a callable.
---> 68             cardinality = int(self._dataset.cardinality())
     69         else:
     70             # However, in the case of `DistributedDataset`, it's a np.int64.

TypeError: int() argument must be a string, a bytes-like object or a real number, not 'SymbolicTensor'

## === cell 3
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
    callbacks=[es],
    verbose=1,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2996155809.py in <cell line: 0>()
      9 )
     10 
---> 11 model.fit(
     12     train_ds,
     13     epochs=5,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/tf_dataset_adapter.py in num_batches(self)
     66         if callable(cardinality):
     67             # `dataset.cardinality` is normally expected to be a callable.
---> 68             cardinality = int(self._dataset.cardinality())
     69         else:
     70             # However, in the case of `DistributedDataset`, it's a np.int64.

TypeError: int() argument must be a string, a bytes-like object or a real number, not 'SymbolicTensor'

## === cell 4
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")

test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
test_paths = [os.path.join(test_dir, f) for f in test_files]
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda p: _load_image(p, 0)[0], num_parallel_calls=AUTOTUNE, deterministic=False
).cache()
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

preds = model.predict(test_ds, verbose=1)
pred_classes = np.argmax(preds, axis=1)

submission = pd.DataFrame({"image_id": test_files, "label": pred_classes})




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1404826115.py in <cell line: 0>()
----> 1 val_loss, val_acc = model.evaluate(val_ds, verbose=0)
      2 print(f"Validation accuracy: {val_acc:.4f}")
      3 
      4 test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
      5 test_paths = [os.path.join(test_dir, f) for f in test_files]

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/tf_dataset_adapter.py in num_batches(self)
     66         if callable(cardinality):
     67             # `dataset.cardinality` is normally expected to be a callable.
---> 68             cardinality = int(self._dataset.cardinality())
     69         else:
     70             # However, in the case of `DistributedDataset`, it's a np.int64.

TypeError: int() argument must be a string, a bytes-like object or a real number, not 'SymbolicTensor'

## === cell 5
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2638925618.py in <cell line: 0>()
      1 output_path = "submission.csv"
----> 2 submission.to_csv(output_path, index=False)
      3 print(f"Submission saved to {output_path}")

NameError: name 'submission' is not defined
