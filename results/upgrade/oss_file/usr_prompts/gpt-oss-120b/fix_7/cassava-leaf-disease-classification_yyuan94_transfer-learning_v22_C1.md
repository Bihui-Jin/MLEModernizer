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

0.2311876699909338

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'The changes enable TensorFlow XLA JIT compilation and set thread parallelism for faster CPU execution, move image resizing into the data pipeline (so the model’s resizing layer becomes a no‑op, saving work), and drop the in‑memory `cache()` which was causing heavy RAM pressure and swapping. These tweaks keep the exact model architecture, loss, and training schedule, so the predictions remain unchanged while the end‑to‑end runtime fits within the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd
import glob
import sys

try:
    import tensorflow as tf

    tf.config.optimizer.set_jit(True)

    tf.config.threading.set_intra_op_parallelism_threads(
        tf.config.threading.cpu_count()
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        tf.config.threading.cpu_count()
    )

    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception as e:
    tf = None
    print("TensorFlow import failed; using fallback baseline.", e)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
with open(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    "r",
) as fp:
    class_map = json.load(fp)




## === cell 2
if tf is None:
    most_common_label = train_csv["label"].mode()[0]

    test_pattern = "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg"
    test_files = glob.glob(test_pattern)

    test_ids = [os.path.basename(p) for p in test_files]
    predictions = np.full(len(test_ids), most_common_label, dtype=int)

    submission = pd.DataFrame({"image_id": test_ids, "label": predictions})
    submission.to_csv("submission.csv", index=False)
    print(
        "Fallback submission written to submission.csv (all predictions = most common class)."
    )
    sys.exit(0)




## --- ERROR in cell 2, traceback:
An exception has occurred, use %tb to see the full traceback.

SystemExit: 0


## === cell 3
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_csv, test_size=0.1, random_state=42, stratify=train_csv["label"]
)




## === cell 4
def process_data(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # uint8 → float32 in [0,1]
    img = tf.image.resize(img, [300, 300])
    return img, tf.one_hot(label, depth=5)




## === cell 5
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

train_paths = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_df["image_id"].astype(str).values
)
val_paths = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + val_df["image_id"].astype(str).values
)

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_df["label"].values))
    .map(process_data, num_parallel_calls=AUTOTUNE)
    .shuffle(2000)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_df["label"].values))
    .map(process_data, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1841955728.py in <cell line: 0>()
      1 batch_size = 32
----> 2 AUTOTUNE = tf.data.AUTOTUNE
      3 
      4 train_paths = (
      5     "/kaggle/input/cassava-leaf-disease-classification/train_images/"

AttributeError: 'NoneType' object has no attribute 'data'

## === cell 6
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom((-0.2, 0.0)),
        tf.keras.layers.RandomContrast((0.0, 0.2)),
    ]
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3640588461.py in <cell line: 0>()
----> 1 data_augmentation = tf.keras.Sequential(
      2     [
      3         tf.keras.layers.RandomFlip("horizontal_and_vertical"),
      4         tf.keras.layers.RandomRotation(0.25),
      5         tf.keras.layers.RandomZoom((-0.2, 0.0)),

AttributeError: 'NoneType' object has no attribute 'keras'

## === cell 7
input_shape = (300, 300, 3)
num_classes = len(class_map)

inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
x = data_augmentation(inputs, training=True)
x = tf.keras.layers.Resizing(input_shape[0], input_shape[1])(x)

base_model = tf.keras.applications.EfficientNetB3(
    weights="imagenet", include_top=False, input_shape=input_shape
)
base_model.trainable = False
x = base_model(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

model = tf.keras.Model(inputs, outputs, name="EfficientNetB3_classifier")
model.summary()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2898730417.py in <cell line: 0>()
      2 num_classes = len(class_map)
      3 
----> 4 inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
      5 x = data_augmentation(inputs, training=True)
      6 # The Resizing layer now receives already‑sized tensors; it adds negligible overhead

AttributeError: 'NoneType' object has no attribute 'keras'

## === cell 8
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=METRICS)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2952105312.py in <cell line: 0>()
      1 METRICS = [
----> 2     tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
      3     tf.keras.metrics.Precision(name="precision"),
      4     tf.keras.metrics.Recall(name="recall"),
      5     tf.keras.metrics.AUC(name="auc"),

AttributeError: 'NoneType' object has no attribute 'keras'

## === cell 9
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=0
    )
]

history = model.fit(
    train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks, verbose=2
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4081964342.py in <cell line: 0>()
      1 callbacks = [
----> 2     tf.keras.callbacks.ModelCheckpoint(
      3         filepath="best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=0
      4     )
      5 ]

AttributeError: 'NoneType' object has no attribute 'keras'

## === cell 10
if os.path.exists("best_model.h5"):
    model.load_weights("best_model.h5")




## === cell 11
test_files = tf.io.gfile.glob(
    "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg"
)


def process_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [300, 300])  # ensure correct size for the model
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_files)
    .map(process_test, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1861210457.py in <cell line: 0>()
----> 1 test_files = tf.io.gfile.glob(
      2     "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg"
      3 )
      4 
      5 

AttributeError: 'NoneType' object has no attribute 'io'

## === cell 12
probabilities = model.predict(test_ds, verbose=0)
predictions = np.argmax(probabilities, axis=-1)

test_ids = [os.path.basename(p) for p in test_files]
submission = pd.DataFrame({"image_id": test_ids, "label": predictions})
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/616565887.py in <cell line: 0>()
----> 1 probabilities = model.predict(test_ds, verbose=0)
      2 predictions = np.argmax(probabilities, axis=-1)
      3 
      4 test_ids = [os.path.basename(p) for p in test_files]
      5 submission = pd.DataFrame({"image_id": test_ids, "label": predictions})

NameError: name 'model' is not defined
