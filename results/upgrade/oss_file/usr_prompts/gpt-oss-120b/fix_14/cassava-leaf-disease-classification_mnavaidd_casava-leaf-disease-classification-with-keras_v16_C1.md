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

0.7840737382895134

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The changes increase data‑loading parallelism during training, enlarge the batch size to cut the number of gradient steps, and batch the test‑time predictions to avoid a costly per‑image `model.predict` call. These tweaks keep the model architecture, training epochs, augmentation, and loss exactly the same, merely making the existing pipeline more efficient, so the final predictions remain unchanged.'
- What this solution (achieved 0.11584) has done: 'Implemented key performance boosts while keeping the model architecture, training loop, and evaluation logic unchanged.

- Added `workers` and `max_queue_size` to `model.fit` for parallel data loading and preprocessing.
- Replaced the explicit Python loop for test‑time predictions with an efficient `tf.data.Dataset` pipeline that reads, decodes, resizes, and normalizes images in parallel, then runs a single batched `model.predict`. This preserves the original ordering and prediction semantics.'
- What this solution (achieved 0.11584) has done: 'The changes freeze the EfficientNet backbone (so only the small head is trained), increase the batch size and enable parallel data loading, and set a deterministic seed. These tweaks keep the same model architecture and training loop but cut the amount of computation per epoch dramatically, allowing the whole notebook to finish within the 600‑second limit while preserving the original logic and accuracy characteristics.'
- What this solution (achieved 0.11024) has done: 'The changes increase the batch size to halve the number of steps per epoch, and enable multi‑process data loading (`workers=4, use_multiprocessing=True`) during model fitting, which speeds up image preprocessing without altering the model architecture, loss, or training logic. These adjustments keep the same augmentation pipeline and training schedule, preserving accuracy while fitting comfortably within the 600‑second limit.'
- What this solution (achieved 0.05531) has done: 'Optimized the script by (1) making the protobuf installation conditional to avoid unnecessary pip calls, (2) enabling parallel data loading in the training loops with `workers` and `use_multiprocessing`, and (3) adding a small comment explaining each speed tweak. These changes keep the model architecture, training schedule, and evaluation unchanged while reducing I/O and setup overhead, ensuring the notebook finishes within the 600‑second limit.'
- What this solution (achieved 0.05531) has done: 'Implemented a fast tf.data pipeline with in‑graph augmentations, removed the costly protobuf reinstall, and reused the same model/optimizer settings. This keeps the architecture, loss, and training loops unchanged while drastically cutting I/O and CPU‑side preprocessing overhead, ensuring the script completes well within the 600‑second limit.'

# 9. Code solution

## === cell 0
import os, sys, subprocess

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_path = data_path + "train.csv"
label_json_path = data_path + "label_num_to_disease_map.json"
train_images_dir = data_path + "train_images/"




## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(str)

label_map = pd.read_json(label_json_path, orient="index")
label_names = label_map.values.flatten().tolist()




## === cell 3
print("Label names:")
for i, name in enumerate(label_names):
    print(f" {i}. {name}")




## === cell 4
BATCH_SIZE = 256
IMG_SIZE = 224
NUM_CLASSES = len(label_names)
AUTOTUNE = tf.data.AUTOTUNE




## === cell 5
df_shuffled = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_frac = 0.15
val_size = int(len(df_shuffled) * val_frac)
val_df = df_shuffled.iloc[:val_size]
train_split_df = df_shuffled.iloc[val_size:]


def df_to_paths_labels(df):
    paths = tf.constant(
        [os.path.join(train_images_dir, fname) for fname in df["image_id"].values]
    )
    labels = tf.constant(df["label"].astype(int).values, dtype=tf.int32)
    one_hot = tf.one_hot(labels, depth=NUM_CLASSES)
    return paths, one_hot


train_paths, train_labels = df_to_paths_labels(train_split_df)
val_paths, val_labels = df_to_paths_labels(val_df)


def _load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img


def _augment(img):
    img = tf.keras.layers.RandomRotation(1.0)(img)
    img = tf.keras.layers.RandomTranslation(0.1, 0.1)(img)
    img = tf.keras.layers.RandomZoom(0.3)(img)
    img = img + tf.random.normal(tf.shape(img), mean=0.0, stddev=0.01)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.4)  # approx.
    return tf.clip_by_value(img, 0.0, 1.0)


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(buffer=1024, seed=42)
    .map(lambda p, l: (_load_and_preprocess(p), l), num_parallel_calls=AUTOTUNE)
    .map(lambda img, lbl: (_augment(img), lbl), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(lambda p, l: (_load_and_preprocess(p), l), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3805628344.py in <cell line: 0>()
     59 train_ds = (
     60     tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
---> 61     .shuffle(buffer=1024, seed=42)
     62     .map(lambda p, l: (_load_and_preprocess(p), l), num_parallel_calls=AUTOTUNE)
     63     .map(lambda img, lbl: (_augment(img), lbl), num_parallel_calls=AUTOTUNE)

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 6
base_model = applications.EfficientNetB0(
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights="imagenet",
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.summary()




## === cell 7
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model.h5",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
)

early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, mode="min", restore_best_weights=True, verbose=1
)




## === cell 8
model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=[checkpoint_cb, early_stop_cb],
    verbose=2,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3997057797.py in <cell line: 0>()
      1 model.fit(
----> 2     train_ds,
      3     validation_data=valid_ds,
      4     epochs=10,
      5     callbacks=[checkpoint_cb, early_stop_cb],

NameError: name 'train_ds' is not defined

## === cell 9
base_model.trainable = True
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=15,
    callbacks=[checkpoint_cb, early_stop_cb],
    verbose=2,
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1471250928.py in <cell line: 0>()
      7 
      8 model.fit(
----> 9     train_ds,
     10     validation_data=valid_ds,
     11     epochs=15,

NameError: name 'train_ds' is not defined

## === cell 10
if os.path.exists("best_model.h5"):
    model.load_weights("best_model.h5")
else:
    print("Checkpoint not found; using the last trained weights.")




## === cell 11
test_csv_path = data_path + "sample_submission.csv"
test_df = pd.read_csv(test_csv_path)

image_paths = [
    os.path.join(data_path, "test_images", img_name) for img_name in test_df["image_id"]
]


def _load_and_preprocess_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(image_paths)
    .map(_load_and_preprocess_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

probs = model.predict(test_ds, verbose=0)
preds = np.argmax(probs, axis=1).astype(str).tolist()

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": preds})
submission.to_csv("submission.csv", index=False)




## === cell 12
print("Submission file created:")
print(submission.head())
