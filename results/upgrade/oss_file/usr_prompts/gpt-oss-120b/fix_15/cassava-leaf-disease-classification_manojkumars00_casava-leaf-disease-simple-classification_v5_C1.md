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

0.0019

# 6. Current score

0.09417

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11061) has done: 'The changes speed up the pipeline by (1) increasing the batch size to reduce the number of training steps, (2) enabling parallel data loading in `model.fit`, and (3) batching test‑image inference instead of predicting one image at a time. These adjustments keep the exact model architecture, training epochs, and augmentation unchanged, so the learned behavior and final predictions remain equivalent while fitting comfortably within the 600‑second limit.'
- What this solution (achieved 0.59753) has done: 'The update adds parallel data loading by increasing the worker count for both training and validation generators and enables multiprocessing during model fitting. These changes keep the model architecture, training epochs, and callbacks unchanged while reducing Python‑side I/O overhead, allowing the same number of iterations to complete well within the 600 s limit. No algorithmic logic or results are altered.'
- What this solution (achieved 0.1151) has done: 'The fix adds a larger batch size to cut the number of steps per epoch and enables parallel data loading during `model.fit` by setting `workers` and `use_multiprocessing`. Both changes keep the model architecture, loss, optimizer, and augmentation pipeline intact while reducing the overall runtime enough to finish within the 600‑second limit.'
- What this solution (achieved 0.43647) has done: 'I speed up the training loop by re‑enabling parallel data loading during `model.fit`. The generators already use multiple workers, but `fit` defaults to a single worker, causing a bottleneck when reading and augmenting images on CPU. Adding `workers=8` and `use_multiprocessing=True` restores the intended parallelism without altering the model, augmentations, or training schedule, preserving exact results while significantly reducing wall‑clock time.'
- What this solution (achieved 0.09417) has done: 'The changes replace the Python‑based `ImageDataGenerator` pipeline with a fast TensorFlow `tf.data` pipeline that reads images directly, resizes and rescales them, and batches them using TensorFlow’s efficient C++ backend. This removes the per‑image Python overhead while keeping the same image size, batch size, and label encoding, so model architecture and training semantics stay unchanged. The validation set is created by a simple pandas split, and a quick visualisation uses the new dataset. All other logic (model definition, training, prediction, submission) is left intact.'

# 9. Code solution

## === cell 0
import os

try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):
        _mf.MessageFactory.GetPrototype = _mf.MessageFactory.GetMessageClass
except Exception:
    pass

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.callbacks import LearningRateScheduler

np.random.seed(42)
tf.random.set_seed(42)

tf.config.optimizer.set_jit(True)

if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_input = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_input, "train.csv")
label_json_path = os.path.join(base_input, "label_num_to_disease_map.json")
images_dir_path = os.path.join(base_input, "train_images")
test_images_dir = os.path.join(base_input, "test_images")



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype(int)  # keep integer labels

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()



## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 4
train_csv.head()



## === cell 5
BATCH_SIZE = 1024
IMG_SIZE = 320



## === cell 6
val_df = train_csv.sample(frac=0.2, random_state=42)
train_df = train_csv.drop(val_df.index)


def _make_dataset(df):
    paths = df["image_id"].apply(lambda x: os.path.join(images_dir_path, x)).tolist()
    labels = df["label"].tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _parse(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img / 255.0
        label_onehot = tf.one_hot(label, depth=5)
        return img, label_onehot

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


train_ds = (
    _make_dataset(train_df).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)
val_ds = _make_dataset(val_df).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



## === cell 7
sample_batch = next(iter(train_ds))
sample_images, sample_labels = sample_batch
plt.figure(figsize=(15, 9))
for i in range(min(15, sample_images.shape[0])):
    plt.subplot(5, 3, i + 1)
    plt.axis("off")
    plt.imshow(sample_images[i].numpy())
    plt.title(label_class[np.argmax(sample_labels[i].numpy())])
plt.show()



## === cell 8
base = applications.InceptionResNetV2(
    include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
)
base.trainable = False



## === cell 9
model = tf.keras.Sequential(
    [
        base,
        BatchNormalization(axis=-1),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dense(5, activation="softmax"),
    ]
)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=["acc"],
)


def scheduler(epoch, lr):
    return lr / 1.25 if epoch > 2 else lr


callback = LearningRateScheduler(scheduler)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[callback],
    verbose=2,
)

model_path = "./CassavaLeafDiseaseModel.h5"
model.save(model_path)



## === cell 10
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Model load failed ({e}), using the freshly trained model.")



## === cell 11
test_img_path = os.path.join(test_images_dir, "2216849948.jpg")
img = cv2.imread(test_img_path)
if img is not None:
    resized_img = (
        cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255
    )
    plt.figure(figsize=(8, 4))
    plt.title("TEST IMAGE")
    plt.imshow(resized_img[0])
    plt.show()
else:
    print("Test image not found; skipping visualisation.")



## === cell 12
sample_sub_path = os.path.join(base_input, "sample_submission.csv")
ss = pd.read_csv(sample_sub_path)

image_paths = ss["image_id"].apply(lambda x: os.path.join(test_images_dir, x)).tolist()
loaded_images = []
valid_indices = []  # indices where the image exists

for idx, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    if img is None:
        continue
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    loaded_images.append(resized)
    valid_indices.append(idx)

if loaded_images:
    batch_array = np.stack(loaded_images, axis=0).astype(np.float32) / 255.0
    raw_preds = np.argmax(model.predict(batch_array, verbose=0), axis=1)
    preds_batch = (raw_preds + 1) % 5
else:
    preds_batch = []

preds = [0] * len(ss)
for idx, pred in zip(valid_indices, preds_batch):
    preds[idx] = int(pred)

ss["label"] = preds
submission_path = "./submission.csv"
ss.to_csv(submission_path, index=False)



## === cell 13
print("Submission File saved to:", submission_path)
print(ss.head())
