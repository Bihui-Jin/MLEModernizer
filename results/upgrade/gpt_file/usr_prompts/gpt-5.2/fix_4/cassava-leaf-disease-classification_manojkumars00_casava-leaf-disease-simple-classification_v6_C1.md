# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)

print("TF version:", tf.__version__)



## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"
test_images_dir = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()



## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 4
train_csv.head()



## === cell 5
BATCH_SIZE = 24
IMG_SIZE = 320



## === cell 6
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)  # XLA can improve throughput for CNNs
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 7
train_gen = ImageDataGenerator(
    rotation_range=270,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.1, 0.9],
    shear_range=25,
    zoom_range=0.3,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
    validation_split=0.2,
)

valid_gen = ImageDataGenerator(
    rescale=1 / 255,
    validation_split=0.2,
)



## === cell 8
classes = [str(i) for i in range(5)]
print("Using classes:", classes)



## === cell 9
from sklearn.model_selection import train_test_split

df = train_csv.copy()
df["label_int"] = df["label"].astype(int)

train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_int"].values,
)

train_paths = (images_dir_path + "/" + train_df["image_id"].astype(str)).values
train_labels = train_df["label_int"].values.astype(np.int32)

valid_paths = (images_dir_path + "/" + valid_df["image_id"].astype(str)).values
valid_labels = valid_df["label_int"].values.astype(np.int32)


def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _apply_augmentations(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    b = tf.random.uniform([], 0.1, 0.9, dtype=tf.float32)
    img = tf.clip_by_value(img * b, 0.0, 1.0)

    return img


geom_aug = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=270.0 / 360.0, fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="reflect"
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3), fill_mode="reflect"
        ),
    ],
    name="geom_aug",
)


def _train_map(path, label):
    img = _read_decode_resize(path)
    img = _apply_augmentations(img)
    img = geom_aug(img, training=True)
    y = tf.one_hot(label, depth=5, dtype=tf.float32)
    return img, y


def _valid_map(path, label):
    img = _read_decode_resize(path)
    y = tf.one_hot(label, depth=5, dtype=tf.float32)
    return img, y


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(buffer_size=len(train_paths), seed=42, reshuffle_each_iteration=True)
    .map(_train_map, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(_valid_map, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)


class _DSWrap:
    def __init__(self, ds, samples):
        self.ds = ds
        self.samples = samples


train_generator = _DSWrap(train_ds, len(train_paths))
valid_generator = _DSWrap(valid_ds, len(valid_paths))



## === cell 10
if False:
    batch = next(iter(train_generator.ds))
    images = batch[0].numpy()
    labels = batch[1].numpy()

    plt.figure(figsize=(15, 9))
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(5, 3, i % 15 + 1)
        plt.axis("off")
        plt.imshow(img)
        plt.title(label_class[np.argmax(label)])
        if i == 15:
            break



## === cell 11
base = applications.InceptionResNetV2(
    include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
)



## === cell 12
model = tf.keras.Sequential()
model.add(base)
model.add(BatchNormalization(axis=-1))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dense(5, activation="softmax"))

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=["acc"],
)




## === cell 13
def scheduler(epoch, lr):
    if epoch > 2:
        return lr / 1.25
    else:
        return lr


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)



## === cell 14
model_path = "../working/CasavaLeafDiseaseModel.h5"



## === cell 15
loaded = False
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        loaded = True
        print("Loaded existing model from:", model_path)
    except Exception as e:
        print("Could not load model, will train. Error:", repr(e))



## === cell 16
if not loaded:
    EPOCHS = 8
    steps_per_epoch = int(np.ceil(train_generator.samples / BATCH_SIZE))
    validation_steps = int(np.ceil(valid_generator.samples / BATCH_SIZE))

    history = model.fit(
        train_generator.ds,
        validation_data=valid_generator.ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[callback],
        verbose=1,
    )
    model.save(model_path)
    print("Saved model to:", model_path)



## === cell 17
if False:
    test_img_path = os.path.join(test_images_dir, "2216849948.jpg")
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
        if img is not None:
            resized_img = (
                cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
                / 255.0
            )
            plt.figure(figsize=(8, 4))
            plt.title("TEST IMAGE")
            plt.imshow(resized_img[0][..., ::-1])
            plt.axis("off")
        else:
            print("cv2.imread returned None for:", test_img_path)
    else:
        print("Test image not found at:", test_img_path)



## === cell 18
preds = []
ss = pd.read_csv(sample_sub_path)


def _load_and_preprocess_image(image_id):
    path = tf.strings.join([test_images_dir, "/", image_id])
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(ss["image_id"].astype(str).values)
    .map(_load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
)

probs = model.predict(test_ds, verbose=0)
preds = probs.argmax(axis=1).astype(int).tolist()

my_submission = pd.DataFrame({"image_id": ss.image_id, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 19
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nLabel value counts:\n", my_submission["label"].value_counts().sort_index())
