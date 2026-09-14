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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory as mf

    if not hasattr(mf.MessageFactory, "GetPrototype"):

        def GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        mf.MessageFactory.GetPrototype = GetPrototype
except Exception:
    pass  # If protobuf not available, let TF raise its own error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import applications, mixed_precision, layers
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import cv2  # for optional visualisation

tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count() or 1)
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() or 1)

tf.random.set_seed(42)
np.random.seed(42)

mixed_precision.set_global_policy("mixed_float16")

try:
    import tensorflow_addons as tfa

    def _rotate(img, angle):
        return tfa.image.rotate(img, angle, fill_mode="nearest")

except Exception:

    def _rotate(img, angle):
        return img  # fallback: no rotation if tfa unavailable




## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"



## === cell 2
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype(str)

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()



## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 4
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 5
EPOCHS = 15
NUM_WORKERS = min(8, os.cpu_count() or 1)

base_model = applications.EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax", dtype="float32")(
    x
)  # keep final logits in float32

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 5
label_to_index = {str(i): i for i in range(NUM_CLASSES)}

train_df = train_csv.sample(frac=0.8, random_state=42)
val_df = train_csv.drop(train_df.index)


def df_to_dataset(df, shuffle=False, augment=False, cache_path=None):
    """Create a tf.data.Dataset from a dataframe.
    Decoding & normalization are cached once; augmentation (if any) is applied after caching.
    """
    file_paths = tf.convert_to_tensor(
        df["image_id"]
        .apply(lambda x: os.path.join(data_path, "train_images/", x))
        .values
    )
    labels = tf.convert_to_tensor(
        df["label"].map(label_to_index).values, dtype=tf.int32
    )

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=False)

    def _decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_path is not None:
        ds = ds.cache(cache_path)
    else:
        ds = ds.cache()

    if augment:

        def _augment(img, label):
            img = tf.image.random_flip_left_right(img, seed=42)
            angle = tf.random.uniform([], -15.0, 15.0, seed=42) * (3.14159265 / 180.0)
            img = _rotate(img, angle)
            scales = tf.random.uniform([2], 0.9, 1.1, seed=42)
            scaled_size = tf.cast(tf.round(IMG_SIZE * scales), tf.int32)
            img = tf.image.resize(img, scaled_size)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            return img, tf.one_hot(label, NUM_CLASSES)

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


val_cache_file = "/tmp/val_dataset_cache"
train_dataset = df_to_dataset(train_df, shuffle=True, augment=True, cache_path=None)
val_dataset = df_to_dataset(
    val_df, shuffle=False, augment=False, cache_path=val_cache_file
)



## === cell 6
steps_per_epoch = np.ceil(len(train_df) / BATCH_SIZE).astype(int)
validation_steps = np.ceil(len(val_df) / BATCH_SIZE).astype(int)

model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)



## === cell 7
test_img_path = data_path + "test_images/2216849948.jpg"
if os.path.exists(test_img_path):
    img = cv2.imread(test_img_path)
    resized_img = (
        cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    )
    plt.figure(figsize=(8, 4))
    plt.title("TEST IMAGE")
    plt.imshow(resized_img[0])
    plt.show()
else:
    print("Sample test image not found; skipping visualisation.")



## === cell 8
sample_submission = pd.read_csv(data_path + "sample_submission.csv")

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=sample_submission,
    directory=os.path.join(data_path, "test_images/"),
    x_col="image_id",
    y_col=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
)

preds_prob = model.predict(test_generator, verbose=0)
preds = np.argmax(preds_prob, axis=1).tolist()

my_submission = pd.DataFrame({"image_id": sample_submission.image_id, "label": preds})
my_submission.to_csv("submission.csv", index=False)



## === cell 9
print("Submission File: \n---------------")
print(my_submission.head())
