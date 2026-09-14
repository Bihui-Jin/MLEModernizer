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

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")

print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))



## === cell 1
import json
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(123)
np.random.seed(123)

print("tf:", tf.__version__)



## === cell 2
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
display(train.head())
print("train shape:", train.shape)



## === cell 3
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)
classes



## === cell 4
train["class"] = train["label"].apply(lambda x: classes[str(x)])

plt.figure(figsize=(15, 7))
sns.countplot(x=train["class"], order=train["class"].value_counts().index)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()



## === cell 5
train = train.copy()
train["path"] = train["image_id"].apply(lambda x: os.path.join(TRAIN_PATH, str(x)))

train = train.astype(
    {"image_id": "string", "label": "string", "path": "string", "class": "string"}
)
train_df, val_df = train_test_split(
    train,
    test_size=0.05,
    random_state=100,
    stratify=train["label"].values,
)
print("train_df:", train_df.shape, "val_df:", val_df.shape)



## === cell 6
batch_size = 4
IMG_SIZE = (512, 512)


def transform(image):
    x = tf.convert_to_tensor(image, dtype=tf.float32)

    x = tf.image.random_flip_left_right(x)

    x = tf.image.random_flip_up_down(x)

    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)

    do_transpose = tf.random.uniform([]) < 0.5
    x = tf.cond(do_transpose, lambda: tf.transpose(x, perm=[1, 0, 2]), lambda: x)

    return x.numpy()


train_gen = ImageDataGenerator(preprocessing_function=transform).flow_from_dataframe(
    batch_size=batch_size,
    dataframe=train_df,
    directory=TRAIN_PATH,
    shuffle=True,
    x_col="image_id",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
)

val_gen = ImageDataGenerator().flow_from_dataframe(
    batch_size=batch_size,
    dataframe=val_df,
    directory=TRAIN_PATH,
    shuffle=False,
    x_col="image_id",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
)

num_classes = len(train_gen.class_indices)
print("Num classes inferred:", num_classes)
print("Class indices:", train_gen.class_indices)



## === cell 7
MODEL_PATH = "../input/pass-4/weightEffnetB4_v6.h5"


def build_fallback_model(num_classes: int, img_size=(512, 512)):
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


loaded_model = None
if os.path.exists(MODEL_PATH):
    try:
        loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Loaded external model:", MODEL_PATH)
    except Exception as e:
        print("Failed to load external model; will train fallback. Error:", repr(e))
        loaded_model = None
else:
    print("External model path not found; will train fallback:", MODEL_PATH)

if loaded_model is None:
    loaded_model = build_fallback_model(num_classes, IMG_SIZE)
    loaded_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        verbose=1,
    )

print("Model input shape:", loaded_model.input_shape)



## === cell 8
sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
test_ids = sub["image_id"].tolist()

input_shape = loaded_model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]
req_h, req_w = int(input_shape[1]), int(input_shape[2])


def load_and_preprocess(img_path, target_size):
    img = Image.open(img_path).convert("RGB")
    if (img.size[1], img.size[0]) != target_size:
        img = img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    return arr


target_size = (req_h, req_w)

preds = []
BATCH = 16
for i in range(0, len(test_ids), BATCH):
    batch_ids = test_ids[i : i + BATCH]
    batch_imgs = []
    for img_id in batch_ids:
        img_path = os.path.join(TEST_PATH, img_id)
        batch_imgs.append(load_and_preprocess(img_path, target_size))
    batch_x = np.stack(batch_imgs, axis=0)
    proba = loaded_model.predict(batch_x, verbose=0)
    preds.extend(np.argmax(proba, axis=1).astype(int).tolist())

print("preds:", len(preds), "test_ids:", len(test_ids))



## === cell 9
submission = pd.DataFrame({"image_id": test_ids, "label": preds})
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

print(submission.head())
print("Wrote:", os.path.join(OUTPUT_DIR, "submission.csv"), "rows:", len(submission))
print("Columns:", submission.columns.tolist())
