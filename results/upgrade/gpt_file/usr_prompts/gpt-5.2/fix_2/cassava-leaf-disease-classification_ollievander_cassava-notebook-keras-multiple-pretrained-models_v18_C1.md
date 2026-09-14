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

0.8597763674826231

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11024) has done: 'I fix the import/runtime issues that prevent the notebook from running end-to-end by (1) removing/adjusting the incompatible Albumentations imports while keeping the same augmentation intent, (2) making Keras/TensorFlow imports consistent so `ImageDataGenerator` and other symbols are defined, and (3) eliminating the failing external weight load by training the existing pipeline’s model locally and then running inference. I also correct the test-time image preprocessing to match the model’s expected input size and dtype, and ensure predictions align exactly to `sample_submission.csv` order. Finally, the script always write a valid `submission.csv` with the required columns and `.csv` suffix.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")
LABEL_MAP_JSON = os.path.join(INPUT_DIR, "label_num_to_disease_map.json")

print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB_CSV exists:", os.path.exists(SAMPLE_SUB_CSV))
print("LABEL_MAP_JSON exists:", os.path.exists(LABEL_MAP_JSON))



## === cell 1
import albumentations as A

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

SEED = 100
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

AUTOTUNE = tf.data.experimental.AUTOTUNE
print("TensorFlow:", tf.__version__)
print("Albumentations:", A.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
with open(LABEL_MAP_JSON) as f:
    classes = json.load(f)

train_df["class"] = train_df["label"].apply(lambda x: classes[str(x)])
train_df.head(), train_df.shape



## === cell 3
train_df = train_df.copy()
train_df["path"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_PATH, str(x))
)

train_df = train_df.astype("str")

train_split, val_split = train_test_split(
    train_df, test_size=0.05, random_state=SEED, stratify=train_df["label"].values
)

print("Train split:", train_split.shape, "Val split:", val_split.shape)
print(train_split[["image_id", "label"]].head())



## === cell 4
IMG_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_CLASSES = 5


def transform(image):
    if image.dtype != np.uint8:
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image

    aug = A.Compose(
        [
            A.Flip(p=0.5),
            A.Rotate(limit=40, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.5),
        ]
    )
    out = aug(image=image_uint8)["image"].astype(np.float32)
    return out


train_gen = ImageDataGenerator(
    preprocessing_function=transform, rescale=1.0 / 255.0
).flow_from_dataframe(
    batch_size=BATCH_SIZE,
    dataframe=train_split,
    directory=TRAIN_PATH,
    shuffle=True,
    x_col="image_id",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
)

val_gen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_dataframe(
    batch_size=BATCH_SIZE,
    dataframe=val_split,
    directory=TRAIN_PATH,
    shuffle=False,
    x_col="image_id",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
)

print("Class indices:", train_gen.class_indices)



## === cell 5

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path = os.path.join(OUTPUT_DIR, "best_model.keras")
callbacks = [
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=1, verbose=1),
    EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
]

EPOCHS = (
    3  # keep small for runtime; no early stopping relaxation beyond callbacks above
)
history = model.fit(
    train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1
)

if os.path.exists(ckpt_path):
    model = keras.models.load_model(ckpt_path)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1375680926.py in <cell line: 0>()
     33     3  # keep small for runtime; no early stopping relaxation beyond callbacks above
     34 )
---> 35 history = model.fit(
     36     train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1
     37 )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/103819909.py in transform(image)
     15     aug = A.Compose(
     16         [
---> 17             A.Flip(p=0.5),
     18             A.Rotate(limit=40, p=0.5),
     19             A.HorizontalFlip(p=0.5),

AttributeError: module 'albumentations' has no attribute 'Flip'

## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_ids = sample_sub["image_id"].tolist()

test_gen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_dataframe(
    dataframe=sample_sub,  # contains image_id column already
    directory=TEST_PATH,
    x_col="image_id",
    y_col=None,
    target_size=IMG_SIZE,
    class_mode=None,
    shuffle=False,
    batch_size=BATCH_SIZE,
)

probs = model.predict(test_gen, verbose=1)
preds = probs.argmax(axis=1).astype(int)

submission = pd.DataFrame({"image_id": test_ids, "label": preds})
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
submission.head()



## === cell 7
assert os.path.exists(submission_path), "submission.csv was not created"
sub_check = pd.read_csv(submission_path)
assert list(sub_check.columns) == ["image_id", "label"], "Wrong submission columns"
assert len(sub_check) == len(sample_sub), "Wrong number of rows in submission"
assert sub_check["label"].between(0, 4).all(), "Labels must be in [0,4]"
print("Submission looks valid with shape:", sub_check.shape)
sub_check.tail()
