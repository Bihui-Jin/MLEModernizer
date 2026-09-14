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
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

AUTOTUNE = tf.data.experimental.AUTOTUNE

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 2
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
print(train.shape)
train.head()



## === cell 3
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)

classes



## === cell 4
train["class"] = train["label"].apply(lambda x: classes[str(x)])
train[["image_id", "label", "class"]].head()



## === cell 5
print("Class distribution:\n", train["class"].value_counts())



## === cell 6
train["path"] = train["image_id"].apply(lambda x: os.path.join(TRAIN_PATH, str(x)))
train = train.astype({"image_id": "str", "label": "str", "class": "str", "path": "str"})


def stratified_split_df(df, label_col, test_size=0.05, seed=100):
    rng = np.random.RandomState(seed)
    parts = []
    val_parts = []
    for label, g in df.groupby(label_col):
        idx = g.index.values.copy()
        rng.shuffle(idx)
        n_val = int(np.ceil(len(idx) * test_size))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_parts.append(df.loc[val_idx])
        parts.append(df.loc[train_idx])
    train_df = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return train_df, val_df


train, val = stratified_split_df(train, "label", test_size=0.05, seed=100)

print("train:", train.shape, "val:", val.shape)
train.head()



## === cell 7
batch_size = 4

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
).flow_from_dataframe(
    batch_size=batch_size,
    dataframe=train,
    directory=TRAIN_PATH,  # x_col is image_id
    shuffle=True,
    x_col="image_id",
    y_col="label",
    target_size=(512, 512),
    class_mode="categorical",
    seed=SEED,
)



## === cell 8
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_dataframe(
    batch_size=batch_size,
    dataframe=val,
    directory=TRAIN_PATH,
    shuffle=False,
    x_col="image_id",
    y_col="label",
    target_size=(512, 512),
    class_mode="categorical",
    seed=SEED,
)



## === cell 9
print("Train batches:", len(datagen), "Val batches:", len(val_datagen))
print("Class indices (generator):", datagen.class_indices)



## === cell 10
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
print(sample_sub.shape)
sample_sub.head()



## === cell 11
test_images = sample_sub["image_id"].astype(str).tolist()
df_test = pd.DataFrame({"image_id": test_images})
df_test["path"] = df_test["image_id"].apply(lambda x: os.path.join(TEST_PATH, str(x)))

missing = (~df_test["path"].apply(os.path.exists)).sum()
print("Missing test files:", int(missing))
df_test.head()



## === cell 12
testgen = ImageDataGenerator(rescale=1.0 / 255.0)

test_gen2 = testgen.flow_from_dataframe(
    dataframe=df_test,
    directory=TEST_PATH,
    x_col="image_id",
    y_col=None,
    batch_size=batch_size,
    seed=SEED,
    shuffle=False,  # critical: preserve ordering for submission
    class_mode=None,
    target_size=(512, 512),
)



## === cell 13
num_classes = len(datagen.class_indices)

inputs = tf.keras.Input(shape=(512, 512, 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(
    datagen,
    validation_data=val_datagen,
    epochs=3,
    verbose=1,
)

model2 = None  # keep variable for downstream ensemble code compatibility



## === cell 14
preds = []
tta = 3  # keep TTA semantics but limit to keep runtime safe

for i in range(tta):
    test_gen2.reset()
    p1 = model.predict(test_gen2, verbose=0)
    if model2 is not None:
        test_gen2.reset()
        p2 = model2.predict(test_gen2, verbose=0)
        preds.append(p1 + p2)
    else:
        preds.append(p1)

predbis = np.mean(preds, axis=0)

idx_to_class = {v: k for k, v in datagen.class_indices.items()}  # index -> string label
pred_idx = np.argmax(predbis, axis=-1).astype(int)
predictions = np.array([int(idx_to_class[i]) for i in pred_idx], dtype=int)

print("predbis shape:", predbis.shape, "predictions shape:", predictions.shape)
print("Unique predicted labels:", np.unique(predictions))



## === cell 15
submission = pd.DataFrame({"image_id": test_images, "label": predictions})

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission"
assert list(submission.columns) == ["image_id", "label"], "Wrong submission columns"

sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
submission.head()



## === cell 16
submission.tail()
