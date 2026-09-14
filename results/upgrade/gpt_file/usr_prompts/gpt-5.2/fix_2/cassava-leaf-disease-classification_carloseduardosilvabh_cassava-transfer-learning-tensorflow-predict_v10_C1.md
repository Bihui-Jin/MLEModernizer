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
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## === cell 1
image = tf.keras.preprocessing.image.load_img(
    r"../input/cassava-leaf-disease-classification/train_images/1000015157.jpg"
)
image



## === cell 2
plt.imread(
    "../input/cassava-leaf-disease-classification/train_images/1000015157.jpg"
).shape



## === cell 3
path = "../input/cassava-leaf-disease-classification"



## === cell 4
train_images = os.listdir(os.path.join(path, "train_images"))
print("Total images for Train: ", len(train_images))



## === cell 5
with open(
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
) as file:
    classes = json.loads(file.read())

print(json.dumps(classes, indent=4))



## === cell 6
df_train = pd.read_csv(os.path.join(path, "train.csv"))
df_train.head()



## === cell 7
df_train["class"] = df_train["label"].map({int(i): c for i, c in classes.items()})
df_train.head()



## === cell 8
plt.subplots(figsize=(12, 8))
ax = sns.countplot(x="class", data=df_train)

for p in ax.patches:
    ax.annotate("{:1}".format(p.get_height()), (p.get_x() + 0.3, p.get_height()))
plt.xticks(rotation=90)
ax.set_title("quantities by classes", fontdict={"fontsize": 15})
plt.show()




## === cell 9
def plot(images, labels, predictions=None):
    n_cols = min(4, len(images))
    n_rows = math.ceil(len(images) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 16))

    if predictions is None:
        predictions = [None] * len(labels)

    for i, (x, y_true, y_pred) in enumerate(zip(images, labels, predictions)):
        ax = axes.flat[i]
        a = plt.imread(os.path.join(path, "train_images", x))
        ax.imshow(a)
        ax.set_title(f"Class: {y_true}")

        if y_pred is not None:
            ax.set_xlabel(f"Pred: {y_pred}", color="blue", fontweight="bold")

        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, n_rows * n_cols):
        axes.flat[j].axis("off")




## === cell 10
df_0 = df_train[df_train["label"] == 0].sample(12, random_state=SEED)
df_0_id = df_0["image_id"].values
df_0_class = df_0["class"].values



## === cell 11
plot(df_0_id, df_0_class)



## === cell 12
df_1 = df_train[df_train["label"] == 1].sample(12, random_state=SEED)
df_1_id = df_1["image_id"].values
df_1_class = df_1["class"].values



## === cell 13
plot(df_1_id, df_1_class)



## === cell 14
df_2 = df_train[df_train["label"] == 2].sample(12, random_state=SEED)
df_2_id = df_2["image_id"].values
df_2_class = df_2["class"].values



## === cell 15
plot(df_2_id, df_2_class)



## === cell 16
df_3 = df_train[df_train["label"] == 3].sample(12, random_state=SEED)
df_3_id = df_3["image_id"].values
df_3_class = df_3["class"].values



## === cell 17
plot(df_3_id, df_3_class)



## === cell 18
df_4 = df_train[df_train["label"] == 4].sample(12, random_state=SEED)
df_4_id = df_4["image_id"].values
df_4_class = df_4["class"].values



## === cell 19
plot(df_4_id, df_4_class)



## === cell 20
train = df_train.astype({"label": str})
from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    train, test_size=0.2, random_state=SEED, stratify=train["label"]
)



## === cell 21
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)



## === cell 22
img_size = 300
size = (img_size, img_size)



## === cell 23
train_generator = train_datagen.flow_from_dataframe(
    train,
    directory=path + "/train_images",
    x_col="image_id",
    y_col="class",
    target_size=size,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=SEED,
    interpolation="nearest",
)



## === cell 24
valid_generator = train_datagen.flow_from_dataframe(
    valid,
    directory=path + "/train_images",
    x_col="image_id",
    y_col="class",
    target_size=size,
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
    seed=SEED,
    interpolation="nearest",
)




## === cell 25
def modelTransf():
    model = models.Sequential()
    model.add(
        EfficientNetB3(
            input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
        )
    )
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation="softmax"))
    return model




## === cell 26
model = modelTransf()



## === cell 27
model.summary()



## === cell 28
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)



## === cell 29
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, mode="min", restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "modelB3.keras", monitor="val_loss", verbose=1, mode="min", save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=10, min_lr=0.001, mode="min", verbose=1
)



## === cell 30
step_size_train = max(1, train_generator.n // train_generator.batch_size)
step_size_valid = max(1, valid_generator.n // valid_generator.batch_size)
step_size_train, step_size_valid



## === cell 31
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=30,
    steps_per_epoch=step_size_train,
    validation_steps=step_size_valid,
    callbacks=[early_stopping, checkpoint, reduce_lr],
)



## === cell 32
best_model_path = "modelB3.keras"
if os.path.exists(best_model_path):
    model_trained = keras.models.load_model(best_model_path)
else:
    model_trained = model



## === cell 33
from sklearn.metrics import accuracy_score, confusion_matrix

valid_generator.reset()
val_probs = model_trained.predict(
    valid_generator,
    steps=math.ceil(valid_generator.n / valid_generator.batch_size),
    verbose=0,
)
val_pred = np.argmax(val_probs, axis=1)[: valid_generator.n]
val_true = valid_generator.classes
print("Validation accuracy:", accuracy_score(val_true, val_pred))



## === cell 34
submission_file = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_file.head()



## === cell 35
test_df = submission_file.copy()
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=path + "/test_images",
    x_col="image_id",
    y_col=None,
    target_size=size,
    class_mode=None,
    batch_size=32,
    shuffle=False,
    seed=SEED,
    interpolation="nearest",
)



## === cell 36
test_generator.reset()
test_probs = model_trained.predict(
    test_generator,
    steps=math.ceil(test_generator.n / test_generator.batch_size),
    verbose=1,
)
test_pred = np.argmax(test_probs, axis=1)[: test_generator.n]
len(test_pred), test_generator.n



## === cell 37
submission = submission_file.copy()
submission["label"] = test_pred.astype(int)
submission.head()



## === cell 38
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Saved at:", os.path.abspath("submission.csv"))
