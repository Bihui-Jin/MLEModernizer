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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from pathlib import Path

print("TensorFlow:", tf.__version__)



## === cell 1
BASE = Path("/kaggle/input/cassava-leaf-disease-classification")
train_csv_path = str(BASE / "train.csv")
label_json_path = str(BASE / "label_num_to_disease_map.json")
train_images_dir = BASE / "train_images"
test_images_dir = BASE / "test_images"
sample_sub_path = str(BASE / "sample_submission.csv")

assert Path(train_csv_path).exists(), f"Missing: {train_csv_path}"
assert Path(sample_sub_path).exists(), f"Missing: {sample_sub_path}"
assert train_images_dir.exists(), f"Missing: {train_images_dir}"
assert test_images_dir.exists(), f"Missing: {test_images_dir}"



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")
train_csv["label_int"] = train_csv["label"].astype(int)

label_class_j = pd.read_json(label_json_path, orient="index")
label_class = label_class_j.values.flatten().tolist()

print(train_csv.head())
print("Num classes:", len(label_class))



## === cell 3
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)



## === cell 4
img_ids = train_csv.image_id.values[:8]
img_lbls = train_csv.label.values[:8]



## === cell 5
images_collection = []
for img_id in img_ids:
    path = str(train_images_dir / str(img_id))
    img_arr = cv2.imread(path)
    if img_arr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    img_arr = cv2.resize(img_arr, (150, 150))
    images_collection.append(img_arr)



## === cell 6
plt.figure(figsize=(18, 10))
for i in range(8):
    plt.subplot(2, 4, i % 8 + 1)
    plt.imshow(images_collection[i])
    plt.title(label_class[int(img_lbls[i])])
    plt.axis("off")
plt.show()



## === cell 7
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(
    train_csv,
    test_size=0.15,
    random_state=SEED,
    stratify=train_csv["label_int"],
)

print("Train size:", len(df_train), "Val size:", len(df_val))
print(
    "Train label distribution:\n",
    df_train["label_int"].value_counts(normalize=True).sort_index(),
)



## === cell 8

NUM_CLASSES = 5
IMG_SIZE = 320
model_1_img_size = 32
model_2_img_size = 320

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE_1 = 128
BATCH_SIZE_2 = 16


def _read_image(path, size):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [size, size], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_ds(df, images_root, size, batch_size, training, label_mode):
    paths = (images_root / df["image_id"].values).astype(str)
    if label_mode == "binary_healthy":
        y = (df["label_int"].values == 4).astype(np.int32)
    elif label_mode == "multiclass":
        y = df["label_int"].values.astype(np.int32)
    else:
        raise ValueError("Unknown label_mode")

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    def _map(p, lbl):
        img = _read_image(p, size)
        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
        return img, lbl

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


train_ds_1 = make_ds(
    df_train, train_images_dir, model_1_img_size, BATCH_SIZE_1, True, "binary_healthy"
)
val_ds_1 = make_ds(
    df_val, train_images_dir, model_1_img_size, BATCH_SIZE_1, False, "binary_healthy"
)


def build_stage1(input_size):
    inp = tf.keras.Input(shape=(input_size, input_size, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


model_1 = build_stage1(model_1_img_size)
model_1.summary()



## === cell 9
EPOCHS_1 = 3
history_1 = model_1.fit(
    train_ds_1,
    validation_data=val_ds_1,
    epochs=EPOCHS_1,
    verbose=2,
)



## === cell 10
train_ds_2 = make_ds(
    df_train, train_images_dir, model_2_img_size, BATCH_SIZE_2, True, "multiclass"
)
val_ds_2 = make_ds(
    df_val, train_images_dir, model_2_img_size, BATCH_SIZE_2, False, "multiclass"
)


def build_stage2(input_size, num_classes):
    inp = tf.keras.Input(shape=(input_size, input_size, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model_2 = build_stage2(model_2_img_size, NUM_CLASSES)
model_2.summary()



## === cell 11
EPOCHS_2 = 3
history_2 = model_2.fit(
    train_ds_2,
    validation_data=val_ds_2,
    epochs=EPOCHS_2,
    verbose=2,
)



## === cell 12
print("IMG_SIZE:", IMG_SIZE)
print("model_1_img_size:", model_1_img_size)
print("model_2_img_size:", model_2_img_size)



## === cell 13
ss_tmp = pd.read_csv(sample_sub_path)
test_img_path = str(test_images_dir / ss_tmp["image_id"].iloc[0])

img = cv2.imread(test_img_path)
if img is None:
    raise FileNotFoundError(f"Could not read test image: {test_img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_img = (
    cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
)

plt.figure(figsize=(8, 4))
plt.title(f"TEST IMAGE: {Path(test_img_path).name}")
plt.imshow(resized_img[0])
plt.axis("off")
plt.show()



## === cell 14
HEALTHY_THRESHOLD = 0.5


def predict_one(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    img1 = tf.keras.preprocessing.image.smart_resize(
        img, (model_1_img_size, model_1_img_size)
    )
    img1 = tf.reshape(img1, (-1, model_1_img_size, model_1_img_size, 3)) / 255.0
    p1 = model_1.predict(img1, verbose=0)[0, 0]  # probability healthy

    if p1 >= HEALTHY_THRESHOLD:
        return 4
    else:
        img2 = tf.keras.preprocessing.image.smart_resize(
            img, (model_2_img_size, model_2_img_size)
        )
        img2 = tf.reshape(img2, (-1, model_2_img_size, model_2_img_size, 3)) / 255.0
        p2 = model_2.predict(img2, verbose=0)[0]
        return int(np.argmax(p2))




## === cell 15
preds = []
ss = pd.read_csv(sample_sub_path)

for image in ss.image_id.values:
    image_path = str(test_images_dir / image)
    preds.append(predict_one(image_path))

my_submission = pd.DataFrame({"image_id": ss.image_id.values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())



## === cell 16
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: /kaggle/working/submission.csv")
