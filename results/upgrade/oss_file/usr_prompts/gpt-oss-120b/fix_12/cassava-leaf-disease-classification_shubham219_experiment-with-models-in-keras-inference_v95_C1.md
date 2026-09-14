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

3.11

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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

SEED = 42
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(str)
df_train["filepath"] = df_train["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x)
)

train_df, val_df = train_test_split(
    df_train,
    test_size=0.2,
    random_state=SEED,
    stratify=df_train["label"],
)




## === cell 1
batch_size = 512
target_size = (256, 256)

if TF_AVAILABLE:
    preprocess = tf.keras.applications.resnet50.preprocess_input

    def _load_and_preprocess(path, label):
        """Read image file, decode, resize, and apply ResNet preprocessing."""
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = preprocess(img)
        return img, label

    train_paths = train_df["filepath"].values
    train_labels = tf.keras.utils.to_categorical(
        train_df["label"].astype(int).values, num_classes=5
    )
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = (
        train_ds.shuffle(
            buffer_size=len(train_paths), seed=SEED, reshuffle_each_iteration=False
        )
        .map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()  # <-- cache after first epoch processing
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_paths = val_df["filepath"].values
    val_labels = tf.keras.utils.to_categorical(
        val_df["label"].astype(int).values, num_classes=5
    )
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = (
        val_ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()  # <-- cache validation set as well
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    base_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(256, 256, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    my_model = tf.keras.Model(inputs, outputs)

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    my_model.fit(
        train_ds,
        epochs=8,
        validation_data=val_ds,
        verbose=2,
    )
else:

    def load_images(df, size=(64, 64)):
        imgs = []
        for path in df["filepath"]:
            img = Image.open(path).convert("RGB")
            img = img.resize(size)
            imgs.append(np.asarray(img).flatten())
        return np.stack(imgs)

    X_train = load_images(train_df, size=(64, 64))
    X_val = load_images(val_df, size=(64, 64))

    y_train = train_df["label"].astype(int).values
    y_val = val_df["label"].astype(int).values

    my_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=30,
            random_state=SEED,
            batch_size=256,
        ),
    )
    my_model.fit(X_train, y_train)




## === cell 2
test_images = [
    os.path.join(TEST_IMG_DIR, fname)
    for fname in sorted(os.listdir(TEST_IMG_DIR))
    if fname.lower().endswith(".jpg")
]
df_test = pd.DataFrame({"path": test_images})

if TF_AVAILABLE:

    def _load_test(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = preprocess(img)
        return img

    test_paths = df_test["path"].values
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = (
        test_ds.map(_load_test, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()  # <-- cache test images after loading once
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
else:
    pass




## === cell 3
if TF_AVAILABLE:
    pred_test = my_model.predict(test_ds, verbose=2)
    pred_test_labels = np.argmax(pred_test, axis=-1)
else:
    X_test = []
    for path in df_test["path"]:
        img = Image.open(path).convert("RGB")
        img = img.resize((64, 64))
        X_test.append(np.asarray(img).flatten())
    X_test = np.stack(X_test)
    pred_test_labels = my_model.predict(X_test)

final_submission = pd.DataFrame(
    {
        "image_id": df_test["path"].apply(lambda p: os.path.basename(p)),
        "label": pred_test_labels.astype(int),
    }
)

final_submission.to_csv("submission.csv", index=False)




## === cell 4
final_submission.head()
