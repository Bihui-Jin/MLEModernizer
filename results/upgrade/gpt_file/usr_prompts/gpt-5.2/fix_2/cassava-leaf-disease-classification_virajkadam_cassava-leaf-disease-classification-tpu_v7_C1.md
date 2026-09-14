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
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array, smart_resize

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_dir = os.path.join(BASE_DIR, "train_images")
test_dir = os.path.join(BASE_DIR, "test_images")

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"

train = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print(train.head())
print(sample_sub.head())
print("Train size:", len(train), " Test size:", len(sample_sub))



## === cell 2

NUM_CLASSES = 5
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # keep conservative for memory
AUTOTUNE = tf.data.AUTOTUNE


def build_model(backbone_name: str, input_shape=(512, 512, 3), num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    if backbone_name == "inceptionresnetv2":
        backbone = keras.applications.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone_name == "efficientnetv2b0":
        backbone = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError("Unknown backbone")

    backbone.trainable = True  # allow fine-tuning to improve accuracy toward target
    x = backbone(x, training=True)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


model1 = build_model(
    "inceptionresnetv2", input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES
)
model2 = build_model(
    "efficientnetv2b0", input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES
)

opt1 = keras.optimizers.Adam(learning_rate=1e-4)
opt2 = keras.optimizers.Adam(learning_rate=1e-4)

model1.compile(
    optimizer=opt1, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model2.compile(
    optimizer=opt2, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("Model1 params:", model1.count_params())
print("Model2 params:", model2.count_params())




## === cell 3
def decode_image_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_train_ds(df, training=True):
    paths = tf.constant([os.path.join(train_dir, x) for x in df["image_id"].values])
    labels = tf.constant(df["label"].values, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(p, y):
        x = decode_image_from_path(p)
        if training:
            x = tf.image.random_flip_left_right(x, seed=SEED)
        return x, y

    if training:
        ds = ds.shuffle(min(len(df), 2048), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_shuffled = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_shuffled) * val_frac)
val_df = train_shuffled.iloc[:val_size].copy()
trn_df = train_shuffled.iloc[val_size:].copy()

train_ds = make_train_ds(trn_df, training=True)
val_ds = make_train_ds(val_df, training=False)

print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())



## === cell 4
EPOCHS = 2  # keep within runtime budget while improving over random; not an approximation like early stopping.

history1 = model1.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
history2 = model2.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## === cell 5
def sample_df(sample_size=50):
    df = train.sample(sample_size, random_state=SEED).reset_index(drop=True)
    return df


dfs = sample_df(sample_size=50)

preds = []
y_true = dfs["label"].values

for im_id in dfs["image_id"].values:
    img = load_img(os.path.join(train_dir, im_id))
    img = img_to_array(img)
    img = smart_resize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    p = (0.5 * model1.predict(img, verbose=0)) + (0.5 * model2.predict(img, verbose=0))
    pred = int(np.argmax(p, axis=1)[0])
    preds.append(pred)

acc = (np.array(preds) == y_true).mean()
print("Sample accuracy (50 imgs):", acc)



## === cell 6
sample_test = pd.DataFrame({"Prediction": preds, "Actual": y_true})
print(sample_test.head(30))



## === cell 7
test_paths = [os.path.join(test_dir, x) for x in sample_sub["image_id"].values]
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _map_test(p):
    x = decode_image_from_path(p)
    return x


test_ds = (
    test_ds.map(_map_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())



## === cell 8
probs1 = model1.predict(test_ds, verbose=1)
probs2 = model2.predict(test_ds, verbose=1)

probs = 0.5 * probs1 + 0.5 * probs2
predictions = np.argmax(probs, axis=1).astype(int)

print("Predictions shape:", predictions.shape, "Unique labels:", np.unique(predictions))



## === cell 9
submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": predictions}
)
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

print(submission.head())
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(submission))
