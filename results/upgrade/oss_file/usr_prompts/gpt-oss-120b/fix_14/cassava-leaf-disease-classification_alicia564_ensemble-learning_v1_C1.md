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

3.13

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
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

NUM_WORKERS = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(NUM_WORKERS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_WORKERS)

BATCH_SIZE = 256  # larger batch reduces steps

tf.keras.mixed_precision.set_global_policy("mixed_float16")

tf.random.set_seed(42)
np.random.seed(42)




## === cell 1
label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)
train_csv["label"] = train_csv["label"].astype(str)

train_df, valid_df = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

label_classes = sorted(train_csv["label"].unique())
label2idx = {lbl: idx for idx, lbl in enumerate(label_classes)}
idx2label = {idx: lbl for lbl, idx in label2idx.items()}




## === cell 2
def preprocess_path(path):
    """Read image file, decode, resize, and apply EfficientNet preprocessing."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)
    return img


def make_dataset(df, shuffle=False, cache=False):
    """
    Build a tf.data pipeline.
    `cache` is False for large datasets (train, test) to avoid huge memory usage.
    """
    paths = df["path"].values
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(preprocess_path, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()  # cache only when beneficial (e.g., validation)

    if shuffle:
        buffer_size = min(10000, len(paths))
        ds = ds.shuffle(
            buffer_size=buffer_size, seed=42, reshuffle_each_iteration=False
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(
    train_df, shuffle=True, cache=False
)  # no caching for training set
valid_ds = make_dataset(
    valid_df, shuffle=False, cache=True
)  # cache validation for fast reuse




## === cell 3
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False
gap = GlobalAveragePooling2D()(base_model.output)
backbone = Model(inputs=base_model.input, outputs=gap, name="backbone")


def extract_features(dataset):
    """Extract features from all batches of `dataset` using the backbone."""
    return backbone.predict(dataset, verbose=0)


train_features = extract_features(train_ds)  # shape (N_train, 1280)
valid_features = extract_features(valid_ds)  # shape (N_valid, 1280)

train_labels = train_df["label"].map(label2idx).values
valid_labels = valid_df["label"].map(label2idx).values

train_labels_oh = tf.keras.utils.to_categorical(train_labels, num_classes=5)
valid_labels_oh = tf.keras.utils.to_categorical(valid_labels, num_classes=5)




## === cell 4
input_feat = Input(shape=(1280,))
output = Dense(5, activation="softmax", dtype="float32")(
    input_feat
)  # force float32 for stability
top_model = Model(inputs=input_feat, outputs=output)

top_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
)
lr_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
)

top_model.fit(
    train_features,
    train_labels_oh,
    epochs=5,
    batch_size=64,
    validation_data=(valid_features, valid_labels_oh),
    callbacks=[early_stop, lr_reduce],
    verbose=2,
)




## === cell 5
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
test_filenames = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_df = pd.DataFrame(
    {
        "image_id": test_filenames,
        "path": [os.path.join(test_dir, fname) for fname in test_filenames],
    }
)

test_ds = make_dataset(test_df, shuffle=False, cache=False)  # no caching for test set
test_features = extract_features(test_ds)

test_probs = top_model.predict(test_features, batch_size=64, verbose=0)
pred_indices = np.argmax(test_probs, axis=1)
pred_labels = [idx2label[idx] for idx in pred_indices]

submission = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
