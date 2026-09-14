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

# 5. Target score

0.7886068298579632

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(tf.config.threading.cpu_count())

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(str)

label_to_index = {str(i): i for i in range(5)}
train_df["label_idx"] = train_df["label"].map(label_to_index)

from sklearn.model_selection import train_test_split

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label_idx"],
)

base_resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_resnet.trainable = False
gap = GlobalAveragePooling2D()(base_resnet.output)
feature_extractor = Model(
    inputs=base_resnet.input, outputs=gap, name="feature_extractor"
)

BATCH_SIZE = 128


def build_feature_dataset(df, augment=False):
    file_paths = (
        df["image_id"].apply(lambda fname: os.path.join(TRAIN_IMG_DIR, fname)).values
    )
    labels = df["label_idx"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def _load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = preprocess_input(img)
        return img, label

    ds = ds.map(
        _load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    if augment:
        try:
            import tensorflow_addons as tfa

            def _augment(img, label):
                img = tf.image.random_flip_left_right(img, seed=SEED)
                angle_rad = tf.random.uniform([], -20, 20, seed=SEED) * (
                    3.14159265 / 180.0
                )
                img = tfa.image.rotate(img, angle_rad, fill_mode="reflect")
                scales = tf.random.uniform([], 0.8, 1.2, seed=SEED)
                new_size = tf.cast(scales * 224, tf.int32)
                img = tf.image.resize(img, [new_size, new_size])
                img = tf.image.resize_with_crop_or_pad(img, 224, 224)
                return img, label

        except ImportError:

            def _augment(img, label):
                img = tf.image.random_flip_left_right(img, seed=SEED)
                return img, label

        ds = ds.map(
            _augment,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    ds = ds.shuffle(buffer_size=1024, seed=SEED, reshuffle_each_iteration=False)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def _extract_features_batch(img_batch, label_batch):
        feats = feature_extractor(img_batch, training=False)  # (B, 2048)
        label_onehot = tf.one_hot(label_batch, depth=5)
        return feats, label_onehot

    ds = (
        ds.map(
            _extract_features_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


train_feature_dataset = build_feature_dataset(train_split, augment=True)
val_feature_dataset = build_feature_dataset(val_split, augment=False)

classifier = Sequential(
    [
        Input(shape=feature_extractor.output_shape[1:]),  # (2048,)
        Dense(5, activation="softmax"),
    ],
    name="classifier",
)

classifier.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

classifier.fit(
    train_feature_dataset,
    epochs=5,
    validation_data=val_feature_dataset,
    verbose=1,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2439176306.py in <cell line: 0>()
----> 1 train_df = pd.read_csv(TRAIN_CSV)
      2 train_df["label"] = train_df["label"].astype(str)
      3 
      4 label_to_index = {str(i): i for i in range(5)}
      5 train_df["label_idx"] = train_df["label"].map(label_to_index)

NameError: name 'TRAIN_CSV' is not defined

## === cell 2
test_filenames = sorted(
    [
        f
        for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)

test_paths = tf.convert_to_tensor(
    [os.path.join(TEST_IMG_DIR, fname) for fname in test_filenames]
)


def _load_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)
    return img


test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
test_dataset = test_dataset.map(
    _load_test, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
)

test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def _extract_test_features_batch(img_batch):
    feats = feature_extractor(img_batch, training=False)  # (B, 2048)
    return feats


test_feature_dataset = test_dataset.map(
    _extract_test_features_batch,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False,
)

pred_probs = classifier.predict(test_feature_dataset, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1970710180.py in <cell line: 0>()
      2     [
      3         f
----> 4         for f in os.listdir(TEST_IMG_DIR)
      5         if f.lower().endswith((".png", ".jpg", ".jpeg"))
      6     ]

NameError: name 'TEST_IMG_DIR' is not defined

## === cell 3
submission = pd.DataFrame(
    {
        "image_id": test_filenames,
        "label": pred_labels,
    }
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1181916537.py in <cell line: 0>()
      1 submission = pd.DataFrame(
      2     {
----> 3         "image_id": test_filenames,
      4         "label": pred_labels,
      5     }

NameError: name 'test_filenames' is not defined
