# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow.keras import mixed_precision

tf.config.optimizer.set_jit(True)

mixed_precision.set_global_policy("mixed_float16")

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)



## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(TEST_CSV)



## === cell 2
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(train_df["label_list"])
label_cols = mlb.classes_
label_matrix = label_matrix.astype(np.float16)
train_df["label_vec"] = list(label_matrix)

train_split, val_split = train_test_split(
    train_df, test_size=0.1, random_state=seed, stratify=label_matrix
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 128  # larger batch reduces iteration overhead

AUTOTUNE = tf.data.AUTOTUNE


def decode_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float16)  # efficient scaling
    return img


def make_image_dataset(df, img_dir, shuffle=False):
    paths = df["image"].apply(lambda img: os.path.join(img_dir, img)).values
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: decode_img(p), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, seed=seed)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds




## === cell 3
base_model = EfficientNetB4(
    weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

feature_extractor = models.Model(
    inputs=base_model.input, outputs=layers.GlobalAveragePooling2D()(base_model.output)
)




## === cell 4
def compute_features(ds):
    return feature_extractor.predict(ds, verbose=0)


train_img_ds = make_image_dataset(train_split, TRAIN_IMG_DIR, shuffle=False)
val_img_ds = make_image_dataset(val_split, TRAIN_IMG_DIR, shuffle=False)
test_img_ds = make_image_dataset(submissions, TEST_IMG_DIR, shuffle=False)

train_features = compute_features(train_img_ds)  # (N_train, 1792)
val_features = compute_features(val_img_ds)  # (N_val,   1792)
test_features = compute_features(test_img_ds)  # (N_test, 1792)



## === cell 5
feature_dim = train_features.shape[-1]

inputs = layers.Input(shape=(feature_dim,))
x = layers.Dropout(0.2)(inputs)
outputs = layers.Dense(len(label_cols), activation="sigmoid", dtype="float16")(x)

classifier = models.Model(inputs, outputs)
classifier.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.Precision(), keras.metrics.Recall()],
)




## === cell 6
def make_feature_dataset(features, labels):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.shuffle(buffer_size=2048, seed=seed)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_labels = np.stack(train_split["label_vec"].values)
val_labels = np.stack(val_split["label_vec"].values)

train_feat_ds = make_feature_dataset(train_features, train_labels)
val_feat_ds = make_feature_dataset(val_features, val_labels)



## === cell 7
classifier.fit(train_feat_ds, validation_data=val_feat_ds, epochs=3, verbose=2)



## === cell 8
preds = classifier.predict(test_features, batch_size=BATCH_SIZE, verbose=0)



## === cell 9
THRESH = 0.2
pred_labels = []
for prob_vec in preds:
    idx = np.where(prob_vec >= THRESH)[0]
    if len(idx) == 0:
        idx = [np.argmax(prob_vec)]
    tags = [label_cols[i] for i in idx]
    if "healthy" in tags and len(tags) > 1:
        tags = [t for t in tags if t != "healthy"]
    pred_labels.append(" ".join(tags))

submissions["labels"] = pred_labels
submissions.to_csv("submission.csv", index=False)



## === cell 10
print(submissions.head())
