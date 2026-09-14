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
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

np.random.seed(42)
tf.random.set_seed(42)


def resolve_path(*parts):
    rel_path = os.path.join(*parts)
    if os.path.exists(rel_path):
        return rel_path
    kaggle_path = os.path.join("/kaggle/input", *parts[1:])  # drop leading ".."
    if os.path.exists(kaggle_path):
        return kaggle_path
    raise FileNotFoundError(f"Could not find path: {rel_path} nor {kaggle_path}")




## === cell 1
train = pd.read_csv(
    resolve_path("..", "input", "plant-pathology-2021-fgvc8", "train.csv")
)
submissions = pd.read_csv(
    resolve_path("..", "input", "plant-pathology-2021-fgvc8", "sample_submission.csv")
)




## === cell 2
h_target, w_target = 256, 256
batch_size = 64

label_split = train["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(label_split)
num_classes = len(mlb.classes_)




## === cell 3
train_dir = resolve_path("..", "input", "plant-pathology-2021-fgvc8", "train_images")
train_paths = train["image"].apply(lambda x: os.path.join(train_dir, x)).values
train_labels = mlb.transform(label_split)


def _load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [h_target, w_target])
    img = img / 255.0
    return img, label


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(
    _load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
)
train_ds = train_ds.cache(filename="/tmp/train_cache")
train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)




## === cell 4
base = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=(h_target, w_target, 3)
)
base.trainable = False  # keep frozen for quick training

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
output = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
model = tf.keras.Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy"
)




## === cell 5
model.fit(train_ds, epochs=5, verbose=1)




## === cell 6
test_dir = resolve_path("..", "input", "plant-pathology-2021-fgvc8", "test_images")
test_paths = submissions["image"].apply(lambda x: os.path.join(test_dir, x)).values


def _load_image_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [h_target, w_target])
    img = img / 255.0
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    _load_image_test, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
)
test_ds = test_ds.cache(filename="/tmp/test_cache")
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)




## === cell 7
preds = model.predict(test_ds, verbose=1)




## === cell 8
threshold = 0.4
for i, probs in enumerate(preds):
    idx = np.where(probs >= threshold)[0]
    if idx.size == 0:
        idx = [np.argmax(probs)]
    selected_labels = mlb.classes_[idx]
    if "healthy" in selected_labels and len(selected_labels) > 1:
        selected_labels = selected_labels[selected_labels != "healthy"]
    submissions.at[i, "labels"] = " ".join(selected_labels)

submissions.to_csv("submission.csv", index=False)
