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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = (
    "python"  # fix protobuf import error
)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm



## === cell 1
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
test_csv_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_img_dir = "../input/plant-pathology-2021-fgvc8/test_images"

train = pd.read_csv(train_csv_path)
submissions = pd.read_csv(test_csv_path)



## === cell 2
h_target = 256
w_target = 256
batch_size = 128  # larger batch halves the number of steps per epoch
threshold = 0.5  # more balanced for F1



## === cell 3
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
labels_array = mlb.fit_transform(label_split)  # (n_samples, n_classes)
label_names = mlb.classes_

train_paths = (
    train["image"].apply(lambda fname: os.path.join(train_img_dir, fname)).values
)
test_paths = (
    submissions["image"].apply(lambda fname: os.path.join(test_img_dir, fname)).values
)



## === cell 4
cache_dir = "/tmp/tf_cache"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(cache_dir, "train_cache")
test_cache_path = os.path.join(cache_dir, "test_cache")


def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [h_target, w_target])
    img = img / 255.0
    return img


def _process_train(path, label):
    img = _load_image(path)
    return img, tf.cast(label, tf.float32)


def _process_test(path):
    img = _load_image(path)
    return img


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, labels_array))
train_ds = train_ds.map(
    _process_train, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
)
train_ds = train_ds.cache(train_cache_path)  # disk‑based cache
train_ds = train_ds.shuffle(1024, seed=42).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    _process_test, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
)
test_ds = test_ds.cache(test_cache_path)  # disk‑based cache
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)



## === cell 5
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
    pooling="avg",
)

base_model.trainable = False

x = base_model.output
output = layers.Dense(len(label_names), activation="sigmoid", dtype="float32")(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)



## === cell 6
model.fit(train_ds, epochs=8, verbose=1)



## === cell 7
preds = model.predict(test_ds, verbose=1)  # (n_test, n_classes)



## === cell 8
pred_label_list = []
for prob in preds:
    idx = np.where(prob >= threshold)[0]
    if len(idx) == 0:  # fallback to strongest prediction
        idx = [np.argmax(prob)]
    pred_label_list.append(" ".join(label_names[idx]))

submissions["labels"] = pred_label_list
submissions.to_csv("submission.csv", index=False)



## === cell 9
submissions.head()
