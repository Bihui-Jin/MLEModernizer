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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
import random

tf.config.optimizer.set_jit(True)

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception as e:
    print("Mixed precision not available:", e)

try:
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
except Exception as e:
    print("Thread config skipped:", e)




## === cell 1
def auto_select_accelerator():
    """
    Reference:
        * https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
        * https://www.kaggle.com/xhlulu/ranzcr-efficientnet-tpu-training
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[3]  # 300 instead of 380

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

class_name = df.labels.unique().tolist()
print("Classes:", class_name)
print("Number of classes:", len(class_name))

df.labels = df.labels.astype(str)
n_labels = len(class_name)




## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = 128

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"

test_filenames = sorted(os.listdir(test_dir))
test_df = pd.DataFrame()
test_df["image"] = test_filenames

AUTOTUNE = tf.data.AUTOTUNE
file_pattern = os.path.join(test_dir, "*.jpg")
list_ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)


def _load_image(file_path):
    """Decode, resize and normalize an image – no randomness."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [im_size, im_size])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def _augment_image(img):
    """Apply cheap random flips for test‑time augmentation."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


cached_ds = (
    list_ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    .cache("/tmp/test_cache")  # disk cache
    .map(_augment_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = cached_ds  # reuse the same pipeline for every TTA iteration

TEST_STEPS = int(np.ceil(len(test_filenames) / BATCH_SIZE))




## === cell 4
with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights="imagenet", include_top=False, input_shape=(im_size, im_size, 3)
    )
    model = Sequential(
        [base, GlobalMaxPooling2D(), Dense(n_labels, activation="softmax")]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
    )
    model.summary()




## === cell 5
weights_path = "/kaggle/input/model222/bestmodel_tpu_aug.h5"
if os.path.exists(weights_path):
    try:
        model.load_weights(weights_path)
        print("Loaded weights from", weights_path)
    except Exception as e:
        print("Failed to load weights:", e)
else:
    print("No fine‑tuned weights found; using ImageNet pretrained model.")




## === cell 6
TTA = 4
tta_ds = test_ds.repeat(TTA)

preds = model.predict(
    tta_ds,
    steps=TEST_STEPS * TTA,
    verbose=0,
)

pred_sum = preds.reshape(TTA, len(test_filenames), n_labels).sum(axis=0)

pred = pred_sum / TTA
argpred = np.argmax(pred, axis=1)
test_df["labels"] = [class_name[idx] for idx in argpred]

submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(test_df.head())
