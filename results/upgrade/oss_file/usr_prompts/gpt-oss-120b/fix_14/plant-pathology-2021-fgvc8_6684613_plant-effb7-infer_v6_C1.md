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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
import random

from tensorflow.keras import mixed_precision

tf.config.threading.set_intra_op_parallelism_threads(min(4, os.cpu_count()))
tf.config.threading.set_inter_op_parallelism_threads(min(4, os.cpu_count()))

tf.config.optimizer.set_jit(True)

mixed_precision.set_global_policy("mixed_float16")

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)




## === cell 1
def auto_select_accelerator():
    """
    Detect TPU if available, otherwise fall back to default strategy.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except (ValueError, RuntimeError):
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[0]  # 224x224

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

class_name = df.labels.unique().tolist()
print("Classes:", class_name)
n_labels = len(class_name)

df.labels = df.labels.astype(str)

train_img_dir = os.path.join(load_dir, "train_images")
train_paths = [os.path.join(train_img_dir, fname) for fname in df["image"]]
label_to_idx = {name: idx for idx, name in enumerate(class_name)}
train_labels_idx = df["labels"].map(label_to_idx).values
train_labels_onehot = to_categorical(train_labels_idx, num_classes=n_labels)


def _load_image_train(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [im_size, im_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


options = tf.data.Options()
options.experimental_deterministic = False  # allow non‑deterministic order for speed
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels_onehot))
train_ds = train_ds.map(
    lambda p, l: (_load_image_train(p), l),
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False,
)
train_ds = train_ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)

cache_file = os.path.join(os.getenv("TMPDIR", "/tmp"), "train_cache")
train_ds = train_ds.cache(cache_file)

train_ds = train_ds.batch(128, drop_remainder=True)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.with_options(options)

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_files = sorted(os.listdir(test_dir))  # keep deterministic order
test_paths = [os.path.join(test_dir, f) for f in test_files]


def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [im_size, im_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def _augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


_base_test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
_base_test_ds = _base_test_ds.map(
    _load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
).cache()


def get_tta_dataset(TTA):
    """
    Build a single dataset that repeats the augmented test data TTA times,
    so we only create the pipeline once and avoid per‑iteration overhead.
    """
    d = _base_test_ds.map(
        _augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    d = d.repeat(TTA)  # repeat TTA times
    d = d.batch(128, drop_remainder=False)
    d = d.prefetch(tf.data.AUTOTUNE)
    return d




## === cell 3
strategy = auto_select_accelerator()
with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights="imagenet", include_top=False, input_shape=(im_size, im_size, 3)
    )
    model = Sequential(
        [
            base,
            GlobalMaxPooling2D(),
            Dense(n_labels, activation="softmax", dtype="float32"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
    )
    model.summary()



## === cell 4
EPOCHS = 3
model.fit(train_ds, epochs=EPOCHS, verbose=1)

TTA = 6
tta_ds = get_tta_dataset(TTA)

preds_concat = model.predict(tta_ds, verbose=0)

num_test = len(test_files)
preds = preds_concat.reshape(TTA, num_test, n_labels)
pred = np.mean(preds, axis=0)  # shape (num_test, n_labels)
argpred = np.argmax(pred, axis=1)

submission_df = pd.DataFrame({"image": test_files})
submission_df["labels"] = [class_name[idx] for idx in argpred]
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission_df.head())
