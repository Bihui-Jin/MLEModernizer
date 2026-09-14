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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing import image as kimage

tf.config.optimizer.set_jit(True)

np.random.seed(42)
tf.random.set_seed(42)

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)


def auto_select_accelerator():
    """
    Return a TensorFlow distribution strategy.
    Simplified to avoid protobuf‑related errors.
    """
    return tf.distribute.get_strategy()




## === cell 1
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[0]

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

strategy = auto_select_accelerator()
BATCH_SIZE = 128  # increased batch size for better GPU utilization
n_labels = 5  # number of disease classes

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_files = [
    f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))
]
test_df = pd.DataFrame({"image": test_files})

print("Building tf.data pipeline for test images...")

test_image_paths = [os.path.join(test_dir, fname) for fname in test_df["image"]]


def _load_and_preprocess(path):
    """Read, decode, resize, and apply EfficientNet preprocessing."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [im_size, im_size])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def _augment(image, seed):
    """Stateless random augmentations matching the original NumPy version."""
    s = tf.cast([seed, 0], tf.int32)
    img = tf.image.stateless_random_flip_left_right(image, s)
    s = tf.cast([seed + 1, 0], tf.int32)
    img = tf.image.stateless_random_flip_up_down(img, s)
    s = tf.cast([seed + 2, 0], tf.int32)
    img = tf.image.stateless_random_brightness(img, max_delta=0.2, seed=s)
    return img


TTA = 3  # number of test‑time augmentations

options = tf.data.Options()
options.experimental_deterministic = False

load_ds = (
    tf.data.Dataset.from_tensor_slices(test_image_paths)
    .with_options(options)
    .map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .repeat(TTA)
    .enumerate()
    .map(
        lambda i, img: _augment(img, seed=42 + tf.cast(tf.math.mod(i, TTA), tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

base_ds = load_ds  # keep the original variable name for downstream code




## === cell 2
with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )
    model = Sequential(
        [base, GlobalAveragePooling2D(), Dense(n_labels, activation="sigmoid")]
    )




## === cell 3
weights_path = "/kaggle/input/effnetb7-2/besteffb7_2.h5"
if os.path.exists(weights_path):
    try:
        model.load_weights(weights_path)
        print("Weights loaded.")
    except Exception as e:
        print("Failed to load weights:", e)
else:
    print("Weights file not found – using randomly initialized model.")




## === cell 4
preds_all = model.predict(base_ds, verbose=0)

num_images = len(test_df)
preds = preds_all.reshape(num_images, TTA, n_labels).mean(axis=1)




## === cell 5
name = {
    0: "complex",
    1: "scab",
    2: "frog_eye_leaf_spot",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",  # fallback label when no disease passes its threshold
}
threshold = {0: 0.25, 1: 0.35, 2: 0.6, 3: 0.8, 4: 0.8}

pred_string = []
for line in preds:
    s = ""
    for i in range(n_labels):
        if line[i] > threshold.get(i, 0.5):
            s += name[i] + " "
    s = s.strip()
    if not s:
        s = name[6]
    pred_string.append(s)

test_df["labels"] = pred_string
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(test_df.head())
