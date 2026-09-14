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
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.keras.mixed_precision.set_global_policy("mixed_float16")
np.random.seed(42)
tf.random.set_seed(42)




## === cell 1
def auto_select_accelerator():
    """
    Detect and initialise TPU if available; otherwise fall back to CPU/GPU.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
        print(f"Running on {strategy.__class__.__name__}")
    print(f"Replicas in sync: {strategy.num_replicas_in_sync}")
    return strategy




## === cell 2
load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_df = pd.read_csv(os.path.join(load_dir, "train.csv"))

strategy = auto_select_accelerator()
BATCH_SIZE = 128  # increased batch size → fewer steps per epoch
im_size = 600
n_labels = 5  # disease classes (healthy handled separately)

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_image_paths = (
    glob.glob(os.path.join(test_dir, "*.jpg"))
    + glob.glob(os.path.join(test_dir, "*.jpeg"))
    + glob.glob(os.path.join(test_dir, "*.png"))
)
test_filenames = [os.path.basename(p) for p in test_image_paths]
test_df = pd.DataFrame({"image": test_filenames})



## === cell 3
label_map = {
    0: "complex",
    1: "scab",
    2: "frog_eye_leaf_spot",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",  # fallback class
}


def encode_labels(lbl_str):
    vec = np.zeros(n_labels, dtype=int)
    for lbl in lbl_str.split():
        if lbl == "healthy":
            continue
        for idx, name in label_map.items():
            if idx == 6:
                continue
            if name == lbl:
                vec[idx] = 1
    return vec.tolist()


train_df["label_vec"] = train_df["labels"].apply(encode_labels)

train_df_split, val_df_split = train_test_split(
    train_df, test_size=0.1, random_state=42, stratify=train_df["labels"]
)

num_workers = min(8, os.cpu_count() or 1)  # efficient parallelism for tf.data


def _load_and_preprocess(path, label=None, training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [im_size, im_size])
    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)  # approx 0.8‑1.2 range
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    if label is None:
        return img
    else:
        return img, label


train_paths = [
    os.path.join(load_dir, "train_images", fname) for fname in train_df_split["image"]
]
train_labels = np.stack(train_df_split["label_vec"].values)
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(buffer_size=len(train_paths), seed=42)
train_ds = train_ds.map(
    lambda p, l: _load_and_preprocess(p, l, training=True),
    num_parallel_calls=num_workers,
)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_paths = [
    os.path.join(load_dir, "train_images", fname) for fname in val_df_split["image"]
]
val_labels = np.stack(val_df_split["label_vec"].values)
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(
    lambda p, l: _load_and_preprocess(p, l, training=False),
    num_parallel_calls=num_workers,
)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



## === cell 4
with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights="imagenet",
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )
    base.trainable = False  # freeze base
    model = tf.keras.Sequential(
        [
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(n_labels, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )

model.fit(
    train_ds,
    epochs=2,
    validation_data=val_ds,
    verbose=2,
)



## === cell 5
test_paths = [os.path.join(test_dir, fname) for fname in test_df["image"]]
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda p: _load_and_preprocess(p, training=True),  # same augmentations as train
    num_parallel_calls=num_workers,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

pred = model.predict(test_ds, verbose=0)



## === cell 6
threshold = {i: 0.25 for i in range(n_labels)}  # uniform threshold

pred_strings = []
for probs in pred:
    tags = ""
    for i in range(n_labels):
        if probs[i] > threshold[i]:
            tags += label_map[i] + " "
    tags = tags.strip()
    if not tags:  # no disease detected → healthy
        tags = label_map[6]
    pred_strings.append(tags)

test_df["labels"] = pred_strings
output_path = "submission.csv"
test_df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
print(test_df.head())
