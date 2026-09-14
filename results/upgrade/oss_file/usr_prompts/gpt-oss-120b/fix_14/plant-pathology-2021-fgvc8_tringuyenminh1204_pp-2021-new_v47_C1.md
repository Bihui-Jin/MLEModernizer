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
import os, re, random, math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print(tf.__version__)
print(tf.keras.__version__)




## === cell 1
def decode_image(filename, label=None, image_size=(224, 224)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 2
BATCH_SIZE = 256  # increased batch size to reduce number of steps per epoch
IMAGE_SIZE = (224, 224)

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_image_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_image_dir = "../input/plant-pathology-2021-fgvc8/test_images"

train_df = pd.read_csv(train_csv_path)
assert {"image", "labels"}.issubset(train_df.columns)

class_names = [
    "scab",
    "frog_eye_leaf_spot",
    "complex",
    "rust",
    "powdery_mildew",
    "healthy",
]

class_to_idx = {c: i for i, c in enumerate(class_names)}


def labels_to_vector(label_str):
    vec = np.zeros(len(class_names), dtype=np.float32)
    for lbl in label_str.split():
        idx = class_to_idx.get(lbl)
        if idx is not None:
            vec[idx] = 1.0
    return vec


train_labels = np.stack([labels_to_vector(s) for s in train_df["labels"]], axis=0)

train_image_paths = [
    os.path.join(train_image_dir, fname) for fname in train_df["image"]
]



## === cell 3
num_samples = len(train_image_paths)
indices = np.arange(num_samples)
np.random.shuffle(indices)

split_idx = int(0.9 * num_samples)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

train_paths = [train_image_paths[i] for i in train_idx]
train_labels_split = train_labels[train_idx]

val_paths = [train_image_paths[i] for i in val_idx]
val_labels_split = train_labels[val_idx]


def tf_dataset(paths, labels, use_disk_cache=False):
    """
    Build a tf.data pipeline.
    When use_disk_cache=True the pipeline caches decoded images on disk,
    reducing RAM pressure without changing any algorithmic behavior.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, l: (decode_image(p, image_size=IMAGE_SIZE), l),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if use_disk_cache:
        cache_path = os.path.join(
            "./tf_cache", f"cache_{'train' if 'train' in paths[0] else 'val'}"
        )
        ds = ds.cache(cache_path)  # disk‑based cache
    else:
        ds = ds.cache()  # in‑memory cache – fast for small val set
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return ds


os.makedirs("./tf_cache", exist_ok=True)

train_dataset = tf_dataset(train_paths, train_labels_split, use_disk_cache=True)
val_dataset = tf_dataset(val_paths, val_labels_split, use_disk_cache=False)



## === cell 4
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,)
)
base_model.trainable = False  # freeze backbone

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(len(class_names), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)



## === cell 5
EPOCHS = 3
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=2,
)



## === cell 6
test_image_paths = [
    os.path.join(test_image_dir, f)
    for f in os.listdir(test_image_dir)
    if re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(\.jpg|\.jpeg|\.png)$", f, re.IGNORECASE)
]

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_image_paths)
    .map(
        lambda p: decode_image(p, image_size=IMAGE_SIZE),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    .cache()  # cache test images in memory – small set
    .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE)
)



## === cell 7
probs = model.predict(test_dataset, verbose=0)

thresholds = {i: 0.4 for i in range(len(class_names))}

pred_strings = []
for line in probs:
    s = ""
    for i in range(len(class_names)):
        if line[i] > thresholds[i]:
            s += class_names[i] + " "
    if not s:  # default to healthy if nothing passes
        s = "healthy "
    pred_strings.append(s.strip())

submission_df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in test_image_paths], "labels": pred_strings}
)
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
display(submission_df.head())
