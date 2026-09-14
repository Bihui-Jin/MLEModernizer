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

# 5. Target score

0.2116897506925212

# 6. Current score

0.30881

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.30881) has done: 'The timeout is dominated by decoding/resizing 512×512 JPEGs for ~15k training images for 2 epochs plus validation, which is heavy CPU work if the input pipeline isn’t fully optimized. I keep the same model, epochs, loss, thresholds, and split logic, but speed up data loading by (1) avoiding costly `os.walk` directory scans that aren’t used, (2) switching to `tf.image.decode_and_crop_jpeg` to decode directly into the target 512×512 region (provably equivalent to decode+resize when the source is already 512×512), (3) using deterministic options while enabling parallel map, caching the validation set (one pass reused every epoch), and adding `drop_remainder=True` for steadier throughput. These changes reduce redundant work and CPU overhead without changing training semantics or output format.'

# 9. Code solution

## === cell 0
import os, random, re, math
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "sample_submission.csv")
sub = pd.read_csv(path + "sample_submission.csv")

train.head(), test.head()



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE



## === cell 3
from matplotlib import pyplot as plt

img = plt.imread(
    "../input/plant-pathology-2021-fgvc8/train_images/800113bb65efe69e.jpg"
)
print(img.shape)
plt.imshow(img)
plt.axis("off")



## === cell 4
import pathlib



## === cell 5
train_paths = None
test_paths = None
print("Skipped os.walk scans (not used).")



## === cell 6
import numpy as np

kind = np.unique(train["labels"])
kind[:10], len(kind)



## === cell 7
labels_onehot_features = train["labels"].str.get_dummies(sep=" ")
new_train = pd.concat([train[["image"]], labels_onehot_features], axis=1).iloc[:]

new_train.head(), labels_onehot_features.columns.tolist()



## === cell 8
new_train.describe(include="all").T.head(12)




## === cell 9
@tf.function
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)

    def _fast():
        image = tf.image.decode_and_crop_jpeg(
            bits, crop_window=[0, 0, image_size[0], image_size[1]], channels=3
        )
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def _slow():
        image = tf.image.decode_jpeg(bits, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, image_size)
        return image

    shape = tf.image.extract_jpeg_shape(bits)
    image = tf.cond(
        tf.logical_and(shape[0] >= image_size[0], shape[1] >= image_size[1]),
        _fast,
        _slow,
    )

    if label is None:
        return image
    else:
        return image, label




## === cell 10
test_images_dir = "../input/plant-pathology-2021-fgvc8/test_images"
test = test.copy()
test["image_path"] = test["image"].apply(lambda x: os.path.join(test_images_dir, x))

missing = test.loc[~test["image_path"].apply(os.path.exists), "image"].head()
if len(missing) > 0:
    raise FileNotFoundError(f"Some test images not found, e.g.: {missing.tolist()}")

test_paths_aligned = test["image_path"].tolist()
test_paths_aligned[:3], len(test_paths_aligned)



## === cell 11
BATCH_SIZE = 64



## === cell 12
opts = tf.data.Options()
opts.deterministic = True

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths_aligned)
    .with_options(opts)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 13
import tensorflow as tf
from tensorflow import keras



## === cell 14
disease_cols = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
for c in disease_cols:
    if c not in new_train.columns:
        new_train[c] = 0

train_images_dir = "../input/plant-pathology-2021-fgvc8/train_images"
new_train["image_path"] = new_train["image"].apply(
    lambda x: os.path.join(train_images_dir, x)
)

if not os.path.exists(new_train["image_path"].iloc[0]):
    raise FileNotFoundError("Train images directory not found or image paths invalid.")

X_paths = new_train["image_path"].values
y = new_train[disease_cols].values.astype(np.float32)

n = len(new_train)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

split = int(n * 0.9)
tr_idx, va_idx = idx[:split], idx[split:]

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_paths[tr_idx], y[tr_idx]))
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .with_options(opts)
    .map(lambda p, lab: decode_image(p, lab), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((X_paths[va_idx], y[va_idx]))
    .with_options(opts)
    .map(lambda p, lab: decode_image(p, lab), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTO)
)

len(tr_idx), len(va_idx), y.shape



## === cell 15
inputs = keras.Input(shape=(512, 512, 3))
x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPool2D()(x)
x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.MaxPool2D()(x)
x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(5, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 16
EPOCHS = 2
history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, verbose=2)



## === cell 17
probs = model.predict(test_dataset, verbose=1)
probs.shape



## === cell 18
temp_probs = probs.copy()
temp_probs[:2]



## === cell 19
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.2, 1: 0.3, 2: 0.15, 3: 0.3, 4: 0.35}

pred_string = []
for line in temp_probs:
    s = ""
    count = 0
    for i in range(5):
        if line[i] > threshold[i]:
            count += 1

    if count >= 3:
        mx = -1.0
        key = 0
        for i in range(5):
            if line[i] > mx:
                mx = float(line[i])
                key = i
        if name[key] != "complex":
            s += name[key] + " "
        s = s + "complex" + " "
    else:
        for i in range(5):
            if line[i] > threshold[i]:
                s = s + name[i] + " "

    if s == "":
        s = name[5]

    pred_string.append(s.strip())

subm = test[["image"]].copy()
subm["labels"] = pred_string
subm.to_csv("submission.csv", index=False)
subm.head()



## === cell 20
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
print(check.shape)
print(check.columns.tolist())
print(check.head(3))
print("Unique example labels:", check["labels"].head(10).tolist())
