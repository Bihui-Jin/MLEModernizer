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
import os, random, re, math

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras

    print("TF:", tf.__version__)
    print("Keras:", keras.__version__)
except Exception as e:
    raise RuntimeError(
        "TensorFlow failed to import in this environment. Original error:\n" + str(e)
    )



## === cell 1
BASE_CANDIDATES = [
    "../input/plant-pathology-2021-fgvc8/",
    "/kaggle/input/plant-pathology-2021-fgvc8/",
    "../input/",
    "/kaggle/input/",
]


def _find_competition_base():
    for base in BASE_CANDIDATES:
        if os.path.isdir(base):
            if os.path.isfile(os.path.join(base, "train.csv")) and os.path.isdir(
                os.path.join(base, "train_images")
            ):
                return base if base.endswith("/") else base + "/"
            try:
                for d in os.listdir(base):
                    cand = os.path.join(base, d)
                    if os.path.isdir(cand) and os.path.isfile(
                        os.path.join(cand, "train.csv")
                    ):
                        if os.path.isdir(
                            os.path.join(cand, "train_images")
                        ) and os.path.isdir(os.path.join(cand, "test_images")):
                            return cand + "/"
            except Exception:
                pass
    cand = "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/"
    if os.path.isfile(os.path.join(cand, "train.csv")):
        return cand if cand.endswith("/") else cand + "/"
    raise RuntimeError(
        "Could not locate competition data folder under ../input or /kaggle/input"
    )


path = _find_competition_base()
print("Using data path:", path)

train = pd.read_csv(os.path.join(path, "train.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

print(train.shape, sub.shape)
print(train.head())



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE



## === cell 3
from matplotlib import pyplot as plt

img_path_example = os.path.join(path, "train_images", train.loc[0, "image"])
img = plt.imread(img_path_example)
print("Example image:", img_path_example)
print(img.shape)
plt.imshow(img)
plt.axis("off")



## === cell 4
import pathlib



## === cell 5
train_dir = os.path.join(path, "train_images")
test_dir = os.path.join(path, "test_images")

train_paths = tf.io.gfile.glob(os.path.join(train_dir, "*.jpg"))
train_paths += tf.io.gfile.glob(os.path.join(train_dir, "*.jpeg"))
train_paths += tf.io.gfile.glob(os.path.join(train_dir, "*.png"))
train_paths = sorted(train_paths)

test_paths = tf.io.gfile.glob(os.path.join(test_dir, "*.jpg"))
test_paths += tf.io.gfile.glob(os.path.join(test_dir, "*.jpeg"))
test_paths += tf.io.gfile.glob(os.path.join(test_dir, "*.png"))
test_paths = sorted(test_paths)

print("Found train images:", len(train_paths))
print("Found test images:", len(test_paths))

if len(train_paths) == 0:
    raise RuntimeError(f"No train images found in {train_dir}. Check input path.")
if len(test_paths) == 0:
    raise RuntimeError(f"No test images found in {test_dir}. Check input path.")



## === cell 6
CLASSES = ["scab", "frog_eye_leaf_spot", "rust", "complex", "powdery_mildew", "healthy"]
print("Classes:", CLASSES)




## === cell 7
def multilabel_onehot(series, classes):
    out = np.zeros((len(series), len(classes)), dtype=np.float32)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for r, s in enumerate(series.astype(str).values):
        for lab in s.split():
            if lab in class_to_idx:
                out[r, class_to_idx[lab]] = 1.0
    return pd.DataFrame(out, columns=classes)


labels_onehot_features = multilabel_onehot(train["labels"], CLASSES)
new_train = pd.concat([train[["image"]], labels_onehot_features], axis=1).iloc[:]
new_train.head()



## === cell 8
new_train




## === cell 9
def decode_image(filename, label=None, image_size=(512, 512)):
    filename = tf.cast(filename, tf.string)

    bits = tf.io.read_file(filename)
    image = tf.image.decode_image(bits, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        label = tf.cast(label, tf.float32)
        return image, label




## === cell 10
test_paths[:5], len(test_paths)



## === cell 11
BATCH_SIZE = 64



## === cell 12
_ds_opts = tf.data.Options()
_ds_opts.experimental_deterministic = True
try:
    _ds_opts.threading.private_threadpool_size = max(8, (os.cpu_count() or 8) - 1)
except Exception:
    pass

test_path_map = {os.path.basename(p): p for p in test_paths}
sub["filepath"] = sub["image"].map(test_path_map)
missing = int(sub["filepath"].isna().sum())
if missing:
    raise RuntimeError(
        f"Missing {missing} test image filepaths. Example missing: "
        f"{sub.loc[sub['filepath'].isna(), 'image'].head(3).tolist()}. "
        f"Check test_images directory and filenames under: {test_dir}"
    )

test_paths_tensor = tf.constant(sub["filepath"].astype(str).values, dtype=tf.string)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths_tensor)
    .with_options(_ds_opts)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 13
import tensorflow as tf
from tensorflow import (
    keras as tf_keras,
)  # keep original import availability for any downstream usage



## === cell 14
SEED = 42

train_path_map = {os.path.basename(p): p for p in train_paths}
train["filepath"] = train["image"].map(train_path_map)

train = train[train["filepath"].notna()].reset_index(drop=True)
if len(train) == 0:
    raise RuntimeError(
        f"After mapping images to filepaths, training set is empty. Check train_images directory: {train_dir}"
    )

labels_onehot_features = multilabel_onehot(train["labels"], CLASSES).astype(np.float32)

rng = np.random.default_rng(SEED)
idx = np.arange(len(train))
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

x_tr = train.loc[tr_idx, "filepath"].astype(str).values
y_tr = labels_onehot_features.loc[tr_idx].values.astype(np.float32)
x_va = train.loc[va_idx, "filepath"].astype(str).values
y_va = labels_onehot_features.loc[va_idx].values.astype(np.float32)

x_tr_t = tf.constant(x_tr, dtype=tf.string)
x_va_t = tf.constant(x_va, dtype=tf.string)

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_tr_t, y_tr))
    .with_options(_ds_opts)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(lambda f, y: decode_image(f, y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((x_va_t, y_va))
    .with_options(_ds_opts)
    .map(lambda f, y: decode_image(f, y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

print("Train/valid sizes:", len(x_tr), len(x_va))



## === cell 15
tf.keras.utils.set_random_seed(SEED)

inputs = keras.Input(shape=(512, 512, 3))
x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 16
EPOCHS = 2
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 17
probs = model.predict(test_dataset, verbose=1)



## === cell 18
probs.shape



## === cell 19
probs[:2]



## === cell 20
temp_probs = probs



## === cell 21
temp_probs[:2]



## === cell 22
name = {i: c for i, c in enumerate(CLASSES)}
threshold = {name_idx: 0.15 for name_idx, cls in name.items() if cls != "healthy"}

pred_string = []
for line in temp_probs:
    s = ""
    count = 0
    for i, cls in name.items():
        if cls == "healthy":
            continue
        if line[i] > threshold[i]:
            s = s + cls + " "
            count += 1

    if count >= 2:
        if "complex" not in s.split():
            s = s + "complex" + " "

    if s.strip() == "":
        s = "healthy"
    else:
        s = s.strip()

    pred_string.append(s)

submission = sub.copy()
if len(pred_string) != len(submission):
    raise ValueError(
        f"Prediction rows ({len(pred_string)}) != submission rows ({len(submission)})"
    )

submission["labels"] = pred_string
submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
