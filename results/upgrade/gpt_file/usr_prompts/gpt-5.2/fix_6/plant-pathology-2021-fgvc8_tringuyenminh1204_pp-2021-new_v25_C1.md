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

0.2739229653080058

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, re, math

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(path + "train.csv")
sub = pd.read_csv(path + "sample_submission.csv")

print(train.shape, sub.shape)
print(train.head())



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
train_paths = tf.io.gfile.glob(
    "../input/plant-pathology-2021-fgvc8/train_images/**/*.jpg"
)
train_paths += tf.io.gfile.glob(
    "../input/plant-pathology-2021-fgvc8/train_images/**/*.jpeg"
)
train_paths += tf.io.gfile.glob(
    "../input/plant-pathology-2021-fgvc8/train_images/**/*.png"
)
train_paths = sorted(train_paths)

test_paths = tf.io.gfile.glob(
    "../input/plant-pathology-2021-fgvc8/test_images/**/*.jpg"
)
test_paths += tf.io.gfile.glob(
    "../input/plant-pathology-2021-fgvc8/test_images/**/*.jpeg"
)
test_paths += tf.io.gfile.glob(
    "../input/plant-pathology-2021-fgvc8/test_images/**/*.png"
)
test_paths = sorted(test_paths)

print("Found train images:", len(train_paths))
print("Found test images:", len(test_paths))

if len(train_paths) == 0:
    raise RuntimeError("No train images found. Check input path.")
if len(test_paths) == 0:
    raise RuntimeError("No test images found. Check input path.")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_10/2128004380.py in <cell line: 0>()
     26 
     27 if len(train_paths) == 0:
---> 28     raise RuntimeError("No train images found. Check input path.")
     29 if len(test_paths) == 0:
     30     raise RuntimeError("No test images found. Check input path.")

RuntimeError: No train images found. Check input path.

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
missing = sub["filepath"].isna().sum()
if missing:
    raise RuntimeError(
        f"Missing {missing} test image filepaths. Check test_images directory and filenames."
    )

test_paths_tensor = tf.constant(sub["filepath"].astype(str).values, dtype=tf.string)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths_tensor)
    .with_options(_ds_opts)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_10/2399109041.py in <cell line: 0>()
     14     # If hidden test set provides more images at submission time, this should still be fine there.
     15     # But locally, all sample_submission images should be present.
---> 16     raise RuntimeError(
     17         f"Missing {missing} test image filepaths. Check test_images directory and filenames."
     18     )

RuntimeError: Missing 3727 test image filepaths. Check test_images directory and filenames.

## === cell 13
import tensorflow as tf
from tensorflow import (
    keras as tf_keras,
)  # keep original import availability for any downstream usage



## === cell 14
train_path_map = {os.path.basename(p): p for p in train_paths}
train["filepath"] = train["image"].map(train_path_map)

train = train[train["filepath"].notna()].reset_index(drop=True)
if len(train) == 0:
    raise RuntimeError(
        "After mapping images to filepaths, training set is empty. Check train_images directory."
    )

labels_onehot_features = multilabel_onehot(train["labels"], CLASSES).astype(np.float32)

SEED = 42
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



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_10/1537119188.py in <cell line: 0>()
      5 train = train[train["filepath"].notna()].reset_index(drop=True)
      6 if len(train) == 0:
----> 7     raise RuntimeError(
      8         "After mapping images to filepaths, training set is empty. Check train_images directory."
      9     )

RuntimeError: After mapping images to filepaths, training set is empty. Check train_images directory.

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



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/350449437.py in <cell line: 0>()
----> 1 tf.keras.utils.set_random_seed(SEED)
      2 
      3 inputs = keras.Input(shape=(512, 512, 3))
      4 x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
      5 x = keras.layers.MaxPooling2D()(x)

NameError: name 'SEED' is not defined

## === cell 16
EPOCHS = 2
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1213112273.py in <cell line: 0>()
      1 EPOCHS = 2
----> 2 history = model.fit(
      3     train_dataset,
      4     validation_data=valid_dataset,
      5     epochs=EPOCHS,

NameError: name 'model' is not defined

## === cell 17
probs = model.predict(test_dataset, verbose=1)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3073497979.py in <cell line: 0>()
----> 1 probs = model.predict(test_dataset, verbose=1)
      2 

NameError: name 'model' is not defined

## === cell 18
probs.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4130349179.py in <cell line: 0>()
----> 1 probs.shape
      2 

NameError: name 'probs' is not defined

## === cell 19
probs[:2]



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2619598287.py in <cell line: 0>()
----> 1 probs[:2]
      2 

NameError: name 'probs' is not defined

## === cell 20
temp_probs = probs



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1438685205.py in <cell line: 0>()
----> 1 temp_probs = probs
      2 

NameError: name 'probs' is not defined

## === cell 21
temp_probs[:2]



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/442628044.py in <cell line: 0>()
----> 1 temp_probs[:2]
      2 

NameError: name 'temp_probs' is not defined

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

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4135533674.py in <cell line: 0>()
      3 
      4 pred_string = []
----> 5 for line in temp_probs:
      6     s = ""
      7     count = 0

NameError: name 'temp_probs' is not defined
