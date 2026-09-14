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

0.2239124126104717

# 6. Current score

0.27088

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.27088) has done: 'The timeout is dominated by decoding and resizing 512×512 JPEGs on-the-fly for ~15k training images over multiple epochs; we keep the exact same model/training logic but make the input pipeline substantially faster and more CPU-parallel. Specifically, we enable TF graph optimizations and deterministic execution, add dataset caching (in-memory if possible, otherwise to disk) so each image is decoded/resized only once across epochs, and tune tf.data options (parallelism, prefetch, non-blocking) without changing semantics. We also replace the slow Python loop that builds submission strings with an equivalent vectorized NumPy implementation (same thresholds and “healthy” fallback). These changes preserve accuracy/evaluation behavior while cutting redundant work to fit within the 600s budget.'

# 9. Code solution

## === cell 0
import os, random, math, re
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

print("tf:", tf.__version__)
print("keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as e:
    print("Determinism setting not available:", e)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Thread config not available:", e)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT not available:", e)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"

train = pd.read_csv(os.path.join(path, "train.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")

print(train.shape, sub.shape)
print(train.head())




## === cell 2
AUTO = tf.data.experimental.AUTOTUNE




## === cell 3
from matplotlib import pyplot as plt

img_path = os.path.join(train_images_dir, train.iloc[0]["image"])
img = plt.imread(img_path)
print("Example image:", img_path, "shape:", img.shape)




## === cell 4
import pathlib




## === cell 5
train_paths = sorted([str(p) for p in pathlib.Path(train_images_dir).glob("*.jpg")])
test_paths = sorted([str(p) for p in pathlib.Path(test_images_dir).glob("*.jpg")])

print("n_train_images:", len(train_paths))
print("n_test_images:", len(test_paths))
print("example test path:", test_paths[0])




## === cell 6
CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
num_classes = len(CLASSES)
print("classes:", CLASSES)




## === cell 7
def multilabel_to_multihot(label_str: str, class_to_idx: dict) -> np.ndarray:
    y = np.zeros(len(class_to_idx), dtype=np.float32)
    for t in str(label_str).split():
        if t in class_to_idx:
            y[class_to_idx[t]] = 1.0
    return y


class_to_idx = {c: i for i, c in enumerate(CLASSES)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

train["filepath"] = train["image"].apply(lambda x: os.path.join(train_images_dir, x))
train["multihot"] = train["labels"].apply(
    lambda s: multilabel_to_multihot(s, class_to_idx)
)

new_train = pd.concat(
    [
        train[["image"]],
        pd.DataFrame(np.stack(train["multihot"].values), columns=CLASSES),
    ],
    axis=1,
)
print(new_train.head())
print("Multihot check sum:", new_train[CLASSES].sum().to_dict())




## === cell 8
new_train




## === cell 9
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 10
test_df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in test_paths], "filepath": test_paths}
)
print(test_df.head(), test_df.shape)




## === cell 11
BATCH_SIZE = 16  # keep memory safe for 512x512 images
IMG_SIZE = (512, 512)




## === cell 12
def with_fast_deterministic_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_optimization.autotune_buffers = True
    ds = ds.with_options(opts)
    return ds




## === cell 13
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_df["filepath"].values)
    .map(lambda x: decode_image(x, None, IMG_SIZE), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
test_dataset = with_fast_deterministic_options(test_dataset)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/722999134.py in <cell line: 0>()
      5     .prefetch(AUTO)
      6 )
----> 7 test_dataset = with_fast_deterministic_options(test_dataset)
      8 
      9 

/tmp/ipykernel_11/3232770457.py in with_fast_deterministic_options(ds)
      9     opts.experimental_optimization.map_parallelization = True
     10     opts.experimental_optimization.parallel_batch = True
---> 11     opts.experimental_optimization.autotune_buffers = True
     12     ds = ds.with_options(opts)
     13     return ds

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 14
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train["filepath"].values,
    np.stack(train["multihot"].values),
    test_size=0.10,
    random_state=SEED,
    shuffle=True,
)

train_cache_path = "/kaggle/working/tfds_train_cache"
val_cache_path = "/kaggle/working/tfds_val_cache"

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(lambda x, y: decode_image(x, y, IMG_SIZE), num_parallel_calls=AUTO)
    .cache(train_cache_path)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
train_dataset = with_fast_deterministic_options(train_dataset)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(lambda x, y: decode_image(x, y, IMG_SIZE), num_parallel_calls=AUTO)
    .cache(val_cache_path)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
val_dataset = with_fast_deterministic_options(val_dataset)

print("train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
print("val batches:", tf.data.experimental.cardinality(val_dataset).numpy())




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2951584654.py in <cell line: 0>()
     22     .prefetch(AUTO)
     23 )
---> 24 train_dataset = with_fast_deterministic_options(train_dataset)
     25 
     26 val_dataset = (

/tmp/ipykernel_11/3232770457.py in with_fast_deterministic_options(ds)
      9     opts.experimental_optimization.map_parallelization = True
     10     opts.experimental_optimization.parallel_batch = True
---> 11     opts.experimental_optimization.autotune_buffers = True
     12     ds = ds.with_options(opts)
     13     return ds

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 15
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({"swish": keras.layers.Activation(tf.nn.swish)})


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 16
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy"
)
model.summary()




## === cell 17
EPOCHS = 2
history = model.fit(
    train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1
)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/390007194.py in <cell line: 0>()
      1 EPOCHS = 2
      2 history = model.fit(
----> 3     train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1
      4 )
      5 

NameError: name 'val_dataset' is not defined

## === cell 18
probs = model.predict(test_dataset, verbose=1)
print("probs shape:", probs.shape)




## === cell 19
temp_probs = probs




## === cell 20
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.30, 1: 0.50, 2: 0.30, 3: 0.50, 4: 0.50}

thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)  # (5,)
mask = temp_probs[:, :5] > thr[None, :]  # (N,5) bool
labels5 = np.array([name[i] for i in range(5)], dtype=object)

pred_string = []
for row in mask:
    if row.any():
        pred_string.append(" ".join(labels5[row].tolist()))
    else:
        pred_string.append(name[5])

submission = pd.DataFrame({"image": test_df["image"].values, "labels": pred_string})

submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
