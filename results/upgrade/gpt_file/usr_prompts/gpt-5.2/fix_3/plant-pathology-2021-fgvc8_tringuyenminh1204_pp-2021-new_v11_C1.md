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

0.1732594644506002

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

print("tf:", tf.__version__)
print("keras:", keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "/kaggle/input/plant-pathology-2021-fgvc8/"

train = pd.read_csv(os.path.join(path, "train.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

train_img_dir = os.path.join(path, "train_images")
test_img_dir = os.path.join(path, "test_images")

assert {"image", "labels"}.issubset(train.columns)
assert {"image", "labels"}.issubset(sub.columns)

train.head()



## === cell 2
AUTO = tf.data.AUTOTUNE

train["filepath"] = train["image"].apply(lambda x: os.path.join(train_img_dir, x))
sub["filepath"] = sub["image"].apply(lambda x: os.path.join(test_img_dir, x))

for fp in train["filepath"].head(5):
    if not tf.io.gfile.exists(fp):
        raise FileNotFoundError(fp)
for fp in sub["filepath"].head(5):
    if not tf.io.gfile.exists(fp):
        raise FileNotFoundError(fp)

train.shape, sub.shape




## === cell 3
def first_token(label_str: str) -> str:
    return str(label_str).split(" ")[0].strip()


train["label_1"] = train["labels"].apply(first_token)

classes = sorted(train["label_1"].unique().tolist())
num_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

train["target"] = train["label_1"].map(class_to_idx).astype(np.int32)

num_classes, classes[:20]



## === cell 4
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


@tf.function
def decode_image(filename, label=None, image_size=IMG_SIZE):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image = tf.cast(image, tf.float32) / 255.0
    if label is None:
        return image
    return image, label




## === cell 5
from sklearn.model_selection import train_test_split

trn_df, val_df = train_test_split(
    train[["filepath", "target"]],
    test_size=0.1,
    random_state=SEED,
    stratify=train["target"],
)

trn_df = trn_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

trn_df.shape, val_df.shape



## === cell 6
trn_files = trn_df["filepath"].values
val_files = val_df["filepath"].values
test_files = sub["filepath"].values

trn_targets_oh = np.eye(num_classes, dtype=np.float32)[trn_df["target"].values]
val_targets_oh = np.eye(num_classes, dtype=np.float32)[val_df["target"].values]


def _with_fast_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    options = tf.data.Options()
    options.experimental_deterministic = True  # preserve determinism / stable results
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.autotune_cpu_budget = 0  # let TF decide
    options.experimental_optimization.autotune_ram_budget = 0  # let TF decide
    return ds.with_options(options)


train_ds = tf.data.Dataset.from_tensor_slices((trn_files, trn_targets_oh))
train_ds = _with_fast_options(train_ds)
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(decode_image, num_parallel_calls=AUTO)
train_ds = (
    train_ds.cache()
)  # cache decoded+resized float32 images for reuse across epochs
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_targets_oh))
val_ds = _with_fast_options(val_ds)
val_ds = val_ds.map(decode_image, num_parallel_calls=AUTO)
val_ds = val_ds.cache()
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
val_ds = val_ds.prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices(test_files)
test_ds = _with_fast_options(test_ds)
test_ds = test_ds.map(decode_image, num_parallel_calls=AUTO)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTO)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2135879300.py in <cell line: 0>()
     22 
     23 train_ds = tf.data.Dataset.from_tensor_slices((trn_files, trn_targets_oh))
---> 24 train_ds = _with_fast_options(train_ds)
     25 train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
     26 train_ds = train_ds.map(decode_image, num_parallel_calls=AUTO)

/tmp/ipykernel_11/2135879300.py in _with_fast_options(ds)
     15     options.experimental_deterministic = True  # preserve determinism / stable results
     16     options.experimental_optimization.apply_default_optimizations = True
---> 17     options.experimental_optimization.autotune_buffers = True
     18     options.experimental_optimization.autotune_cpu_budget = 0  # let TF decide
     19     options.experimental_optimization.autotune_ram_budget = 0  # let TF decide

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 7
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 8
EPOCHS = 3

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/403388791.py in <cell line: 0>()
      1 EPOCHS = 3
      2 
----> 3 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      4 

NameError: name 'val_ds' is not defined

## === cell 9
probs = model.predict(test_ds, verbose=1)
probs.shape



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1936401604.py in <cell line: 0>()
----> 1 probs = model.predict(test_ds, verbose=1)
      2 probs.shape
      3 

NameError: name 'test_ds' is not defined

## === cell 10
pred_idx = probs.argmax(axis=1)
pred_labels = np.array([idx_to_class[i] for i in pred_idx], dtype=str)

assert len(pred_labels) == len(sub), (len(pred_labels), len(sub))

pred_labels[:10], len(pred_labels)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3794003017.py in <cell line: 0>()
----> 1 pred_idx = probs.argmax(axis=1)
      2 pred_labels = np.array([idx_to_class[i] for i in pred_idx], dtype=str)
      3 
      4 assert len(pred_labels) == len(sub), (len(pred_labels), len(sub))
      5 

NameError: name 'probs' is not defined

## === cell 11
submission = sub[["image"]].copy()
submission["labels"] = pred_labels

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Saved submission.csv with shape:", submission.shape)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/177495624.py in <cell line: 0>()
      1 submission = sub[["image"]].copy()
----> 2 submission["labels"] = pred_labels
      3 
      4 submission.to_csv("submission.csv", index=False)
      5 print(submission.head())

NameError: name 'pred_labels' is not defined
