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

0.1662788550323177

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

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

train = pd.read_csv(os.path.join(path, "train.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))
test = sub.copy()

train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")

print(train.shape, test.shape)
print(train.head(2))



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE



## === cell 3
import pathlib




## === cell 4
def _list_images_fast(directory):
    patterns = [
        os.path.join(directory, "**", "*.jpg"),
        os.path.join(directory, "**", "*.jpeg"),
        os.path.join(directory, "**", "*.png"),
        os.path.join(directory, "**", "*.JPG"),
        os.path.join(directory, "**", "*.JPEG"),
        os.path.join(directory, "**", "*.PNG"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(tf.io.gfile.glob(pat))
    return sorted(set(paths))


train_paths = _list_images_fast(train_images_dir)
test_paths = _list_images_fast(test_images_dir)

print("n_train_images:", len(train_paths))
print("n_test_images:", len(test_paths))
print("example test path:", test_paths[0] if test_paths else None)



## === cell 5
kind = np.unique(train["labels"])
kind



## === cell 6
all_classes = sorted(
    {c for s in train["labels"].astype(str).tolist() for c in s.split()}
)
print("classes:", all_classes)

mlb = pd.Series(train["labels"].astype(str)).str.get_dummies(sep=" ")
labels_onehot_features = mlb.reindex(columns=all_classes, fill_value=0).astype(
    np.float32
)

new_train = pd.concat([train[["image"]], labels_onehot_features], axis=1).iloc[:]
new_train.head()



## === cell 7
new_train




## === cell 8
def decode_image(filename, label=None, image_size=(512, 512)):
    filename = tf.cast(filename, tf.string)

    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 9
test_paths[:5]



## === cell 10
BATCH_SIZE = 64



## === cell 11
import tensorflow as tf
from tensorflow import keras



## === cell 12
IMG_SIZE = (512, 512)
N_CLASSES = len(all_classes)

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
x = base.output
out = Dense(N_CLASSES, activation="sigmoid")(x)
model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)

print(model.output_shape)



## === cell 13
image_to_path = {os.path.basename(p): p for p in train_paths}

df = new_train.copy()
df["path"] = df["image"].map(image_to_path)
df = df.dropna(subset=["path"]).reset_index(drop=True)

idx = np.arange(len(df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(df))
train_idx, val_idx = idx[:split], idx[split:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)

y_cols = all_classes


def make_dataset(frame, training=True):
    paths = frame["path"].astype(str).values
    labels = frame[y_cols].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img, y = decode_image(p, y, image_size=IMG_SIZE)
        img = tf.keras.applications.resnet.preprocess_input(img * 255.0)
        return img, y

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    ds = (
        ds.map(_map_fn, num_parallel_calls=AUTO)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    return ds


train_ds = make_dataset(df_train, training=True)
val_ds = make_dataset(df_val, training=False)

print("train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("val batches:", tf.data.experimental.cardinality(val_ds).numpy())



## === cell 14
EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1368826698.py in <cell line: 0>()
      1 EPOCHS = 2
----> 2 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      3 
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py in update(self, current, values, finalize)
    117 
    118             if self.target is not None:
--> 119                 numdigits = int(math.log10(self.target)) + 1
    120                 bar = ("%" + str(numdigits) + "d/%d") % (current, self.target)
    121                 bar = f"\x1b[1m{bar}\x1b[0m "

ValueError: math domain error

## === cell 15
def make_test_dataset(paths):
    paths = np.asarray([str(p) for p in paths], dtype=object)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(p):
        img = decode_image(p, label=None, image_size=IMG_SIZE)
        img = tf.keras.applications.resnet.preprocess_input(img * 255.0)
        return img

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    ds = (
        ds.map(_map_fn, num_parallel_calls=AUTO)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    return ds




## === cell 16
test_image_to_path = {os.path.basename(p): p for p in test_paths}

missing = [
    img for img in test["image"].astype(str).values if img not in test_image_to_path
]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images in {test_images_dir}. "
        f"First few missing: {missing[:5]}"
    )

ordered_test_paths = [test_image_to_path[i] for i in test["image"].astype(str).values]
ordered_test_ds = make_test_dataset(ordered_test_paths)

probs = model.predict(ordered_test_ds, verbose=1)
print("probs shape:", probs.shape)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4174755587.py in <cell line: 0>()
      7 if missing:
      8     # In a valid Kaggle run this should be empty; if not, fail early with context.
----> 9     raise FileNotFoundError(
     10         f"Missing {len(missing)} test images in {test_images_dir}. "
     11         f"First few missing: {missing[:5]}"

FileNotFoundError: Missing 3727 test images in ../input/plant-pathology-2021-fgvc8/test_images. First few missing: ['ca6a50c5d2adb8ae.jpg', 'b686d217a1e2e3a5.jpg', 'c9a5345ec78b4ac5.jpg', 'acdc8d88549873d7.jpg', 'cf7d6b1e5856a021.jpg']

## === cell 17
temp_probs = probs
temp_probs[:2]



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2943143751.py in <cell line: 0>()
----> 1 temp_probs = probs
      2 temp_probs[:2]
      3 

NameError: name 'probs' is not defined

## === cell 18
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.27, 1: 0.5, 2: 0.3, 3: 0.5, 4: 0.5, 5: 0.5}

class_to_idx = {c: i for i, c in enumerate(all_classes)}

pred_string = []
for line in temp_probs:
    chosen = []
    for cls_name, thr in [
        ("scab", threshold[0]),
        ("frog_eye_leaf_spot", threshold[1]),
        ("complex", threshold[2]),
        ("rust", threshold[3]),
        ("powdery_mildew", threshold[4]),
    ]:
        if cls_name in class_to_idx:
            j = class_to_idx[cls_name]
            if float(line[j]) > float(thr):
                chosen.append(cls_name)

    if len(chosen) == 0:
        chosen = ["healthy"]

    pred_string.append(" ".join(chosen))

test["labels"] = pred_string

submission = test[["image", "labels"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3827380397.py in <cell line: 0>()
     13 
     14 pred_string = []
---> 15 for line in temp_probs:
     16     chosen = []
     17     for cls_name, thr in [

NameError: name 'temp_probs' is not defined
