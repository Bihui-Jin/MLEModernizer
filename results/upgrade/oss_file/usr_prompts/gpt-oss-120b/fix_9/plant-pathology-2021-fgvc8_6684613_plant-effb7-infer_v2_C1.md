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

0.7559372114496771

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import random
import math




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def auto_select_accelerator():
    """
    Reference:
        * https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
        * https://www.kaggle.com/xhlulu/ranzcr-efficientnet-tpu-training
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[0]  # previously IMSIZES[7] (600)

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

class_name = df.labels.unique().tolist()
print("Classes:", class_name)
print("Number of classes:", len(class_name))

df["labels"] = df["labels"].astype(str)
n_labels = len(class_name)




## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 32

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_df = pd.DataFrame()
test_df["image"] = os.listdir(test_dir)




## === cell 4
test_paths = tf.convert_to_tensor(
    [os.path.join(test_dir, fname) for fname in test_df["image"].values]
)




## === cell 5
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights="imagenet", include_top=False, input_shape=(im_size, im_size, 3)
    )
    model = tf.keras.Sequential(
        [
            base,
            tf.keras.layers.GlobalMaxPooling2D(),
            tf.keras.layers.Dense(n_labels, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4),
        metrics=["accuracy"],
    )
    model.summary()




## === cell 6
label_to_index = {name: idx for idx, name in enumerate(class_name)}
df["label_idx"] = df["labels"].map(label_to_index)

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["labels"]
)


def decode_and_preprocess(path):
    """Deterministic decoding, resizing and EfficientNet preprocessing."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [im_size, im_size])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def augment_image(img):
    """Random augmentations applied after caching."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


def preprocess_train_image(path, label):
    img = decode_and_preprocess(path)
    img = augment_image(img)  # random flips
    return img, tf.one_hot(label, n_labels)


def preprocess_val_image(path, label):
    img = decode_and_preprocess(path)
    return img, tf.one_hot(label, n_labels)


def make_dataset(df_subset, training=True):
    paths = tf.convert_to_tensor(
        [os.path.join(load_dir, "train_images/", fname) for fname in df_subset["image"]]
    )
    labels = tf.convert_to_tensor(df_subset["label_idx"].values, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer=1000, seed=42)
    ds = ds.map(
        lambda p, l: (decode_and_preprocess(p), l), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.cache()
    if training:
        ds = ds.map(
            lambda img, l: (augment_image(img), l), num_parallel_calls=tf.data.AUTOTUNE
        )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = make_dataset(train_df, training=True)
val_dataset = make_dataset(val_df, training=False)

steps_per_epoch = math.ceil(len(train_df) / BATCH_SIZE)
validation_steps = math.ceil(len(val_df) / BATCH_SIZE)

base_test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
base_test_dataset = (
    base_test_dataset.map(decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1866249011.py in <cell line: 0>()
     54 
     55 
---> 56 train_dataset = make_dataset(train_df, training=True)
     57 val_dataset = make_dataset(val_df, training=False)
     58 

/tmp/ipykernel_11/1866249011.py in make_dataset(df_subset, training)
     41     ds = tf.data.Dataset.from_tensor_slices((paths, labels))
     42     if training:
---> 43         ds = ds.shuffle(buffer=1000, seed=42)
     44     ds = ds.map(
     45         lambda p, l: (decode_and_preprocess(p), l), num_parallel_calls=tf.data.AUTOTUNE

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 7
weight_path = "/kaggle/input/modelplant1/bestmodel_tpu_aug.h5"
weights_loaded = False
if os.path.exists(weight_path):
    try:
        model.load_weights(weight_path)
        weights_loaded = True
        print("Loaded external weights, skipping training.")
    except Exception as e:
        print(f"Could not load external weights: {e}")

if not weights_loaded:
    model.fit(
        train_dataset,
        epochs=2,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        verbose=1,
    )




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/447171173.py in <cell line: 0>()
     11 if not weights_loaded:
     12     model.fit(
---> 13         train_dataset,
     14         epochs=2,
     15         steps_per_epoch=steps_per_epoch,

NameError: name 'train_dataset' is not defined

## === cell 8
TTA = 5
test_steps = math.ceil(len(test_df) / BATCH_SIZE)
preds = []

for i in range(TTA):
    tta_dataset = (
        base_test_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    preds.append(model.predict(tta_dataset, steps=test_steps, verbose=0))

pred = np.mean(np.array(preds), axis=0)
argpred = np.argmax(pred, axis=1)
test_df["labels"] = argpred
test_df["labels"] = test_df["labels"].apply(lambda x: class_name[x])

submission_path = "submission.csv"
test_df[["image", "labels"]].to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
test_df.head()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3262780385.py in <cell line: 0>()
      6     # Apply random flips for this TTA round, then batch & prefetch.
      7     tta_dataset = (
----> 8         base_test_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
      9         .batch(BATCH_SIZE)
     10         .prefetch(tf.data.AUTOTUNE)

NameError: name 'base_test_dataset' is not defined
