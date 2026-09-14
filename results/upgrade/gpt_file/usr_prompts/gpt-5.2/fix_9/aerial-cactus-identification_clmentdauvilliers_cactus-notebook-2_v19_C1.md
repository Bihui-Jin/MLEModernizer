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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-image==0.25.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9438

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
import random

random.seed(42)
np.random.seed(42)

print("Listing a few input files:")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
from zipfile import ZipFile



## === cell 2
path = "/kaggle/input/aerial-cactus-identification/"
train_csv_path = os.path.join(path, "train.csv")
train_zip_path = os.path.join(path, "train.zip")
test_zip_path = os.path.join(path, "test.zip")
sample_sub_path = os.path.join(path, "sample_submission.csv")

files_dataframe = pd.read_csv(train_csv_path, dtype={"id": str, "has_cactus": np.int32})
print(files_dataframe.head())



## === cell 3
os.makedirs("./training", exist_ok=True)
os.makedirs("./test", exist_ok=True)


def _dir_has_files(d):
    try:
        it = os.scandir(d)
    except FileNotFoundError:
        return False
    with it:
        for _ in it:
            return True
    return False


if not _dir_has_files("./training"):
    with ZipFile(train_zip_path, "r") as zipper:
        zipper.extractall("./training")
else:
    print("Train already extracted; skipping unzip.")

if not _dir_has_files("./test"):
    with ZipFile(test_zip_path, "r") as zipper:
        zipper.extractall("./test")
else:
    print("Test already extracted; skipping unzip.")


def _find_image_dir(root, leaf):
    """
    Returns directory containing .jpg files for the requested leaf folder name.
    Handles common Kaggle zip layouts.
    """
    cands = [
        os.path.join(root, leaf),
        os.path.join(root, "aerial-cactus-identification", leaf),
        os.path.join(
            root, "aerial-cactus-identification", "aerial-cactus-identification", leaf
        ),
    ]
    for d in cands:
        if os.path.isdir(d):
            try:
                files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
                if len(files) > 0:
                    return d
            except Exception:
                pass
    for base, _, files in os.walk(root):
        if os.path.basename(base) == leaf:
            jpgs = [f for f in files if f.lower().endswith(".jpg")]
            if len(jpgs) > 0:
                return base
    raise FileNotFoundError(f"Could not find image directory '{leaf}' under {root}")


train_extracted_dir = _find_image_dir("./training", "train")
test_extracted_dir = _find_image_dir("./test", "test")

print("Resolved train_extracted_dir:", train_extracted_dir)
print("Resolved test_extracted_dir:", test_extracted_dir)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2948592394.py in <cell line: 0>()
     57 
     58 
---> 59 train_extracted_dir = _find_image_dir("./training", "train")
     60 test_extracted_dir = _find_image_dir("./test", "test")
     61 

/tmp/ipykernel_11/2948592394.py in _find_image_dir(root, leaf)
     54             if len(jpgs) > 0:
     55                 return base
---> 56     raise FileNotFoundError(f"Could not find image directory '{leaf}' under {root}")
     57 
     58 

FileNotFoundError: Could not find image directory 'train' under ./training

## === cell 4
class_reparts = files_dataframe["has_cactus"].value_counts()
total_samples = int(files_dataframe["has_cactus"].size)
print("Total number of samples:", total_samples)
has_cactus_weight = total_samples / (2.0 * float(class_reparts[1]))
no_cactus_weight = total_samples / (2.0 * float(class_reparts[0]))
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights:", class_weights)



## === cell 5
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TensorFlow:", tf.__version__)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
def _tf_percentile_rescale_3_97(img_u8):
    img = tf.cast(img_u8, tf.float32)  # 0..255
    flat = tf.reshape(img, [-1])

    n = tf.size(flat)
    n_f = tf.cast(n, tf.float32)

    k3 = tf.cast(tf.floor(0.03 * (n_f - 1.0)), tf.int32)
    k97 = tf.cast(tf.floor(0.97 * (n_f - 1.0)), tf.int32)

    neg = -flat
    k3_top = tf.maximum(1, n - k3)
    k97_top = tf.maximum(1, n - k97)

    p2 = -tf.nn.top_k(neg, k=k3_top, sorted=True).values[-1]
    p98 = -tf.nn.top_k(neg, k=k97_top, sorted=True).values[-1]

    denom = tf.maximum(p98 - p2, 1e-6)
    img = (img - p2) / denom
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = img * 255.0
    return img


def _tf_samplewise_center_std(img_f32):
    mean = tf.reduce_mean(img_f32)
    std = tf.math.reduce_std(img_f32)
    std = tf.maximum(std, 1e-6)
    return (img_f32 - mean) / std




## === cell 7
BATCH_SIZE = 128
IMG_SIZE = (32, 32)


def _decode_jpeg(path):
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_jpeg(bytes_, channels=3)  # uint8
    img = tf.image.resize(img, IMG_SIZE, method="bilinear", antialias=False)  # float32
    img = tf.cast(img, tf.uint8)  # keep percentile semantics on 0..255 like original
    return img


def _augment(img_u8, seed):
    seed = tf.cast(seed, tf.int32)
    img = tf.image.stateless_random_flip_left_right(
        img_u8, seed=seed + tf.constant([1, 0], tf.int32)
    )
    img = tf.image.stateless_random_flip_up_down(
        img, seed=seed + tf.constant([2, 0], tf.int32)
    )

    k = tf.random.stateless_uniform(
        shape=[],
        seed=seed + tf.constant([3, 0], tf.int32),
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    img = tf.image.rot90(img, k=k)

    img_f = tf.cast(img, tf.float32) / 255.0
    delta = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 0], tf.int32), minval=-0.08, maxval=0.08
    )
    img_f = tf.clip_by_value(img_f + delta, 0.0, 1.0)
    contrast = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([5, 0], tf.int32), minval=0.9, maxval=1.1
    )
    mean = tf.reduce_mean(img_f, axis=[0, 1], keepdims=True)
    img_f = tf.clip_by_value((img_f - mean) * contrast + mean, 0.0, 1.0)
    img = tf.cast(img_f * 255.0, tf.uint8)

    return img


def _preprocess(img_u8):
    img = _tf_percentile_rescale_3_97(img_u8)
    img = _tf_samplewise_center_std(img)
    return img


def _make_dataset(filepaths, labels, training, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=len(filepaths), seed=seed, reshuffle_each_iteration=True
        )

    ds = ds.enumerate()  # (idx, (path,label))

    def _map_fn(idx, data):
        path, y = data
        img_u8 = _decode_jpeg(path)
        if training:
            img_u8 = _augment(
                img_u8, seed=tf.stack([tf.cast(seed, tf.int32), tf.cast(idx, tf.int32)])
            )
        x = _preprocess(img_u8)
        y_oh = tf.one_hot(tf.cast(y, tf.int32), depth=2, dtype=tf.float32)
        return x, y_oh

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 8
all_paths = np.array(
    [os.path.join(train_extracted_dir, fn) for fn in files_dataframe["id"].values],
    dtype=object,
)
all_labels = files_dataframe["has_cactus"].astype(int).values

exists_mask = np.array([os.path.isfile(p) for p in all_paths])
if not np.all(exists_mask):
    missing = int((~exists_mask).sum())
    print(f"Warning: {missing} training images missing; dropping them.")
    all_paths = all_paths[exists_mask]
    all_labels = all_labels[exists_mask]

n = len(all_paths)
val_frac = 0.25
n_val = int(np.floor(n * val_frac))
rng = np.random.RandomState(42)
perm = rng.permutation(n)

val_idx = perm[:n_val]
train_idx = perm[n_val:]

train_paths, train_labels = all_paths[train_idx], all_labels[train_idx]
val_paths, val_labels = all_paths[val_idx], all_labels[val_idx]

training_ds = _make_dataset(train_paths, train_labels, training=True, seed=42)
validation_ds = _make_dataset(val_paths, val_labels, training=False, seed=42)

class_indices = {"0": 0, "1": 1}
print("Class indices (label -> column):", class_indices)
print("Train samples:", len(train_paths), "Val samples:", len(val_paths))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3471420527.py in <cell line: 0>()
      1 all_paths = np.array(
----> 2     [os.path.join(train_extracted_dir, fn) for fn in files_dataframe["id"].values],
      3     dtype=object,
      4 )
      5 all_labels = files_dataframe["has_cactus"].astype(int).values

/tmp/ipykernel_11/3471420527.py in <listcomp>(.0)
      1 all_paths = np.array(
----> 2     [os.path.join(train_extracted_dir, fn) for fn in files_dataframe["id"].values],
      3     dtype=object,
      4 )
      5 all_labels = files_dataframe["has_cactus"].astype(int).values

NameError: name 'train_extracted_dir' is not defined

## === cell 9
from tensorflow.keras import layers, models

model = models.Sequential()

model.add(
    layers.Conv2D(
        32, (5, 5), padding="valid", activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 10
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

checkpoint_path = "/tmp/checkpoint.weights.h5"
save_best_model = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
)



## === cell 11
history = model.fit(
    training_ds,
    validation_data=validation_ds,
    verbose=1,
    epochs=10,
    callbacks=[reduce_lr, save_best_model],
    class_weight=class_weights,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2641607698.py in <cell line: 0>()
      1 history = model.fit(
----> 2     training_ds,
      3     validation_data=validation_ds,
      4     verbose=1,
      5     epochs=10,

NameError: name 'training_ds' is not defined

## === cell 12
test_ids = sorted(
    [fn for fn in os.listdir(test_extracted_dir) if fn.lower().endswith(".jpg")]
)
test_paths = np.array(
    [os.path.join(test_extracted_dir, fn) for fn in test_ids], dtype=object
)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda p: _preprocess(_decode_jpeg(p)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("Test images:", len(test_ids))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4081647150.py in <cell line: 0>()
      1 test_ids = sorted(
----> 2     [fn for fn in os.listdir(test_extracted_dir) if fn.lower().endswith(".jpg")]
      3 )
      4 test_paths = np.array(
      5     [os.path.join(test_extracted_dir, fn) for fn in test_ids], dtype=object

NameError: name 'test_extracted_dir' is not defined

## === cell 13
if os.path.isfile(checkpoint_path):
    model.load_weights(checkpoint_path)
else:
    print("Warning: checkpoint not found; using last-epoch model weights.")

probs = model.predict(test_ds, verbose=1)

idx_pos = class_indices.get("1", 1)
has_cactus_prob = probs[:, idx_pos].astype(float)

sample_sub = pd.read_csv(sample_sub_path, dtype={"id": str})

pred_df = pd.DataFrame({"id": test_ids, "has_cactus": has_cactus_prob})
pred_df = sample_sub[["id"]].merge(pred_df, on="id", how="left")
pred_df["has_cactus"] = pred_df["has_cactus"].fillna(0.5)

print("Pred rows:", len(pred_df), "Sample rows:", len(sample_sub))
print(
    "Any missing predictions filled with 0.5:",
    int((pred_df["has_cactus"] == 0.5).sum()),
)

pred_df.to_csv("submission.csv", index=False)
print(pred_df.head())
print("Wrote submission.csv with shape:", pred_df.shape)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/554430026.py in <cell line: 0>()
      5     print("Warning: checkpoint not found; using last-epoch model weights.")
      6 
----> 7 probs = model.predict(test_ds, verbose=1)
      8 
      9 idx_pos = class_indices.get("1", 1)

NameError: name 'test_ds' is not defined

## === cell 14
import shutil

try:
    shutil.rmtree("test")
except OSError:
    print("Test files already erased or not present")
try:
    shutil.rmtree("training")
except OSError:
    print("Training files already erased or not present")
