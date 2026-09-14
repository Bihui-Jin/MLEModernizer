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

0.8993

# 6. Current score

0.99349

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.99349) has done: 'I fix the TensorFlow import crash by switching to the already-installed `tf_keras` backend (so protobuf incompatibilities don’t break runtime), while keeping your model/training logic intact. Then I fix the dataset pipeline errors by using valid `tf.image.resize` method constants (instead of the unsupported `"BILINEAR"` string) and by replacing `tf.image.rotate` with `tensorflow_addons`-free augmentation (via KerasCV layers) so the map() graph can build. Finally, I ensure the test pipeline outputs image tensors (not strings) and that `submission.csv` is written with the correct columns and row alignment.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd



## === cell 1
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

from tf_keras import backend as K
import tensorflow as tf  # tf_keras depends on tensorflow; in Kaggle this works reliably with tf_keras usage

tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

from zipfile import ZipFile



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
path = "/kaggle/input/aerial-cactus-identification/"
files_dataframe = pd.read_csv(path + "train.csv", dtype={"id": str, "has_cactus": int})
files_dataframe.head()



## === cell 3
os.makedirs("./train", exist_ok=True)
os.makedirs("./test", exist_ok=True)


def _maybe_extract(zip_path, out_dir):
    marker = os.path.join(out_dir, ".extracted")
    if os.path.exists(marker):
        return
    with ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    with open(marker, "w") as f:
        f.write("ok")


_maybe_extract(path + "train.zip", "./train")
_maybe_extract(path + "test.zip", "./test")

training_files = "train/" + files_dataframe["id"]
print("Training sample:")
print(training_files.head(2))

print("Sanity check exists:", os.path.exists("./" + training_files.iloc[0]))



## === cell 4
class_reparts = files_dataframe["has_cactus"].value_counts()
ax = class_reparts.plot.bar()



## === cell 5
total_samples = files_dataframe["has_cactus"].size
print("Total number of samples: ", total_samples)
has_cactus_weight = total_samples / (2 * class_reparts[1])
no_cactus_weight = total_samples / (2 * class_reparts[0])
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights: ", class_weights)



## === cell 6
import skimage.exposure as exposure


def preprocess(img):
    p2, p98 = np.percentile(img, (3, 97))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale




## === cell 7
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = (32, 32)

files_df = files_dataframe.copy()
files_df["id"] = files_df["id"].astype(str)

perm = np.random.RandomState(SEED).permutation(len(files_df))
files_df = files_df.iloc[perm].reset_index(drop=True)

val_frac = 0.25
val_n = int(round(len(files_df) * val_frac))
val_df = files_df.iloc[:val_n].reset_index(drop=True)
train_df = files_df.iloc[val_n:].reset_index(drop=True)

print("Train size:", len(train_df), "Val size:", len(val_df))

_p_cache_path = "/kaggle/working/p2p98_cache.npz"


def _build_p2p98_cache(df, root_dir="./train/"):
    import imageio.v2 as imageio  # fast JPEG reader in Python

    ids = df["id"].to_numpy()
    p2 = np.empty(len(ids), dtype=np.float32)
    p98 = np.empty(len(ids), dtype=np.float32)
    for i, fn in enumerate(ids):
        fpath = os.path.join(root_dir, fn)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing image: {fpath}")
        img = imageio.imread(fpath)
        imgf = img.astype(np.float32)
        p2[i], p98[i] = np.percentile(imgf, (3, 97))
    return ids.astype(str), p2, p98


if os.path.exists(_p_cache_path):
    cache = np.load(_p_cache_path, allow_pickle=False)
    cache_ids = cache["id"].astype(str)
    cache_p2 = cache["p2"].astype(np.float32)
    cache_p98 = cache["p98"].astype(np.float32)
else:
    cache_ids, cache_p2, cache_p98 = _build_p2p98_cache(files_df, root_dir="./train/")
    np.savez_compressed(_p_cache_path, id=cache_ids, p2=cache_p2, p98=cache_p98)

keys = tf.constant(cache_ids, dtype=tf.string)
vals_p2 = tf.constant(cache_p2, dtype=tf.float32)
vals_p98 = tf.constant(cache_p98, dtype=tf.float32)

table_p2 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals_p2),
    default_value=tf.constant(0.0, dtype=tf.float32),
)
table_p98 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals_p98),
    default_value=tf.constant(255.0, dtype=tf.float32),
)


def _decode_jpeg(path_):
    img_bytes = tf.io.read_file(path_)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) * 255.0  # match skimage domain
    return img


def _rescale_intensity(img, p2, p98):
    p2 = tf.maximum(p2, 0.0)
    p98 = tf.minimum(p98, 255.0)
    denom = tf.maximum(p98 - p2, 1e-6)
    img = (img - p2) * (255.0 / denom)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def _samplewise_center_std(img):
    mean = tf.reduce_mean(img)
    std = tf.math.reduce_std(img)
    img = img - mean
    img = img / tf.maximum(std, 1e-6)
    return img


import keras_cv

_aug = keras_cv.layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED)
_rot = keras_cv.layers.RandomRotation(
    factor=45.0 / 360.0, fill_mode="reflect", seed=SEED
)
_shear = keras_cv.layers.RandomShear(
    x_factor=10.0 / 180.0, y_factor=0.0, fill_mode="reflect", seed=SEED
)


def _augment(img):
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = _aug(img)
    img = _rot(img)
    img = _shear(img)
    return img


def _prep_example(filename, label, do_augment=True, use_preprocess=True):
    fpath = tf.strings.join(["./train/", filename])
    img = _decode_jpeg(fpath)

    if use_preprocess:
        p2 = table_p2.lookup(filename)
        p98 = table_p98.lookup(filename)
        img = _rescale_intensity(img, p2, p98)

    if do_augment:
        img = _augment(img)
    else:
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    img = _samplewise_center_std(img)

    label = tf.cast(label, tf.int32)
    y = tf.one_hot(label, depth=2, dtype=tf.float32)
    return img, y


def make_ds(df, training=True):
    filenames = tf.constant(df["id"].to_numpy(dtype=str), dtype=tf.string)
    labels = tf.constant(df["has_cactus"].to_numpy(dtype=np.int32), dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda f, y: _prep_example(f, y, do_augment=True, use_preprocess=True),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda f, y: _prep_example(f, y, do_augment=False, use_preprocess=True),
            num_parallel_calls=AUTOTUNE,
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


training_ds = make_ds(train_df, training=True)
validation_ds = make_ds(val_df, training=False)

training_n = len(train_df)
validation_n = len(val_df)



## === cell 8
print("training_ds samples:", training_n)
print("validation_ds samples:", validation_n)



## === cell 9
pass



## === cell 10
from tf_keras import layers, models
from tf_keras.models import clone_model



## === cell 11
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

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 12
from tf_keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

checkpoint_path = "/tmp/checkpoint.keras"
save_best_model = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)



## === cell 13
steps_per_epoch = max(1, training_n // BATCH_SIZE)
val_steps = max(1, validation_n // BATCH_SIZE)

history = model.fit(
    training_ds,
    validation_data=validation_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
    epochs=10,
    callbacks=[reduce_lr, save_best_model],
    class_weight=class_weights,
)



## === cell 14
sample_sub = pd.read_csv(path + "sample_submission.csv", dtype={"id": str})
test_df = sample_sub.copy()


def _prep_test_example(filename):
    fpath = tf.strings.join(["./test/", filename])
    img = _decode_jpeg(fpath)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    flat = tf.reshape(img, [-1])
    flat_sorted = tf.sort(flat)
    n = tf.shape(flat_sorted)[0]
    i2 = tf.cast(tf.round(0.03 * tf.cast(n - 1, tf.float32)), tf.int32)
    i98 = tf.cast(tf.round(0.97 * tf.cast(n - 1, tf.float32)), tf.int32)
    p2 = flat_sorted[i2]
    p98 = flat_sorted[i98]

    img = _rescale_intensity(img, p2, p98)
    img = _samplewise_center_std(img)
    return img


test_filenames = tf.constant(test_df["id"].to_numpy(dtype=str), dtype=tf.string)
test_ds = tf.data.Dataset.from_tensor_slices(test_filenames)
test_ds = test_ds.map(_prep_test_example, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

print("test samples:", len(test_df))



## === cell 15
from tf_keras.models import load_model

if os.path.exists(checkpoint_path):
    model = load_model(checkpoint_path)



## === cell 16
proba = model.predict(test_ds, verbose=0)
if proba.ndim == 2 and proba.shape[1] == 2:
    has_cactus_proba = proba[:, 1]
else:
    has_cactus_proba = proba.reshape(-1)

output = sample_sub.copy()
output["has_cactus"] = has_cactus_proba.astype(float)
output["has_cactus"] = output["has_cactus"].clip(0.0, 1.0)

output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print(output.head())



## === cell 17
import shutil

for d in ["test", "train"]:
    try:
        shutil.rmtree(d)
    except OSError:
        print(f"{d} files already erased or not present")
