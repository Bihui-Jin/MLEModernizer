# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.9

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import math
import random
import shutil
import collections
import pathlib
import errno

import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf
from tensorflow import keras

print("TF version:", tf.__version__)



## === cell 1
root = "/kaggle/working/dog-breed-identification"
if os.path.exists(root):
    for p in (
        "labels.csv",
        "sample_submission.csv",
        "train",
        "test",
    ):
        print(
            "Exists:",
            os.path.join(root, p),
            "->",
            os.path.exists(os.path.join(root, p)),
        )
else:
    print(f"{root} does not exist yet (will be created next).")



## === cell 2
print(
    "Skipping `pip install d2lzh` because mxnet is not available; using a local mkdir helper instead."
)



## === cell 3
src = "/kaggle/input/dog-breed-identification"
dst = "/kaggle/working/dog-breed-identification"
if os.path.exists(dst):
    data_root = dst
else:
    data_root = src
print("Using data_root:", data_root)
print("pwd:", os.getcwd())




## === cell 4
def mkdir_if_not_exist(path_parts):
    path = (
        os.path.join(*path_parts)
        if isinstance(path_parts, (list, tuple))
        else path_parts
    )
    os.makedirs(path, exist_ok=True)


def _is_image_file(fn: str) -> bool:
    fnl = fn.lower()
    return fnl.endswith(".jpg") or fnl.endswith(".jpeg") or fnl.endswith(".png")


def _find_image_dir(base_dir: str, expected_name: str) -> str:
    """
    BUGFIX: Kaggle dataset sometimes contains nested folders like train/train and test/test.
    This finds the directory that actually contains image files.
    """
    cand1 = os.path.join(base_dir, expected_name)
    cand2 = os.path.join(base_dir, expected_name, expected_name)
    for cand in (cand2, cand1):
        if os.path.isdir(cand):
            try:
                files = os.listdir(cand)
            except Exception:
                continue
            if any(_is_image_file(f) for f in files):
                return cand
    for root, _, files in os.walk(base_dir):
        if os.path.basename(root) == expected_name and any(
            _is_image_file(f) for f in files
        ):
            return root
    raise FileNotFoundError(
        f"Could not find an image directory for '{expected_name}' under {base_dir}"
    )


def _fast_link_or_copy(src_path: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        if os.path.exists(dst_path):
            return
        os.link(src_path, dst_path)  # hardlink: fastest, preserves bytes exactly
    except OSError:
        try:
            if os.path.exists(dst_path):
                return
            os.symlink(src_path, dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)


def reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label):
    train_img_dir = _find_image_dir(data_dir, train_dir)

    train_files = [f for f in os.listdir(train_img_dir) if _is_image_file(f)]
    if len(train_files) == 0:
        raise RuntimeError(f"No training images found in: {train_img_dir}")

    min_n_train_per_label = collections.Counter(idx_label.values()).most_common()[
        :-2:-1
    ][0][1]
    n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)

    label_count = {}
    for train_file in train_files:
        idx = train_file.split(".")[0]
        if idx not in idx_label:
            continue
        label = idx_label[idx]

        src_path = os.path.join(train_img_dir, train_file)

        dst_tv = os.path.join(data_dir, input_dir, "train_valid", label, train_file)
        _fast_link_or_copy(src_path, dst_tv)

        if label not in label_count or label_count[label] < n_valid_per_label:
            dst_v = os.path.join(data_dir, input_dir, "valid", label, train_file)
            _fast_link_or_copy(src_path, dst_v)
            label_count[label] = label_count.get(label, 0) + 1
        else:
            dst_t = os.path.join(data_dir, input_dir, "train", label, train_file)
            _fast_link_or_copy(src_path, dst_t)


def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    with open(os.path.join(data_dir, label_file), "r") as f:
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(",") for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))

    reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label)

    test_img_dir = _find_image_dir(data_dir, test_dir)
    mkdir_if_not_exist([data_dir, input_dir, "test", "unknown"])
    test_files = [f for f in os.listdir(test_img_dir) if _is_image_file(f)]
    for test_file in test_files:
        src_path = os.path.join(test_img_dir, test_file)
        dst_path = os.path.join(data_dir, input_dir, "test", "unknown", test_file)
        _fast_link_or_copy(src_path, dst_path)




## === cell 5
data_dir = data_root
label_file, train_dir, test_dir = "labels.csv", "train", "test"
input_dir, batch_size, valid_ratio = "train_valid_test", 128, 0.1

expected_dir = os.path.join(data_dir, input_dir)
print("Reorg step skipped for speed; using original folders directly.")
print("Expected (unused) reorg dir exists:", os.path.exists(expected_dir))



## === cell 6
MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


@tf.function
def _decode_and_resize400(imgpath):
    imgbytes = tf.io.read_file(imgpath)
    feature = tf.image.decode_jpeg(imgbytes, channels=3)
    feature = tf.image.resize(feature, size=[400, 400])
    feature.set_shape([400, 400, 3])
    return feature


@tf.function
def _augment_from_resized400(feature400, imgpath, label):
    seed_base = tf.strings.to_hash_bucket_fast(imgpath, 2**31 - 1)
    seed = tf.stack(
        [tf.cast(seed_base, tf.int32), tf.cast(seed_base ^ 0x9E3779B9, tf.int32)]
    )

    u = tf.random.stateless_uniform([], seed=seed, minval=0, maxval=93, dtype=tf.int32)
    scale = tf.cast(u + 8, tf.float32) / 100.0

    h = tf.shape(feature400)[0]
    w = tf.shape(feature400)[1]
    crop_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    crop_h = tf.maximum(1, tf.minimum(crop_h, h))
    crop_w = tf.maximum(1, tf.minimum(crop_w, w))
    feature = tf.image.stateless_random_crop(
        feature400, size=[crop_h, crop_w, 3], seed=seed + 1
    )

    feature = tf.image.resize(feature, size=[224, 224])
    feature.set_shape([224, 224, 3])

    feature = tf.image.stateless_random_flip_left_right(feature, seed=seed + 2)
    feature = tf.image.stateless_random_flip_up_down(feature, seed=seed + 3)

    feature = tf.cast(feature, tf.float32) / 255.0
    feature = (feature - MEAN) / STD
    return feature, label


@tf.function
def transform_train_from_path(imgpath, label):
    feature400 = _decode_and_resize400(imgpath)
    return _augment_from_resized400(feature400, imgpath, label)


@tf.function
def transform_test_from_path(imgpath, label):
    imgbytes = tf.io.read_file(imgpath)
    feature = tf.image.decode_jpeg(imgbytes, channels=3)
    feature = tf.image.resize(feature, [224, 224])
    feature.set_shape([224, 224, 3])
    feature = tf.cast(feature, tf.float32) / 255.0
    feature = (feature - MEAN) / STD
    return feature, label




## === cell 7
data_root = data_dir

train_img_dir = _find_image_dir(data_root, "train")
test_img_dir = _find_image_dir(data_root, "test")

sample_path = os.path.join(data_root, "sample_submission.csv")
sample_df = pd.read_csv(sample_path)
breed_cols = list(sample_df.columns[1:])
label_names = breed_cols[:]  # preserve order
label_to_index = {name: index for index, name in enumerate(label_names)}

labels_path = os.path.join(data_root, "labels.csv")
labels_df = pd.read_csv(labels_path)
idx_to_label = dict(zip(labels_df["id"].astype(str).values, labels_df["breed"].values))


def _list_image_filenames_flat(dir_path: str):
    out = []
    with os.scandir(dir_path) as it:
        for e in it:
            if e.is_file() and _is_image_file(e.name):
                out.append(e.name)
    out.sort()  # deterministic order
    return out


train_files = _list_image_filenames_flat(train_img_dir)
if len(train_files) == 0:
    raise RuntimeError(f"No training images found in: {train_img_dir}")

min_n_train_per_label = collections.Counter(idx_to_label.values()).most_common()[
    :-2:-1
][0][1]
n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)

train_ids = pd.Series([fn.split(".")[0] for fn in train_files], name="id")
train_df = pd.DataFrame({"id": train_ids, "filename": train_files})
train_df = train_df.merge(labels_df, on="id", how="left")
train_df = train_df.dropna(subset=["breed"]).reset_index(drop=True)

train_df["within_label_rank"] = train_df.groupby("breed").cumcount()
is_valid = train_df["within_label_rank"] < n_valid_per_label

train_df["y"] = train_df["breed"].map(label_to_index).astype(np.int32)

train_all_image_paths = (
    train_img_dir + "/" + train_df.loc[~is_valid, "filename"]
).tolist()
train_all_image_labels = train_df.loc[~is_valid, "y"].to_numpy(np.int32)

valid_all_image_paths = (
    train_img_dir + "/" + train_df.loc[is_valid, "filename"]
).tolist()
valid_all_image_labels = train_df.loc[is_valid, "y"].to_numpy(np.int32)

train_valid_all_image_paths = (train_img_dir + "/" + train_df["filename"]).tolist()
train_valid_all_image_labels = train_df["y"].to_numpy(np.int32)

test_files = _list_image_filenames_flat(test_img_dir)
test_all_image_paths = [os.path.join(test_img_dir, fn) for fn in test_files]
test_all_image_labels = np.full((len(test_all_image_paths),), -1, dtype=np.int32)

print("n_labels:", len(label_names))
print(
    "n_train:",
    len(train_all_image_paths),
    "n_valid:",
    len(valid_all_image_paths),
    "n_test:",
    len(test_all_image_paths),
)



## === cell 8
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_and_batch_fusion = True
options.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
options.threading.max_intra_op_parallelism = 1


@tf.function
def _decode400_pair(imgpath, label):
    return _decode_and_resize400(imgpath), imgpath, label


def _make_train_like_ds(paths, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(options)

    if shuffle:
        buf = min(len(paths), 2048)
        ds = ds.shuffle(buf, reshuffle_each_iteration=True)

    ds = ds.map(
        _decode400_pair, num_parallel_calls=AUTOTUNE, deterministic=True
    ).cache()

    @tf.function
    def _aug_triple(feature400, imgpath, label):
        return _augment_from_resized400(feature400, imgpath, label)

    ds = ds.map(_aug_triple, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_like_ds(
    train_all_image_paths,
    train_all_image_labels,
    shuffle=True,
)
valid_ds = _make_train_like_ds(
    valid_all_image_paths,
    valid_all_image_labels,
    shuffle=True,
)
train_valid_ds = _make_train_like_ds(
    train_valid_all_image_paths,
    train_valid_all_image_labels,
    shuffle=True,
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_all_image_paths, test_all_image_labels))
    .with_options(options)
    .map(transform_test_from_path, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

train_steps = int(math.ceil(len(train_all_image_paths) / batch_size))
valid_steps = int(math.ceil(len(valid_all_image_paths) / batch_size))
train_valid_steps = int(math.ceil(len(train_valid_all_image_paths) / batch_size))

train_ds = train_ds.repeat()
valid_ds = valid_ds.repeat()
train_valid_ds = train_valid_ds.repeat()

train_ds



## === cell 9
from tensorflow.keras.applications import ResNet50

net = ResNet50(input_shape=(224, 224, 3), weights="imagenet", include_top=False)

model = tf.keras.Sequential(
    [
        net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu", dtype=tf.float32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_names), activation="softmax", dtype=tf.float32),
    ]
)

model.summary()



## === cell 10
lr = 0.1
lr_decay = 0.01


def scheduler(epoch):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(lr_decay * (10 - epoch))


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    loss="sparse_categorical_crossentropy",
)



## === cell 11
model.fit(
    train_ds,
    epochs=1,
    steps_per_epoch=train_steps,
    validation_data=valid_ds,
    validation_steps=valid_steps,
    callbacks=[callback],
)



## === cell 12
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    loss="sparse_categorical_crossentropy",
)
model.fit(
    train_valid_ds,
    epochs=1,
    steps_per_epoch=train_valid_steps,
    callbacks=[callback],
)



## === cell 13
probabilities = model.predict(test_ds, verbose=1)
print("probabilities shape:", probabilities.shape)



## === cell 14
df = sample_df.copy()

if probabilities.shape[1] != len(breed_cols):
    raise ValueError(
        f"Model outputs {probabilities.shape[1]} classes, but submission expects {len(breed_cols)}"
    )

df.loc[:, breed_cols] = probabilities

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(df.head())



## === cell 15
print("Done. Submission file is ready at ./submission.csv")
