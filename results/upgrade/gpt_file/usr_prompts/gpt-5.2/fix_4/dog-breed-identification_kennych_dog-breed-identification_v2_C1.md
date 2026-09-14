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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import shutil
import collections
import math
import random
import time
import zipfile
import pathlib

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt


def mkdir_if_not_exist(path_parts):
    path = (
        os.path.join(*path_parts)
        if isinstance(path_parts, (list, tuple))
        else path_parts
    )
    os.makedirs(path, exist_ok=True)


print("TensorFlow:", tf.__version__)

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## === cell 1
src = "/kaggle/input/dog-breed-identification"
dst = "/kaggle/working/dog-breed-identification"
if not os.path.exists(dst):
    shutil.copytree(src, dst)
print("Data copied/exists at:", dst)
print("PWD:", os.getcwd())




## === cell 2
def _is_image_file(fn: str) -> bool:
    fn_low = fn.lower()
    return (
        fn_low.endswith(".jpg") or fn_low.endswith(".jpeg") or fn_low.endswith(".png")
    )


def _fast_link(src_path: str, dst_path: str):
    if os.path.exists(dst_path):
        return
    try:
        os.symlink(src_path, dst_path)
    except Exception:
        try:
            os.link(src_path, dst_path)  # hardlink fallback
        except Exception:
            shutil.copy2(src_path, dst_path)  # last resort (correctness-preserving)


def reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label):
    train_path = os.path.join(data_dir, train_dir)
    train_files = [f for f in os.listdir(train_path) if _is_image_file(f)]

    min_n_train_per_label = collections.Counter(idx_label.values()).most_common()[
        :-2:-1
    ][0][1]
    n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)

    label_count = {}
    for train_file in train_files:
        idx = os.path.splitext(train_file)[0]
        if idx not in idx_label:
            continue
        label = idx_label[idx]

        mkdir_if_not_exist([data_dir, input_dir, "train_valid", label])
        _fast_link(
            os.path.join(train_path, train_file),
            os.path.join(data_dir, input_dir, "train_valid", label, train_file),
        )

        if label not in label_count or label_count[label] < n_valid_per_label:
            mkdir_if_not_exist([data_dir, input_dir, "valid", label])
            _fast_link(
                os.path.join(train_path, train_file),
                os.path.join(data_dir, input_dir, "valid", label, train_file),
            )
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, "train", label])
            _fast_link(
                os.path.join(train_path, train_file),
                os.path.join(data_dir, input_dir, "train", label, train_file),
            )


def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    with open(os.path.join(data_dir, label_file), "r") as f:
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(",") for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))

    reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label)

    mkdir_if_not_exist([data_dir, input_dir, "test", "unknown"])
    test_path = os.path.join(data_dir, test_dir)
    test_files = [f for f in os.listdir(test_path) if _is_image_file(f)]
    for test_file in test_files:
        _fast_link(
            os.path.join(test_path, test_file),
            os.path.join(data_dir, input_dir, "test", "unknown", test_file),
        )




## === cell 3
data_dir = "/kaggle/working/dog-breed-identification"
label_file, train_dir, test_dir = "labels.csv", "train", "test"
input_dir, batch_size, valid_ratio = "train_valid_test", 128, 0.1

prepared_root = os.path.join(data_dir, input_dir)
prepared_flag = os.path.join(prepared_root, "_PREPARED")

if not os.path.exists(prepared_flag) and os.path.exists(prepared_root):
    shutil.rmtree(prepared_root, ignore_errors=True)

if not os.path.exists(prepared_flag):
    reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)
    mkdir_if_not_exist(os.path.dirname(prepared_flag))
    with open(prepared_flag, "w") as f:
        f.write("ok")

print("Prepared data root:", prepared_root)




## === cell 4
MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def _normalize(feature):
    feature = tf.divide(feature, 255.0)
    feature = tf.divide(tf.subtract(feature, MEAN), STD)
    return tf.image.convert_image_dtype(feature, tf.float32)


def transform_train(imgpath, label):
    feature = tf.io.read_file(imgpath)
    feature = tf.image.decode_jpeg(feature, channels=3)
    feature = tf.image.resize(feature, size=[400, 400])

    seed = tf.random.experimental.stateless_fold_in(
        tf.constant([SEED, 0], dtype=tf.int32),
        tf.strings.to_hash_bucket_fast(imgpath, 2**31 - 1),
    )
    rint = tf.random.stateless_uniform(
        shape=[],
        seed=seed,
        minval=8,
        maxval=101,
        dtype=tf.int32,
    )
    scale = tf.cast(rint, tf.float32) / 100.0

    h = tf.cast(tf.shape(feature)[0], tf.float32)
    w = tf.cast(tf.shape(feature)[1], tf.float32)
    ch = tf.cast(tf.maximum(1.0, tf.floor(scale * h)), tf.int32)
    cw = tf.cast(tf.maximum(1.0, tf.floor(scale * w)), tf.int32)

    feature = tf.image.random_crop(feature, size=[ch, cw, 3], seed=SEED)
    feature = tf.image.resize(feature, size=[224, 224])
    feature = tf.image.random_flip_left_right(feature, seed=SEED)
    feature = tf.image.random_flip_up_down(feature, seed=SEED)
    feature = _normalize(feature)
    return feature, label


def transform_test(imgpath, label):
    feature = tf.io.read_file(imgpath)
    feature = tf.image.decode_jpeg(feature, channels=3)
    feature = tf.image.resize(feature, [224, 224])
    feature = _normalize(feature)
    return feature, label




## === cell 5
labels_df = pd.read_csv(os.path.join(data_dir, label_file))
label_names = sorted(labels_df["breed"].unique().tolist())
label_to_index = {name: index for index, name in enumerate(label_names)}

data_root = os.path.join(data_dir, input_dir)
train_data_root = pathlib.Path(os.path.join(data_root, "train"))
valid_data_root = pathlib.Path(os.path.join(data_root, "valid"))
train_valid_data_root = pathlib.Path(os.path.join(data_root, "train_valid"))
test_data_root = pathlib.Path(os.path.join(data_root, "test"))

train_all_image_paths = [str(path) for path in list(train_data_root.glob("*/*"))]
valid_all_image_paths = [str(path) for path in list(valid_data_root.glob("*/*"))]
train_valid_all_image_paths = [
    str(path) for path in list(train_valid_data_root.glob("*/*"))
]
test_all_image_paths = [str(path) for path in list(test_data_root.glob("*/*"))]

train_all_image_labels = [
    label_to_index[pathlib.Path(path).parent.name] for path in train_all_image_paths
]
valid_all_image_labels = [
    label_to_index[pathlib.Path(path).parent.name] for path in valid_all_image_paths
]
train_valid_all_image_labels = [
    label_to_index[pathlib.Path(path).parent.name]
    for path in train_valid_all_image_paths
]
test_all_image_labels = [-1 for _ in range(len(test_all_image_paths))]

print("Num classes:", len(label_names))
print(
    "Train/Valid/TrainValid/Test sizes:",
    len(train_all_image_paths),
    len(valid_all_image_paths),
    len(train_valid_all_image_paths),
    len(test_all_image_paths),
)




## === cell 6
AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_all_image_paths, train_all_image_labels))
    .with_options(options)
    .shuffle(len(train_all_image_paths), seed=SEED, reshuffle_each_iteration=True)
    .map(transform_train, num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_all_image_paths, valid_all_image_labels))
    .with_options(options)
    .map(
        transform_test, num_parallel_calls=AUTOTUNE
    )  # no augmentation; preserves evaluation semantics and speeds up
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

train_valid_ds = (
    tf.data.Dataset.from_tensor_slices(
        (train_valid_all_image_paths, train_valid_all_image_labels)
    )
    .with_options(options)
    .shuffle(len(train_valid_all_image_paths), seed=SEED, reshuffle_each_iteration=True)
    .map(transform_train, num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_all_image_paths, test_all_image_labels))
    .with_options(options)
    .map(transform_test, num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## === cell 7
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




## === cell 8
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




## === cell 9
model.fit(train_ds, epochs=1, validation_data=valid_ds, callbacks=[callback])




## === cell 10
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    loss="sparse_categorical_crossentropy",
)
model.fit(train_valid_ds, epochs=1, callbacks=[callback])




## === cell 11
probabilities = model.predict(test_ds, verbose=1)
print("Pred shape:", probabilities.shape)

probabilities = probabilities / np.clip(
    probabilities.sum(axis=1, keepdims=True), 1e-12, None
)




## === cell 12
sub_path = os.path.join(data_dir, "sample_submission.csv")
df = pd.read_csv(sub_path)

test_ids = np.array(
    [os.path.splitext(os.path.basename(p))[0] for p in test_all_image_paths]
)
pred_df = pd.DataFrame(probabilities, index=test_ids, columns=label_names)

breed_cols = list(df.columns[1:])
if set(label_names) != set(breed_cols):
    raise ValueError(
        f"Label names do not match submission columns. "
        f"Got {len(label_names)} labels vs {len(breed_cols)} submission cols."
    )

aligned = pred_df.reindex(index=df["id"].values, columns=breed_cols)
if aligned.isna().any().any():
    missing = int(aligned.isna().any(axis=1).sum())
    raise ValueError(
        f"Missing predictions for {missing} test ids; check test set paths."
    )

df.loc[:, breed_cols] = aligned.values.astype(np.float32, copy=False)
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)




## === cell 13
print("Done.")
