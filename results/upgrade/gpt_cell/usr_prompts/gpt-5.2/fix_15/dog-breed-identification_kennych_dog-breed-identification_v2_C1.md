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

No external packages required in the script and installed.

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
import sys
import time
import math
import shutil
import random
import pathlib
import collections

import numpy as np
import pandas as pd



## === cell 1
pass



## === cell 2
src = "/kaggle/input/dog-breed-identification"
dst = "/kaggle/working/dog-breed-identification"
if not os.path.exists(dst):
    try:
        os.symlink(src, dst)
    except Exception:
        pass



## === cell 3
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass


class _D2LShim:
    @staticmethod
    def mkdir_if_not_exist(path):
        if isinstance(path, (list, tuple)):
            path = os.path.join(*path)
        os.makedirs(path, exist_ok=True)


d2l = _D2LShim()


## === cell 4
def reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label):
    min_n_train_per_label = collections.Counter(idx_label.values()).most_common()[
        :-2:-1
    ][0][1]
    n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = train_file.split(".")[0]
        label = idx_label[idx]
        d2l.mkdir_if_not_exist([data_dir, input_dir, "train_valid", label])
        shutil.copy(
            os.path.join(data_dir, train_dir, train_file),
            os.path.join(data_dir, input_dir, "train_valid", label),
        )
        if label not in label_count or label_count[label] < n_valid_per_label:
            d2l.mkdir_if_not_exist([data_dir, input_dir, "valid", label])
            shutil.copy(
                os.path.join(data_dir, train_dir, train_file),
                os.path.join(data_dir, input_dir, "valid", label),
            )
            label_count[label] = label_count.get(label, 0) + 1
        else:
            d2l.mkdir_if_not_exist([data_dir, input_dir, "train", label])
            shutil.copy(
                os.path.join(data_dir, train_dir, train_file),
                os.path.join(data_dir, input_dir, "train", label),
            )


def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    with open(os.path.join(data_dir, label_file), "r") as f:
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(",") for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))
    reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label)
    d2l.mkdir_if_not_exist([data_dir, input_dir, "test", "unknown"])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(
            os.path.join(data_dir, test_dir, test_file),
            os.path.join(data_dir, input_dir, "test", "unknown"),
        )




## === cell 5
data_dir = "/kaggle/working/dog-breed-identification"
if not os.path.exists(os.path.join(data_dir, "labels.csv")):
    data_dir = "/kaggle/input/dog-breed-identification"

label_file, train_dir, test_dir = "labels.csv", "train", "test"
input_dir, batch_size, valid_ratio = "train_valid_test", 128, 0.1

labels_df = pd.read_csv(os.path.join(data_dir, label_file))
label_names = sorted(labels_df["breed"].unique().tolist())
label_to_index = {name: i for i, name in enumerate(label_names)}

idx_label = dict(
    zip(labels_df["id"].values.tolist(), labels_df["breed"].values.tolist())
)

min_n_train_per_label = collections.Counter(idx_label.values()).most_common()[:-2:-1][
    0
][1]
n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)

train_root = os.path.join(data_dir, train_dir)
train_files = sorted([f for f in os.listdir(train_root) if f.lower().endswith(".jpg")])

label_count = collections.defaultdict(int)
train_all_image_paths, train_all_image_labels = [], []
valid_all_image_paths, valid_all_image_labels = [], []
train_valid_all_image_paths, train_valid_all_image_labels = [], []

for train_file in train_files:
    idx = train_file.split(".")[0]
    label = idx_label[idx]
    full_path = os.path.join(train_root, train_file)
    y = label_to_index[label]

    train_valid_all_image_paths.append(full_path)
    train_valid_all_image_labels.append(y)

    if label_count[label] < n_valid_per_label:
        valid_all_image_paths.append(full_path)
        valid_all_image_labels.append(y)
        label_count[label] += 1
    else:
        train_all_image_paths.append(full_path)
        train_all_image_labels.append(y)

test_root = os.path.join(data_dir, test_dir)
test_files = sorted([f for f in os.listdir(test_root) if f.lower().endswith(".jpg")])
test_all_image_paths = [os.path.join(test_root, f) for f in test_files]
test_all_image_labels = [-1 for _ in range(len(test_all_image_paths))]

print("First 10 images indices: ", train_valid_all_image_labels[:10])
print("First 10 labels indices: ", train_valid_all_image_labels[:10])



## === cell 6
MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


@tf.function
def _decode_and_resize_400(imgpath, label):
    img = tf.io.read_file(imgpath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=[400, 400])
    return img, label


@tf.function
def _augment_and_normalize(img, label, seed_pair):
    scale = tf.random.stateless_uniform(
        [], seed=seed_pair, minval=0.08, maxval=1.0, dtype=tf.float32
    )
    h = tf.cast(tf.round(scale * tf.cast(tf.shape(img)[0], tf.float32)), tf.int32)
    w = tf.cast(tf.round(scale * tf.cast(tf.shape(img)[1], tf.float32)), tf.int32)
    h = tf.maximum(h, 1)
    w = tf.maximum(w, 1)
    img = tf.image.random_crop(img, size=[h, w, 3], seed=seed_pair[0])
    img = tf.image.resize(img, size=[224, 224])

    img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)
    img = tf.image.stateless_random_flip_up_down(
        img, seed=seed_pair + tf.constant([1, 1], tf.int32)
    )

    img = tf.cast(img, tf.float32) / 255.0
    img = (img - MEAN) / STD
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, label


@tf.function
def _transform_test(imgpath, label):
    img = tf.io.read_file(imgpath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - MEAN) / STD
    return img, label




## === cell 7
data_root = os.path.join(data_dir, input_dir)
train_data_root = pathlib.Path(data_root + "/train")
valid_data_root = pathlib.Path(data_root + "/valid")
train_valid_data_root = pathlib.Path(data_root + "/train_valid")
test_data_root = pathlib.Path(data_root + "/test")

print("Num train:", len(train_all_image_paths))
print("Num valid:", len(valid_all_image_paths))
print("Num train_valid:", len(train_valid_all_image_paths))
print("Num test:", len(test_all_image_paths))



## === cell 8
AUTOTUNE = tf.data.AUTOTUNE


def build_train_like_ds(paths, labels, shuffle, cache_name, seed_offset):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 2048),
            seed=SEED + seed_offset,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(_decode_and_resize_400, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(os.path.join("/kaggle/working", cache_name))

    ds = ds.enumerate()

    def _apply_aug(i, x):
        img, y = x
        seed_pair = tf.stack(
            [tf.cast(SEED + seed_offset, tf.int32), tf.cast(i, tf.int32)], axis=0
        )
        return _augment_and_normalize(img, y, seed_pair)

    ds = ds.map(_apply_aug, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


options = tf.data.Options()
options.experimental_deterministic = False

if "_augment_and_normalize" in globals():
    _orig_augment_and_normalize = _augment_and_normalize

    @tf.function
    def _augment_and_normalize(img, label, seed_pair):
        scale = tf.random.stateless_uniform(
            [], seed=seed_pair, minval=0.08, maxval=1.0, dtype=tf.float32
        )
        h = tf.cast(tf.round(scale * tf.cast(tf.shape(img)[0], tf.float32)), tf.int32)
        w = tf.cast(tf.round(scale * tf.cast(tf.shape(img)[1], tf.float32)), tf.int32)
        h = tf.maximum(h, 1)
        w = tf.maximum(w, 1)
        img = tf.image.stateless_random_crop(img, size=[h, w, 3], seed=seed_pair)
        img = tf.image.resize(img, size=[224, 224])

        img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)
        img = tf.image.stateless_random_flip_up_down(
            img, seed=seed_pair + tf.constant([1, 1], tf.int32)
        )

        img = tf.cast(img, tf.float32) / 255.0
        img = (img - MEAN) / STD
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label


train_ds = build_train_like_ds(
    train_all_image_paths,
    train_all_image_labels,
    shuffle=True,
    cache_name="cache_train_400",
    seed_offset=1,
).with_options(options)

valid_ds = build_train_like_ds(
    valid_all_image_paths,
    valid_all_image_labels,
    shuffle=True,
    cache_name="cache_valid_400",
    seed_offset=2,
).with_options(options)

train_valid_ds = build_train_like_ds(
    train_valid_all_image_paths,
    train_valid_all_image_labels,
    shuffle=True,
    cache_name="cache_trainvalid_400",
    seed_offset=3,
).with_options(options)

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_all_image_paths, test_all_image_labels))
    .map(_transform_test, num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
).with_options(options)


## === cell 9
train_ds



## === cell 10
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



## === cell 11
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



## === cell 12
model.fit(train_ds, epochs=1, validation_data=valid_ds, callbacks=[callback])



## === cell 13
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    loss="sparse_categorical_crossentropy",
)
model.fit(train_valid_ds, epochs=1, callbacks=[callback])



## === cell 14
probabilities = model.predict(test_ds)
predictions = np.argmax(probabilities, axis=-1)



## === cell 15
print(probabilities)



## === cell 16
df = pd.read_csv("/kaggle/working/dog-breed-identification/sample_submission.csv")
if not os.path.exists("/kaggle/working/dog-breed-identification/sample_submission.csv"):
    df = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")

for i, c in enumerate(df.columns[1:]):
    df[c] = probabilities[:, i]

df.to_csv("submission.csv", index=None)



## === cell 17
try:
    if os.path.islink("/kaggle/working/dog-breed-identification"):
        pass
    elif os.path.isdir("/kaggle/working/dog-breed-identification"):
        pass
except Exception:
    pass
