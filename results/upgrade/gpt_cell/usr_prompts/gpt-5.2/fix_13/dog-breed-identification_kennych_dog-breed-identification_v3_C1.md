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
import numpy as np
import pandas as pd
import os

DATA_ROOT = "/kaggle/working/dog-breed-identification"
print("DATA_ROOT exists:", os.path.exists(DATA_ROOT))



## === cell 1
import sys, subprocess, importlib.util

if importlib.util.find_spec("d2lzh") is None:
    print("d2lzh not installed; using fallback implementation provided later.")
else:
    print("d2lzh installed.")



## === cell 2
import shutil, os

src = "/kaggle/input/dog-breed-identification"
dst = "/kaggle/working/dog-breed-identification"
if not os.path.exists(dst):
    shutil.copytree(src, dst)
print("Dataset ready at:", dst)



## === cell 3
import collections
import math
import os
import shutil
import time
import zipfile
import sys
import subprocess
import random

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
except Exception:
    google = None


def _ensure_protobuf_compatible():
    try:
        import google.protobuf
        from packaging import version

        pb_ver = version.parse(google.protobuf.__version__)
        if pb_ver.major >= 4:
            raise RuntimeError(
                f"Incompatible protobuf version {google.protobuf.__version__}"
            )
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<4"]
        )


_ensure_protobuf_compatible()

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


SEED = 1234
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


class _D2LFallback:
    @staticmethod
    def mkdir_if_not_exist(path):
        if isinstance(path, (list, tuple)):
            path = os.path.join(*path)
        os.makedirs(path, exist_ok=True)


d2l = _D2LFallback()

autograd = gluon = init = nd = None
gdata = gloss = model_zoo = nn = None




## === cell 4
def _fast_copy_or_link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)  # hardlink; fast and storage-free
    except Exception:
        shutil.copy2(src, dst)  # safe fallback


def reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label):
    min_n_train_per_label = collections.Counter(idx_label.values()).most_common()[
        :-2:-1
    ][0][1]
    n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)
    label_count = {}

    train_root = os.path.join(data_dir, train_dir)
    out_root = os.path.join(data_dir, input_dir)

    for train_file in os.listdir(train_root):
        full_path = os.path.join(train_root, train_file)
        if not os.path.isfile(full_path):
            continue
        idx = train_file.split(".")[0]
        if idx not in idx_label:
            continue
        label = idx_label[idx]

        dst_train_valid_dir = os.path.join(out_root, "train_valid", label)
        d2l.mkdir_if_not_exist(dst_train_valid_dir)
        _fast_copy_or_link(full_path, os.path.join(dst_train_valid_dir, train_file))

        if label not in label_count or label_count[label] < n_valid_per_label:
            dst_valid_dir = os.path.join(out_root, "valid", label)
            d2l.mkdir_if_not_exist(dst_valid_dir)
            _fast_copy_or_link(full_path, os.path.join(dst_valid_dir, train_file))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            dst_train_dir = os.path.join(out_root, "train", label)
            d2l.mkdir_if_not_exist(dst_train_dir)
            _fast_copy_or_link(full_path, os.path.join(dst_train_dir, train_file))


def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    with open(os.path.join(data_dir, label_file), "r") as f:
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(",") for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))
    reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label)
    d2l.mkdir_if_not_exist([data_dir, input_dir, "test", "unknown"])
    test_root = os.path.join(data_dir, test_dir)
    out_test = os.path.join(data_dir, input_dir, "test", "unknown")
    for test_file in os.listdir(test_root):
        src = os.path.join(test_root, test_file)
        if os.path.isfile(src):
            _fast_copy_or_link(src, os.path.join(out_test, test_file))




## === cell 5
data_dir = "/kaggle/working/dog-breed-identification"
label_file, train_dir, test_dir = "labels.csv", "train", "test"
input_dir, batch_size, valid_ratio = "train_valid_test", 128, 0.1

out_root = os.path.join(data_dir, input_dir)
expected_train = os.path.join(out_root, "train")
expected_valid = os.path.join(out_root, "valid")
expected_train_valid = os.path.join(out_root, "train_valid")
expected_test = os.path.join(out_root, "test")

already_done = all(
    os.path.isdir(p)
    for p in [expected_train, expected_valid, expected_train_valid, expected_test]
)

if not already_done:
    if os.path.exists(out_root):
        shutil.rmtree(out_root)

    _label_path = os.path.join(data_dir, label_file)
    with open(_label_path, "r") as _f:
        _lines = _f.read().splitlines()[1:]
    _label_ids = set(l.split(",")[0] for l in _lines if l)

    _train_path = os.path.join(data_dir, train_dir)
    if os.path.isdir(_train_path):
        for _name in list(os.listdir(_train_path)):
            _full = os.path.join(_train_path, _name)
            if os.path.isdir(_full) or os.path.islink(_full):
                try:
                    shutil.rmtree(_full)
                except Exception:
                    try:
                        os.remove(_full)
                    except Exception:
                        pass
                continue
            if not os.path.isfile(_full):
                continue
            _idx = _name.split(".")[0]
            if _idx not in _label_ids:
                os.remove(_full)

    reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)

print("Reorg complete:", out_root)



## === cell 6
MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def transform_train(imgpath, label):
    feature = tf.io.read_file(imgpath)
    feature = tf.image.decode_jpeg(feature, channels=3)
    feature = tf.image.resize(feature, size=[400, 400])
    seed = random.randint(8, 100) / 100
    feature = tf.image.random_crop(
        feature, size=[int(seed * feature.shape[0]), int(seed * feature.shape[1]), 3]
    )
    feature = tf.image.resize(feature, size=[224, 224])
    feature = tf.image.random_flip_left_right(feature)
    feature = tf.image.random_flip_up_down(feature)
    feature = tf.divide(feature, 255.0)
    feature = tf.divide(tf.subtract(feature, MEAN), STD)
    return tf.image.convert_image_dtype(feature, tf.float32), label


def transform_test(imgpath, label):
    feature = tf.io.read_file(imgpath)
    feature = tf.image.decode_jpeg(feature, channels=3)
    feature = tf.image.resize(feature, [224, 224])
    feature = tf.divide(feature, 255.0)
    feature = tf.divide(tf.subtract(feature, MEAN), STD)
    return feature, label




## === cell 7
import pathlib

data_root = "/kaggle/working/dog-breed-identification/train_valid_test"
train_data_root = pathlib.Path(data_root + "/train")
valid_data_root = pathlib.Path(data_root + "/valid")
train_valid_data_root = pathlib.Path(data_root + "/train_valid")
test_data_root = pathlib.Path(data_root + "/test")
label_names = sorted(item.name for item in train_data_root.glob("*/") if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))

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
test_all_image_labels = [-1 for i in range(len(test_all_image_paths))]
print("First 10 images indices: ", train_valid_all_image_labels[:10])
print("First 10 labels indices: ", train_valid_all_image_labels[:10])



## === cell 8
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_all_image_paths, train_all_image_labels))
    .map(transform_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    .shuffle(len(train_all_image_paths), seed=SEED, reshuffle_each_iteration=True)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_all_image_paths, valid_all_image_labels))
    .map(transform_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    .shuffle(len(valid_all_image_paths), seed=SEED, reshuffle_each_iteration=True)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

train_valid_ds = (
    tf.data.Dataset.from_tensor_slices(
        (train_valid_all_image_paths, train_valid_all_image_labels)
    )
    .map(transform_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    .shuffle(len(train_valid_all_image_paths), seed=SEED, reshuffle_each_iteration=True)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_all_image_paths, test_all_image_labels))
    .map(transform_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    .shuffle(len(test_all_image_paths), seed=SEED, reshuffle_each_iteration=False)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)



## === cell 9
train_ds



## === cell 10
from tensorflow.keras.applications import ResNet50

net = ResNet50(
    input_shape=(224, 224, 3),
    weights="imagenet",
    include_top=False,
)
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
df.iloc[:, 1:] = probabilities[:, : (df.shape[1] - 1)]
df.to_csv("submission.csv", index=None)



## === cell 17
print("Done. submission.csv written.")
