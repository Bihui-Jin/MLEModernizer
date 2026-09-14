# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.9

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split



## === cell 1
import zipfile

INPUT_ROOT = "../input/dogs-vs-cats-redux-kernels-edition"
WORK_ROOT = "./dogs-vs-cats-data"

os.makedirs(WORK_ROOT, exist_ok=True)

train_zip = os.path.join(INPUT_ROOT, "train.zip")
test_zip = os.path.join(INPUT_ROOT, "test.zip")

train_extract_root = os.path.join(WORK_ROOT, "train")
test_extract_root = os.path.join(WORK_ROOT, "test")


def _has_jpgs_somewhere(root_dir: str) -> bool:
    if not os.path.isdir(root_dir):
        return False
    for cand in (
        root_dir,
        os.path.join(root_dir, "train"),
        os.path.join(root_dir, "test"),
        os.path.join(root_dir, "unknown"),
    ):
        if os.path.isdir(cand):
            try:
                with os.scandir(cand) as it:
                    for e in it:
                        if e.is_file() and e.name.lower().endswith(".jpg"):
                            return True
            except FileNotFoundError:
                pass
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(fn.lower().endswith(".jpg") for fn in filenames):
            return True
        rel = os.path.relpath(dirpath, root_dir)
        if rel != "." and rel.count(os.sep) >= 4:
            dirnames[:] = []
    return False


if (not _has_jpgs_somewhere(train_extract_root)) and os.path.isfile(train_zip):
    with zipfile.ZipFile(train_zip, "r") as z:
        z.extractall(WORK_ROOT)

if (not _has_jpgs_somewhere(test_extract_root)) and os.path.isfile(test_zip):
    with zipfile.ZipFile(test_zip, "r") as z:
        z.extractall(WORK_ROOT)


def _find_image_dir(root_dir: str) -> str:
    """
    Robustly find a directory containing .jpg files, supporting common nested layouts after extraction:
      - WORK_ROOT/train/train/*.jpg
      - WORK_ROOT/test/test/*.jpg
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root dir does not exist: {root_dir}")

    for sub in ("", "train", "test", "unknown"):
        cand = os.path.join(root_dir, sub) if sub else root_dir
        if os.path.isdir(cand):
            try:
                with os.scandir(cand) as it:
                    for e in it:
                        if e.is_file() and e.name.lower().endswith(".jpg"):
                            return cand
            except FileNotFoundError:
                pass

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(fn.lower().endswith(".jpg") for fn in filenames):
            return dirpath
        rel = os.path.relpath(dirpath, root_dir)
        if rel != "." and rel.count(os.sep) >= 4:
            dirnames[:] = []

    raise FileNotFoundError(f"Could not find any .jpg files under: {root_dir}")


def _list_jpgs(dirpath: str):
    with os.scandir(dirpath) as it:
        return [e.name for e in it if e.is_file() and e.name.lower().endswith(".jpg")]


_train_root = (
    train_extract_root
    if _has_jpgs_somewhere(train_extract_root)
    else os.path.join(INPUT_ROOT, "train")
)
_test_root = (
    test_extract_root
    if _has_jpgs_somewhere(test_extract_root)
    else os.path.join(INPUT_ROOT, "test")
)

TRAIN_DIR = _find_image_dir(_train_root)
TEST_DIR = _find_image_dir(_test_root)

_train_jpgs = _list_jpgs(TRAIN_DIR)
_test_jpgs = _list_jpgs(TEST_DIR)

print("TRAIN_DIR:", TRAIN_DIR, "num_files:", len(_train_jpgs))
print("TEST_DIR:", TEST_DIR, "num_files:", len(_test_jpgs))



## === cell 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3
BATCH_SIZE = 256



## === cell 3
filenames = _train_jpgs
categories = (
    (pd.Series(filenames).str.split(".").str[0].eq("dog")).astype(np.int32).tolist()
)



## === cell 4
all_data = pd.DataFrame(
    {
        "filename": filenames,
        "category": pd.Series(categories, dtype="int32").astype(str),
    }
)

print("Unique classes:", sorted(all_data["category"].unique().tolist()))
all_data.head(), all_data.shape



## === cell 5
index = min(357, len(all_data) - 1)
sample_img_filename, sample_img_label = all_data.iloc[index, :]
sample_img_label_int = int(sample_img_label)
print(
    "Sample file:",
    sample_img_filename,
    "Label: {}({})".format(["Cat", "Dog"][sample_img_label_int], sample_img_label_int),
)



## === cell 6
train_data, validation_data = train_test_split(
    all_data,
    test_size=0.05,
    shuffle=True,
    random_state=2,
    stratify=all_data["category"],
)
train_data = train_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)

train_data.shape, validation_data.shape



## === cell 7
num_train = train_data.shape[0]
num_val = validation_data.shape[0]
num_train, num_val



## === cell 8
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Activation,
)

tf.random.set_seed(2)
np.random.seed(2)
try:
    tf.keras.utils.set_random_seed(2)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    _CPU_COUNT = os.cpu_count() or 2
except Exception:
    _CPU_COUNT = 2

GEN_WORKERS = max(2, min(8, _CPU_COUNT))
GEN_MAX_QUEUE = 32



## === cell 9
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory=TRAIN_DIR,
    x_col="filename",
    y_col="category",
    classes=["0", "1"],
    class_mode="binary",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=2,
)



## === cell 10
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_dataframe(
    validation_data,
    directory=TRAIN_DIR,
    x_col="filename",
    y_col="category",
    classes=["0", "1"],
    class_mode="binary",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=2,
)



## === cell 11
example_generator = train_datagen.flow_from_dataframe(
    train_data.sample(n=1, random_state=0),
    directory=TRAIN_DIR,
    x_col="filename",
    y_col="category",
    classes=["0", "1"],
    class_mode="binary",
    target_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=False,
    seed=2,
)
example_x, example_y = next(example_generator)

label = int(np.ravel(example_y)[0])
print(
    "Augmentation pipeline check - Label: {}({})".format(["Cat", "Dog"][label], label)
)



## === cell 12
pretrained_base = VGG19(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
    pooling=None,
)

for layer in pretrained_base.layers[:5]:
    layer.trainable = True
for layer in pretrained_base.layers[5:]:
    layer.trainable = False

pretrained_base.summary()



## === cell 13
model = Sequential(
    [
        pretrained_base,
        Flatten(),
        Dropout(0.2),
        Dense(512),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Dense(128),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Dense(32),
        BatchNormalization(),
        Activation("relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.summary()



## === cell 14
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 15
steps_per_epoch = max(1, num_train // BATCH_SIZE)
validation_steps = max(1, num_val // BATCH_SIZE)

history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_steps,
)



## === cell 16
test_filenames_all = _test_jpgs


def _try_parse_id(fn: str):
    stem = os.path.splitext(fn)[0]
    if stem.isdigit():
        return int(stem)
    return None


pairs = [(fn, _try_parse_id(fn)) for fn in test_filenames_all]
pairs = [(fn, i) for fn, i in pairs if i is not None]

if len(pairs) == 0:
    raise RuntimeError(
        f"No numeric test images found in TEST_DIR={TEST_DIR}. "
        f"First few files: {test_filenames_all[:10]}"
    )

pairs.sort(key=lambda t: t[1])
test_filenames = [fn for fn, _ in pairs]
test_ids = np.array([i for _, i in pairs], dtype=int)

test_data = pd.DataFrame({"filename": test_filenames})
print(
    "Prepared test_data:",
    test_data.shape,
    "id range:",
    (int(test_ids.min()), int(test_ids.max())),
)
test_data.head()



## === cell 17
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

PRED_BATCH_SIZE = 256

test_generator = test_datagen.flow_from_dataframe(
    test_data,
    directory=TEST_DIR,
    x_col="filename",
    y_col=None,
    target_size=IMAGE_SIZE,
    class_mode=None,
    shuffle=False,
    batch_size=PRED_BATCH_SIZE,
)



## === cell 18
predictions = model.predict(
    x=test_generator,
    verbose=1,
)
predictions = np.squeeze(predictions)
predictions.shape



## === cell 19
ids = test_ids
ids.shape, ids[:5]



## === cell 20
submission = pd.DataFrame({"id": ids, "label": predictions.astype(float)})
submission = submission.sort_values("id").reset_index(drop=True)
submission["label"] = submission["label"].clip(1e-7, 1 - 1e-7)

submission.head()



## === cell 21
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.describe(include="all"))
