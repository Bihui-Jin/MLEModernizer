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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

print("Input root exists:", os.path.exists("/kaggle/input"))
print("Input top-level entries:", sorted(os.listdir("/kaggle/input"))[:50])



## === cell 1
import zipfile  # zipファイルの解凍に必要
import tensorflow as tf  # 機械学習に必要
from tensorflow.keras import layers, models  # レイヤークラス, 学習モデル
import random
import re
from pathlib import Path

print("Python:", os.sys.version)
print("TensorFlow:", tf.__version__)




## === cell 2
def _resolve_dataset_root():
    candidates = [
        Path("/kaggle/input/dogs-vs-cats-redux-kernels-edition"),
        Path("/kaggle/input")
        / "dogs-vs-cats-redux-kernels-edition"
        / "dogs-vs-cats-redux-kernels-edition",
        Path("/kaggle/data/dogs-vs-cats-redux-kernels-edition"),
        Path("/kaggle/input"),
        Path("/kaggle/data"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("/kaggle/input")


DATASET_ROOT = _resolve_dataset_root()
TRAIN_CAT_DIR = DATASET_ROOT / "train" / "cat"
TRAIN_DOG_DIR = DATASET_ROOT / "train" / "dog"

TEST_DIR_CANDIDATES = [
    DATASET_ROOT / "test" / "unknown",
    DATASET_ROOT / "test" / "test" / "unknown",
    DATASET_ROOT / "test",
]


def _first_existing_dir(paths):
    for p in paths:
        if p.is_dir():
            return p
    return None


DIRECT_TEST_DIR = _first_existing_dir(TEST_DIR_CANDIDATES)

print("Resolved DATASET_ROOT:", str(DATASET_ROOT))
print("TRAIN_CAT_DIR exists:", TRAIN_CAT_DIR.is_dir(), str(TRAIN_CAT_DIR))
print("TRAIN_DOG_DIR exists:", TRAIN_DOG_DIR.is_dir(), str(TRAIN_DOG_DIR))
print(
    "DIRECT_TEST_DIR exists:",
    (DIRECT_TEST_DIR is not None),
    str(DIRECT_TEST_DIR) if DIRECT_TEST_DIR else None,
)

if not (TRAIN_CAT_DIR.is_dir() and TRAIN_DOG_DIR.is_dir()):
    raise FileNotFoundError(
        "Expected extracted train directories /train/cat and /train/dog under dataset root."
    )



## === cell 3
hyper_epochs = 10
hyper_split_rate = 0.8
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception as e:
    print("Threading config skipped:", e)

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.optimizer.set_jit(False)
    print("XLA JIT: disabled for faster end-to-end runtime")
except Exception as e:
    print("XLA JIT disable skipped:", e)



## === cell 4
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)



## === cell 5
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)



## === cell 6
cat_files = []
with os.scandir(TRAIN_CAT_DIR) as it:
    for e in it:
        if e.is_file() and e.name.lower().endswith(".jpg"):
            cat_files.append(str(TRAIN_CAT_DIR / e.name))

dog_files = []
with os.scandir(TRAIN_DOG_DIR) as it:
    for e in it:
        if e.is_file() and e.name.lower().endswith(".jpg"):
            dog_files.append(str(TRAIN_DOG_DIR / e.name))

if len(cat_files) == 0 or len(dog_files) == 0:
    raise ValueError("No training images found under train/cat or train/dog.")

all_paths = cat_files + dog_files
all_labels = [0] * len(cat_files) + [1] * len(dog_files)

idx = list(range(len(all_paths)))
random.shuffle(idx)
split_point = int(len(idx) * hyper_split_rate)
train_idx = idx[:split_point]
val_idx = idx[split_point:]

train_paths = [all_paths[i] for i in train_idx]
train_labels = [all_labels[i] for i in train_idx]
val_paths = [all_paths[i] for i in val_idx]
val_labels = [all_labels[i] for i in val_idx]

df_train = pd.DataFrame(
    {"filename": train_paths, "class": [str(x) for x in train_labels]}
)
df_val = pd.DataFrame({"filename": val_paths, "class": [str(x) for x in val_labels]})

print("Split counts:", len(df_train), len(df_val))
print("Train source dirs:", str(TRAIN_CAT_DIR), str(TRAIN_DOG_DIR))




## === cell 7
def devide_class(any_dir):
    return


devide_class("unused")



## === cell 8
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

train_paths_np = df_train["filename"].to_numpy(dtype=object)
train_labels_np = df_train["class"].astype(np.int32).to_numpy()
val_paths_np = df_val["filename"].to_numpy(dtype=object)
val_labels_np = df_val["class"].astype(np.int32).to_numpy()

if train_paths_np.size == 0 or val_paths_np.size == 0:
    raise ValueError(
        f"Empty dataset: train={train_paths_np.size}, val={val_paths_np.size}"
    )


@tf.function
def _load_and_preprocess(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    label = tf.cast(label, tf.float32)
    return img, label


_SHUFFLE_BUFFER = int(min(8192, train_paths_np.size))

data_opts = tf.data.Options()
data_opts.deterministic = False
data_opts.experimental_optimization.apply_default_optimizations = True

TRAIN_CACHE_PATH = "/kaggle/working/tf_cache_train"
VAL_CACHE_PATH = "/kaggle/working/tf_cache_val"
TEST_CACHE_PATH = "/kaggle/working/tf_cache_test"
import shutil

for _p in (TRAIN_CACHE_PATH, VAL_CACHE_PATH, TEST_CACHE_PATH):
    try:
        if os.path.isdir(_p):
            shutil.rmtree(_p)
        elif os.path.exists(_p):
            os.remove(_p)
    except Exception:
        pass


def _make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(data_opts)
    ds = ds.shuffle(buffer_size=_SHUFFLE_BUFFER, seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.cache()  # in-memory cache (same semantics, much less I/O)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(data_opts)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.cache()  # in-memory cache (same semantics, much less I/O)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds(train_paths_np, train_labels_np)
val_ds = _make_val_ds(val_paths_np, val_labels_np)

print(
    "Train samples:", int(train_paths_np.size), "Val samples:", int(val_paths_np.size)
)
print("Class indices:", {"0": 0, "1": 1})



## === cell 9
history = model.fit(
    train_ds,
    epochs=hyper_epochs,
    validation_data=val_ds,
)



## === cell 10
model_file = "/kaggle/working/cnn.h5"
model.save(model_file)
print("Saved model to:", model_file)



## === cell 11
if DIRECT_TEST_DIR is None:
    raise FileNotFoundError(
        "Could not find extracted test images directory under dataset root (expected test/unknown or test/test/unknown)."
    )
extracted_test_images_dir = str(DIRECT_TEST_DIR)
print("Extracted test images dir:", extracted_test_images_dir)



## === cell 12
test_dir = extracted_test_images_dir

test_files = []
with os.scandir(test_dir) as it:
    for e in it:
        if not e.is_file():
            continue
        n = e.name
        nl = n.lower()
        if not nl.endswith(".jpg"):
            continue
        stem = nl[:-4]
        if stem.isdigit():
            test_files.append(n)

test_ids = np.fromiter(
    (int(os.path.splitext(f)[0]) for f in test_files), dtype=np.int64
)
order = np.argsort(test_ids)
sorted_files = [test_files[i] for i in order]
sorted_ids = test_ids[order].tolist()
image_path_list = [os.path.join(test_dir, f) for f in sorted_files]

if len(image_path_list) == 0:
    raise ValueError(f"No test images found in {test_dir}")

print("Num test images:", len(image_path_list))
print("First/last test file:", sorted_files[0], sorted_files[-1])

PRED_BATCH = 64
test_paths_np = np.array(image_path_list, dtype=object)


@tf.function
def _load_and_preprocess_test(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths_np)
test_ds = test_ds.with_options(data_opts)
test_ds = test_ds.map(
    _load_and_preprocess_test, num_parallel_calls=AUTOTUNE, deterministic=False
)
test_ds = test_ds.cache()  # in-memory cache; same outputs, less I/O
test_ds = test_ds.batch(PRED_BATCH, drop_remainder=False).prefetch(AUTOTUNE)

preds = model.predict(
    test_ds,
    verbose=0,
).reshape(-1)

labels = np.clip(preds.astype(np.float64), 1e-6, 1 - 1e-6)

df = pd.DataFrame({"id": sorted_ids, "label": labels})
df = df.sort_values("id").reset_index(drop=True)

sub_path = "/kaggle/working/submission.csv"
df.to_csv(sub_path, index=False)
print("Wrote submission to:", sub_path)
print(df.head())
print(df.tail())
