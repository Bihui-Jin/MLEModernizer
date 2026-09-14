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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

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
def find_dir(root, target_name):
    root = Path(root)
    for p in root.rglob(target_name):
        if p.is_dir() and p.name == target_name:
            return str(p)
    return None


def find_flat_train_images_dir(root):
    root = Path(root)
    candidates = [
        root / "train",
        root / "dogs-vs-cats-redux-kernels-edition" / "train",
        root
        / "dogs-vs-cats-redux-kernels-edition"
        / "dogs-vs-cats-redux-kernels-edition"
        / "train",
    ]
    for c in candidates:
        if c.is_dir():
            try:
                files = [p.name.lower() for p in c.iterdir() if p.is_file()]
            except Exception:
                continue
            has_cat = any(fn.startswith("cat.") and fn.endswith(".jpg") for fn in files)
            has_dog = any(fn.startswith("dog.") and fn.endswith(".jpg") for fn in files)
            if has_cat and has_dog:
                return str(c)

    for dirpath, _, filenames in os.walk(str(root)):
        filenames_l = [f.lower() for f in filenames]
        if any(
            f.startswith("cat.") and f.endswith(".jpg") for f in filenames_l
        ) and any(f.startswith("dog.") and f.endswith(".jpg") for f in filenames_l):
            return dirpath
    return None


def find_flat_test_images_dir(root):
    root = Path(root)
    candidates = [
        root / "test",
        root / "dogs-vs-cats-redux-kernels-edition" / "test",
        root
        / "dogs-vs-cats-redux-kernels-edition"
        / "dogs-vs-cats-redux-kernels-edition"
        / "test",
    ]
    for c in candidates:
        if c.is_dir():
            try:
                for p in c.iterdir():
                    if (
                        p.is_file()
                        and p.suffix.lower() == ".jpg"
                        and re.fullmatch(r"\d+", p.stem)
                    ):
                        return str(c)
            except Exception:
                pass

    for dirpath, _, filenames in os.walk(str(root)):
        numeric_jpgs = 0
        for f in filenames:
            if f.lower().endswith(".jpg") and re.fullmatch(r"\d+\.jpg", f):
                numeric_jpgs += 1
                if numeric_jpgs >= 20:
                    return dirpath
    return None


def extract_zip_once(zip_file, out_dir, sentinel, expect_find_func):
    if os.path.exists(sentinel):
        return
    if expect_find_func(out_dir) is not None:
        with open(sentinel, "w") as f:
            f.write("ok\n")
        return
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(out_dir)
    with open(sentinel, "w") as f:
        f.write("ok\n")




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
zip_file = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
out_dir = "/kaggle/working"
sentinel = os.path.join(out_dir, ".train_zip_extracted.ok")

extract_zip_once(zip_file, out_dir, sentinel, find_flat_train_images_dir)

extracted_train_images_dir = find_flat_train_images_dir(out_dir)
if extracted_train_images_dir is None:
    raise FileNotFoundError(
        "Could not find extracted train images directory containing cat.*.jpg and dog.*.jpg under /kaggle/working"
    )

print("Extracted train images dir:", extracted_train_images_dir)



## === cell 7
learn_dir = extracted_train_images_dir

learn_file_names = []
with os.scandir(learn_dir) as it:
    for e in it:
        if e.is_file():
            n = e.name
            if n.lower().endswith(".jpg"):
                learn_file_names.append(n)

if len(learn_file_names) == 0:
    raise ValueError(f"No training images found in {learn_dir}")

random.shuffle(learn_file_names)
split_point = int(len(learn_file_names) * hyper_split_rate)
train_file_names = learn_file_names[:split_point]
val_file_names = learn_file_names[split_point:]


def _label_from_fname(fname: str):
    f = fname.lower()
    if f.startswith("cat"):
        return 0
    if f.startswith("dog"):
        return 1
    return None


train_paths, train_labels = [], []
for fn in train_file_names:
    y = _label_from_fname(fn)
    if y is None:
        continue
    train_paths.append(os.path.join(learn_dir, fn))
    train_labels.append(y)

val_paths, val_labels = [], []
for fn in val_file_names:
    y = _label_from_fname(fn)
    if y is None:
        continue
    val_paths.append(os.path.join(learn_dir, fn))
    val_labels.append(y)

df_train = pd.DataFrame(
    {"filename": train_paths, "class": [str(x) for x in train_labels]}
)
df_val = pd.DataFrame({"filename": val_paths, "class": [str(x) for x in val_labels]})

print("Split counts:", len(df_train), len(df_val))
print("Train source dir:", learn_dir)




## === cell 8
def devide_class(any_dir):
    return


devide_class("unused")



## === cell 9
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


def _load_and_preprocess(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    label = tf.cast(label, tf.float32)
    return img, label


def _make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds(train_paths_np, train_labels_np)
val_ds = _make_val_ds(val_paths_np, val_labels_np)

print(
    "Train samples:", int(train_paths_np.size), "Val samples:", int(val_paths_np.size)
)
print("Class indices:", {"0": 0, "1": 1})



## === cell 10
history = model.fit(
    train_ds,
    epochs=hyper_epochs,
    validation_data=val_ds,
)



## === cell 11
model_file = "/kaggle/working/cnn.h5"
model.save(model_file)
print("Saved model to:", model_file)



## === cell 12
zip_file = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"
out_dir = "/kaggle/working"
sentinel = os.path.join(out_dir, ".test_zip_extracted.ok")

extract_zip_once(zip_file, out_dir, sentinel, find_flat_test_images_dir)

extracted_test_images_dir = find_flat_test_images_dir(out_dir)
if extracted_test_images_dir is None:
    raise FileNotFoundError(
        "Could not find extracted test images directory containing numeric *.jpg under /kaggle/working"
    )

print("Extracted test images dir:", extracted_test_images_dir)



## === cell 13
test_dir = extracted_test_images_dir

test_files = []
with os.scandir(test_dir) as it:
    for e in it:
        if not e.is_file():
            continue
        n = e.name
        nl = n.lower()
        if nl.endswith(".jpg") and re.fullmatch(r"\d+\.jpg", nl):
            test_files.append(n)

test_ids = np.array([int(os.path.splitext(f)[0]) for f in test_files], dtype=np.int64)
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


def _load_and_preprocess_test(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths_np)
test_ds = test_ds.map(_load_and_preprocess_test, num_parallel_calls=AUTOTUNE)
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
