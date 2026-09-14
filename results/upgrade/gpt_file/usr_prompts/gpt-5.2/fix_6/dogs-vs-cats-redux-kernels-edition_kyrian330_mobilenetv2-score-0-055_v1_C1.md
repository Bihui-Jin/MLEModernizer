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
import os, glob, zipfile, random

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers, callbacks

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("✅ Imports OK")
print("TF version:", tf.__version__)




## === cell 1
train_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"

extract_dir = "/kaggle/working/"

train_marker = os.path.join(extract_dir, ".train_unzipped")
test_marker = os.path.join(extract_dir, ".test_unzipped")


def _has_any_jpg_fast(root_candidates):
    for root in root_candidates:
        if os.path.isdir(root):
            hit = next(glob.iglob(os.path.join(root, "*.jpg")), None)
            if hit is not None:
                return True
            hit = next(glob.iglob(os.path.join(root, "*", "*.jpg")), None)
            if hit is not None:
                return True
    return False


train_extracted = _has_any_jpg_fast(
    [
        os.path.join(extract_dir, "train"),
        os.path.join(extract_dir, "dogs-vs-cats-redux-kernels-edition", "train"),
    ]
)
test_extracted = _has_any_jpg_fast(
    [
        os.path.join(extract_dir, "test"),
        os.path.join(extract_dir, "dogs-vs-cats-redux-kernels-edition", "test"),
    ]
)

if not os.path.exists(train_marker) and not train_extracted:
    with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    open(train_marker, "w").close()
elif not os.path.exists(train_marker) and train_extracted:
    open(train_marker, "w").close()

if not os.path.exists(test_marker) and not test_extracted:
    with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    open(test_marker, "w").close()
elif not os.path.exists(test_marker) and test_extracted:
    open(test_marker, "w").close()

print("✅ Unzip OK")




## === cell 2
IMG_SIZE = 256
BATCH_SIZE = 32

TRAIN_GLOB_CANDIDATES = [
    "/kaggle/working/train/*.jpg",
    "/kaggle/working/train/train/*.jpg",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/*.jpg",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/train/*.jpg",
    "/kaggle/working/train/*/*.jpg",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/*/*.jpg",
]
TEST_GLOB_CANDIDATES = [
    "/kaggle/working/test/*.jpg",
    "/kaggle/working/test/test/*.jpg",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test/*.jpg",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test/test/*.jpg",
    "/kaggle/working/test/*/*.jpg",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test/*/*.jpg",
]


def first_nonempty_glob(patterns):
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            return pat, files
    return None, []


train_pat, all_images = first_nonempty_glob(TRAIN_GLOB_CANDIDATES)
test_pat, _test_images = first_nonempty_glob(TEST_GLOB_CANDIDATES)

print(f"Resolved train pattern: {train_pat} ({len(all_images)} files)")
print(f"Resolved test pattern:  {test_pat} ({len(_test_images)} files)")

if len(all_images) == 0:
    raise FileNotFoundError(
        "No training images found after unzip. Checked patterns:\n"
        + "\n".join(TRAIN_GLOB_CANDIDATES)
    )
if len(_test_images) == 0:
    raise FileNotFoundError(
        "No test images found after unzip. Checked patterns:\n"
        + "\n".join(TEST_GLOB_CANDIDATES)
    )

print("✅ Paths OK")




## === cell 3
def infer_labels_vectorized(paths):
    paths = np.asarray(paths, dtype=object)
    bases = np.char.lower(np.array([os.path.basename(p) for p in paths], dtype=str))
    parents = np.char.lower(
        np.array([os.path.basename(os.path.dirname(p)) for p in paths], dtype=str)
    )
    is_dog = np.char.find(bases, "dog") >= 0
    is_dog |= parents == "dog"
    is_cat = np.char.find(bases, "cat") >= 0
    is_cat |= parents == "cat"
    labels = np.where(is_dog, 1, np.where(is_cat, 0, 0)).astype(np.int32)
    return labels.tolist()


labels = infer_labels_vectorized(all_images)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_images, labels, test_size=0.15, stratify=labels, random_state=SEED
)


def decode_img_with_label(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, tf.cast(label, tf.float32)


def decode_img_only(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img


def build_dataset(paths, labels, is_train=True, cache_path=None):
    options = tf.data.Options()
    options.experimental_deterministic = True  # keep deterministic iteration order
    options.threading.private_threadpool_size = 8

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(options)
    ds = ds.map(decode_img_with_label, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_path is None:
        ds = ds.cache()
    else:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ds = ds.cache(cache_path)

    if is_train:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_cache = "/kaggle/working/tf_cache/train.cache"
val_cache = "/kaggle/working/tf_cache/val.cache"
train_ds = build_dataset(
    train_paths, train_labels, is_train=True, cache_path=train_cache
)
val_ds = build_dataset(val_paths, val_labels, is_train=False, cache_path=val_cache)

print("✅ Datasets OK")
print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:  ", tf.data.experimental.cardinality(val_ds).numpy())




## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
)

base_model.trainable = False
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="MobileNetV2")(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(2e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()
print("✅ Model OK")




## === cell 5
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=1, restore_best_weights=True
)

history = model.fit(
    train_ds, validation_data=val_ds, epochs=40, callbacks=[early_stop], verbose=2
)

print("✅ Train OK")




## === cell 6
model.save("/kaggle/working/MobileNetV2.keras")
print("✅ Saved model")




## === cell 7
print("✅ Skipped plots to save time")




## === cell 8
def test_id_from_path(p):
    return int(os.path.splitext(os.path.basename(p))[0])


test_basenames = np.array([os.path.basename(p) for p in _test_images], dtype=str)
test_ids = np.char.partition(test_basenames, ".")[:, 0].astype(np.int64)
order = np.argsort(test_ids)
test_paths = [_test_images[i] for i in order]
test_ids_sorted = test_ids[order].tolist()


def build_test_ds(paths, cache_path=None):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 8

    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
    ds = ds.map(decode_img_only, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_path is None:
        ds = ds.cache()
    else:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ds = ds.cache(cache_path)
    return ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


test_cache = "/kaggle/working/tf_cache/test.cache"
test_ds = build_test_ds(test_paths, cache_path=test_cache)

preds = model.predict(test_ds, verbose=0).ravel()
preds = preds.clip(min=0.005, max=0.995)

print("✅ Predict OK")




## === cell 9
print(f"预测最大值：{preds.max():.4f}")
print(f"预测最小值：{preds.min():.4f}")
print(f"预测均值：{preds.mean():.4f}")
print("✅ Pred stats OK")




## === cell 10
submission = pd.DataFrame(
    {
        "id": test_ids_sorted,
        "label": preds.astype(np.float64),
    }
)

submission = submission.sort_values("id").reset_index(drop=True)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(f"✅ Wrote submission: {out_path}  shape={submission.shape}")
print(submission.head())




## === cell 11
import shutil


def safe_rmtree(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"Removed: {path}")
        except Exception as e:
            print(f"Skip removing {path}, reason: {e}")
    else:
        print(f"Not found, skip: {path}")


safe_rmtree("/kaggle/working/train")
safe_rmtree("/kaggle/working/test")

print("✅ Cleanup OK")
