# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os
import random
import zipfile

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_INPUT = "/kaggle/input/aerial-cactus-identification"
BASE_WORK = "/kaggle/working"



## === cell 1
train_zip = os.path.join(BASE_INPUT, "train.zip")
test_zip = os.path.join(BASE_INPUT, "test.zip")

train_extract_dir = os.path.join(BASE_WORK, "train")
test_extract_dir = os.path.join(BASE_WORK, "test")

os.makedirs(train_extract_dir, exist_ok=True)
os.makedirs(test_extract_dir, exist_ok=True)


def _has_any_jpg(dir_path: str) -> bool:
    try:
        with os.scandir(dir_path) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(".jpg"):
                    return True
    except FileNotFoundError:
        return False
    return False


def _already_extracted_fast(root_dir: str) -> bool:
    for cand in (
        root_dir,
        os.path.join(root_dir, "train"),
        os.path.join(root_dir, "test"),
        os.path.join(root_dir, "train", "train"),
        os.path.join(root_dir, "test", "test"),
    ):
        if _has_any_jpg(cand):
            return True
    return False


input_train_dir = os.path.join(BASE_INPUT, "train")
input_test_dir = os.path.join(BASE_INPUT, "test")

if _already_extracted_fast(input_train_dir):
    train_extract_dir = input_train_dir
else:
    if not _already_extracted_fast(train_extract_dir):
        with zipfile.ZipFile(train_zip, "r") as zip_ref:
            zip_ref.extractall(train_extract_dir)

if _already_extracted_fast(input_test_dir):
    test_extract_dir = input_test_dir
else:
    if not _already_extracted_fast(test_extract_dir):
        with zipfile.ZipFile(test_zip, "r") as zip_ref:
            zip_ref.extractall(test_extract_dir)

print("Extracted/available train in:", train_extract_dir)
print("Extracted/available test in:", test_extract_dir)




## === cell 2
def _find_image_dir(root_dir, expected_count_min=1000):
    candidates = [
        root_dir,
        os.path.join(root_dir, "train"),
        os.path.join(root_dir, "test"),
        os.path.join(root_dir, "train", "train"),
        os.path.join(root_dir, "test", "test"),
    ]

    def _count_jpgs(p, stop_at):
        try:
            n = 0
            with os.scandir(p) as it:
                for e in it:
                    name = e.name
                    if name and name[-4:].lower() == ".jpg" and e.is_file():
                        n += 1
                        if n >= stop_at:
                            return n
            return n
        except FileNotFoundError:
            return 0

    best = None
    best_count = -1
    for c in candidates:
        cnt = _count_jpgs(c, expected_count_min)
        if cnt > best_count:
            best_count = cnt
            best = c
        if cnt >= expected_count_min:
            pass

    if best is None or best_count < expected_count_min:
        raise FileNotFoundError(
            f"Could not find image directory under {root_dir}. Best={best} count={best_count}"
        )
    return best, best_count


train_dir, train_count = _find_image_dir(train_extract_dir, expected_count_min=10000)
test_dir, test_count = _find_image_dir(test_extract_dir, expected_count_min=1000)

print("Resolved train_dir:", train_dir, "count:", train_count)
print("Resolved test_dir:", test_dir, "count:", test_count)



## === cell 3
print(
    "Skipping external pip install; using tensorflow.keras EfficientNet implementation."
)



## === cell 4
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf
from tensorflow import keras

print("TensorFlow:", tf.__version__)

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True
try:
    DATASET_OPTIONS.threading.private_threadpool_size = 0
except Exception:
    pass
try:
    DATASET_OPTIONS.experimental_optimization.map_parallelization = True
    DATASET_OPTIONS.experimental_optimization.parallel_batch = True
    DATASET_OPTIONS.experimental_optimization.autotune_buffers = True
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 5
train_df = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
train_df.head(20)



## === cell 6
train_count2 = train_count
test_count2 = test_count
print(f"Train images: {train_count2}")
print(f"Test images: {test_count2}")



## === cell 7
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 8
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)



## === cell 9
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



## === cell 10
train_df["has_cactus"] = train_df["has_cactus"].astype("str")




## === cell 11
@tf.function
def _decode_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, (32, 32), method="bilinear", antialias=False)
    return img


@tf.function
def _augment_indexed(img, label, idx):
    k = tf.random.stateless_uniform(
        (),
        seed=tf.stack([SEED, tf.cast(idx, tf.int32)]),
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    img = tf.image.rot90(img, k=k)

    seed2 = tf.stack([SEED + 1, tf.cast(idx, tf.int32)])
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)
    seed3 = tf.stack([SEED + 2, tf.cast(idx, tf.int32)])
    img = tf.image.stateless_random_flip_up_down(img, seed=seed3)

    seed4 = tf.stack([SEED + 3, tf.cast(idx, tf.int32)])
    factor = tf.random.stateless_uniform(
        (), seed=seed4, minval=0.8, maxval=1.2, dtype=tf.float32
    )
    img_255 = img * 255.0
    img_255 = tf.clip_by_value(img_255 * factor, 0.0, 255.0)
    img = img_255 / 255.0
    return img, label


def _make_train_val_datasets(df, img_dir, val_split=0.10, seed=SEED):
    df = df.copy()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_val = int(round(n * val_split))
    val_df = df.iloc[:n_val].reset_index(drop=True)
    tr_df = df.iloc[n_val:].reset_index(drop=True)

    tr_y_np = (tr_df["has_cactus"].values.astype(str) == "1").astype(np.float32)
    va_y_np = (val_df["has_cactus"].values.astype(str) == "1").astype(np.float32)

    base = img_dir.rstrip("/") + "/"
    tr_paths_np = (base + tr_df["id"].astype(str)).to_numpy(dtype=object)
    va_paths_np = (base + val_df["id"].astype(str)).to_numpy(dtype=object)

    tr_paths = tf.constant(tr_paths_np)
    va_paths = tf.constant(va_paths_np)
    tr_y = tf.constant(tr_y_np, dtype=tf.float32)
    va_y = tf.constant(va_y_np, dtype=tf.float32)

    tr_img_ds = tf.data.Dataset.from_tensor_slices(tr_paths).with_options(
        DATASET_OPTIONS
    )
    va_img_ds = tf.data.Dataset.from_tensor_slices(va_paths).with_options(
        DATASET_OPTIONS
    )

    tr_img_ds = tr_img_ds.map(_decode_resize_from_path, num_parallel_calls=AUTOTUNE)
    va_img_ds = va_img_ds.map(_decode_resize_from_path, num_parallel_calls=AUTOTUNE)

    tr_imgs = np.stack(list(tr_img_ds.as_numpy_iterator()), axis=0)
    va_imgs = np.stack(list(va_img_ds.as_numpy_iterator()), axis=0)

    tr_imgs_tf = tf.constant(tr_imgs, dtype=tf.float32)
    va_imgs_tf = tf.constant(va_imgs, dtype=tf.float32)

    tr_ds = tf.data.Dataset.from_tensor_slices((tr_imgs_tf, tr_y)).with_options(
        DATASET_OPTIONS
    )
    va_ds = tf.data.Dataset.from_tensor_slices((va_imgs_tf, va_y)).with_options(
        DATASET_OPTIONS
    )

    shuffle_buf = int(min(len(tr_df), 4096))
    tr_ds = tr_ds.shuffle(
        buffer_size=shuffle_buf, seed=seed, reshuffle_each_iteration=True
    )

    tr_ds = tr_ds.enumerate()
    tr_ds = tr_ds.map(
        lambda idx, x: _augment_indexed(x[0], x[1], idx), num_parallel_calls=AUTOTUNE
    )

    tr_ds = tr_ds.batch(128, drop_remainder=True)
    va_ds = va_ds.batch(256, drop_remainder=False)

    if tf.config.list_physical_devices("GPU"):
        try:
            tr_ds = tr_ds.apply(
                tf.data.experimental.prefetch_to_device(
                    "/device:GPU:0", buffer_size=AUTOTUNE
                )
            )
            va_ds = va_ds.apply(
                tf.data.experimental.prefetch_to_device(
                    "/device:GPU:0", buffer_size=AUTOTUNE
                )
            )
        except Exception:
            tr_ds = tr_ds.prefetch(AUTOTUNE)
            va_ds = va_ds.prefetch(AUTOTUNE)
    else:
        tr_ds = tr_ds.prefetch(AUTOTUNE)
        va_ds = va_ds.prefetch(AUTOTUNE)

    return tr_ds, va_ds, len(tr_df), len(val_df)


train_ds, val_ds, n_train, n_val = _make_train_val_datasets(
    train_df, train_dir, val_split=0.10, seed=SEED
)
print("Train samples:", n_train, "Val samples:", n_val)




## === cell 12
def _make_test_dataset(img_dir):
    pattern = os.path.join(img_dir, "*.jpg")
    paths = tf.io.gfile.glob(pattern)
    paths = sorted(paths, key=lambda p: os.path.basename(p))
    files = [os.path.basename(p) for p in paths]

    paths_tf = tf.constant(np.asarray(paths, dtype=object))
    img_ds = tf.data.Dataset.from_tensor_slices(paths_tf).with_options(DATASET_OPTIONS)
    img_ds = img_ds.map(_decode_resize_from_path, num_parallel_calls=AUTOTUNE)

    test_imgs = np.stack(list(img_ds.as_numpy_iterator()), axis=0)
    test_imgs_tf = tf.constant(test_imgs, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(test_imgs_tf).with_options(DATASET_OPTIONS)
    ds = ds.batch(512, drop_remainder=False)

    if tf.config.list_physical_devices("GPU"):
        try:
            ds = ds.apply(
                tf.data.experimental.prefetch_to_device(
                    "/device:GPU:0", buffer_size=AUTOTUNE
                )
            )
        except Exception:
            ds = ds.prefetch(AUTOTUNE)
    else:
        ds = ds.prefetch(AUTOTUNE)

    return ds, files


test_ds, test_files = _make_test_dataset(test_dir)
print("Test samples:", len(test_files))



## === cell 13
efficient_net = EfficientNetB3(
    weights="imagenet", input_shape=(32, 32, 3), include_top=False, pooling="max"
)

efficient_net.trainable = False

model = Sequential()
model.add(efficient_net)
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.summary()



## === cell 14
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)



## === cell 15
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    class_weight=class_weights_dict,
)



## === cell 16
preds = model.predict(test_ds, verbose=1)



## === cell 17
image_ids = test_files
predictions = preds.reshape(-1)

submission = pd.DataFrame({"id": image_ids, "has_cactus": predictions})

sample_sub = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))
submission = sample_sub[["id"]].merge(submission, on="id", how="left")

submission["has_cactus"] = submission["has_cactus"].astype(np.float32)
submission["has_cactus"] = submission["has_cactus"].fillna(0.5)

print(submission.head(20))
print("Submission shape:", submission.shape)



## === cell 18
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv")



## === cell 19
print(os.listdir("/kaggle/working"))
print("submission.csv exists:", os.path.exists("/kaggle/working/submission.csv"))
