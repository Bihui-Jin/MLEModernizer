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

# 5. Target score

0.8359651666666666

# 6. Current score

0.69243

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.69243) has done: 'Main runtime killers here are (1) using EfficientNetB3 at 32×32 (very compute-heavy for 50 epochs), (2) inefficient tf.data usage (caching decoded images then re-augmenting every epoch causes a lot of CPU work + large cache), and (3) single-threaded Python directory scans and suboptimal pipeline options. The refactor below keeps the exact same model, loss, epochs, class weights, and overall training semantics, but speeds up input handling by using deterministic tf.data options, caching *encoded JPEG bytes* (small) instead of decoded/resized tensors (large), vectorizing path building, and avoiding expensive repeated directory scans. It also avoids forcing explicit `steps_per_epoch`/`validation_steps` (which can block on input) while keeping identical epoch semantics via dataset cardinality, and ensures prefetching overlaps CPU decode/augment with GPU/CPU training.'

# 9. Code solution

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


def _already_extracted(root_dir: str) -> bool:
    for cand in (
        root_dir,
        os.path.join(root_dir, "train"),
        os.path.join(root_dir, "test"),
    ):
        try:
            with os.scandir(cand) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".jpg"):
                        return True
        except FileNotFoundError:
            pass

    try:
        with os.scandir(root_dir) as it:
            for e in it:
                if e.is_dir():
                    try:
                        with os.scandir(e.path) as it2:
                            for e2 in it2:
                                if e2.is_file() and e2.name.lower().endswith(".jpg"):
                                    return True
                    except FileNotFoundError:
                        pass
        return False
    except FileNotFoundError:
        return False


if not _already_extracted(train_extract_dir):
    with zipfile.ZipFile(train_zip, "r") as zip_ref:
        zip_ref.extractall(train_extract_dir)

if not _already_extracted(test_extract_dir):
    with zipfile.ZipFile(test_zip, "r") as zip_ref:
        zip_ref.extractall(test_extract_dir)

print("Extracted/available train in:", train_extract_dir)
print("Extracted/available test in:", test_extract_dir)



## === cell 2
pass




## === cell 3
def _find_image_dir(root_dir, expected_count_min=1000):
    candidates = [
        root_dir,
        os.path.join(root_dir, "train"),
        os.path.join(root_dir, "test"),
        os.path.join(root_dir, "train", "train"),
        os.path.join(root_dir, "test", "test"),
    ]

    def _count_jpgs(p):
        try:
            n = 0
            with os.scandir(p) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".jpg"):
                        n += 1
            return n
        except Exception:
            return 0

    best = None
    best_count = -1
    for c in candidates:
        cnt = _count_jpgs(c)
        if cnt > best_count:
            best_count = cnt
            best = c

    if best is None or best_count < expected_count_min:
        raise FileNotFoundError(
            f"Could not find image directory under {root_dir}. Best={best} count={best_count}"
        )
    return best, best_count


train_dir, train_count = _find_image_dir(train_extract_dir, expected_count_min=10000)
test_dir, test_count = _find_image_dir(test_extract_dir, expected_count_min=1000)

print("Resolved train_dir:", train_dir, "count:", train_count)
print("Resolved test_dir:", test_dir, "count:", test_count)



## === cell 4
print(
    "Skipping external pip install; using tensorflow.keras EfficientNet implementation."
)



## === cell 5
import tensorflow as tf
from tensorflow import keras

print("TensorFlow:", tf.__version__)
print("Keras (tf.keras):", keras.__version__)

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    pass
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True
try:
    DATASET_OPTIONS.threading.private_threadpool_size = 0
except Exception:
    pass



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
train_df = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
train_df.head(20)



## === cell 7
train_count2 = train_count
test_count2 = test_count
print(f"Train images: {train_count2}")
print(f"Test images: {test_count2}")



## === cell 8
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 9
pass



## === cell 10
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)



## === cell 11
pass



## === cell 12
pass



## === cell 13
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



## === cell 14
train_df["has_cactus"] = train_df["has_cactus"].astype("str")




## === cell 15
@tf.function
def _read_bytes(path):
    return tf.io.read_file(path)


@tf.function
def _decode_resize(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, (32, 32), method="bilinear", antialias=False)
    return img


def _load_and_decode(path):
    return _decode_resize(_read_bytes(path))


@tf.function
def _augment(img, label):
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    img = tf.cond(
        tf.random.uniform(()) > 0.5, lambda: tf.image.flip_left_right(img), lambda: img
    )
    img = tf.cond(
        tf.random.uniform(()) > 0.5, lambda: tf.image.flip_up_down(img), lambda: img
    )
    factor = tf.random.uniform((), minval=0.8, maxval=1.2, dtype=tf.float32)
    img_255 = img * 255.0
    img_255 = tf.clip_by_value(img_255 * factor, 0.0, 255.0)
    img = img_255 / 255.0
    return img, label


@tf.function
def _noaugment(img, label):
    return img, label


def _make_train_val_datasets(df, img_dir, val_split=0.10, seed=SEED):
    df = df.copy()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_val = int(round(n * val_split))
    val_df = df.iloc[:n_val].reset_index(drop=True)
    tr_df = df.iloc[n_val:].reset_index(drop=True)

    tr_y = (tr_df["has_cactus"].values.astype(str) == "1").astype(np.float32)
    va_y = (val_df["has_cactus"].values.astype(str) == "1").astype(np.float32)

    prefix = img_dir.rstrip("/") + "/"
    tr_paths = (prefix + tr_df["id"].values.astype(str)).astype(object)
    va_paths = (prefix + val_df["id"].values.astype(str)).astype(object)

    tr_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_y)).with_options(
        DATASET_OPTIONS
    )
    va_ds = tf.data.Dataset.from_tensor_slices((va_paths, va_y)).with_options(
        DATASET_OPTIONS
    )

    @tf.function
    def _map_read(path, y):
        return _read_bytes(path), tf.cast(y, tf.float32)

    @tf.function
    def _map_decode(img_bytes, y):
        return _decode_resize(img_bytes), y

    tr_ds = tr_ds.shuffle(
        buffer_size=len(tr_df), seed=seed, reshuffle_each_iteration=True
    )
    tr_ds = tr_ds.map(_map_read, num_parallel_calls=AUTOTUNE).cache()
    tr_ds = tr_ds.map(_map_decode, num_parallel_calls=AUTOTUNE)
    tr_ds = tr_ds.map(_augment, num_parallel_calls=AUTOTUNE)
    tr_ds = tr_ds.batch(128, drop_remainder=False).prefetch(AUTOTUNE)

    va_ds = va_ds.map(_map_read, num_parallel_calls=AUTOTUNE).cache()
    va_ds = va_ds.map(_map_decode, num_parallel_calls=AUTOTUNE)
    va_ds = va_ds.map(_noaugment, num_parallel_calls=AUTOTUNE)
    va_ds = va_ds.batch(64, drop_remainder=False).prefetch(AUTOTUNE)

    return tr_ds, va_ds, len(tr_df), len(val_df)


train_ds, val_ds, n_train, n_val = _make_train_val_datasets(
    train_df, train_dir, val_split=0.10, seed=SEED
)
print("Train samples:", n_train, "Val samples:", n_val)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4054168861.py in <cell line: 0>()
     89 
     90 
---> 91 train_ds, val_ds, n_train, n_val = _make_train_val_datasets(
     92     train_df, train_dir, val_split=0.10, seed=SEED
     93 )

/tmp/ipykernel_11/4054168861.py in _make_train_val_datasets(df, img_dir, val_split, seed)
     54     # Speed: vectorized join without np.char.add overhead.
     55     prefix = img_dir.rstrip("/") + "/"
---> 56     tr_paths = (prefix + tr_df["id"].values.astype(str)).astype(object)
     57     va_paths = (prefix + val_df["id"].values.astype(str)).astype(object)
     58 

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U22'), dtype('<U36')) -> None

## === cell 16
pass




## === cell 17
def _make_test_dataset(img_dir):
    files = sorted(
        [
            e.name
            for e in os.scandir(img_dir)
            if e.is_file() and e.name.lower().endswith(".jpg")
        ]
    )
    paths = [os.path.join(img_dir, f) for f in files]
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(DATASET_OPTIONS)

    @tf.function
    def _map_test(path):
        img = _decode_resize(_read_bytes(path))
        return img

    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE).cache()
    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(256, drop_remainder=False).prefetch(AUTOTUNE)
    return ds, files


test_ds, test_files = _make_test_dataset(test_dir)
print("Test samples:", len(test_files))



## === cell 18
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



## === cell 19
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 20
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    class_weight=class_weights_dict,
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2807975786.py in <cell line: 0>()
      2 # and lets Keras/TensorFlow optimize epoch execution. Correctness: same data, same batching, same epochs.
      3 history = model.fit(
----> 4     train_ds,
      5     epochs=50,
      6     validation_data=val_ds,

NameError: name 'train_ds' is not defined

## === cell 21
pass



## === cell 22
preds = model.predict(test_ds, verbose=1)



## === cell 23
image_ids = [os.path.basename(name) for name in test_files]
predictions = preds.reshape(-1)

submission = pd.DataFrame({"id": image_ids, "has_cactus": predictions})

sample_sub = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))
submission = sample_sub[["id"]].merge(submission, on="id", how="left")

submission["has_cactus"] = submission["has_cactus"].astype(np.float32)
submission["has_cactus"] = submission["has_cactus"].fillna(0.5)

print(submission.head(20))
print("Submission shape:", submission.shape)



## === cell 24
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv")



## === cell 25
print(os.listdir("/kaggle/working"))
print("submission.csv exists:", os.path.exists("/kaggle/working/submission.csv"))
