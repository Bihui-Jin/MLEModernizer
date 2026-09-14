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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import zipfile

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)
try:
    import tensorflow as tf

    tf.random.set_seed(42)
except Exception:
    pass




## === cell 1
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 64
NO_EPOCHS = 20
NUM_CLASSES = 2
SAMPLE_SIZE = 20000
IMG_SIZE = 128

TRAIN_FOLDER = "/kaggle/working/train"
TEST_FOLDER = "/kaggle/working/test"
PATH_TRAIN = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
PATH_TEST = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"




## === cell 2
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)


def _has_jpgs_fast(root_dir: str) -> bool:
    for dirpath, _, _ in os.walk(root_dir):
        try:
            with os.scandir(dirpath) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".jpg"):
                        return True
        except FileNotFoundError:
            continue
    return False


if not _has_jpgs_fast(TRAIN_FOLDER):
    with zipfile.ZipFile(PATH_TRAIN, "r") as zip_ref:
        zip_ref.extractall(TRAIN_FOLDER)

if not _has_jpgs_fast(TEST_FOLDER):
    with zipfile.ZipFile(PATH_TEST, "r") as zip_ref:
        zip_ref.extractall(TEST_FOLDER)


def find_images_dir(root_dir: str, kind: str) -> str:
    expected = os.path.join(root_dir, kind)
    if os.path.isdir(expected):
        expected2 = os.path.join(expected, kind)
        for cand in (expected2, expected):
            if os.path.isdir(cand):
                try:
                    with os.scandir(cand) as it:
                        for e in it:
                            if e.is_file() and e.name.lower().endswith(".jpg"):
                                return cand
                except FileNotFoundError:
                    pass

    candidates = []
    for dirpath, _, _ in os.walk(root_dir):
        cnt = 0
        try:
            with os.scandir(dirpath) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".jpg"):
                        cnt += 1
        except FileNotFoundError:
            continue
        if cnt:
            candidates.append((dirpath, cnt))
    if not candidates:
        raise FileNotFoundError(f"No .jpg files found under: {root_dir}")

    candidates.sort(key=lambda x: (kind not in x[0].lower(), -x[1], len(x[0])))
    return candidates[0][0]


train_inner_folder = find_images_dir(TRAIN_FOLDER, "train")
test_inner_folder = find_images_dir(TEST_FOLDER, "test")

with os.scandir(train_inner_folder) as it:
    _train_all = [e.name for e in it if e.is_file() and e.name.lower().endswith(".jpg")]
with os.scandir(test_inner_folder) as it:
    _test_all = [e.name for e in it if e.is_file() and e.name.lower().endswith(".jpg")]

train_image_list = _train_all[:SAMPLE_SIZE]
test_image_list = _test_all

print("train_inner_folder:", train_inner_folder)
print("test_inner_folder :", test_inner_folder)
print("Found train images:", len(_train_all))
print("Using SAMPLE_SIZE:", len(train_image_list))
print("Found test images:", len(test_image_list))




## === cell 3
def label_pet_image_one_hot_encoder(img_filename):
    pet = img_filename.split(".")[0]
    if pet == "cat":
        return [1, 0]
    elif pet == "dog":
        return [0, 1]
    return [1, 0]




## === cell 4
from concurrent.futures import ThreadPoolExecutor


def _read_resize_one(path: str) -> np.ndarray | None:
    img_array = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_array is None:
        return None
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(
        img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
    )
    return img_array


def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    """
    Kept for correctness/compatibility with existing later cells (e.g., visualization).
    NOTE: Training/inference uses tf.data below; this function is only used for small visualization.
    """
    n = len(data_image_list)

    if isTrain:
        X = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        y = np.empty((n, 2), dtype=np.int64)
    else:
        X = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        y = np.empty((n,), dtype=object)

    paths = [os.path.join(DATA_FOLDER, img) for img in data_image_list]

    max_workers = min(32, (os.cpu_count() or 4) * 2)
    valid = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for img, img_array in zip(data_image_list, ex.map(_read_resize_one, paths)):
            if img_array is None:
                continue
            X[valid] = img_array
            if isTrain:
                y[valid] = np.array(
                    label_pet_image_one_hot_encoder(img), dtype=np.int64
                )
            else:
                y[valid] = img
            valid += 1

    if valid != n:
        X = X[:valid]
        y = y[:valid]

    rng = np.random.RandomState(RANDOM_STATE)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X, y




## === cell 5
def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split(".")[0])
    plt.figure(figsize=(6, 4))
    sns.countplot(x=labels)
    plt.title("Cats and Dogs")
    plt.show()




## === cell 6
train_X_u8, train_y = process_data(train_image_list[:128], train_inner_folder, True)




## === cell 7
def show_images(data, isTest=False):
    f, ax = plt.subplots(5, 5, figsize=(12, 12))
    for i in range(25):
        img_data = data[0][i] if isinstance(data, tuple) else data[i][0]
        img_num = data[1][i] if isinstance(data, tuple) else data[i][1]
        if isTest:
            str_label = "None"
        else:
            label = int(np.argmax(img_num))
            str_label = "Dog" if label == 1 else "Cat"
        ax[i // 5, i % 5].imshow(img_data.astype(np.uint8))
        ax[i // 5, i % 5].axis("off")
        ax[i // 5, i % 5].set_title("Label: {}".format(str_label))
    plt.tight_layout()
    plt.show()




## === cell 8
test_filenames_arr = np.array([str(f) for f in test_image_list], dtype=object)




## === cell 9
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def _tf_label_from_path(path: tf.Tensor) -> tf.Tensor:
    fname = tf.strings.split(path, os.sep)[-1]
    pet = tf.strings.split(fname, ".")[0]
    is_cat = tf.equal(pet, "cat")
    return tf.cast(tf.stack([is_cat, tf.logical_not(is_cat)]), tf.int32)


@tf.function
def _decode_resize_preprocess(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


def make_trainval_datasets(train_files, val_files, batch_size: int):
    def _map_train(path):
        img = _decode_resize_preprocess(path)
        y = _tf_label_from_path(path)
        return img, y

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True

    shuffle_buf = int(min(len(train_files), 4096))

    train_ds = tf.data.Dataset.from_tensor_slices(train_files).with_options(opts)
    train_ds = train_ds.shuffle(
        buffer_size=shuffle_buf,
        seed=RANDOM_STATE,
        reshuffle_each_iteration=True,
    )
    train_ds = train_ds.map(_map_train, num_parallel_calls=AUTOTUNE)

    cache_root = "/kaggle/working/tfdata_cache"
    os.makedirs(cache_root, exist_ok=True)
    train_cache = os.path.join(cache_root, f"train_img{IMG_SIZE}_n{len(train_files)}")
    val_cache = os.path.join(cache_root, f"val_img{IMG_SIZE}_n{len(val_files)}")

    train_ds = train_ds.cache(train_cache)
    train_ds = train_ds.batch(batch_size, drop_remainder=False)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(val_files).with_options(opts)
    val_ds = val_ds.map(_map_train, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache(val_cache)
    val_ds = val_ds.batch(batch_size, drop_remainder=False)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


def make_test_dataset(test_files, batch_size: int):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True

    test_ds = tf.data.Dataset.from_tensor_slices(test_files).with_options(opts)
    test_ds = test_ds.map(_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)

    cache_root = "/kaggle/working/tfdata_cache"
    os.makedirs(cache_root, exist_ok=True)
    test_cache = os.path.join(cache_root, f"test_img{IMG_SIZE}_n{len(test_files)}")

    test_ds = test_ds.cache(test_cache)
    test_ds = test_ds.batch(batch_size, drop_remainder=False)
    test_ds = test_ds.prefetch(AUTOTUNE)
    return test_ds




## === cell 10
base = ResNet50(
    include_top=False,
    pooling="max",
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
model = Sequential([base, Dense(NUM_CLASSES, activation="softmax")])
base.trainable = False
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()




## === cell 11
all_train_paths = np.array(
    [os.path.join(train_inner_folder, f) for f in train_image_list], dtype=object
)
train_paths, val_paths = train_test_split(
    all_train_paths, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

train_ds, val_ds = make_trainval_datasets(train_paths, val_paths, BATCH_SIZE)

train_model = model.fit(
    train_ds,
    epochs=NO_EPOCHS,
    verbose=1,
    validation_data=val_ds,
)




## === cell 12
def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist["accuracy"] if "accuracy" in hist else hist.get("acc", [])
    val_acc = (
        hist["val_accuracy"] if "val_accuracy" in hist else hist.get("val_acc", [])
    )
    loss = hist["loss"]
    val_loss = hist["val_loss"]
    epochs = range(len(acc))
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(epochs, acc, "g", label="Training accuracy")
    ax[0].plot(epochs, val_acc, "r", label="Validation accuracy")
    ax[0].set_title("Training and validation accuracy")
    ax[0].legend()
    ax[1].plot(epochs, loss, "g", label="Training loss")
    ax[1].plot(epochs, val_loss, "r", label="Validation loss")
    ax[1].set_title("Training and validation loss")
    ax[1].legend()
    plt.show()




## === cell 13
pass




## === cell 14
score = model.evaluate(val_ds, verbose=0)
print("Validation loss:", score[0])
print("Validation accuracy:", score[1])




## === cell 15
y_pred_proba = model.predict(val_ds, verbose=0)
predicted_classes = np.argmax(y_pred_proba, axis=1)

val_basenames = np.char.rpartition(np.asarray(val_paths, dtype=str), os.sep)[:, 2]
val_prefix = np.char.partition(val_basenames, ".")[:, 0]
y_true = (val_prefix == "dog").astype(np.int64)




## === cell 16
correct = np.nonzero(predicted_classes == y_true)[0]
incorrect = np.nonzero(predicted_classes != y_true)[0]
print("Correct:", len(correct), "Incorrect:", len(incorrect))




## === cell 17
target_names = ["Class 0 (cat)", "Class 1 (dog)"]
print(classification_report(y_true, predicted_classes, target_names=target_names))




## === cell 18
ss = pd.read_csv(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
)




## === cell 19
pass




## === cell 20
test_filenames = [str(x) for x in test_filenames_arr]
test_ids = [int(os.path.splitext(f)[0]) for f in test_filenames]

test_paths = np.array(
    [os.path.join(test_inner_folder, f) for f in test_filenames], dtype=object
)
test_ds = make_test_dataset(test_paths, BATCH_SIZE)

y_pred_proba_test = model.predict(test_ds, verbose=1)
dog_proba = y_pred_proba_test[:, 1].astype(float)

pred_df = pd.DataFrame({"id": test_ids, "label": dog_proba})

submission = ss[["id"]].merge(pred_df, on="id", how="left")
submission["label"] = submission["label"].fillna(0.5).clip(1e-7, 1 - 1e-7)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)




## === cell 21
pass




## === cell 22
submission.shape
