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
Label images of animals with their species.

## Metric
Macro F1 score

## Submission Format
```
Id,Predicted
58857ccf-23d2-11e8-a6a3-ec086b02610b,1
591e4006-23d2-11e8-a6a3-ec086b02610b,5
```

The `Id` column corresponds to the test image id. The `Category` is an integer value that indicates the class of the animal, or `0` to represent the absence of an animal.

## Dataset
The training set contains 196,157 images from 138 different locations in Southern California. 

The test set contains 153,730 images from 100 locations in Idaho.

The task is to label each image with one of the following label ids:

```
name, id
empty, 0
deer, 1
moose, 2
squirrel, 3
rodent, 4
small_mammal, 5
elk, 6
pronghorn_antelope, 7
rabbit, 8
bighorn_sheep, 9
fox, 10
coyote, 11
black_bear, 12
raccoon, 13
skunk, 14
wolf, 15
bobcat, 16
cat, 17
dog, 18
opossum, 19
bison, 20
mountain_goat, 21
mountain_lion, 22
```

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (109 lines)
            sample_submission.csv (16878 lines)
            sample_submission.csv.zip (133.7 kB)
            test.csv (16878 lines)
            test.csv.zip (415.9 kB)
            test.zip (160 Bytes)
            test_images.zip (1.8 GB)
            train.csv (179423 lines)
            train.csv.zip (4.7 MB)
            train.zip (162 Bytes)
            train_images.zip (26.1 GB)
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
            test/
                test/
            test_images/
                59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                ... and 16860 other files
                test_images/
            train/
                train/
            train_images/
                598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                ... and 179222 other files
                train_images/
        input/
            description.md (109 lines)
            sample_submission.csv (16878 lines)
            sample_submission.csv.zip (133.7 kB)
            test.csv (16878 lines)
            test.csv.zip (415.9 kB)
            test.zip (160 Bytes)
            test_images.zip (1.8 GB)
            train.csv (179423 lines)
            train.csv.zip (4.7 MB)
            train.zip (162 Bytes)
            train_images.zip (26.1 GB)
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
            test/
                test/
                    test/
            test_images/
                59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                ... and 16860 other files
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                ... and 179222 other files
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
        working/
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
```

-> data/iwildcam-2019-fgvc6/sample_submission.csv has 16877 rows and 3 columns.
The columns are: Unnamed: 0, Id, Category

-> data/iwildcam-2019-fgvc6/test.csv has 16877 rows and 10 columns.
The columns are: date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/iwildcam-2019-fgvc6/train.csv has 179422 rows and 11 columns.
The columns are: category_id, date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/sample_submission.csv has 16877 rows and 3 columns.
The columns are: Unnamed: 0, Id, Category

-> data/test.csv has 16877 rows and 10 columns.
The columns are: date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/train.csv has 179422 rows and 11 columns.
The columns are: category_id, date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> (stopped after 10 files for performance)

# 5. Target score

0.0808795909190302

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_ROOT = "../input/iwildcam-2019-fgvc6"
WORKING_ROOT = "../working"

os.makedirs(WORKING_ROOT, exist_ok=True)

print("Input root exists:", os.path.isdir("../input"))
print("Competition folder exists:", os.path.isdir(INPUT_ROOT))



## === cell 1
train = pd.read_csv(f"{INPUT_ROOT}/train.csv")
test = pd.read_csv(f"{INPUT_ROOT}/test.csv")
sample_submission = pd.read_csv(f"{INPUT_ROOT}/sample_submission.csv")

print("train.shape:", train.shape)
print("test.shape:", test.shape)
print("sample_submission.shape:", sample_submission.shape)
print("sample_submission columns:", sample_submission.columns.tolist())

TRAIN_IMG_DIR = f"{INPUT_ROOT}/train_images"
TEST_IMG_DIR = f"{INPUT_ROOT}/test_images"

print("TRAIN_IMG_DIR exists:", os.path.isdir(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.isdir(TEST_IMG_DIR))



## === cell 2
import cv2
import math
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

os.environ.setdefault("PYTHONHASHSEED", "1")
np.random.seed(1)
tf.random.set_seed(1)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    cv2.setNumThreads(max(1, min(8, os.cpu_count() or 4)))
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
except Exception:
    pass

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
print(sample_submission.head())

vc = train["category_id"].value_counts()
print("Top category_id counts:\n", vc.head(10))




## === cell 4
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[np.array(Y).reshape(-1)].T
    return Y




## === cell 5
etiquetas = train["category_id"].values.astype(np.int64)
y_train = np.eye(23, dtype=np.float32)[etiquetas]  # (m,23)

y_train_path = os.path.join(WORKING_ROOT, "y_train.npy")
np.save(y_train_path, y_train)
print("Saved:", y_train_path, "shape:", y_train.shape)



## === cell 6
from concurrent.futures import ThreadPoolExecutor


def _read_resize_one_into(p, target_size):
    im = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR uint8
    if im is None:
        return None
    im = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)
    return im


def load_or_build_memmap(
    file_names,
    img_dir,
    target_size=(32, 32),
    out_dat_path=None,
    num_workers=None,
    chunksize=2048,
):
    if out_dat_path is None:
        raise ValueError("out_dat_path is required for memmap caching")

    n = len(file_names)
    shape = (n, target_size[0], target_size[1], 3)

    if os.path.exists(out_dat_path):
        mm = np.memmap(out_dat_path, dtype=np.uint8, mode="r", shape=shape)
        return mm

    if num_workers is None:
        num_workers = min(8, (os.cpu_count() or 4))

    mm = np.memmap(out_dat_path, dtype=np.uint8, mode="w+", shape=shape)

    missing = 0

    def _worker(fn):
        p = os.path.join(img_dir, fn)
        return _read_resize_one_into(p, target_size)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        it = ex.map(_worker, file_names, chunksize=chunksize)
        for i, im in enumerate(it):
            if im is None:
                mm[i] = 0
                missing += 1
            else:
                mm[i] = im

    mm.flush()
    mm = np.memmap(out_dat_path, dtype=np.uint8, mode="r", shape=shape)

    if missing:
        print(f"Warning: {missing}/{n} images missing or unreadable in {img_dir}")

    return mm


train_files = train["file_name"].tolist()
test_files = test["file_name"].tolist()

x_train_dat_path = os.path.join(WORKING_ROOT, "X_train_32x32_uint8.dat")
x_test_dat_path = os.path.join(WORKING_ROOT, "X_test_32x32_uint8.dat")

print("Preparing disk-backed caches (memmap) for images...")


def _expected_bytes(n, h, w, c, dtype=np.uint8):
    return int(np.dtype(dtype).itemsize) * int(n) * int(h) * int(w) * int(c)


def _cache_ok(path, expected_n):
    if not os.path.exists(path):
        return False
    try:
        sz = os.path.getsize(path)
    except OSError:
        return False
    return sz == _expected_bytes(expected_n, 32, 32, 3, np.uint8)


if _cache_ok(x_train_dat_path, len(train_files)) and _cache_ok(
    x_test_dat_path, len(test_files)
):
    x_train_arr = np.memmap(
        x_train_dat_path,
        dtype=np.uint8,
        mode="r",
        shape=(len(train_files), 32, 32, 3),
    )
    x_test_arr = np.memmap(
        x_test_dat_path,
        dtype=np.uint8,
        mode="r",
        shape=(len(test_files), 32, 32, 3),
    )
    print("Loaded cached memmaps from ../working")
else:
    if os.path.exists(x_train_dat_path) and not _cache_ok(
        x_train_dat_path, len(train_files)
    ):
        try:
            os.remove(x_train_dat_path)
        except Exception:
            pass
    if os.path.exists(x_test_dat_path) and not _cache_ok(
        x_test_dat_path, len(test_files)
    ):
        try:
            os.remove(x_test_dat_path)
        except Exception:
            pass

    print("Building train memmap cache (one-time cost)...")
    x_train_arr = load_or_build_memmap(
        train_files,
        TRAIN_IMG_DIR,
        target_size=(32, 32),
        out_dat_path=x_train_dat_path,
    )
    print("Building test memmap cache (one-time cost)...")
    x_test_arr = load_or_build_memmap(
        test_files,
        TEST_IMG_DIR,
        target_size=(32, 32),
        out_dat_path=x_test_dat_path,
    )
    print("Saved cached memmaps to ../working")

y_train_arr = np.load(y_train_path, mmap_mode="r")

print("x_train_arr shape:", x_train_arr.shape, x_train_arr.dtype)
print("x_test_arr shape:", x_test_arr.shape, x_test_arr.dtype)
print("y_train_arr shape:", y_train_arr.shape, y_train_arr.dtype)



## === cell 7
x_train = x_train_arr  # uint8 memmap
y_train = y_train_arr.astype(np.float32, copy=False)
x_test = x_test_arr  # uint8 memmap

print("x_train.shape:", x_train.shape, x_train.dtype)
print("y_train.shape:", y_train.shape, y_train.dtype)
print("x_test.shape:", x_test.shape, x_test.dtype)



## === cell 8
labels_int = train["category_id"].values.astype(np.int64)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=32)
train_idx, dev_idx = next(sss.split(np.zeros_like(labels_int), labels_int))

x_train_idx = train_idx
x_dev_idx = dev_idx

y_train_idx = train_idx
y_dev_idx = dev_idx

lab_train = labels_int[train_idx]
lab_dev = labels_int[dev_idx]

print("train/dev sizes:", len(train_idx), len(dev_idx))




## === cell 9
def _split_indices(n, test_size, seed):
    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    if isinstance(test_size, float):
        n_test = int(np.ceil(n * test_size))
    else:
        n_test = int(test_size)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


n_test = x_test.shape[0]
idx_a, idx_b = _split_indices(n_test, test_size=0.5, seed=32)
idx_1, idx_2 = _split_indices(len(idx_a), test_size=0.5, seed=32)
idx_3, idx_4 = _split_indices(len(idx_b), test_size=0.5, seed=32)

test1_idx = idx_a[idx_1]
test2_idx = idx_a[idx_2]
test3_idx = idx_b[idx_3]
test4_idx = idx_b[idx_4]

print(
    "test parts sizes:", len(test1_idx), len(test2_idx), len(test3_idx), len(test4_idx)
)
print(
    "Total test parts:",
    len(test1_idx) + len(test2_idx) + len(test3_idx) + len(test4_idx),
)



## === cell 10
print("test id head:", test["id"].head().tolist())
print("sample_submission Id head:", sample_submission["Id"].head().tolist())
print(
    "Id match ratio:",
    (
        np.mean(sample_submission["Id"].values == test["id"].values)
        if len(sample_submission) == len(test)
        else "len mismatch"
    ),
)




## === cell 11
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.keras.Input(shape=(n_H0, n_W0, n_C0), dtype=tf.float32, name="X")
    Y = tf.keras.Input(shape=(n_y,), dtype=tf.float32, name="Y")
    return X, Y


def initialize_parameters():
    return {}


def forward_propagation(X, parameters):
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
        name="conv1",
    )(X)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(8, 8), strides=(8, 8), padding="same", name="pool1"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
        name="conv2",
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(4, 4), strides=(4, 4), padding="same", name="pool2"
    )(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    Z3 = tf.keras.layers.Dense(23, activation=None, name="dense")(x)
    return Z3


def compute_cost(Z3, Y):
    return tf.reduce_mean(tf.keras.losses.mse(Y, Z3))




## === cell 12
def iter_minibatches_indices(m, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    perm = np.random.permutation(m)
    for start in range(0, m, mini_batch_size):
        end = min(start + mini_batch_size, m)
        yield perm[start:end]




## === cell 13
import matplotlib.pyplot as plt


class MemmapBatchSequence(tf.keras.utils.Sequence):
    def __init__(self, X_uint8, Y_float32, indices, batch_size, seed):
        self.X = X_uint8
        self.Y = Y_float32
        self.indices = (
            np.asarray(indices, dtype=np.int64) if indices is not None else None
        )
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self._order = None
        self.on_epoch_end()

    def __len__(self):
        n = len(self.indices) if self.indices is not None else self.X.shape[0]
        return (n + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        n = len(self.indices) if self.indices is not None else self.X.shape[0]
        rs = np.random.RandomState(self.seed + self.epoch)
        order = rs.permutation(n).astype(np.int64)
        if self.indices is not None:
            order = self.indices[order]  # absolute indices into underlying memmaps
        self._order = order
        self.epoch += 1

    def __getitem__(self, i):
        start = i * self.batch_size
        end = min((i + 1) * self.batch_size, self._order.shape[0])
        batch_idx = self._order[start:end]
        xb = self.X[batch_idx]  # uint8
        yb = self.Y[batch_idx]  # float32 already
        return xb, yb


def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    X_test_test,
    X_test_test1,
    X_test_test2,
    X_test_test3,
    X_test_test4,
    learning_rate=0.009,
    num_epochs=20,
    minibatch_size=64,
    print_cost=True,
):
    tf.random.set_seed(1)
    np.random.seed(1)

    def _resolve_xy(X, Y):
        if isinstance(X, tuple):
            X_arr, X_idx = X
        else:
            X_arr, X_idx = X, None
        if isinstance(Y, tuple):
            Y_arr, Y_idx = Y
        else:
            Y_arr, Y_idx = Y, None
        return X_arr, X_idx, Y_arr, Y_idx

    X_train_arr, X_train_idx, Y_train_arr, Y_train_idx = _resolve_xy(X_train, Y_train)
    X_dev_arr, X_dev_idx, Y_dev_arr, Y_dev_idx = _resolve_xy(X_test, Y_test)

    (m_full, n_H0, n_W0, n_C0) = X_train_arr.shape
    n_y = Y_train_arr.shape[1]
    costs = []

    X_in, Y_in = create_placeholders(n_H0, n_W0, n_C0, n_y)
    X_norm = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_255")(X_in)

    parameters = initialize_parameters()
    Z3 = forward_propagation(X_norm, parameters)
    net = tf.keras.Model(inputs=X_in, outputs=Z3)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    net.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred)),
        run_eagerly=False,
        steps_per_execution=32,
    )

    def _make_epoch_order(indices_abs, n_total, seed, epoch):
        n = len(indices_abs) if indices_abs is not None else n_total
        rs = np.random.RandomState(seed + epoch)
        order_local = rs.permutation(n).astype(np.int64)
        if indices_abs is not None:
            return indices_abs[order_local]
        return order_local

    def _dataset_from_memmap(Xmm, Ymm, indices_abs, batch_size, seed):
        n_total = Xmm.shape[0]

        def gen():
            epoch = 0
            while True:
                order = _make_epoch_order(indices_abs, n_total, seed, epoch)
                for start in range(0, order.shape[0], batch_size):
                    idx = order[start : start + batch_size]
                    yield Xmm[idx], Ymm[idx]
                epoch += 1

        output_signature = (
            tf.TensorSpec(shape=(None, n_H0, n_W0, n_C0), dtype=tf.uint8),
            tf.TensorSpec(shape=(None, n_y), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    n_train = int(len(X_train_idx) if X_train_idx is not None else X_train_arr.shape[0])
    steps_per_epoch = (n_train + int(minibatch_size) - 1) // int(minibatch_size)

    train_ds = _dataset_from_memmap(
        X_train_arr, Y_train_arr, X_train_idx, minibatch_size, seed=3
    )

    class CostHistory(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.costs = []

        def on_epoch_end(self, epoch, logs=None):
            loss = float((logs or {}).get("loss", np.nan))
            self.costs.append(loss)
            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, loss))

    hist_cb = CostHistory()

    net.fit(
        train_ds,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
        callbacks=[hist_cb],
        shuffle=False,
    )
    costs = hist_cb.costs

    def _as_array_or_indexed(x):
        if isinstance(x, tuple):
            arr, idx = x
            if idx is None:
                return arr
            return arr[np.asarray(idx, dtype=np.int64)]
        return x

    def predict_classes_batched(x_uint8, batch_size=8192):
        x_uint8 = _as_array_or_indexed(x_uint8)
        logits = net.predict(
            x_uint8,
            batch_size=batch_size,
            verbose=0,
            workers=1,
            use_multiprocessing=False,
        )
        return np.argmax(logits, axis=1).astype(np.int64, copy=False)

    def accuracy_eval(x_uint8, y, batch_size=8192):
        x_uint8 = _as_array_or_indexed(x_uint8)
        y = _as_array_or_indexed(y)
        logits = net.predict(
            x_uint8,
            batch_size=batch_size,
            verbose=0,
            workers=1,
            use_multiprocessing=False,
        )
        pred = np.argmax(logits, axis=1)
        true = np.argmax(y, axis=1)
        return float(np.mean(pred == true))

    X_train_for_acc = (
        (X_train_arr, X_train_idx[:25000])
        if (X_train_idx is not None and len(X_train_idx) > 25000)
        else (X_train_arr, X_train_idx)
    )
    Y_train_for_acc = (
        (Y_train_arr, Y_train_idx[:25000])
        if (Y_train_idx is not None and len(Y_train_idx) > 25000)
        else (Y_train_arr, Y_train_idx)
    )

    if X_train_idx is None and X_train_arr.shape[0] > 25000:
        X_train_for_acc = X_train_arr[:25000]
        Y_train_for_acc = Y_train_arr[:25000]
        number_for_train_accuracy = 25000
    else:
        number_for_train_accuracy = (
            25000
            if (X_train_idx is not None and len(X_train_idx) > 25000)
            else (len(X_train_idx) if X_train_idx is not None else X_train_arr.shape[0])
        )

    X_dev_for_acc = (
        (X_dev_arr, X_dev_idx[:25000])
        if (X_dev_idx is not None and len(X_dev_idx) > 25000)
        else (X_dev_arr, X_dev_idx)
    )
    Y_dev_for_acc = (
        (Y_dev_arr, Y_dev_idx[:25000])
        if (Y_dev_idx is not None and len(Y_dev_idx) > 25000)
        else (Y_dev_arr, Y_dev_idx)
    )

    if X_dev_idx is None and X_dev_arr.shape[0] > 25000:
        X_dev_for_acc = X_dev_arr[:25000]
        Y_dev_for_acc = Y_dev_arr[:25000]
        number_for_test_accuracy = 25000
    else:
        number_for_test_accuracy = (
            25000
            if (X_dev_idx is not None and len(X_dev_idx) > 25000)
            else (len(X_dev_idx) if X_dev_idx is not None else X_dev_arr.shape[0])
        )

    train_accuracy = accuracy_eval(X_train_for_acc, Y_train_for_acc)
    test_accuracy = accuracy_eval(X_dev_for_acc, Y_dev_for_acc)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    _ = (X_test_test1, X_test_test2, X_test_test3, X_test_test4)

    test_results_arr = predict_classes_batched(X_test_test)

    if test_results_arr.shape[0] != _as_array_or_indexed(X_test_test).shape[0]:
        raise ValueError(
            f"Prediction length mismatch: got {test_results_arr.shape[0]} expected {_as_array_or_indexed(X_test_test).shape[0]}"
        )

    submission = pd.DataFrame(
        {
            "Id": test["id"].values[: _as_array_or_indexed(X_test_test).shape[0]],
            "Predicted": test_results_arr.astype(np.int64, copy=False),
        }
    )
    out_path = os.path.join(WORKING_ROOT, "submission_got_it2.csv")
    submission.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print(
        "Used in training set:",
        (len(X_train_idx) if X_train_idx is not None else X_train_arr.shape[0]),
        "elements",
    )
    print(
        "Used in validation set:",
        (len(X_dev_idx) if X_dev_idx is not None else X_dev_arr.shape[0]),
        "elements",
    )
    print("Used in prediction set:", int(test_results_arr.shape[0]), "elements")
    print("Used for train accuracy:", number_for_train_accuracy, "elements")
    print("Used for test accuracy:", number_for_test_accuracy, "elements")
    print(submission.head())
    print("Summary of predictions:")
    print(submission["Predicted"].value_counts().head(10))

    return train_accuracy, test_accuracy, net




## === cell 14
x_train_pair = (x_train, x_train_idx)
y_train_pair = (y_train, y_train_idx)
x_dev_pair = (x_train, x_dev_idx)  # dev comes from train memmap
y_dev_pair = (y_train, y_dev_idx)

x_test_full = x_test
x_test1 = (x_test, test1_idx)
x_test2 = (x_test, test2_idx)
x_test3 = (x_test, test3_idx)
x_test4 = (x_test, test4_idx)

_, _, parameters = model(
    x_train_pair,
    y_train_pair,
    x_dev_pair,
    y_dev_pair,
    x_test_full,
    x_test1,
    x_test2,
    x_test3,
    x_test4,
    learning_rate=0.0009,
    num_epochs=100,
    minibatch_size=64,
    print_cost=True,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1843528888.py in <cell line: 0>()
     10 x_test4 = (x_test, test4_idx)
     11 
---> 12 _, _, parameters = model(
     13     x_train_pair,
     14     y_train_pair,

/tmp/ipykernel_11/3961651776.py in model(X_train, Y_train, X_test, Y_test, X_test_test, X_test_test1, X_test_test2, X_test_test3, X_test_test4, learning_rate, num_epochs, minibatch_size, print_cost)
    228         )
    229 
--> 230     train_accuracy = accuracy_eval(X_train_for_acc, Y_train_for_acc)
    231     test_accuracy = accuracy_eval(X_dev_for_acc, Y_dev_for_acc)
    232 

/tmp/ipykernel_11/3961651776.py in accuracy_eval(x_uint8, y, batch_size)
    173         x_uint8 = _as_array_or_indexed(x_uint8)
    174         y = _as_array_or_indexed(y)
--> 175         logits = net.predict(
    176             x_uint8,
    177             batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 15
print("Files in ../working:")
print(os.listdir("../working")[:50])
print("Submission exists:", os.path.exists("../working/submission_got_it2.csv"))

sub_path = os.path.join(WORKING_ROOT, "submission_got_it2.csv")
sub = pd.read_csv(sub_path)
print("submission shape:", sub.shape)
print("submission columns:", sub.columns.tolist())
print(sub.head())

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/998556608.py in <cell line: 0>()
      4 
      5 sub_path = os.path.join(WORKING_ROOT, "submission_got_it2.csv")
----> 6 sub = pd.read_csv(sub_path)
      7 print("submission shape:", sub.shape)
      8 print("submission columns:", sub.columns.tolist())

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../working/submission_got_it2.csv'
