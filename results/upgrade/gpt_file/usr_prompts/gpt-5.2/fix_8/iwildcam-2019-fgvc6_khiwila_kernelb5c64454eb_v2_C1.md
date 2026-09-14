# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

print("TensorFlow version:", tf.__version__)



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
    im = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)
    return im


def load_or_build_memmap(
    file_names,
    img_dir,
    target_size=(32, 32),
    out_dat_path=None,
    num_workers=None,
    chunksize=512,
):
    if out_dat_path is None:
        raise ValueError("out_dat_path is required for memmap caching")

    n = len(file_names)
    shape = (n, target_size[0], target_size[1], 3)

    if os.path.exists(out_dat_path):
        mm = np.memmap(out_dat_path, dtype=np.uint8, mode="r", shape=shape)
        return mm

    if num_workers is None:
        num_workers = min(16, (os.cpu_count() or 4))

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
if os.path.exists(x_train_dat_path) and os.path.exists(x_test_dat_path):
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
        xb = xb.astype(np.float32) * (1.0 / 255.0)
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
    parameters = initialize_parameters()
    Z3 = forward_propagation(X_in, parameters)
    net = tf.keras.Model(inputs=X_in, outputs=Z3)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    net.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred)),
        run_eagerly=False,
    )

    train_seq = MemmapBatchSequence(
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
        train_seq,
        epochs=num_epochs,
        verbose=0,
        callbacks=[hist_cb],
        workers=0,  # keep deterministic and avoid multiprocessing overhead on memmaps
        use_multiprocessing=False,
        shuffle=False,  # shuffling handled deterministically inside Sequence
    )
    costs = hist_cb.costs

    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("epochs")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    def _as_array_or_indexed(x):
        if isinstance(x, tuple):
            arr, idx = x
            if idx is None:
                return arr
            return arr[np.asarray(idx, dtype=np.int64)]
        return x

    def _predict_logits_ds(x_uint8, batch_size):
        ds = tf.data.Dataset.from_tensor_slices(x_uint8)
        ds = ds.batch(batch_size, drop_remainder=False)

        def _prep(xb):
            return tf.cast(xb, tf.float32) * (1.0 / 255.0)

        ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE
        )
        return ds

    def predict_classes_batched(x_uint8, batch_size=8192):
        x_uint8 = _as_array_or_indexed(x_uint8)
        ds = _predict_logits_ds(x_uint8, batch_size)
        outs = []
        for xb in ds:
            logits = net(xb, training=False)
            outs.append(tf.argmax(logits, axis=1, output_type=tf.int64))
        return tf.concat(outs, axis=0).numpy()

    def accuracy_eval(x_uint8, y, batch_size=8192):
        x_uint8 = _as_array_or_indexed(x_uint8)
        y = _as_array_or_indexed(y)

        dsx = _predict_logits_ds(x_uint8, batch_size)
        dsy = (
            tf.data.Dataset.from_tensor_slices(y)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        correct = 0
        total = 0
        for xb, yb in tf.data.Dataset.zip((dsx, dsy)):
            logits = net(xb, training=False)
            pred = tf.argmax(logits, axis=1, output_type=tf.int64)
            true = tf.argmax(yb, axis=1, output_type=tf.int64)
            correct += int(
                tf.reduce_sum(tf.cast(tf.equal(pred, true), tf.int32)).numpy()
            )
            total += int(xb.shape[0])
        return float(correct / total)

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



## === cell 15
print("Files in ../working:")
print(os.listdir("../working")[:50])
print("Submission exists:", os.path.exists("../working/submission_got_it2.csv"))

sub_path = os.path.join(WORKING_ROOT, "submission_got_it2.csv")
sub = pd.read_csv(sub_path)
print("submission shape:", sub.shape)
print("submission columns:", sub.columns.tolist())
print(sub.head())
