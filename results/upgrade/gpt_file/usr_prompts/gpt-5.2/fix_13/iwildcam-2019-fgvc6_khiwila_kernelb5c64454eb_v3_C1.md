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

0.0812003971200206

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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

INPUT_BASE = "../input/iwildcam-2019-fgvc6"
WORKING_BASE = "../working"

print("Listing ../input:", os.listdir("../input")[:20])




## === cell 1
train = pd.read_csv(f"{INPUT_BASE}/train.csv")
test = pd.read_csv(f"{INPUT_BASE}/test.csv")
sample_submission = pd.read_csv(f"{INPUT_BASE}/sample_submission.csv")

print("train.shape:", train.shape)
print("test.shape:", test.shape)
print("sample_submission.shape:", sample_submission.shape)
print("sample_submission.columns:", list(sample_submission.columns))

TRAIN_IMG_DIR = f"{INPUT_BASE}/train_images"
TEST_IMG_DIR = f"{INPUT_BASE}/test_images"

assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"




## === cell 2
import glob
import cv2
import matplotlib.pyplot as plt
import tqdm
import math
from sklearn.model_selection import train_test_split
from PIL import Image

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

print("TensorFlow version:", tf.__version__)

np.random.seed(1)
tf.compat.v1.set_random_seed(1)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
print(train["category_id"].value_counts().head(10))
first_fn = train["file_name"].iloc[0]
first_path = os.path.join(TRAIN_IMG_DIR, first_fn)
print("Example image path (not loaded for speed):", first_path)




## === cell 4
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[np.array(Y).reshape(-1)].T
    return Y




## === cell 5
etiquetas = train["category_id"].values.astype(np.int64)
num_classes = 23
y_train = np.eye(num_classes, dtype=np.float32)[etiquetas]  # shape (m, 23)

os.makedirs(WORKING_BASE, exist_ok=True)
np.save(os.path.join(WORKING_BASE, "y_train.npy"), y_train)

print("y_train shape:", y_train.shape, "dtype:", y_train.dtype)




## === cell 6
import concurrent.futures as cf


def _read_resize_one(args):
    fn, img_dir, th, tw = args
    p = os.path.join(img_dir, fn)

    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        return None

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
    return im


def _load_images_to_memmap_processpool(
    file_names, img_dir, th, tw, tmp_dat, max_workers
):
    Xmm = np.memmap(
        tmp_dat, mode="w+", dtype=np.uint8, shape=(len(file_names), th, tw, 3)
    )

    def _load_one_indexed(i_fn):
        i, fn = i_fn
        im = _read_resize_one((fn, img_dir, th, tw))
        return i, im

    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(_load_one_indexed, enumerate(file_names), chunksize=256)
        for i, im in tqdm.tqdm(
            it,
            total=len(file_names),
            desc=f"Loading from {os.path.basename(img_dir)} ({max_workers} procs)",
            mininterval=2.0,
        ):
            if im is None:
                Xmm[i].fill(0)
            else:
                Xmm[i] = im

    Xmm.flush()
    return Xmm


def load_images_32x32(file_names, img_dir, target_size=(32, 32), cache_path=None):
    th, tw = target_size
    n = len(file_names)

    dat_cache = None
    if cache_path is not None:
        base, _ = os.path.splitext(cache_path)
        dat_cache = base + ".dat"

    if dat_cache is not None and os.path.exists(dat_cache):
        return np.memmap(dat_cache, mode="r", dtype=np.uint8, shape=(n, th, tw, 3))

    if cache_path is not None and os.path.exists(cache_path):
        X = np.load(cache_path, mmap_mode="r")
        if X.shape == (n, th, tw, 3) and X.dtype == np.uint8:
            if dat_cache is not None and not os.path.exists(dat_cache):
                tmp_dat = dat_cache + ".tmp"
                mm = np.memmap(tmp_dat, mode="w+", dtype=np.uint8, shape=(n, th, tw, 3))
                for s in range(0, n, 8192):
                    mm[s : min(s + 8192, n)] = X[s : min(s + 8192, n)]
                mm.flush()
                os.replace(tmp_dat, dat_cache)
                return np.memmap(
                    dat_cache, mode="r", dtype=np.uint8, shape=(n, th, tw, 3)
                )
            return X

    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    if dat_cache is not None:
        tmp_dat = dat_cache + ".tmp"
    else:
        tmp_dat = None

    cpu = os.cpu_count() or 2
    max_workers = max(1, min(8, cpu))  # keep IPC overhead reasonable on Kaggle VMs

    if dat_cache is not None:
        _load_images_to_memmap_processpool(
            file_names, img_dir, th, tw, tmp_dat, max_workers
        )
        os.replace(tmp_dat, dat_cache)
        return np.memmap(dat_cache, mode="r", dtype=np.uint8, shape=(n, th, tw, 3))

    X = np.empty((n, th, tw, 3), dtype=np.uint8)
    for i, fn in enumerate(
        tqdm.tqdm(file_names, total=n, desc=f"Loading {os.path.basename(img_dir)}")
    ):
        im = _read_resize_one((fn, img_dir, th, tw))
        if im is None:
            X[i].fill(0)
        else:
            X[i] = im

    if cache_path is not None:
        tmp_path = cache_path + ".tmp.npy"
        np.save(tmp_path, np.asarray(X))
        os.replace(tmp_path, cache_path)

    return X


train_fns = train["file_name"].values
test_fns = test["file_name"].values

x_train_cache = os.path.join(WORKING_BASE, "x_train_32x32_uint8.npy")
x_test_cache = os.path.join(WORKING_BASE, "x_test_32x32_uint8.npy")

x_train_arr = load_images_32x32(
    train_fns, TRAIN_IMG_DIR, target_size=(32, 32), cache_path=x_train_cache
)
x_test_arr = load_images_32x32(
    test_fns, TEST_IMG_DIR, target_size=(32, 32), cache_path=x_test_cache
)
y_train_arr = np.load(os.path.join(WORKING_BASE, "y_train.npy"), mmap_mode=None)

print("x_train_arr shape:", x_train_arr.shape, "dtype:", x_train_arr.dtype)
print("x_test_arr shape:", x_test_arr.shape, "dtype:", x_test_arr.dtype)
print("y_train_arr shape:", y_train_arr.shape)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't pickle local object '_load_images_to_memmap_processpool.<locals>._load_one_indexed'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2214300096.py in <cell line: 0>()
    124 x_test_cache = os.path.join(WORKING_BASE, "x_test_32x32_uint8.npy")
    125 
--> 126 x_train_arr = load_images_32x32(
    127     train_fns, TRAIN_IMG_DIR, target_size=(32, 32), cache_path=x_train_cache
    128 )

/tmp/ipykernel_11/2214300096.py in load_images_32x32(file_names, img_dir, target_size, cache_path)
     93 
     94     if dat_cache is not None:
---> 95         _load_images_to_memmap_processpool(
     96             file_names, img_dir, th, tw, tmp_dat, max_workers
     97         )

/tmp/ipykernel_11/2214300096.py in _load_images_to_memmap_processpool(file_names, img_dir, th, tw, tmp_dat, max_workers)
     37         # Use moderately large chunks in mapping to reduce IPC overhead.
     38         it = ex.map(_load_one_indexed, enumerate(file_names), chunksize=256)
---> 39         for i, im in tqdm.tqdm(
     40             it,
     41             total=len(file_names),

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/multiprocessing/queues.py in _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem)
    242 
    243                         # serialize the data before acquiring the lock
--> 244                         obj = _ForkingPickler.dumps(obj)
    245                         if wacquire is None:
    246                             send_bytes(obj)

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object '_load_images_to_memmap_processpool.<locals>._load_one_indexed'

## === cell 7
x_train = x_train_arr  # keep uint8 memmap/ndarray
y_train = y_train_arr.astype("float32", copy=False)
x_test = x_test_arr  # keep uint8 memmap/ndarray

print("x_train.shape:", x_train.shape, "dtype:", x_train.dtype)
print("y_train.shape:", y_train.shape, "dtype:", y_train.dtype)
print("x_test.shape:", x_test.shape, "dtype:", x_test.dtype)
print("First label one-hot:", y_train[0])




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/803400663.py in <cell line: 0>()
----> 1 x_train = x_train_arr  # keep uint8 memmap/ndarray
      2 y_train = y_train_arr.astype("float32", copy=False)
      3 x_test = x_test_arr  # keep uint8 memmap/ndarray
      4 
      5 print("x_train.shape:", x_train.shape, "dtype:", x_train.dtype)

NameError: name 'x_train_arr' is not defined

## === cell 8
n_total = x_train.shape[0]
idx_all = np.arange(n_total, dtype=np.int64)
idx_tr, idx_dev = train_test_split(idx_all, test_size=0.4, random_state=32)

x_train_idx = np.asarray(idx_tr, dtype=np.int64)
x_dev_idx = np.asarray(idx_dev, dtype=np.int64)
y_train_idx = y_train[x_train_idx]
y_dev_idx = y_train[x_dev_idx]

print("x_train_idx.shape:", x_train_idx.shape)
print("y_train_idx.shape:", y_train_idx.shape)
print("x_dev_idx.shape:", x_dev_idx.shape)
print("y_dev_idx.shape:", y_dev_idx.shape)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/366508434.py in <cell line: 0>()
----> 1 n_total = x_train.shape[0]
      2 idx_all = np.arange(n_total, dtype=np.int64)
      3 idx_tr, idx_dev = train_test_split(idx_all, test_size=0.4, random_state=32)
      4 
      5 x_train_idx = np.asarray(idx_tr, dtype=np.int64)

NameError: name 'x_train' is not defined

## === cell 9
n_test_total = x_test.shape[0]
idx_test_all = np.arange(n_test_total, dtype=np.int64)
idx_testa, idx_testb = train_test_split(idx_test_all, test_size=0.5, random_state=32)
idx_test1, idx_test2 = train_test_split(idx_testa, test_size=0.5, random_state=32)
idx_test3, idx_test4 = train_test_split(idx_testb, test_size=0.5, random_state=32)

idx_test1 = np.asarray(idx_test1, dtype=np.int64)
idx_test2 = np.asarray(idx_test2, dtype=np.int64)
idx_test3 = np.asarray(idx_test3, dtype=np.int64)
idx_test4 = np.asarray(idx_test4, dtype=np.int64)

print("idx_test1.shape:", idx_test1.shape)
print("idx_test2.shape:", idx_test2.shape)
print("idx_test3.shape:", idx_test3.shape)
print("idx_test4.shape:", idx_test4.shape)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/134659565.py in <cell line: 0>()
----> 1 n_test_total = x_test.shape[0]
      2 idx_test_all = np.arange(n_test_total, dtype=np.int64)
      3 idx_testa, idx_testb = train_test_split(idx_test_all, test_size=0.5, random_state=32)
      4 idx_test1, idx_test2 = train_test_split(idx_testa, test_size=0.5, random_state=32)
      5 idx_test3, idx_test4 = train_test_split(idx_testb, test_size=0.5, random_state=32)

NameError: name 'x_test' is not defined

## === cell 10
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.compat.v1.placeholder(tf.uint8, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.compat.v1.placeholder(tf.float32, shape=(None, n_y))
    return X, Y


def initialize_parameters():
    tf.compat.v1.set_random_seed(1)
    W1 = tf.compat.v1.get_variable(
        "W1", [4, 4, 3, 16], initializer=tf.compat.v1.glorot_uniform_initializer(seed=0)
    )
    W2 = tf.compat.v1.get_variable(
        "W2",
        [2, 2, 16, 32],
        initializer=tf.compat.v1.glorot_uniform_initializer(seed=0),
    )
    return {"W1": W1, "W2": W2}


def forward_propagation(X, parameters):
    Xf = tf.cast(X, tf.float32) * (1.0 / 255.0)

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    Z1 = tf.nn.conv2d(Xf, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    P2_flat = tf.reshape(P2, [-1, 32])

    with tf.compat.v1.variable_scope("fc", reuse=tf.compat.v1.AUTO_REUSE):
        W3 = tf.compat.v1.get_variable(
            "kernel",
            shape=[32, 23],
            initializer=tf.compat.v1.glorot_uniform_initializer(seed=0),
        )
        b3 = tf.compat.v1.get_variable(
            "bias",
            shape=[23],
            initializer=tf.compat.v1.zeros_initializer(),
        )
        Z3 = tf.matmul(P2_flat, W3) + b3

    return Z3


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(
        tf.compat.v1.losses.mean_squared_error(labels=Y, predictions=Z3)
    )
    return cost




## === cell 11
def iter_mini_batches_idx_from_perm(permutation, mini_batch_size):
    m = permutation.shape[0]
    num_complete = m // mini_batch_size
    for k in range(num_complete):
        yield permutation[k * mini_batch_size : (k + 1) * mini_batch_size]
    if m % mini_batch_size != 0:
        yield permutation[num_complete * mini_batch_size : m]




## === cell 12
def _predict_in_batches(sess, predict_op, X_ph, X_uint8, idx, batch_size=4096):
    out = np.empty((len(idx),), dtype=np.int64)
    for start in range(0, len(idx), batch_size):
        sl = slice(start, min(start + batch_size, len(idx)))
        xb = X_uint8[idx[sl]]
        out[sl] = sess.run(predict_op, feed_dict={X_ph: xb})
    return out


def _accuracy_in_batches(
    sess, accuracy_op, X_ph, Y_ph, X_uint8, Y, idx, batch_size=4096
):
    correct_sum = 0.0
    n = len(idx)
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        xb = X_uint8[idx[sl]]
        yb = Y[idx[sl]]
        acc_b = sess.run(accuracy_op, feed_dict={X_ph: xb, Y_ph: yb})
        correct_sum += float(acc_b) * (sl.stop - sl.start)
    return correct_sum / n if n else 0.0


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
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(1)
    seed = 3

    n_H0, n_W0, n_C0 = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        cost
    )
    init = tf.compat.v1.global_variables_initializer()

    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=max(1, (os.cpu_count() or 2)),
        inter_op_parallelism_threads=2,
        allow_soft_placement=True,
    )

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)

        rng = np.random.RandomState(seed)

        idx_ph = tf.compat.v1.placeholder(tf.int64, shape=(None,))

        ds = tf.data.Dataset.from_tensor_slices(idx_ph)

        for epoch in range(num_epochs):
            minibatch_cost = 0.0
            m = X_train.shape[0]
            num_minibatches = int(m / minibatch_size) if m >= minibatch_size else 1

            seed = seed + 1
            rng.seed(seed)
            permutation = rng.permutation(m).astype(np.int64, copy=False)

            for mb_idx in iter_mini_batches_idx_from_perm(permutation, minibatch_size):
                xb = X_train[mb_idx]
                yb = Y_train[mb_idx]
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: xb, Y: yb})
                minibatch_cost += temp_cost / num_minibatches

            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost and epoch % 1 == 0:
                costs.append(minibatch_cost)

        try:
            if False:
                plt.plot(np.squeeze(costs))
                plt.ylabel("cost")
                plt.xlabel("iterations (per tens)")
                plt.title("Learning rate =" + str(learning_rate))
                plt.show()
        except Exception as e:
            print("Plotting skipped:", repr(e))

        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        if X_train.shape[0] > 25000:
            number_for_train_accuracy = 25000
            train_idx = np.arange(number_for_train_accuracy)
        else:
            number_for_train_accuracy = X_train.shape[0]
            train_idx = np.arange(number_for_train_accuracy)

        if X_test.shape[0] > 25000:
            number_for_test_accuracy = 25000
            test_idx = np.arange(number_for_test_accuracy)
        else:
            number_for_test_accuracy = X_test.shape[0]
            test_idx = np.arange(number_for_test_accuracy)

        train_accuracy = _accuracy_in_batches(
            sess, accuracy, X, Y, X_train, Y_train, train_idx, batch_size=4096
        )
        test_accuracy = _accuracy_in_batches(
            sess, accuracy, X, Y, X_test, Y_test, test_idx, batch_size=4096
        )

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        idx_pred = np.concatenate(
            [X_test_test1, X_test_test2, X_test_test3, X_test_test4], axis=0
        ).astype(np.int64, copy=False)

        order = np.argsort(idx_pred)
        idx_pred_sorted = idx_pred[order]

        preds_sorted = _predict_in_batches(
            sess, predict_op, X, X_test_test, idx_pred_sorted, batch_size=4096
        )
        test_results = np.empty((len(idx_pred),), dtype=np.int64)
        test_results[order] = preds_sorted
        test_results = test_results.tolist()

        id_col = (
            "Id"
            if "Id" in sample_submission.columns
            else ("id" if "id" in sample_submission.columns else None)
        )
        if id_col is None:
            raise ValueError(
                f"Could not find an Id column in sample_submission. Columns={list(sample_submission.columns)}"
            )

        if len(test_results) != sample_submission.shape[0]:
            print(
                "Warning: prediction length",
                len(test_results),
                "!= sample_submission length",
                sample_submission.shape[0],
            )
            if len(test_results) > sample_submission.shape[0]:
                test_results = test_results[: sample_submission.shape[0]]
            else:
                test_results = test_results + [0] * (
                    sample_submission.shape[0] - len(test_results)
                )

        submission = pd.DataFrame(
            {"Id": sample_submission[id_col].values, "Predicted": test_results}
        )
        out_path = os.path.join(WORKING_BASE, "submission.csv")
        submission.to_csv(out_path, index=False)

        print("Saved submission to:", out_path)
        print("Used in training set:", X_train.shape[0], "elements")
        print("Used in validation set:", X_test.shape[0], "elements")
        print("Used in prediction set:", len(test_results), "elements")
        print("Used for train accuracy:", number_for_train_accuracy, "elements")
        print("Used for test accuracy:", number_for_test_accuracy, "elements")
        print(submission.head())
        print("Summary of predictions:")
        print(submission.Predicted.value_counts().head(20))

        return train_accuracy, test_accuracy, parameters




## === cell 13
x_train_view = x_train  # no copy
y_train_view = y_train  # no copy
x_dev_view = x_train  # no copy
y_dev_view = y_train  # no copy

_, _, parameters = model(
    x_train[x_train_idx],
    y_train_idx,
    x_train[x_dev_idx],
    y_dev_idx,
    x_test,  # full test uint8 array/memmap
    idx_test1,  # indices into x_test
    idx_test2,
    idx_test3,
    idx_test4,
    learning_rate=0.0009,
    num_epochs=100,
    minibatch_size=64,
    print_cost=True,
)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3826079540.py in <cell line: 0>()
      2 # Instead, train directly on the original x_train memmap/ndarray using the already-created index arrays.
      3 # This preserves the same split and data; only reduces memory/time.
----> 4 x_train_view = x_train  # no copy
      5 y_train_view = y_train  # no copy
      6 x_dev_view = x_train  # no copy

NameError: name 'x_train' is not defined

## === cell 14
print("Files in ../working:", os.listdir(WORKING_BASE)[:50])
if "submission.csv" in os.listdir(WORKING_BASE):
    sub = pd.read_csv(os.path.join(WORKING_BASE, "submission.csv"))
    print("Submission shape:", sub.shape)
    print("Submission columns:", list(sub.columns))
    print(sub.head())
