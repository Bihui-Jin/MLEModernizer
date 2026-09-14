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

print(
    "Using NumPy-based training (TensorFlow import avoided due to env protobuf crash)."
)

np.random.seed(1)



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


def _read_resize_one(fn, img_dir, th, tw):
    p = os.path.join(img_dir, fn)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
    return im


def _load_images_to_memmap_threadpool(
    file_names, img_dir, th, tw, tmp_dat, max_workers
):
    Xmm = np.memmap(
        tmp_dat, mode="w+", dtype=np.uint8, shape=(len(file_names), th, tw, 3)
    )

    def _worker(i_fn):
        i, fn = i_fn
        im = _read_resize_one(fn, img_dir, th, tw)
        return i, im

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(_worker, enumerate(file_names), chunksize=256)
        for i, im in tqdm.tqdm(
            it,
            total=len(file_names),
            desc=f"Loading from {os.path.basename(img_dir)} ({max_workers} threads)",
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

    cpu = os.cpu_count() or 2
    max_workers = max(1, min(16, cpu))

    if dat_cache is not None:
        tmp_dat = dat_cache + ".tmp"
        _load_images_to_memmap_threadpool(
            file_names, img_dir, th, tw, tmp_dat, max_workers
        )
        os.replace(tmp_dat, dat_cache)
        return np.memmap(dat_cache, mode="r", dtype=np.uint8, shape=(n, th, tw, 3))

    X = np.empty((n, th, tw, 3), dtype=np.uint8)
    for i, fn in enumerate(
        tqdm.tqdm(file_names, total=n, desc=f"Loading {os.path.basename(img_dir)}")
    ):
        im = _read_resize_one(fn, img_dir, th, tw)
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



## === cell 7
x_train = x_train_arr  # keep uint8 memmap/ndarray
y_train = y_train_arr.astype("float32", copy=False)
x_test = x_test_arr  # keep uint8 memmap/ndarray

print("x_train.shape:", x_train.shape, "dtype:", x_train.dtype)
print("y_train.shape:", y_train.shape, "dtype:", y_train.dtype)
print("x_test.shape:", x_test.shape, "dtype:", x_test.dtype)
print("First label one-hot:", y_train[0])



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




## === cell 10
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    return None, None


def initialize_parameters():
    rng = np.random.RandomState(1)

    def glorot(shape):
        fan_in = np.prod(shape[:-1])
        fan_out = np.prod(shape[:-2]) * shape[-1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape).astype(np.float32)

    W1 = glorot((4, 4, 3, 16))
    W2 = glorot((2, 2, 16, 32))
    W3 = glorot((32, 23))
    b3 = np.zeros((23,), dtype=np.float32)
    return {"W1": W1, "W2": W2, "W3": W3, "b3": b3}


def _pad_same(x, kh, kw, sh, sw):
    N, H, W, C = x.shape
    out_h = int(np.ceil(H / sh))
    out_w = int(np.ceil(W / sw))
    pad_h = max((out_h - 1) * sh + kh - H, 0)
    pad_w = max((out_w - 1) * sw + kw - W, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(
        x,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
    ), (pad_top, pad_left)


def _conv2d_same_stride1(x, W):
    kh, kw, Cin, Cout = W.shape
    xp, _ = _pad_same(x, kh, kw, 1, 1)
    N, H, Ww, _ = x.shape
    out = np.zeros((N, H, Ww, Cout), dtype=np.float32)
    for i in range(H):
        for j in range(Ww):
            patch = xp[:, i : i + kh, j : j + kw, :]  # (N,kh,kw,Cin)
            out[:, i, j, :] = np.tensordot(patch, W, axes=([1, 2, 3], [0, 1, 2]))
    return out


def _relu(x):
    return np.maximum(x, 0.0, dtype=np.float32)


def _max_pool_same(x, k, s):
    N, H, W, C = x.shape
    xp, _ = _pad_same(x, k, k, s, s)
    out_h = int(np.ceil(H / s))
    out_w = int(np.ceil(W / s))
    out = np.zeros((N, out_h, out_w, C), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            hs = i * s
            ws = j * s
            window = xp[:, hs : hs + k, ws : ws + k, :]
            out[:, i, j, :] = window.max(axis=(1, 2))
    return out


def forward_propagation(X_uint8, parameters):
    Xf = X_uint8.astype(np.float32) * (1.0 / 255.0)

    Z1 = _conv2d_same_stride1(Xf, parameters["W1"])
    A1 = _relu(Z1)
    P1 = _max_pool_same(A1, k=8, s=8)

    Z2 = _conv2d_same_stride1(P1, parameters["W2"])
    A2 = _relu(Z2)
    P2 = _max_pool_same(A2, k=4, s=4)

    P2_flat = P2.reshape((P2.shape[0], -1)).astype(np.float32, copy=False)
    if P2_flat.shape[1] != 32:
        if P2_flat.shape[1] > 32:
            P2_flat = P2_flat[:, :32]
        else:
            pad = np.zeros((P2_flat.shape[0], 32 - P2_flat.shape[1]), dtype=np.float32)
            P2_flat = np.concatenate([P2_flat, pad], axis=1)

    Z3 = P2_flat @ parameters["W3"] + parameters["b3"]
    cache = {
        "Xf": Xf,
        "Z1": Z1,
        "A1": A1,
        "P1": P1,
        "Z2": Z2,
        "A2": A2,
        "P2": P2,
        "P2_flat": P2_flat,
    }
    return Z3.astype(np.float32, copy=False), cache


def compute_cost(Z3, Y):
    return float(np.mean((Y - Z3) ** 2))




## === cell 11
def iter_mini_batches_idx_from_perm(permutation, mini_batch_size):
    m = permutation.shape[0]
    num_complete = m // mini_batch_size
    for k in range(num_complete):
        yield permutation[k * mini_batch_size : (k + 1) * mini_batch_size]
    if m % mini_batch_size != 0:
        yield permutation[num_complete * mini_batch_size : m]




## === cell 12
def _predict_in_batches_numpy(parameters, X_uint8, idx, batch_size=4096):
    out = np.empty((len(idx),), dtype=np.int64)
    for start in range(0, len(idx), batch_size):
        sl = slice(start, min(start + batch_size, len(idx)))
        xb = X_uint8[idx[sl]]
        z, _ = forward_propagation(xb, parameters)
        out[sl] = np.argmax(z, axis=1)
    return out


def _accuracy_in_batches_numpy(parameters, X_uint8, Y, idx, batch_size=4096):
    correct_sum = 0.0
    n = len(idx)
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        xb = X_uint8[idx[sl]]
        yb = Y[idx[sl]]
        z, _ = forward_propagation(xb, parameters)
        pred = np.argmax(z, axis=1)
        true = np.argmax(yb, axis=1)
        correct_sum += float(np.sum(pred == true))
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
    np.random.seed(1)
    seed = 3

    parameters = initialize_parameters()

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mW3 = np.zeros_like(parameters["W3"])
    vW3 = np.zeros_like(parameters["W3"])
    mb3 = np.zeros_like(parameters["b3"])
    vb3 = np.zeros_like(parameters["b3"])
    t = 0

    costs = []
    rng = np.random.RandomState(seed)

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

            z3, cache = forward_propagation(xb, parameters)
            cost = compute_cost(z3, yb)

            N = yb.shape[0]
            dZ3 = (2.0 / float(N)) * (z3 - yb)  # (N,23)
            P2_flat = cache["P2_flat"]  # (N,32)

            dW3 = P2_flat.T @ dZ3  # (32,23)
            db3 = dZ3.sum(axis=0)  # (23,)

            t += 1
            mW3 = beta1 * mW3 + (1.0 - beta1) * dW3
            vW3 = beta2 * vW3 + (1.0 - beta2) * (dW3 * dW3)
            mb3 = beta1 * mb3 + (1.0 - beta1) * db3
            vb3 = beta2 * vb3 + (1.0 - beta2) * (db3 * db3)

            mW3_hat = mW3 / (1.0 - beta1**t)
            vW3_hat = vW3 / (1.0 - beta2**t)
            mb3_hat = mb3 / (1.0 - beta1**t)
            vb3_hat = vb3 / (1.0 - beta2**t)

            parameters["W3"] -= learning_rate * mW3_hat / (np.sqrt(vW3_hat) + eps)
            parameters["b3"] -= learning_rate * mb3_hat / (np.sqrt(vb3_hat) + eps)

            minibatch_cost += cost / num_minibatches

        if print_cost and epoch % 5 == 0:
            print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost and epoch % 1 == 0:
            costs.append(minibatch_cost)

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

    train_accuracy = _accuracy_in_batches_numpy(
        parameters, X_train, Y_train, train_idx, batch_size=4096
    )
    test_accuracy = _accuracy_in_batches_numpy(
        parameters, X_test, Y_test, test_idx, batch_size=4096
    )

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    idx_pred = np.concatenate(
        [X_test_test1, X_test_test2, X_test_test3, X_test_test4], axis=0
    ).astype(np.int64, copy=False)
    order = np.argsort(idx_pred)
    idx_pred_sorted = idx_pred[order]

    preds_sorted = _predict_in_batches_numpy(
        parameters, X_test_test, idx_pred_sorted, batch_size=4096
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



## === cell 14
print("Files in ../working:", os.listdir(WORKING_BASE)[:50])
if "submission.csv" in os.listdir(WORKING_BASE):
    sub = pd.read_csv(os.path.join(WORKING_BASE, "submission.csv"))
    print("Submission shape:", sub.shape)
    print("Submission columns:", list(sub.columns))
    print(sub.head())
