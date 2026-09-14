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

0.0390505687738178

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/iwildcam-2019-fgvc6/train.csv")
test = pd.read_csv("../input/iwildcam-2019-fgvc6/test.csv")
sample_submission = pd.read_csv("../input/iwildcam-2019-fgvc6/sample_submission.csv")

print("test.shape:", test.shape)
print("sample_submmission.shape:", sample_submission.shape)
print("train.shape:", train.shape)
print("sample_submission columns:", sample_submission.columns.tolist())



## === cell 2
import cv2
import glob
import matplotlib.pyplot as plt
import tqdm
import math
from sklearn.model_selection import train_test_split
from PIL import Image

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1
from tensorflow.python.framework import ops

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_id = train["file_name"]
labels = train["category_id"]
test_id = sample_submission["Id"]

img = plt.imread(
    "../input/iwildcam-2019-fgvc6/train_images/" + train["file_name"].iloc[0]
)
print("Example image shape:", img.shape)



## === cell 4
TRAIN_IMAGES_DIR = "../input/iwildcam-2019-fgvc6/train_images"
TEST_IMAGES_DIR = "../input/iwildcam-2019-fgvc6/test_images"


def _read_resize_rgb(path, size=(32, 32)):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    return im


from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = "../working/cache_iwildcam32"
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_TRAIN_IMAGES = 20000  # keep user's existing cap (core logic expectation)
MAX_TEST_IMAGES = len(sample_submission)  # must predict all for a valid submission

train_subset = train.iloc[:MAX_TRAIN_IMAGES].copy().reset_index(drop=True)

num_classes = 23
y_trains1 = train_subset["category_id"].values.astype(np.int64)
y_trains1_oh = np.eye(num_classes, dtype=np.float32)[y_trains1]

test_join = sample_submission[["Id"]].merge(
    test[["id", "file_name"]], left_on="Id", right_on="id", how="left"
)
if test_join["file_name"].isna().any():
    missing = int(test_join["file_name"].isna().sum())
    raise RuntimeError(f"Could not map {missing} test Ids to file_name from test.csv")


def _cache_path(name):
    return os.path.join(CACHE_DIR, name)


def _load_or_build_cache_uint8(cache_path, paths, size=(32, 32), max_workers=8):
    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        return arr

    n = len(paths)
    out = np.zeros((n, size[1], size[0], 3), dtype=np.uint8)

    def _worker(i_path):
        i, p = i_path
        return i, _read_resize_rgb(p, size=size)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, im in tqdm.tqdm(ex.map(_worker, enumerate(paths)), total=n):
            out[i] = im

    np.save(cache_path, out)
    return out


train_paths = [
    os.path.join(TRAIN_IMAGES_DIR, fn) for fn in train_subset["file_name"].values
]
test_paths = [os.path.join(TEST_IMAGES_DIR, fn) for fn in test_join["file_name"].values]

x_trains1 = _load_or_build_cache_uint8(
    _cache_path(f"train_uint8_{len(train_subset)}.npy"),
    train_paths,
    size=(32, 32),
    max_workers=min(8, (os.cpu_count() or 4)),
)
x_test_arr = _load_or_build_cache_uint8(
    _cache_path(f"test_uint8_{len(test_join)}.npy"),
    test_paths,
    size=(32, 32),
    max_workers=min(8, (os.cpu_count() or 4)),
)

print("x_trains1.shape:", x_trains1.shape)
print("y_trains1_oh.shape:", y_trains1_oh.shape)
print("x_test_arr shape:", x_test_arr.shape)



## === cell 5
x_test = x_test_arr.astype("float32")
x_test /= 255.0

x_trains1_f = x_trains1.astype("float32") / 255.0

print("x_test.shape:", x_test.shape)
print("x_trains1_f.shape:", x_trains1_f.shape)



## === cell 6
y_trains1 = y_trains1_oh
print("x_trains1.shape:", x_trains1_f.shape)
print("y_trains1.shape:", y_trains1.shape)



## === cell 7
x_train, x_dev, y_train, y_dev = train_test_split(
    x_trains1_f,
    y_trains1,
    test_size=0.10,
    random_state=32,
    stratify=np.argmax(y_trains1, axis=1),
)

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_dev.shape:", x_dev.shape)
print("y_dev.shape:", y_dev.shape)

x_test1 = x_test2 = x_test3 = x_test4 = None
print("Skipping test split into 4 parts; prediction will be batched over full x_test.")
print("x_test.shape:", x_test.shape)



## === cell 8
pass




## === cell 9
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf1.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf1.placeholder(tf.float32, shape=(None, n_y))
    return X, Y


def initialize_parameters():
    tf1.set_random_seed(1)

    glorot = tf1.keras.initializers.glorot_uniform(seed=0)
    W1 = tf1.get_variable("W1", [4, 4, 3, 16], initializer=glorot)
    W2 = tf1.get_variable("W2", [2, 2, 16, 32], initializer=glorot)

    parameters = {"W1": W1, "W2": W2}
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    P2 = tf1.layers.flatten(P2)
    Z3 = tf1.layers.dense(P2, 23, activation=None)

    return Z3


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf1.losses.mean_squared_error(Y, Z3))
    return cost




## === cell 10
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :, :
        ]
        mini_batch_Y = shuffled_Y[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size, :
        ]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[
            num_complete_minibatches * mini_batch_size : m, :, :, :
        ]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches




## === cell 11
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
    ops.reset_default_graph()
    tf1.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)

    optimizer = tf1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf1.global_variables_initializer()

    def _iter_minibatch_slices(n, batch_size, seed_for_epoch):
        np.random.seed(seed_for_epoch)
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            yield idx

    def _predict_in_batches(sess, predict_op, X_ph, X_data, batch_size=2048):
        n = X_data.shape[0]
        out = np.empty((n,), dtype=np.int64)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            out[start:end] = sess.run(predict_op, feed_dict={X_ph: X_data[start:end]})
        return out

    with tf1.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0.0
            num_minibatches = int(np.ceil(m / minibatch_size))
            seed = seed + 1

            for idx in _iter_minibatch_slices(m, minibatch_size, seed):
                minibatch_X = X_train[idx]
                minibatch_Y = Y_train[idx]
                _, temp_cost = sess.run(
                    [optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y}
                )
                minibatch_cost += temp_cost / num_minibatches

            if print_cost is True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost is True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per tens)")
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        if X_train.shape[0] > 30000:
            train_accuracy = accuracy.eval({X: X_train[:25000], Y: Y_train[:25000]})
            number_for_train_accuracy = 25000
        else:
            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
            number_for_train_accuracy = X_train.shape[0]

        if X_test.shape[0] > 25000:
            test_accuracy = accuracy.eval({X: X_test[:25000], Y: Y_test[:25000]})
            number_for_test_accuracy = 25000
        else:
            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
            number_for_test_accuracy = X_test.shape[0]

        print("Train Accuracy:", train_accuracy)
        print("Dev Accuracy:", test_accuracy)

        test_results = _predict_in_batches(
            sess, predict_op, X, X_test_test, batch_size=2048
        )

        if len(test_results) != X_test_test.shape[0]:
            raise RuntimeError(
                f"Prediction length mismatch: got {len(test_results)} vs expected {X_test_test.shape[0]}"
            )

        submission = pd.DataFrame(
            {
                "Id": sample_submission["Id"].values,
                "Predicted": np.array(test_results, dtype=np.int64),
            }
        )
        submission.to_csv("submission_got_it10.csv", index=False)

        print("Used in training set:", X_train.shape[0], "elements")
        print("Used in validation set:", X_test.shape[0], "elements")
        print("Used in prediction set:", len(test_results), "elements")
        print("Used for train accuracy:", number_for_train_accuracy, "elements")
        print("Used for dev accuracy:", number_for_test_accuracy, "elements")
        print(submission.head())
        print("Summary of predictions:")
        print(submission.Predicted.value_counts())

        return train_accuracy, test_accuracy, parameters




## === cell 12
_, _, parameters = model(
    x_train,
    y_train,
    x_dev,
    y_dev,
    x_test,
    x_test1,
    x_test2,
    x_test3,
    x_test4,
    learning_rate=0.0050,
    num_epochs=150,
    minibatch_size=64,
    print_cost=True,
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1691280542.py in <cell line: 0>()
----> 1 _, _, parameters = model(
      2     x_train,
      3     y_train,
      4     x_dev,
      5     y_dev,

/tmp/ipykernel_11/188585619.py in model(X_train, Y_train, X_test, Y_test, X_test_test, X_test_test1, X_test_test2, X_test_test3, X_test_test4, learning_rate, num_epochs, minibatch_size, print_cost)
     23     X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
     24     parameters = initialize_parameters()
---> 25     Z3 = forward_propagation(X, parameters)
     26     cost = compute_cost(Z3, Y)
     27 

/tmp/ipykernel_11/4217611828.py in forward_propagation(X, parameters)
     28     P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
     29 
---> 30     P2 = tf1.layers.flatten(P2)
     31     Z3 = tf1.layers.dense(P2, 23, activation=None)
     32 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/lazy_loader.py in __getattr__(self, item)
    205           "__internal__.legacy."
    206       ):
--> 207         raise AttributeError(
    208             f"`{item}` is not available with Keras 3."
    209         )

AttributeError: `flatten` is not available with Keras 3.

## === cell 13
print(os.listdir("../working"))
print("Wrote submission:", os.path.abspath("submission_got_it10.csv"))
print(pd.read_csv("submission_got_it10.csv").head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/97498840.py in <cell line: 0>()
      1 print(os.listdir("../working"))
      2 print("Wrote submission:", os.path.abspath("submission_got_it10.csv"))
----> 3 print(pd.read_csv("submission_got_it10.csv").head())

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

FileNotFoundError: [Errno 2] No such file or directory: 'submission_got_it10.csv'
