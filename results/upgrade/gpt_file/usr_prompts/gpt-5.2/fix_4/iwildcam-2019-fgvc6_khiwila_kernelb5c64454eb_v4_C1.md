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

0.0765526631913625

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

INPUT_DIR = "../input/iwildcam-2019-fgvc6"
WORKING_DIR = "../working"

print("Listing ../input:", os.listdir("../input")[:20])
print("Listing competition dir:", os.listdir(INPUT_DIR)[:20])



## === cell 1
train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
sample_submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

print("train.shape:", train.shape)
print("test.shape:", test.shape)
print("sample_submission.shape:", sample_submission.shape)

train_images_dir = f"{INPUT_DIR}/train_images"
test_images_dir = f"{INPUT_DIR}/test_images"

assert os.path.isdir(train_images_dir), f"Missing dir: {train_images_dir}"
assert os.path.isdir(test_images_dir), f"Missing dir: {test_images_dir}"



## === cell 2
import cv2
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

import tensorflow as tf

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, (os.cpu_count() or 2) - 1))
except Exception as e:
    print("OpenCV threading setup skipped:", repr(e))

tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
print(sample_submission.head())
print("Train category distribution (top 10):")
print(train["category_id"].value_counts().head(10))

img_path0 = os.path.join(train_images_dir, train["file_name"].iloc[0])
img0 = cv2.imread(img_path0)
if img0 is None:
    raise FileNotFoundError(f"Could not read image: {img_path0}")
print("Example image shape (H,W,C):", img0.shape)




## === cell 4
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[np.array(Y).reshape(-1)].T
    return Y




## === cell 5
etiquetas = train["category_id"].values.astype(np.int64)
y_train = convert_to_one_hot(etiquetas, 23).T  # shape (m, 23)

y_path = os.path.join(WORKING_DIR, "y_train.npy")
np.save(y_path, y_train)
print("y_train.shape:", y_train.shape, "saved to", y_path)



## === cell 6
import multiprocessing as mp


def _read_resize_one(args):
    images_dir, fn, img_size = args
    p = os.path.join(images_dir, fn)
    im = cv2.imread(p)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return im


def load_images_as_array_parallel(
    df,
    images_dir,
    file_col="file_name",
    img_size=32,
    max_items=None,
    n_workers=None,
    chunksize=128,
):
    files = df[file_col].values
    if max_items is not None:
        files = files[:max_items]
    n = len(files)

    X = np.empty((n, img_size, img_size, 3), dtype=np.uint8)

    if n_workers is None:
        n_workers = max(1, min(8, (os.cpu_count() or 2) // 2))

    with mp.get_context("fork").Pool(processes=n_workers) as pool:
        it = pool.imap(
            _read_resize_one,
            ((images_dir, fn, img_size) for fn in files),
            chunksize=chunksize,
        )
        for i, im in enumerate(it):
            if im is None:
                X[i].fill(0)
            else:
                X[i] = im
            if (i + 1) % 5000 == 0:
                print(f"Loaded {i+1}/{n} images from {images_dir}")
    return X


MAX_TRAIN_IMAGES = int(os.environ.get("MAX_TRAIN_IMAGES", "30000"))

x_train_path = os.path.join(WORKING_DIR, f"X_train_32_{MAX_TRAIN_IMAGES}.npy")
x_test_path = os.path.join(WORKING_DIR, "X_test_32.npy")

if os.path.exists(x_train_path) and os.path.exists(x_test_path):
    x_train_arr = np.load(x_train_path, mmap_mode="r")
    x_test_arr = np.load(x_test_path, mmap_mode="r")
    print("Loaded cached arrays:", x_train_arr.shape, x_test_arr.shape)
else:
    train_sub = train.iloc[:MAX_TRAIN_IMAGES].copy()
    y_train = y_train[:MAX_TRAIN_IMAGES]

    x_train_arr = load_images_as_array_parallel(
        train_sub, train_images_dir, img_size=32, max_items=None
    )
    x_test_arr = load_images_as_array_parallel(
        test, test_images_dir, img_size=32, max_items=None
    )

    np.save(x_train_path, x_train_arr)
    np.save(x_test_path, x_test_arr)
    print("Saved arrays:", x_train_arr.shape, x_test_arr.shape)

y_train_arr = y_train.astype(np.float32)

print("x_train_arr shape:", x_train_arr.shape)
print("x_test_arr shape:", x_test_arr.shape)
print("y_train_arr shape:", y_train_arr.shape)



## === cell 7
x_train = np.asarray(x_train_arr, dtype=np.float32) * (1.0 / 255.0)
x_test = np.asarray(x_test_arr, dtype=np.float32) * (1.0 / 255.0)
y_train = y_train_arr.astype(np.float32, copy=False)

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape)
print("First label one-hot:", y_train[0])



## === cell 8
x_train, x_dev, y_train, y_dev = train_test_split(
    x_train, y_train, test_size=0.5, random_state=32
)
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_dev.shape:", x_dev.shape)
print("y_dev.shape:", y_dev.shape)



## === cell 9
print("Skipping unused x_test splits to save time/memory. x_test.shape:", x_test.shape)



## === cell 10
pass




## === cell 11
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf1.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf1.placeholder(tf.float32, shape=(None, n_y))
    return X, Y


def initialize_parameters():
    tf1.set_random_seed(1)
    xavier = tf.keras.initializers.GlorotUniform(seed=0)
    W1 = tf1.get_variable("W1", [4, 4, 3, 16], initializer=xavier)
    W2 = tf1.get_variable("W2", [2, 2, 16, 32], initializer=xavier)
    return {"W1": W1, "W2": W2}


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    P2_flat = tf.reshape(P2, [tf.shape(P2)[0], -1])
    Z3 = tf.keras.layers.Dense(23, activation=None)(P2_flat)
    return Z3


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.keras.losses.mean_squared_error(Y, Z3))
    return cost




## === cell 12
def minibatch_index_generator(m, mini_batch_size=64, seed=0):
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(m)

    num_complete_minibatches = m // mini_batch_size
    for k in range(num_complete_minibatches):
        start = k * mini_batch_size
        end = start + mini_batch_size
        yield permutation[start:end]

    if m % mini_batch_size != 0:
        yield permutation[num_complete_minibatches * mini_batch_size : m]




## === cell 13
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

    tf1.reset_default_graph()
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

    config = tf1.ConfigProto(
        intra_op_parallelism_threads=max(1, (os.cpu_count() or 2) - 1),
        inter_op_parallelism_threads=2,
        allow_soft_placement=True,
    )

    with tf1.Session(config=config) as sess:
        sess.run(init)

        run = sess.run
        opt_cost_fetch = [optimizer, cost]

        for epoch in range(num_epochs):
            minibatch_cost = 0.0
            num_minibatches = int(m / minibatch_size) if m >= minibatch_size else 1
            seed = seed + 1

            for mb_idx in minibatch_index_generator(m, minibatch_size, seed):
                minibatch_X = X_train[mb_idx]
                minibatch_Y = Y_train[mb_idx]
                _, temp_cost = run(
                    opt_cost_fetch, feed_dict={X: minibatch_X, Y: minibatch_Y}
                )
                minibatch_cost += temp_cost / num_minibatches

            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost:
                costs.append(minibatch_cost)

        try:
            plt.plot(np.squeeze(costs))
            plt.ylabel("cost")
            plt.xlabel("epochs")
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
        except Exception as e:
            print("Plot skipped:", repr(e))

        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        if X_train.shape[0] > 25000:
            train_accuracy = run(
                accuracy, feed_dict={X: X_train[:25000], Y: Y_train[:25000]}
            )
            number_for_train_accuracy = 25000
        else:
            train_accuracy = run(accuracy, feed_dict={X: X_train, Y: Y_train})
            number_for_train_accuracy = X_train.shape[0]

        if X_test.shape[0] > 25000:
            test_accuracy = run(
                accuracy, feed_dict={X: X_test[:25000], Y: Y_test[:25000]}
            )
            number_for_test_accuracy = 25000
        else:
            test_accuracy = run(accuracy, feed_dict={X: X_test, Y: Y_test})
            number_for_test_accuracy = X_test.shape[0]

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        test_results = run(predict_op, feed_dict={X: X_test_test}).astype(int)

        submission = pd.DataFrame(
            {
                "Id": sample_submission["Id"].values[: X_test_test.shape[0]],
                "Predicted": test_results,
            }
        )
        sub_path = os.path.join(WORKING_DIR, "submission_got_it5.csv")
        submission.to_csv(sub_path, index=False)

        print("Submission saved to:", sub_path)
        print("Used in training set:", X_train.shape[0], "elements")
        print("Used in validation set:", X_test.shape[0], "elements")
        print("Used in prediction set:", len(test_results), "elements")
        print("Used for train accuracy:", number_for_train_accuracy, "elements")
        print("Used for test accuracy:", number_for_test_accuracy, "elements")
        print(submission.head())
        print("Summary of predictions:")
        print(submission.Predicted.value_counts().head(25))

        return train_accuracy, test_accuracy, parameters




## === cell 14
_, _, parameters = model(
    x_train,
    y_train,
    x_dev,
    y_dev,
    x_test,
    None,
    None,
    None,
    None,
    learning_rate=0.0010,
    num_epochs=100,
    minibatch_size=64,
    print_cost=True,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3925766436.py in <cell line: 0>()
----> 1 _, _, parameters = model(
      2     x_train,
      3     y_train,
      4     x_dev,
      5     y_dev,

/tmp/ipykernel_11/174604817.py in model(X_train, Y_train, X_test, Y_test, X_test_test, X_test_test1, X_test_test2, X_test_test3, X_test_test4, learning_rate, num_epochs, minibatch_size, print_cost)
     25     X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
     26     parameters = initialize_parameters()
---> 27     Z3 = forward_propagation(X, parameters)
     28     cost = compute_cost(Z3, Y)
     29 

/tmp/ipykernel_11/3137748452.py in forward_propagation(X, parameters)
     26 
     27     P2_flat = tf.reshape(P2, [tf.shape(P2)[0], -1])
---> 28     Z3 = tf.keras.layers.Dense(23, activation=None)(P2_flat)
     29     return Z3
     30 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in _validate_shape(self, shape)
    207         shape = standardize_shape(shape)
    208         if None in shape:
--> 209             raise ValueError(
    210                 "Shapes used to initialize variables must be "
    211                 "fully-defined (no `None` dimensions). Received: "

ValueError: Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape=(None, 23) for variable path='dense/kernel'

## === cell 15
print("Working dir files:", os.listdir("../working")[:50])
print("Check submission exists:", os.path.exists("../working/submission_got_it5.csv"))
sub = pd.read_csv("../working/submission_got_it5.csv")
print(sub.head())
print(sub.columns)
print("Rows:", len(sub))

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/898600175.py in <cell line: 0>()
      1 print("Working dir files:", os.listdir("../working")[:50])
      2 print("Check submission exists:", os.path.exists("../working/submission_got_it5.csv"))
----> 3 sub = pd.read_csv("../working/submission_got_it5.csv")
      4 print(sub.head())
      5 print(sub.columns)

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

FileNotFoundError: [Errno 2] No such file or directory: '../working/submission_got_it5.csv'
