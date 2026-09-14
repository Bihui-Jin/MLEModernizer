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


MAX_TRAIN_IMAGES = 20000  # minimal pragmatic cap for runtime
MAX_TEST_IMAGES = len(sample_submission)  # must predict all for a valid submission

train_subset = train.iloc[:MAX_TRAIN_IMAGES].copy().reset_index(drop=True)

x_trains1 = np.zeros((len(train_subset), 32, 32, 3), dtype=np.uint8)
y_trains1 = train_subset["category_id"].values.astype(np.int64)

print("Loading train images:", len(train_subset))
for i, fn in enumerate(
    tqdm.tqdm(train_subset["file_name"].values, total=len(train_subset))
):
    x_trains1[i] = _read_resize_rgb(os.path.join(TRAIN_IMAGES_DIR, fn), size=(32, 32))

num_classes = 23
y_trains1_oh = np.eye(num_classes, dtype=np.float32)[y_trains1]

test_join = sample_submission[["Id"]].merge(
    test[["id", "file_name"]], left_on="Id", right_on="id", how="left"
)
if test_join["file_name"].isna().any():
    missing = int(test_join["file_name"].isna().sum())
    raise RuntimeError(f"Could not map {missing} test Ids to file_name from test.csv")

x_test_arr = np.zeros((len(test_join), 32, 32, 3), dtype=np.uint8)
print("Loading test images:", len(test_join))
for i, fn in enumerate(tqdm.tqdm(test_join["file_name"].values, total=len(test_join))):
    x_test_arr[i] = _read_resize_rgb(os.path.join(TEST_IMAGES_DIR, fn), size=(32, 32))

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

print("GENERATE Testsets: ")
print("----------------- ")

x_testa, x_testb = train_test_split(x_test, test_size=0.5, random_state=32)
x_test1, x_test2 = train_test_split(x_testa, test_size=0.5, random_state=32)
x_test3, x_test4 = train_test_split(x_testb, test_size=0.5, random_state=32)

print("x_test1.shape:", x_test1.shape)
print("x_test2.shape:", x_test2.shape)
print("x_test3.shape:", x_test3.shape)
print("x_test4.shape:", x_test4.shape)
print(
    "Total test parts:",
    x_test1.shape[0] + x_test2.shape[0] + x_test3.shape[0] + x_test4.shape[0],
)



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

    with tf1.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0.0
            num_minibatches = (
                int(m / minibatch_size) if int(m / minibatch_size) > 0 else 1
            )
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
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

        test_results = []
        prediction1 = predict_op.eval({X: X_test_test1})
        test_results.extend(prediction1)

        prediction2 = predict_op.eval({X: X_test_test2})
        test_results.extend(prediction2)

        prediction3 = predict_op.eval({X: X_test_test3})
        test_results.extend(prediction3)

        prediction4 = predict_op.eval({X: X_test_test4})
        test_results.extend(prediction4)

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



## === cell 13
print(os.listdir("../working"))
print("Wrote submission:", os.path.abspath("submission_got_it10.csv"))
print(pd.read_csv("submission_got_it10.csv").head())
