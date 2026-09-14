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

0.0727196069603995

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

INPUT_ROOT = "../input/iwildcam-2019-fgvc6"
print("Listing ../input:", os.listdir("../input"))
print("Listing competition root:", os.listdir(INPUT_ROOT))



## === cell 1
train = pd.read_csv(f"{INPUT_ROOT}/train.csv")
test = pd.read_csv(f"{INPUT_ROOT}/test.csv")
sample_submission = pd.read_csv(f"{INPUT_ROOT}/sample_submission.csv")

print("train.shape:", train.shape)
print("test.shape:", test.shape)
print("sample_submission.shape:", sample_submission.shape)
print("sample_submission columns:", sample_submission.columns.tolist())



## === cell 2
import glob
import math
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("TF_INTRA_OP", "2"))
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get("TF_INTER_OP", "2"))
    )
except Exception as _e:
    pass

tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_id = train["file_name"]
labels = train["category_id"]
test_id = sample_submission["Id"]

img_path0 = os.path.join(INPUT_ROOT, "train_images", train["file_name"].iloc[0])
img0 = plt.imread(img_path0)
print("Example image path:", img_path0)
print("Example image shape:", getattr(img0, "shape", None))



## === cell 4
from concurrent.futures import ThreadPoolExecutor


def _read_resize_rgb(path, target_size):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)
    return im


def load_images_to_array(
    df, images_dir, filename_col="file_name", target_size=(32, 32), max_images=None
):
    n = len(df) if max_images is None else min(len(df), max_images)
    X = np.zeros((n, target_size[0], target_size[1], 3), dtype=np.uint8)

    fns = df[filename_col].values[:n]
    paths = [os.path.join(images_dir, fn) for fn in fns]

    max_workers = int(
        os.environ.get("IMG_WORKERS", str(min(16, (os.cpu_count() or 4) * 2)))
    )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, im in enumerate(
            ex.map(lambda p: _read_resize_rgb(p, target_size), paths, chunksize=256)
        ):
            if im is not None:
                X[i] = im
            if (i + 1) % 2000 == 0:
                print(f"Loaded {i+1}/{n} images from {images_dir}")
    return X


def one_hot(y, num_classes=23):
    y = np.asarray(y, dtype=np.int64)
    Y = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


TRAIN_MAX = int(os.environ.get("TRAIN_MAX_IMAGES", "25000"))
TEST_MAX = None  # use all test rows from sample_submission/test.csv (16877)

train_images_dir = os.path.join(INPUT_ROOT, "train_images")
test_images_dir = os.path.join(INPUT_ROOT, "test_images")

train_small = train.iloc[:TRAIN_MAX].reset_index(drop=True)
print("Using TRAIN_MAX images:", len(train_small))

x_trains1 = load_images_to_array(
    train_small, train_images_dir, filename_col="file_name", target_size=(32, 32)
)
y_trains1 = one_hot(train_small["category_id"].values, num_classes=23)

test_meta = test.rename(columns={"id": "Id"})
test_join = sample_submission[["Id"]].merge(
    test_meta[["Id", "file_name"]], on="Id", how="left"
)
if test_join["file_name"].isna().any():
    print(
        "Warning: some test file_names missing after merge; falling back to test.csv order"
    )
    test_join = test_meta[["Id", "file_name"]].copy()

x_test_arr = load_images_to_array(
    test_join,
    test_images_dir,
    filename_col="file_name",
    target_size=(32, 32),
    max_images=TEST_MAX,
)

print("x_trains1.shape:", x_trains1.shape)
print("y_trains1.shape:", y_trains1.shape)
print("x_test_arr shape:", x_test_arr.shape)
print(x_test_arr.shape[0], "test samples")



## === cell 5
x_test = x_test_arr.astype(np.float32, copy=False) / 255.0
print("x_test.shape:", x_test.shape)

x_trains1 = x_trains1.astype(np.float32, copy=False) / 255.0
print("x_trains1 scaled dtype:", x_trains1.dtype)



## === cell 6
print("Prepared training and test arrays.")
print("x_trains1.shape:", x_trains1.shape)
print("y_trains1.shape:", y_trains1.shape)



## === cell 7
x_train, x_dev, y_train, y_dev = train_test_split(
    x_trains1, y_trains1, test_size=0.050, random_state=32
)

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_dev.shape:", x_dev.shape)
print("y_dev.shape:", y_dev.shape)

print("GENERATE Testsets: ")
print("----------------- ")

x_testa, x_testb = train_test_split(x_test, test_size=0.5, random_state=123)
x_test1, x_test2 = train_test_split(x_testa, test_size=0.5, random_state=123)
x_test3, x_test4 = train_test_split(x_testb, test_size=0.5, random_state=123)

print("x_test1.shape:", x_test1.shape)
print("x_test2.shape:", x_test2.shape)
print("x_test3.shape:", x_test3.shape)
print("x_test4.shape:", x_test4.shape)



## === cell 8
pass




## === cell 9
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf1.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf1.placeholder(tf.float32, shape=(None, n_y))
    return X, Y


def initialize_parameters():
    tf1.set_random_seed(1)

    glorot = tf.keras.initializers.GlorotUniform(seed=0)

    W1 = tf1.get_variable("W1", [4, 4, 3, 32], initializer=glorot)
    W2 = tf1.get_variable("W2", [2, 2, 32, 64], initializer=glorot)

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

    P2 = tf.keras.layers.Flatten()(P2)
    Z3 = tf.keras.layers.Dense(23, activation=None)(P2)
    return Z3


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf1.losses.mean_squared_error(labels=Y, predictions=Z3))
    return cost




## === cell 10
def random_mini_batch_indices(m, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    perm = np.random.permutation(m)
    for start in range(0, m, mini_batch_size):
        yield perm[start : start + mini_batch_size]




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
        intra_op_parallelism_threads=int(os.environ.get("TF1_INTRA_OP", "2")),
        inter_op_parallelism_threads=int(os.environ.get("TF1_INTER_OP", "2")),
        allow_soft_placement=True,
    )

    with tf1.Session(config=config) as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0.0
            num_minibatches = int(m / minibatch_size) if m >= minibatch_size else 1
            seed = seed + 1

            for batch_idx in random_mini_batch_indices(m, minibatch_size, seed):
                minibatch_X = X_train[batch_idx, :, :, :]
                minibatch_Y = Y_train[batch_idx, :]
                _, temp_cost = sess.run(
                    [optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y}
                )
                minibatch_cost += temp_cost / num_minibatches

            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost and epoch % 1 == 0:
                costs.append(minibatch_cost)

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

        if len(test_results) != sample_submission.shape[0]:
            print(
                "Warning: prediction length mismatch:",
                len(test_results),
                "vs",
                sample_submission.shape[0],
            )
            if len(test_results) < sample_submission.shape[0]:
                test_results = list(test_results) + [0] * (
                    sample_submission.shape[0] - len(test_results)
                )
            else:
                test_results = list(test_results)[: sample_submission.shape[0]]

        submission = pd.DataFrame(
            {
                "Id": sample_submission["Id"].values,
                "Predicted": np.array(test_results, dtype=np.int64),
            }
        )
        submission.to_csv("submission_got_it11.csv", index=False)

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
    num_epochs=130,
    minibatch_size=64,
    print_cost=True,
)

print("Saved submission to: submission_got_it11.csv")
print("Submission preview:")
print(pd.read_csv("submission_got_it11.csv").head())



## === cell 13
print("Files in ../working:", os.listdir("../working"))
if os.path.exists("submission_got_it11.csv"):
    print(
        "submission_got_it11.csv size (bytes):",
        os.path.getsize("submission_got_it11.csv"),
    )

## --- ERROR in outputing the csv:
Invalid submission: Submission must have 'Category' column
