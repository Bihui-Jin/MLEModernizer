# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

print(os.listdir("../input"))



## === cell 1
CANDIDATE_BASES = [
    "../input/aerial-cactus-identification",
    "../input",
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input",
]
BASE = None
for b in CANDIDATE_BASES:
    if (
        os.path.exists(os.path.join(b, "train.csv"))
        and os.path.exists(os.path.join(b, "train"))
        and os.path.exists(os.path.join(b, "test"))
    ):
        BASE = b
        break
if BASE is None:
    for b in CANDIDATE_BASES:
        if os.path.isdir(b):
            for d in os.listdir(b):
                p = os.path.join(b, d)
                if (
                    os.path.isdir(p)
                    and os.path.exists(os.path.join(p, "train.csv"))
                    and os.path.exists(os.path.join(p, "train"))
                    and os.path.exists(os.path.join(p, "test"))
                ):
                    BASE = p
                    break
        if BASE is not None:
            break

if BASE is None:
    raise FileNotFoundError(
        "Could not find dataset base folder containing train.csv, train/, and test/"
    )

TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_CSV = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")

print("Using BASE:", BASE)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_CSV:", SAMPLE_CSV)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)



## === cell 2
train = pd.read_csv(TRAIN_CSV)
sample = pd.read_csv(SAMPLE_CSV)
print(train.shape)

train_images = os.path.join(TRAIN_DIR, "*")
test_images = os.path.join(TEST_DIR, "*")

train.head()



## === cell 3
train.has_cactus.unique()



## === cell 4
train.has_cactus.hist()
train.has_cactus.value_counts()



## === cell 5
IMAGES = train_images
all_images = glob.glob(IMAGES)

if len(all_images) == 0:
    raise FileNotFoundError(f"No training images found with glob pattern: {IMAGES}")

idx1 = min(1, len(all_images) - 1)
idx2 = min(234, len(all_images) - 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(plt.imread(all_images[idx1]))
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(plt.imread(all_images[idx2]))
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()


## === cell 6
train_id = train["id"]
labels = train["has_cactus"]
test_id = sample["id"]



## === cell 7
import tqdm

img = plt.imread(os.path.join(TRAIN_DIR, train["id"][0]))
img.shape



## === cell 8
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

x_train, x_dev, y_train, y_dev = train_test_split(
    train["id"], train["has_cactus"], test_size=0.1, random_state=32
)

x_train_arr = []
for images in tqdm.tqdm(x_train):
    img = plt.imread(os.path.join(TRAIN_DIR, images))
    x_train_arr.append(img)

x_train_arr = np.array(x_train_arr)
print(x_train_arr.shape)



## === cell 9
print(x_train.shape)
print(x_dev.shape)
print(y_train.shape)
print(y_dev.shape)



## === cell 10
x_dev_arr = []
for images in tqdm.tqdm(x_dev):
    img = plt.imread(os.path.join(TRAIN_DIR, images))
    x_dev_arr.append(img)

x_dev_arr = np.array(x_dev_arr)
print(x_dev_arr.shape)



## === cell 11
X_test = []
for images in tqdm.tqdm(sample["id"]):
    img = plt.imread(os.path.join(TEST_DIR, images))
    X_test.append(img)

X_test = np.array(X_test)
print("X_test.shape:", X_test.shape)



## === cell 12
X_train = x_train_arr.astype("float32")
X_dev = x_dev_arr.astype("float32")
X_train = X_train / 255.0
X_dev = X_dev / 255.0
X_test = X_test.astype("float32") / 255.0

y_train = np.asarray(y_train)
y_dev = np.asarray(y_dev)

y_train = np.reshape(y_train, (-1, 1)).astype("float32")
y_dev = np.reshape(y_dev, (-1, 1)).astype("float32")

print("X_train.shape:", X_train.shape)
print("X_dev.shape:", X_dev.shape)
print("y_train.shape:", y_train.shape)
print("y_dev.shape:", y_dev.shape)



## === cell 13
import os
import sys
import importlib
import subprocess

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf
    from google.protobuf import __version__ as _pb_ver

    _major = int(_pb_ver.split(".")[0])
    if _major >= 4:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.*"]
        )
        importlib.invalidate_caches()
        importlib.reload(google.protobuf)
except Exception:
    pass

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tensorflow.python.framework import ops
import scipy
from scipy import ndimage
import math




## === cell 14
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    return X, Y




## === cell 15
def initialize_parameters():
    tf.set_random_seed(1)
    xavier_init = tf.keras.initializers.GlorotUniform(seed=0)
    W1 = tf.get_variable("W1", [4, 4, 3, 16], initializer=xavier_init)
    W2 = tf.get_variable("W2", [2, 2, 16, 64], initializer=xavier_init)
    parameters = {"W1": W1, "W2": W2}
    return parameters




## === cell 16
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 1, activation_fn=None)
    return Z3




## === cell 17
def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.losses.mean_squared_error(Y, Z3))
    return cost




## === cell 18
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation,]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :, :
        ]
        mini_batch_Y = shuffled_Y[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size,
        ]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[
            num_complete_minibatches * mini_batch_size : m, :, :, :
        ]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches




## === cell 19
def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    X_test_test,
    learning_rate=0.009,
    num_epochs=20,
    minibatch_size=64,
    print_cost=True,
):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0.0
            num_minibatches = (
                int(m / minibatch_size) if int(m / minibatch_size) > 0 else 1
            )
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch_X, minibatch_Y in minibatches:
                _, temp_cost = sess.run(
                    [optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y}
                )
                minibatch_cost += temp_cost / num_minibatches

            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost:
                costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per epoch)")
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        predict_op_round = tf.round(Z3)
        correct_prediction = tf.equal(predict_op_round, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        prob_op = tf.sigmoid(Z3)
        prediction_proba = prob_op.eval({X: X_test_test}).reshape(-1)

        submission = pd.DataFrame(
            {"id": sample["id"].values, "has_cactus": prediction_proba}
        )
        submission.to_csv("submission.csv", index=False)

        print("Prediction_test_test set:", prediction_proba.shape[0])
        print(submission.head())

        return train_accuracy, test_accuracy, parameters




## === cell 20
train_acc, dev_acc, params = model(X_train, y_train, X_dev, y_dev, X_test)



## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2823478854.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Train and write submission.csv[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mtrain_acc[0m[0;34m,[0m [0mdev_acc[0m[0;34m,[0m [0mparams[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0my_train[0m[0;34m,[0m [0mX_dev[0m[0;34m,[0m [0my_dev[0m[0;34m,[0m [0mX_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1492997302.py[0m in [0;36mmodel[0;34m(X_train, Y_train, X_test, Y_test, X_test_test, learning_rate, num_epochs, minibatch_size, print_cost)[0m
[1;32m     19[0m     [0mX[0m[0;34m,[0m [0mY[0m [0;34m=[0m [0mcreate_placeholders[0m[0;34m([0m[0mn_H0[0m[0;34m,[0m [0mn_W0[0m[0;34m,[0m [0mn_C0[0m[0;34m,[0m [0mn_y[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m     [0mparameters[0m [0;34m=[0m [0minitialize_parameters[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 21[0;31m     [0mZ3[0m [0;34m=[0m [0mforward_propagation[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mparameters[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m     [0mcost[0m [0;34m=[0m [0mcompute_cost[0m[0;34m([0m[0mZ3[0m[0;34m,[0m [0mY[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m     [0moptimizer[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mtrain[0m[0;34m.[0m[0mAdamOptimizer[0m[0;34m([0m[0mlearning_rate[0m[0;34m=[0m[0mlearning_rate[0m[0;34m)[0m[0;34m.[0m[0mminimize[0m[0;34m([0m[0mcost[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1889247424.py[0m in [0;36mforward_propagation[0;34m(X, parameters)[0m
[1;32m     11[0m     [0mP2[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mnn[0m[0;34m.[0m[0mmax_pool[0m[0;34m([0m[0mA2[0m[0;34m,[0m [0mksize[0m[0;34m=[0m[0;34m[[0m[0;36m1[0m[0;34m,[0m [0;36m4[0m[0;34m,[0m [0;36m4[0m[0;34m,[0m [0;36m1[0m[0;34m][0m[0;34m,[0m [0mstrides[0m[0;34m=[0m[0;34m[[0m[0;36m1[0m[0;34m,[0m [0;36m4[0m[0;34m,[0m [0;36m4[0m[0;34m,[0m [0;36m1[0m[0;34m][0m[0;34m,[0m [0mpadding[0m[0;34m=[0m[0;34m"SAME"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;34m[0m[0m
[0;32m---> 13[0;31m     [0mP2[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mcontrib[0m[0;34m.[0m[0mlayers[0m[0;34m.[0m[0mflatten[0m[0;34m([0m[0mP2[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m     [0mZ3[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mcontrib[0m[0;34m.[0m[0mlayers[0m[0;34m.[0m[0mfully_connected[0m[0;34m([0m[0mP2[0m[0;34m,[0m [0;36m1[0m[0;34m,[0m [0mactivation_fn[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m     [0;32mreturn[0m [0mZ3[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/module_wrapper.py[0m in [0;36m_getattr[0;34m(self, name)[0m
[1;32m    230[0m     """
[1;32m    231[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 232[0;31m       [0mattr[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_tfmw_wrapped_module[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    233[0m     [0;32mexcept[0m [0mAttributeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    234[0m     [0;31m# Placeholder for Google-internal contrib error[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: module 'tensorflow.compat.v1' has no attribute 'contrib'

## === cell 21
print(os.listdir("../working"))
if os.path.exists("submission.csv"):
    print("Wrote submission.csv with shape:", pd.read_csv("submission.csv").shape)
    print(pd.read_csv("submission.csv").head())
