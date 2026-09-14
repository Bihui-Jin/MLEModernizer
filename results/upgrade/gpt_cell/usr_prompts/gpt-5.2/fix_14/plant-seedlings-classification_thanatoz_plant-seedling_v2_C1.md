# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401

    try:
        from google.protobuf import __version__ as _pb_ver
    except Exception:
        _pb_ver = None

    def _pb_major(ver):
        try:
            return int(str(ver).split(".")[0])
        except Exception:
            return None

    if _pb_ver is None or (_pb_major(_pb_ver) is not None and _pb_major(_pb_ver) >= 4):
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "protobuf<4"]
        )
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
except Exception:
    pass

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import math
from tensorflow.python.framework import ops


os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(1)
try:
    tf.compat.v1.set_random_seed(1)
except Exception:
    tf.set_random_seed(1)

print(os.listdir("../input"))




## === cell 1
def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 33
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp




## === cell 2
if False:
    root = "../input/train/Maize/3a6d4d007.png"
    img = cv2.imread(root)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))
    image_segmented = segment_plant(img)
    image_sharpen = sharpen_image(image_segmented)
    plt.imshow(image_sharpen)



## === cell 3
root_candidates = [
    "../input/plant-seedlings-classification/train",
    "../input/plant-seedlings-classification/plant-seedlings-classification/train",
    "../input/train",
]
root = next((p for p in root_candidates if os.path.isdir(p)), root_candidates[0])

folders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

names = {i: folder for i, folder in enumerate(folders)}

paths = []
labels = []
for ptr, folder in enumerate(folders):
    fdir = os.path.join(root, folder)
    for ent in os.scandir(fdir):
        if ent.is_file():
            paths.append(ent.path)
            labels.append(ptr)

N = len(paths)
X = np.empty((N, 128, 128, 3), dtype=np.float32)
Y = np.asarray(labels, dtype=np.int64)

for i, p in enumerate(paths):
    img = cv2.imread(p)
    if img is None:
        X[i] = 0.0
        Y[i] = -1
        continue
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image_segmented = segment_plant(img)
    image_sharpen = sharpen_image(image_segmented)
    img = cv2.resize(image_sharpen, (128, 128), interpolation=cv2.INTER_LINEAR)
    X[i] = img * (1.0 / 255.0)

valid = Y >= 0
X = X[valid]
Y = Y[valid]

print("Loaded:", X.shape, Y.shape)
names




## === cell 4
def display_dataset(X, Y, h=128, w=128, rows=5, cols=2, display_labels=True):
    f, ax = plt.subplots(cols, rows)
    for i in range(rows):
        for j in range(cols):
            index = np.random.randint(0, X.shape[0])
            ax[j, i].imshow(X[index].reshape(h, w, 3), cmap="binary")
            ax[j, i].set_title(Y[index])
    plt.xticks()
    plt.show()




## === cell 5
X = X.reshape(X.shape[0], -1)



## === cell 6
X.shape



## === cell 7
if False:
    display_dataset(X, Y)



## === cell 8
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, shuffle=True, test_size=0.1, random_state=1
)



## === cell 9
if False:
    display_dataset(X_train, Y_train)



## === cell 10
if False:
    display_dataset(X_test, Y_test)



## === cell 11
if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
    tf.compat.v1.disable_eager_execution()
    tf.placeholder = tf.compat.v1.placeholder
    tf.Session = tf.compat.v1.Session
    tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
    tf.set_random_seed = tf.compat.v1.set_random_seed
    tf.get_variable = tf.compat.v1.get_variable
    tf.reset_default_graph = tf.compat.v1.reset_default_graph
    tf.train = tf.compat.v1.train

if not hasattr(tf, "contrib"):

    class _Contrib(object):
        pass

    tf.contrib = _Contrib()

if not hasattr(tf.contrib, "layers"):

    class _ContribLayers(object):
        pass

    tf.contrib.layers = _ContribLayers()

if not hasattr(tf.contrib.layers, "xavier_initializer"):

    def _xavier_initializer(seed=1):
        return tf.compat.v1.keras.initializers.glorot_uniform(seed=seed)

    tf.contrib.layers.xavier_initializer = _xavier_initializer




## === cell 12
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None], name="X")
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name="Y")
    return X, Y




## === cell 13
def initialize_parameters(n_classes=12):
    tf.set_random_seed(1)

    W1 = tf.get_variable(
        "W1", [50, 49152], initializer=tf.contrib.layers.xavier_initializer(seed=1)
    )
    b1 = tf.get_variable("b1", [50, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable(
        "W2", [15, 50], initializer=tf.contrib.layers.xavier_initializer(seed=1)
    )
    b2 = tf.get_variable("b2", [15, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable(
        "W3", [n_classes, 15], initializer=tf.contrib.layers.xavier_initializer(seed=1)
    )
    b3 = tf.get_variable("b3", [n_classes, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return parameters




## === cell 14
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3




## === cell 15
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    return cost




## === cell 16
def random_mini_batches_indices(m, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    permutation = np.random.permutation(m)

    num_complete_minibatches = m // mini_batch_size
    for k in range(num_complete_minibatches):
        yield permutation[k * mini_batch_size : (k + 1) * mini_batch_size]
    if m % mini_batch_size != 0:
        yield permutation[num_complete_minibatches * mini_batch_size : m]




## === cell 17
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y




## === cell 18
def forward_propagation_for_predict(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3




## === cell 19
def predict_batch(X_np, parameters_np):
    g = tf.Graph()
    with g.as_default():
        W1 = tf.constant(parameters_np["W1"])
        b1 = tf.constant(parameters_np["b1"])
        W2 = tf.constant(parameters_np["W2"])
        b2 = tf.constant(parameters_np["b2"])
        W3 = tf.constant(parameters_np["W3"])
        b3 = tf.constant(parameters_np["b3"])
        params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

        x = tf.placeholder(tf.float32, [49152, None], name="x")
        z3 = forward_propagation_for_predict(x, params)
        p = tf.argmax(z3, axis=0)

        with tf.Session(graph=g) as sess:
            pred = sess.run(p, feed_dict={x: X_np})
    return pred




## === cell 20
def model(
    train_X,
    train_Y,
    test_X,
    test_Y,
    learning_rate=0.0001,
    num_epochs=1000,
    minibatch_size=32,
    print_cost=True,
):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = train_X.shape
    n_y = train_Y.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters(n_classes=n_y)

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1

            for idx in random_mini_batches_indices(m, minibatch_size, seed):
                minibatch_X = train_X[:, idx]
                minibatch_Y = train_Y[:, idx]
                _, minibatch_cost = sess.run(
                    [optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y}
                )

            epoch_cost += minibatch_cost / max(1, num_minibatches)

            if print_cost and epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

        if False:
            plt.plot(np.squeeze(costs))
            plt.ylabel("cost")
            plt.xlabel("iterations (per tens)")
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        parameters_np = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3, axis=0), tf.argmax(Y, axis=0))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: train_X, Y: train_Y}))
        print("Test Accuracy:", accuracy.eval({X: test_X, Y: test_Y}))

    return parameters_np




## === cell 21
num_classes = int(max(np.max(Y_train), np.max(Y_test))) + 1
Y_train = convert_to_one_hot(Y_train, num_classes)
Y_test = convert_to_one_hot(Y_test, num_classes)
print(Y_train.shape, Y_test.shape)



## === cell 22
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T
print(X_train.shape, X_test.shape)



## === cell 23
parameters1 = model(X_train, Y_train, X_test, Y_test)



## === cell 24
if False:
    root = "../input/train/Maize/3a6d4d007.png"
    imag = cv2.imread(root)
    imag = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)
    imag = cv2.resize(imag, (128, 128))
    image_segmented = segment_plant(imag)
    image_sharpen = sharpen_image(image_segmented)
    imag = cv2.resize(image_sharpen, (128, 128))
    imag = imag / 255.0
    imb = imag.reshape(1, 128 * 128 * 3).T
    my_image_prediction = predict_batch(imb, parameters1)
    plt.imshow(imb.reshape(128, 128, 3))
    print(
        "Your algorithm predicts: y = "
        + str(names[int(np.squeeze(my_image_prediction))])
    )



## === cell 25
root = "../input/test/"
test_files = sorted([ent.name for ent in os.scandir(root) if ent.is_file()])
n_test = len(test_files)

x = np.empty((n_test, 49152), dtype=np.float32)
y = np.asarray(test_files)

for i, file in enumerate(test_files):
    imag = cv2.imread(os.path.join(root, file))
    imag = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)
    imag = cv2.resize(imag, (128, 128), interpolation=cv2.INTER_LINEAR)
    image_segmented = segment_plant(imag)
    imag = sharpen_image(image_segmented)
    imag = imag * (1.0 / 255.0)
    x[i] = imag.reshape(-1)

x = x.T
print(x.shape, y.shape)



## === cell 26
preda = predict_batch(x, parameters1)



## === cell 27
import pickle

pickle.dump(parameters1, open("parameters_pickle.p", "wb"))
pickle.dump(names, open("names_pickle.p", "wb"))



## === cell 28
put = [names[int(i)] for i in preda]
put[:5]



## === cell 29
res = pd.DataFrame({"file": y, "species": put})
res.head()



## === cell 30
res.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", res.shape)
