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

3.6

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

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

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import math
from tensorflow.python.framework import ops
import seaborn as sns

print(os.listdir("../input"))


## === cell 1
def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 33
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


## === cell 2
root = '../input/train/Maize/3a6d4d007.png'
img = cv2.imread(root)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = cv2.resize(img,(128,128))
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

folders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
X = []
Y = []
names = {}
ptr = 0

for folder in folders:
    names[ptr] = folder
    files = os.listdir(os.path.join(root, folder))
    for file in files:
        image_path = os.path.join(root, folder, file)
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image_segmented = segment_plant(img)
        image_sharpen = sharpen_image(image_segmented)
        img = cv2.resize(image_sharpen, (128, 128))
        img = img / 255
        X.append(img)
        Y.append(ptr)
    ptr += 1

X = np.array(X)
Y = np.array(Y)
names


## === cell 4
def display_dataset(X,Y, h=128, w=128, rows=5, cols=2, display_labels=True):
    f, ax = plt.subplots(cols, rows)
    for i in range(rows):
        for j in range(cols):
            index=np.random.randint(0,X.shape[0])
            ax[j,i].imshow(X[index].reshape(h,w,3), cmap='binary')
            ax[j,i].set_title(Y[index])
    plt.xticks()
    plt.show()


## === cell 5
X = X.reshape(X.shape[0],-1)


## === cell 6
X.shape


## === cell 7
display_dataset(X, Y)


## === cell 8
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle=True, test_size=0.1)


## === cell 9
display_dataset(X_train, Y_train)


## === cell 10
display_dataset(X_test, Y_test)


## === cell 11
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')
    
    return X, Y


## === cell 12
def initialize_parameters():
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    W1 = tf.get_variable("W1", [50, 49152], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [50, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [15, 50],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [15, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [12, 15], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [12, 1], initializer=tf.zeros_initializer())
   
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


## === cell 13
def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2) # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3) # Z3 = np.dot(W3,Z2) + b3
    
    return Z3


## === cell 14

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    print(logits, labels)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost


## === cell 15
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


## === cell 16
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


## === cell 17
def predict(X, parameters, pred_val=1):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [49152, pred_val])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction


## === cell 18
def forward_propagation_for_predict(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3


## === cell 19
def model(train_X, train_Y, test_X, test_Y, learning_rate = 0.0001,
          num_epochs = 1000, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    """
    
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)      # to keep consistent results
    seed = 3                   # to keep consistent results
    (n_x, m) = train_X.shape   # (n_x: input size, m : number of examples in the train set)
    n_y = train_Y.shape[0]     # n_y : output size
    costs = []                 # To keep track of the cost
    
    X, Y = create_placeholders(n_x, n_y)
    
    parameters = initialize_parameters()
    
    Z3 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z3, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(train_X, train_Y, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                
                _ , minibatch_cost = sess.run([optimizer, cost], 
                                             feed_dict={X: minibatch_X, Y: minibatch_Y})

            epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: train_X, Y: train_Y}))
        print ("Test Accuracy:", accuracy.eval({X: test_X, Y: test_Y}))
        
        return parameters


## === cell 20
num_classes = int(max(np.max(Y_train), np.max(Y_test))) + 1

Y_train = convert_to_one_hot(Y_train, num_classes)
Y_test = convert_to_one_hot(Y_test, num_classes)
print(Y_train.shape, Y_test.shape)


## === cell 21
X_train=X_train.reshape(X_train.shape[0],-1).T
X_test=X_test.reshape(X_test.shape[0],-1).T
print(X_train.shape, X_test.shape)


## === cell 23
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
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.compat.v1.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = (
        train_X.shape
    )  # (n_x: input size, m : number of examples in the train set)
    n_y = train_Y.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.0  # Defines a cost related to an epoch
            num_minibatches = int(
                m / minibatch_size
            )  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(train_X, train_Y, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run(
                    [optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y}
                )

            epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per tens)")
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: train_X, Y: train_Y}))
        print("Test Accuracy:", accuracy.eval({X: test_X, Y: test_Y}))

        return parameters


## === cell 24
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

root = "../input/train/Maize/3a6d4d007.png"
imag = cv2.imread(root)
imag = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)
imag = cv2.resize(imag, (128, 128))
image_segmented = segment_plant(imag)
image_sharpen = sharpen_image(image_segmented)
imag = cv2.resize(image_sharpen, (128, 128))
imag = imag / 255

imb = imag.reshape(1, 128 * 128 * 3).T

_params = globals().get("parameters1", globals().get("parameters", None))
if _params is None:
    parameters = model(X_train, Y_train, X_test, Y_test)
    _params = parameters

my_image_prediction = predict(imb, _params)
plt.imshow(imb.reshape(128, 128, 3))
print(
    "Your algorithm predicts: y = " + str(names[int(np.squeeze(my_image_prediction))])
)


## --- ERROR in cell 24, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mInvalidArgumentError[0m                      Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/client/session.py[0m in [0;36m_do_call[0;34m(self, fn, *args)[0m
[1;32m   1406[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1407[0;31m       [0;32mreturn[0m [0mfn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1408[0m     [0;32mexcept[0m [0merrors[0m[0;34m.[0m[0mOpError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/client/session.py[0m in [0;36m_run_fn[0;34m(feed_dict, fetch_list, target_list, options, run_metadata)[0m
[1;32m   1389[0m       [0mself[0m[0;34m.[0m[0m_extend_graph[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1390[0;31m       return self._call_tf_sessionrun(options, feed_dict, fetch_list,
[0m[1;32m   1391[0m                                       target_list, run_metadata)

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/client/session.py[0m in [0;36m_call_tf_sessionrun[0;34m(self, options, feed_dict, fetch_list, target_list, run_metadata)[0m
[1;32m   1482[0m                           run_metadata):
[0;32m-> 1483[0;31m     return tf_session.TF_SessionRun_wrapper(self._session, options, feed_dict,
[0m[1;32m   1484[0m                                             [0mfetch_list[0m[0;34m,[0m [0mtarget_list[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mInvalidArgumentError[0m: logits and labels must be broadcastable: logits_size=[32,12] labels_size=[32,13]
	 [[{{node softmax_cross_entropy_with_logits}}]]

During handling of the above exception, another exception occurred:

[0;31mInvalidArgumentError[0m                      Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3985883161.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     44[0m [0m_params[0m [0;34m=[0m [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mget[0m[0;34m([0m[0;34m"parameters1"[0m[0;34m,[0m [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mget[0m[0;34m([0m[0;34m"parameters"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     45[0m [0;32mif[0m [0m_params[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 46[0;31m     [0mparameters[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0mY_train[0m[0;34m,[0m [0mX_test[0m[0;34m,[0m [0mY_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     47[0m     [0m_params[0m [0;34m=[0m [0mparameters[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/4201288969.py[0m in [0;36mmodel[0;34m(train_X, train_Y, test_X, test_Y, learning_rate, num_epochs, minibatch_size, print_cost)[0m
[1;32m     52[0m                 [0;34m([0m[0mminibatch_X[0m[0;34m,[0m [0mminibatch_Y[0m[0;34m)[0m [0;34m=[0m [0mminibatch[0m[0;34m[0m[0;34m[0m[0m
[1;32m     53[0m [0;34m[0m[0m
[0;32m---> 54[0;31m                 _, minibatch_cost = sess.run(
[0m[1;32m     55[0m                     [0;34m[[0m[0moptimizer[0m[0;34m,[0m [0mcost[0m[0;34m][0m[0;34m,[0m [0mfeed_dict[0m[0;34m=[0m[0;34m{[0m[0mX[0m[0;34m:[0m [0mminibatch_X[0m[0;34m,[0m [0mY[0m[0;34m:[0m [0mminibatch_Y[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[1;32m     56[0m                 )

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/client/session.py[0m in [0;36mrun[0;34m(self, fetches, feed_dict, options, run_metadata)[0m
[1;32m    975[0m [0;34m[0m[0m
[1;32m    976[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 977[0;31m       result = self._run(None, fetches, feed_dict, options_ptr,
[0m[1;32m    978[0m                          run_metadata_ptr)
[1;32m    979[0m       [0;32mif[0m [0mrun_metadata[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/client/session.py[0m in [0;36m_run[0;34m(self, handle, fetches, feed_dict, options, run_metadata)[0m
[1;32m   1218[0m     [0;31m# or if the call is a partial run that specifies feeds.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1219[0m     [0;32mif[0m [0mfinal_fetches[0m [0;32mor[0m [0mfinal_targets[0m [0;32mor[0m [0;34m([0m[0mhandle[0m [0;32mand[0m [0mfeed_dict_tensor[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1220[0;31m       results = self._do_run(handle, final_targets, final_fetches,
[0m[1;32m   1221[0m                              feed_dict_tensor, options, run_metadata)
[1;32m   1222[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/client/session.py[0m in [0;36m_do_run[0;34m(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)[0m
[1;32m   1398[0m [0;34m[0m[0m
[1;32m   1399[0m     [0;32mif[0m [0mhandle[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1400[0;31m       return self._do_call(_run_fn, feeds, fetches, targets, options,
[0m[1;32m   1401[0m                            run_metadata)
[1;32m   1402[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/client/session.py[0m in [0;36m_do_call[0;34m(self, fn, *args)[0m
[1;32m   1424[0m                     [0;34m'\nsession_config.graph_options.rewrite_options.'[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1425[0m                     'disable_meta_optimizer = True')
[0;32m-> 1426[0;31m       [0;32mraise[0m [0mtype[0m[0;34m([0m[0me[0m[0;34m)[0m[0;34m([0m[0mnode_def[0m[0;34m,[0m [0mop[0m[0;34m,[0m [0mmessage[0m[0;34m)[0m  [0;31m# pylint: disable=no-value-for-parameter[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1427[0m [0;34m[0m[0m
[1;32m   1428[0m   [0;32mdef[0m [0m_extend_graph[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mInvalidArgumentError[0m: Graph execution error:

Detected at node 'softmax_cross_entropy_with_logits' defined at (most recent call last):
    File "<frozen runpy>", line 198, in _run_module_as_main
    File "<frozen runpy>", line 88, in _run_code
    File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>
    File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance
    File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start
    File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start
    File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever
    File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once
    File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run
    File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue
    File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one
    File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell
    File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request
    File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute
    File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell
    File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell
    File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell
    File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner
    File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async
    File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes
    File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
    File "/tmp/ipykernel_11/3985883161.py", line 46, in <cell line: 0>
    File "/tmp/ipykernel_11/4201288969.py", line 31, in model
    File "/tmp/ipykernel_11/368211685.py", line 8, in compute_cost
Node: 'softmax_cross_entropy_with_logits'
logits and labels must be broadcastable: logits_size=[32,12] labels_size=[32,13]
	 [[{{node softmax_cross_entropy_with_logits}}]]

Original stack trace for 'softmax_cross_entropy_with_logits':
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>
  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance
  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start
  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start
  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever
  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once
  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run
  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue
  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one
  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell
  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request
  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute
  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell
  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell
  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell
  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner
  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async
  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes
  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
  File "/tmp/ipykernel_11/3985883161.py", line 46, in <cell line: 0>
  File "/tmp/ipykernel_11/4201288969.py", line 31, in model
  File "/tmp/ipykernel_11/368211685.py", line 8, in compute_cost
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py", line 150, in error_handler
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/dispatch.py", line 1260, in op_dispatch_handler
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/nn_ops.py", line 4048, in softmax_cross_entropy_with_logits_v2
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py", line 150, in error_handler
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/dispatch.py", line 1260, in op_dispatch_handler
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py", line 588, in new_func
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/nn_ops.py", line 4154, in softmax_cross_entropy_with_logits_v2_helper
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_nn_ops.py", line 12082, in softmax_cross_entropy_with_logits
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py", line 796, in _apply_op_helper
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 2701, in _create_op_internal
  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 1196, in from_node_def


## === cell 25
root = '../input/test/'
files = os.listdir(root)
x=[]
y=[]
for file in files:
    y.append(file)
    imag = cv2.imread(os.path.join(root,file))
    imag = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)
    imag = cv2.resize(imag,(128,128))
    image_segmented = segment_plant(imag)
    imag = sharpen_image(image_segmented)
    imag = imag/255
    imb = imag.reshape(-1)
    x.append(imb)
x=np.array(x)
y=np.array(y)
x = x.reshape(x.shape[0],-1).T
print(x.shape, y.shape)
