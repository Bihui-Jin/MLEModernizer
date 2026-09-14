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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

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

# 5. Target score

0.961

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, matplotlib.pyplot as plt, glob, os, tqdm

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/train.csv")
sample = pd.read_csv("../input/sample_submission.csv")
print(train.shape)
train_images = "../input/train/*"
test_images = "../input/test/*"
train.head()



## === cell 2
print(train.has_cactus.unique())



## === cell 3
train.has_cactus.hist()
print(train.has_cactus.value_counts())



## === cell 4
IMAGES = os.path.join(train_images, "*")
all_images = glob.glob(IMAGES)
plt.figure(figsize=(12, 10))
plt.subplot(1, 3, 1)
plt.imshow(plt.imread(all_images[1]))
plt.xticks([])
plt.yticks([])
plt.figure(figsize=(12, 10))
plt.subplot(1, 3, 1)
plt.imshow(plt.imread(all_images[234]))
plt.xticks([])
plt.yticks([])



## === cell 5
train_id = train["id"]
labels = train["has_cactus"]
test_id = sample["id"]



## === cell 6
img = plt.imread("../input/train/train/" + train["id"][0])
print(img.shape)



## === cell 7
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

x_train, x_dev, y_train, y_dev = train_test_split(
    train["id"], train["has_cactus"], test_size=0.1, random_state=32
)

x_train_arr = []
for img_name in tqdm.tqdm(x_train):
    img = plt.imread(f"../input/train/train/{img_name}")
    x_train_arr.append(img)
x_train_arr = np.array(x_train_arr)
print("x_train_arr.shape:", x_train_arr.shape)



## === cell 8
print(x_train.shape, x_dev.shape, y_train.shape, y_dev.shape)



## === cell 9
x_dev_arr = []
for img_name in tqdm.tqdm(x_dev):
    img = plt.imread(f"../input/train/train/{img_name}")
    x_dev_arr.append(img)
x_dev_arr = np.array(x_dev_arr)
print("x_dev_arr.shape:", x_dev_arr.shape)



## === cell 10
X_test = []
for img_name in tqdm.tqdm(sample["id"]):
    img = plt.imread(f"../input/test/test/{img_name}")
    X_test.append(img)
X_test = np.array(X_test)
print("X_test.shape:", X_test.shape)



## === cell 11
import tensorflow as tf

tf.compat.v1.disable_eager_execution()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 12
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name="X")
    Y = tf.compat.v1.placeholder(tf.float32, shape=(None, n_y), name="Y")
    return X, Y




## === cell 13
def initialize_parameters():
    tf.compat.v1.set_random_seed(1)
    initializer = tf.keras.initializers.GlorotUniform(seed=1)
    W1 = tf.compat.v1.get_variable("W1", [4, 4, 3, 8], initializer=initializer)
    W2 = tf.compat.v1.get_variable("W2", [2, 2, 8, 32], initializer=initializer)
    return {"W1": W1, "W2": W2}




## === cell 14
def forward_propagation(X, parameters):
    W1, W2 = parameters["W1"], parameters["W2"]
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
    P2_flat = tf.reshape(P2, [tf.shape(P2)[0], -1])
    dense_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )
    Z3 = dense_layer(P2_flat)
    return Z3




## === cell 15
def compute_cost(Z3, Y):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Z3)
    cost = tf.reduce_mean(loss)
    return cost




## === cell 16
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]
    np.random.seed(seed)
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    mini_batches = []
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size :]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    return mini_batches




## === cell 17
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
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(1)
    seed = 3
    m, n_H0, n_W0, n_C0 = X_train.shape
    n_y = Y_train.shape[1]

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.compat.v1.global_variables_initializer()

    costs = []
    with tf.compat.v1.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0.0
            num_minibatches = int(m / minibatch_size)
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch_X, minibatch_Y in minibatches:
                _, temp_cost = sess.run(
                    [optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y}
                )
                minibatch_cost += temp_cost / num_minibatches

            if print_cost and epoch % 5 == 0:
                print(f"Cost after epoch {epoch}: {minibatch_cost:.6f}")
            if print_cost:
                costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations")
        plt.title(f"Learning rate = {learning_rate}")
        plt.show()

        prob_op = tf.nn.sigmoid(Z3)
        predict_op = tf.round(prob_op)

        correct_prediction = tf.equal(predict_op, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        pred_probs = prob_op.eval({X: X_test_test})
        submission = pd.DataFrame(
            {"id": sample["id"], "has_cactus": pred_probs.reshape(-1)}
        )
        submission.to_csv("submission_set.csv", index=False)
        print("Submission file saved as submission_set.csv")
        print(submission.head())
        return train_accuracy, test_accuracy, parameters




## === cell 18
X_train = x_train_arr.astype("float32") / 255.0
X_dev = x_dev_arr.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

y_train = y_train.values.reshape(-1, 1).astype("float32")
y_dev = y_dev.values.reshape(-1, 1).astype("float32")

print(
    "Shapes -> X_train:",
    X_train.shape,
    "X_dev:",
    X_dev.shape,
    "y_train:",
    y_train.shape,
    "y_dev:",
    y_dev.shape,
)

_, _, parameters = model(
    X_train,
    y_train,
    X_dev,
    y_dev,
    X_test,
    learning_rate=0.0009,
    num_epochs=150,
    minibatch_size=64,
    print_cost=True,
)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3331173443.py in <cell line: 0>()
     18 )
     19 
---> 20 _, _, parameters = model(
     21     X_train,
     22     y_train,

/tmp/ipykernel_11/2471361638.py in model(X_train, Y_train, X_test, Y_test, X_test_test, learning_rate, num_epochs, minibatch_size, print_cost)
     18     X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
     19     parameters = initialize_parameters()
---> 20     Z3 = forward_propagation(X, parameters)
     21     cost = compute_cost(Z3, Y)
     22     optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)

/tmp/ipykernel_11/3994033841.py in forward_propagation(X, parameters)
     13         kernel_initializer=tf.keras.initializers.GlorotUniform(),
     14     )
---> 15     Z3 = dense_layer(P2_flat)
     16     return Z3
     17 

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

ValueError: Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape=(None, 1) for variable path='dense/kernel'
