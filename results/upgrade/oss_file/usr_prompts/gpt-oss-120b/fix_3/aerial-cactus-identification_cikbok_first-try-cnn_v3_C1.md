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

geopandas==0.14.4
google-api-python-client==2.177.0
imageio==2.37.0
imageio-ffmpeg==0.6.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

0.9968

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
import cv2
import random
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
print("Input folder contents:", os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



## === cell 2
train_dir = "../input/train/train"
test_dir = "../input/test/test"
train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/sample_submission.csv")



## === cell 3
print("Training data preview:")
print(train.head())



## === cell 4
print(f"out dataset has {train.shape[0]} rows and {train.shape[1]} columns")



## === cell 5
print("Class distribution (train):")
print(train["has_cactus"].value_counts(normalize=True))



## === cell 6
print(f"The number of rows in test set is {len(os.listdir(test_dir))}")



## === cell 7
plt.imshow(keras_image.load_img(os.path.join(train_dir, train.iloc[0, 0])))
plt.axis("off")
plt.show()




## === cell 8
def prepare_data(dataframe, m, direc):
    """Load images, resize to 32x32, preprocess and return as a NumPy array."""
    X = np.zeros((m, 32, 32, 3), dtype=np.float32)
    for i, fid in enumerate(dataframe["id"]):
        img_path = os.path.join(direc, fid)
        img = keras_image.load_img(img_path, target_size=(32, 32))
        x = keras_image.img_to_array(img)
        x = preprocess_input(x) / 255.0
        X[i] = x
    return X




## === cell 9
print("Preparing training data …")
X_data = prepare_data(train, train.shape[0], train_dir)
print("Preparing test data …")
X_test = prepare_data(df_test, len(df_test), test_dir)



## === cell 10
y = train["has_cactus"].values.astype(np.int32)
train_x, val_x, train_y, val_y = train_test_split(
    X_data, y, test_size=0.2, random_state=42, stratify=y
)



## === cell 11
print("Validation set size:", val_x.shape[0])



## === cell 12
height, width, channels = 32, 32, 3
n_outputs = 2

tf.reset_default_graph()
with tf.name_scope("inputs"):
    X_ph = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
    y_ph = tf.placeholder(tf.int32, shape=[None], name="y")
    training_ph = tf.placeholder_with_default(False, shape=[], name="training")

conv3 = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    strides=1,
    padding="same",
    activation=tf.nn.relu,
    name="conv3",
)(X_ph)

conv1 = tf.keras.layers.Conv2D(
    filters=128,
    kernel_size=3,
    strides=1,
    padding="same",
    activation=tf.nn.relu,
    name="conv1",
)(conv3)

conv2 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    strides=1,
    padding="same",
    activation=tf.nn.relu,
    name="conv2",
)(conv1)

pool3 = tf.keras.layers.MaxPooling2D(
    pool_size=2, strides=2, padding="valid", name="pool3"
)(conv2)
pool3_flat = tf.keras.layers.Flatten(name="flatten")(pool3)
pool3_drop = tf.keras.layers.Dropout(rate=0.25, name="pool3_drop")(
    pool3_flat, training=training_ph
)

fc1 = tf.keras.layers.Dense(128, activation=tf.nn.relu, name="fc1")(pool3_drop)
fc1_drop = tf.keras.layers.Dropout(rate=0.5, name="fc1_drop")(fc1, training=training_ph)

logits = tf.keras.layers.Dense(n_outputs, name="output")(fc1_drop)
Y_proba = tf.nn.softmax(logits, name="Y_proba")
y_pred = tf.argmax(Y_proba, axis=1, output_type=tf.int32, name="y_pred")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y_ph
    )
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y_ph, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {v.op.name: v for v in gvars}


def restore_model_params(param_dict):
    assign_ops = []
    feed_dict = {}
    for name, var in param_dict.items():
        placeholder = tf.placeholder(var.dtype, shape=var.shape)
        assign_ops.append(tf.assign(var, placeholder))
        feed_dict[placeholder] = tf.get_default_session().run(var)
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


def shuffle_batch(X, y, batch_size):
    indices = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(indices, n_batches):
        yield X[batch_idx], y[batch_idx]




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_55/13590291.py in <cell line: 0>()
     42 )(conv2)
     43 pool3_flat = tf.keras.layers.Flatten(name="flatten")(pool3)
---> 44 pool3_drop = tf.keras.layers.Dropout(rate=0.25, name="pool3_drop")(
     45     pool3_flat, training=training_ph
     46 )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in _disallow(self, task)
    301 
    302   def _disallow(self, task):
--> 303     raise errors.OperatorNotAllowedInGraphError(
    304         f"{task} is not allowed."
    305         " You can attempt the following resolutions to the problem:"

OperatorNotAllowedInGraphError: Exception encountered when calling Dropout.call().

Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.

Arguments received by Dropout.call():
  • inputs=tf.Tensor(shape=(None, 16384), dtype=float32)
  • training=tf.Tensor(shape=(), dtype=bool)

## === cell 13
n_epochs = 20
batch_size = 100
iteration = 0
best_loss_val = np.inf
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_params = None

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(train_x, train_y, batch_size):
            iteration += 1
            sess.run(
                training_op, feed_dict={X_ph: X_batch, y_ph: y_batch, training_ph: True}
            )
            if iteration % check_interval == 0:
                val_loss = sess.run(loss, feed_dict={X_ph: val_x, y_ph: val_y})
                if val_loss < best_loss_val:
                    best_loss_val = val_loss
                    checks_since_last_progress = 0
                    best_params = get_model_params()
                else:
                    checks_since_last_progress += 1
        acc_batch = sess.run(accuracy, feed_dict={X_ph: X_batch, y_ph: y_batch})
        acc_val = sess.run(accuracy, feed_dict={X_ph: val_x, y_ph: val_y})
        print(
            f"Epoch {epoch}, last batch acc: {acc_batch*100:.2f}%, "
            f"val acc: {acc_val*100:.2f}%, best val loss: {best_loss_val:.6f}"
        )
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping triggered.")
            break

    if best_params is not None:
        restore_model_params(best_params)

    test_proba = sess.run(Y_proba, feed_dict={X_ph: X_test})
    prediction = test_proba[:, 1]  # probability of class 1 (has_cactus)
    saver.save(sess, "./cactus_model")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/106308927.py in <cell line: 0>()
      9 
     10 with tf.Session() as sess:
---> 11     sess.run(init)
     12     for epoch in range(n_epochs):
     13         for X_batch, y_batch in shuffle_batch(train_x, train_y, batch_size):

NameError: name 'init' is not defined

## === cell 14
submission = pd.DataFrame({"id": df_test["id"], "has_cactus": prediction})
submission_path = "sample_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4278000570.py in <cell line: 0>()
      1 # Create submission file
----> 2 submission = pd.DataFrame({"id": df_test["id"], "has_cactus": prediction})
      3 submission_path = "sample_submission.csv"
      4 submission.to_csv(submission_path, index=False)
      5 print(f"Submission saved to {submission_path}, shape: {submission.shape}")

NameError: name 'prediction' is not defined
