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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



## === cell 1
import numpy as np
import pandas as pd
import imageio as im
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import gc
import glob
import os
import cv2
import random



## === cell 2
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os, cv2
from IPython.display import Image
from keras.preprocessing import image
from keras import optimizers
from keras import layers, models
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir("../input"))

import numpy as np



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_dir = "../input/train/train"
test_dir = "../input/test/test"
train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/sample_submission.csv")



## === cell 4
train.head(5)



## === cell 5
print("out dataset has {} rows and {} columns".format(train.shape[0], train.shape[1]))



## === cell 6
train["has_cactus"].value_counts(normalize=True)



## === cell 7
print("The number of rows in test set is %d" % (len(os.listdir("../input/test/test"))))



## === cell 8
Image(os.path.join("../input/train/train", train.iloc[0, 0]), width=250, height=250)




## === cell 9
def prepare_data(data, m, direc):
    print("preparing data")
    X_train = np.zeros((m, 32, 32, 3))
    count = 0
    for fig in data["id"]:
        img = image.load_img(os.path.join(direc, fig), target_size=(32, 32, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x / 255
        count += 1
    print("Done")
    return X_train




## === cell 10
data = prepare_data(train, train.shape[0], train_dir)
test = prepare_data(df_test, len(df_test), test_dir)



## === cell 11
X = data
y = train["has_cactus"].values.astype(np.int32)

train_x, val_x, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



## === cell 12
df_test.head(1)



## === cell 13
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

height = 32
width = 32
channels = 3
n_inputs = height * width * channels

conv3_fmaps = 32
conv3_ksize = 3
conv3_stride = 1
conv3_pad = "SAME"

conv1_fmaps = 128
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

pool3_dropout_rate = 0.25
pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0.5

n_outputs = 2

tf.reset_default_graph()

with tf.name_scope("inputs"):
    X_ph = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="X")
    y_ph = tf.placeholder(tf.int32, shape=[None], name="y")
    training_ph = tf.placeholder_with_default(False, shape=[], name="training")

conv3 = tf.layers.conv2d(
    X_ph,
    filters=conv3_fmaps,
    kernel_size=conv3_ksize,
    strides=conv3_stride,
    padding=conv3_pad,
    activation=tf.nn.relu,
    name="conv3",
)
conv1 = tf.layers.conv2d(
    conv3,
    filters=conv1_fmaps,
    kernel_size=conv1_ksize,
    strides=conv1_stride,
    padding=conv1_pad,
    activation=tf.nn.relu,
    name="conv1",
)
conv2 = tf.layers.conv2d(
    conv1,
    filters=conv2_fmaps,
    kernel_size=conv2_ksize,
    strides=conv2_stride,
    padding=conv2_pad,
    activation=tf.nn.relu,
    name="conv2",
)

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(
        conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
    )
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 16 * 16])
    pool3_flat_drop = tf.layers.dropout(
        pool3_flat, rate=pool3_dropout_rate, training=training_ph
    )

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, rate=fc1_dropout_rate, training=training_ph)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1_drop, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
    y_pred = tf.argmax(Y_proba, axis=1, output_type=tf.int32)

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

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {
        gvar.op.name: value
        for gvar, value in zip(gvars, tf.get_default_session().run(gvars))
    }


def restore_model_params(model_params):
    assign_ops = []
    feed_dict = {}
    for name, val in model_params.items():
        var = tf.get_default_graph().get_tensor_by_name(name + ":0")
        placeholder = tf.placeholder(var.dtype, shape=var.shape)
        assign_op = tf.assign(var, placeholder)
        assign_ops.append(assign_op)
        feed_dict[placeholder] = val
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


n_epochs = 20
batch_size = 100
iteration = 0

best_loss_val = np.infty
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(train_x, train_y, batch_size):
            iteration += 1
            sess.run(
                training_op, feed_dict={X_ph: X_batch, y_ph: y_batch, training_ph: True}
            )
            if iteration % check_interval == 0:
                loss_val = sess.run(loss, feed_dict={X_ph: val_x, y_ph: val_y})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
        acc_batch = sess.run(accuracy, feed_dict={X_ph: X_batch, y_ph: y_batch})
        acc_val = sess.run(accuracy, feed_dict={X_ph: val_x, y_ph: val_y})
        print(
            "Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, best loss: {:.6f}".format(
                epoch, acc_batch * 100, acc_val * 100, best_loss_val
            )
        )
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break
    if best_model_params:
        restore_model_params(best_model_params)
    prediction = sess.run(y_pred, feed_dict={X_ph: test})
    saver.save(sess, "./cactus_model")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3877611075.py in <cell line: 0>()
     39     training_ph = tf.placeholder_with_default(False, shape=[], name="training")
     40 
---> 41 conv3 = tf.layers.conv2d(
     42     X_ph,
     43     filters=conv3_fmaps,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/lazy_loader.py in __getattr__(self, item)
    205           "__internal__.legacy."
    206       ):
--> 207         raise AttributeError(
    208             f"`{item}` is not available with Keras 3."
    209         )

AttributeError: `conv2d` is not available with Keras 3.

## === cell 14
df = pd.DataFrame({"id": df_test["id"]})
df["has_cactus"] = prediction
df.to_csv("sample_submission.csv", index=False)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/956409253.py in <cell line: 0>()
      1 df = pd.DataFrame({"id": df_test["id"]})
----> 2 df["has_cactus"] = prediction
      3 df.to_csv("sample_submission.csv", index=False)

NameError: name 'prediction' is not defined
