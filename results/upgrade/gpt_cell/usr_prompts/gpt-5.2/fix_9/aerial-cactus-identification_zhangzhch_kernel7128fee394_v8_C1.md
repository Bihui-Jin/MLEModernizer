# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.992

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            raise AttributeError(
                "MessageFactory has no GetPrototype/GetMessageClass in this protobuf version."
            )

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from IPython.display import Image
from keras.preprocessing import image
from keras import optimizers
from keras import layers, models
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16
import tensorflow as tf
import numpy as np
import time


## === cell 2
os.getcwd()


## === cell 3
train_y=pd.read_csv('../input/train.csv')
test_y=pd.read_csv('../input/sample_submission.csv')
train_y['has_cactus']=train_y['has_cactus'].astype(str)

train_y['has_cactus'][16500:].value_counts()


## === cell 4
train_dir = "../input/train/train"
test_dir = "../input/test/test"
train_x = []
test_x = []
filename = []
filenamey = []

for datefile in os.listdir(train_dir):
    path = os.path.join(train_dir, datefile)
    if not os.path.isfile(path):
        continue
    img = cv2.imread(path)
    if img is None:
        continue
    filename.append(datefile)
    train_x.append(img)

for datefile in os.listdir(test_dir):
    path = os.path.join(test_dir, datefile)
    if not os.path.isfile(path):
        continue
    img = cv2.imread(path)
    if img is None:
        continue
    filenamey.append(datefile)
    test_x.append(img)

train_x = [i / 255 for i in train_x]
train_x = np.array(train_x)
test_x = [i / 255 for i in test_x]
test = np.array(test_x)


## === cell 5
len(filenamey)


## === cell 6
label=pd.DataFrame({'id':filename})
train_y=label.merge(train_y,how='left',on='id')

train_Y=train_y['has_cactus']
train_Y=np.array(pd.get_dummies(train_Y))


## === cell 7
train_Y.shape


## === cell 8
keepprob = 0.5

tf = tf.compat.v1
tf.disable_eager_execution()
tf.reset_default_graph()


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x_image")
y = tf.placeholder(tf.float32, [None, 2], name="y")

W_conv1 = tf.get_variable(
    "W1",
    shape=[3, 3, 3, 12],
    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_conv1 = tf.get_variable("b1", shape=[12], initializer=tf.constant_initializer(0.1))

h_conv1 = tf.nn.relu(
    tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1
)

W_conv1_2 = tf.get_variable(
    "W1_2",
    shape=[3, 3, 12, 12],
    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_conv1_2 = tf.get_variable(
    "b1_2", shape=[12], initializer=tf.constant_initializer(0.1)
)

h_conv1_2 = tf.nn.relu(
    tf.nn.conv2d(h_conv1, W_conv1_2, strides=[1, 1, 1, 1], padding="SAME") + b_conv1_2
)

h_pool1 = max_pool_2x2(h_conv1_2)

W_conv2 = tf.get_variable(
    "W2_1",
    shape=[3, 3, 12, 24],
    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_conv2 = tf.get_variable("b2_1", shape=[24], initializer=tf.constant_initializer(0.1))

h_conv2 = tf.nn.relu(
    tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2
)

W_conv2_2 = tf.get_variable(
    "W2_2",
    shape=[3, 3, 24, 24],
    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_conv2_2 = tf.get_variable(
    "b2_2", shape=[24], initializer=tf.constant_initializer(0.1)
)

h_conv2_2 = tf.nn.relu(
    tf.nn.conv2d(h_conv2, W_conv2_2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2_2
)

h_pool2 = max_pool_2x2(h_conv2_2)


W_fc1 = tf.get_variable(
    "Wf1_2",
    shape=[8 * 8 * 24, 128],
    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_fc1 = tf.get_variable("bf2", shape=[128], initializer=tf.constant_initializer(0.1))
h_pool_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 24])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = tf.get_variable(
    "Wf3_2",
    shape=[128, 84],
    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_fc2 = tf.get_variable("bf3", shape=[84], initializer=tf.constant_initializer(0.1))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


W_fc3 = tf.get_variable(
    "Wf4_2",
    shape=[84, 2],
    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.05),
)
b_fc3 = tf.get_variable("bf4", shape=[2], initializer=tf.constant_initializer(0.05))


y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
)
train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(
    cross_entropy
)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.add_to_collection("y_conv", y_conv)
tf.add_to_collection("cross_entropy", cross_entropy)


## === cell 9
mybatch=50
iterations=10

def train_model(mybatch,iterations,m,n,keepprob):
    
    init = tf.global_variables_initializer()
    
    saver=tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:      
        sess.run(init)
        a=0.5
        for j in range(4,6):
             print("fold:%d"%(j))
             if j<6:
        
                 x2=m[range(j*2500,(j+1)*2500)]
                 y2=n[range(j*2500,(j+1)*2500)]
            
                 x1=m[np.append(np.array(range(0,j*2500)),np.array(range((j+1)*2500,17500)))]
                 y1=n[np.append(np.array(range(0,j*2500)),np.array(range((j+1)*2500,17500)))]
             else:
                 x2=m[range(j*2500,17500)]
                 y2=n[range(j*2500,17500)]
                 x1=m[:j*2500]
                 y1=n[:j*2500]
     
             for s in range(iterations):
                t0=time.time()
                myindice=list(range(len(x1)))
                np.random.shuffle(myindice)
           
                x_=x1[myindice]
                y_=y1[myindice]

                for i in range(len(x1)//mybatch):
                    sess.run(train_step,feed_dict={x:x_[i*mybatch:i*mybatch+mybatch],y:y_[i*mybatch:i*mybatch+mybatch],keep_prob:keepprob})
      
                t1=time.time()
                print("traintime:%.2f s"%(t1-t0))
                print(sess.run(accuracy,feed_dict={x:x_,y:y_,keep_prob:1.0}))  
            
                train_acc=sess.run(accuracy,feed_dict={x:x2,y:y2,keep_prob:1.0})
                t2=time.time()
                print("evaltime:%.2f s"%(t2-t0))
                print("eval:%1.4f"%(train_acc))
                cross_val=sess.run(cross_entropy,feed_dict={x:x2,y:y2,keep_prob:1.0})
                print("evalcross:%1.4f"%(cross_val))
                if cross_val<a:
                    saver.save(sess,'basemode.ckpt')               
                    a=cross_val
                    pre_plate=sess.run(y_conv,feed_dict={x:test,keep_prob:1.0})
                    predict=np.argmax(pre_plate,axis=1)
        return predict


## === cell 10
mybatch = 50
iterations = 10


def train_model(mybatch, iterations, m, n, keepprob):
    N = len(m)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(init)
        a = 0.5
        for j in range(4, 6):
            print("fold:%d" % (j))
            if j < 6:
                start = j * 2500
                end = min((j + 1) * 2500, N)

                x2 = m[range(start, end)]
                y2 = n[range(start, end)]

                idx1 = np.append(np.array(range(0, start)), np.array(range(end, N)))
                x1 = m[idx1]
                y1 = n[idx1]
            else:
                start = j * 2500
                x2 = m[range(start, N)]
                y2 = n[range(start, N)]
                x1 = m[:start]
                y1 = n[:start]

            for s in range(iterations):
                t0 = time.time()
                myindice = list(range(len(x1)))
                np.random.shuffle(myindice)

                x_ = x1[myindice]
                y_ = y1[myindice]

                for i in range(len(x1) // mybatch):
                    sess.run(
                        train_step,
                        feed_dict={
                            x: x_[i * mybatch : i * mybatch + mybatch],
                            y: y_[i * mybatch : i * mybatch + mybatch],
                            keep_prob: keepprob,
                        },
                    )

                t1 = time.time()
                print("traintime:%.2f s" % (t1 - t0))
                print(sess.run(accuracy, feed_dict={x: x_, y: y_, keep_prob: 1.0}))

                train_acc = sess.run(accuracy, feed_dict={x: x2, y: y2, keep_prob: 1.0})
                t2 = time.time()
                print("evaltime:%.2f s" % (t2 - t0))
                print("eval:%1.4f" % (train_acc))
                cross_val = sess.run(
                    cross_entropy, feed_dict={x: x2, y: y2, keep_prob: 1.0}
                )
                print("evalcross:%1.4f" % (cross_val))
                if cross_val < a:
                    saver.save(sess, "basemode.ckpt")
                    a = cross_val
                    pre_plate = sess.run(y_conv, feed_dict={x: test, keep_prob: 1.0})
                    predict = np.argmax(pre_plate, axis=1)
        return predict


## === cell 11
mybatch = 50
iterations = 10


def train_model(mybatch, iterations, m, n, keepprob):
    N = len(m)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(init)
        a = 0.5
        for j in range(4, 6):
            print("fold:%d" % (j))
            if j < 6:
                start = j * 2500
                end = min((j + 1) * 2500, N)

                x2 = m[range(start, end)]
                y2 = n[range(start, end)]

                idx1 = np.append(
                    np.array(range(0, start), dtype=np.int64),
                    np.array(range(end, N), dtype=np.int64),
                ).astype(np.int64, copy=False)
                x1 = m[idx1]
                y1 = n[idx1]
            else:
                start = j * 2500
                x2 = m[range(start, N)]
                y2 = n[range(start, N)]
                x1 = m[:start]
                y1 = n[:start]

            for s in range(iterations):
                t0 = time.time()
                myindice = list(range(len(x1)))
                np.random.shuffle(myindice)

                x_ = x1[myindice]
                y_ = y1[myindice]

                for i in range(len(x1) // mybatch):
                    sess.run(
                        train_step,
                        feed_dict={
                            x: x_[i * mybatch : i * mybatch + mybatch],
                            y: y_[i * mybatch : i * mybatch + mybatch],
                            keep_prob: keepprob,
                        },
                    )

                t1 = time.time()
                print("traintime:%.2f s" % (t1 - t0))
                print(sess.run(accuracy, feed_dict={x: x_, y: y_, keep_prob: 1.0}))

                train_acc = sess.run(accuracy, feed_dict={x: x2, y: y2, keep_prob: 1.0})
                t2 = time.time()
                print("evaltime:%.2f s" % (t2 - t0))
                print("eval:%1.4f" % (train_acc))
                cross_val = sess.run(
                    cross_entropy, feed_dict={x: x2, y: y2, keep_prob: 1.0}
                )
                print("evalcross:%1.4f" % (cross_val))
                if cross_val < a:
                    saver.save(sess, "basemode.ckpt")
                    a = cross_val
                    pre_plate = sess.run(y_conv, feed_dict={x: test, keep_prob: 1.0})
                    predict = np.argmax(pre_plate, axis=1)
        return predict
