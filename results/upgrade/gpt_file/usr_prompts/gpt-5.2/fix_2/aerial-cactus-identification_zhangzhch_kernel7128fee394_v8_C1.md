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

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import cv2
import time
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

print("TF version:", tf.__version__)

CANDIDATE_ROOTS = [
    "../input/aerial-cactus-identification",
    "../input",
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(r):
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find Kaggle input data root in expected locations."
    )

print("Using DATA_ROOT:", DATA_ROOT)
print("Top-level files:", sorted(os.listdir(DATA_ROOT))[:20])




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


TRAIN_CSV = first_existing(
    [
        os.path.join(DATA_ROOT, "train.csv"),
        "../input/train.csv",
        "/kaggle/input/train.csv",
    ]
)

SAMPLE_SUB = first_existing(
    [
        os.path.join(DATA_ROOT, "sample_submission.csv"),
        "../input/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)

TRAIN_DIR = first_existing(
    [
        os.path.join(DATA_ROOT, "train"),
        os.path.join(DATA_ROOT, "train", "train"),
        "../input/train",
        "../input/train/train",
        "/kaggle/input/train",
        "/kaggle/input/train/train",
    ]
)

TEST_DIR = first_existing(
    [
        os.path.join(DATA_ROOT, "test"),
        os.path.join(DATA_ROOT, "test", "test"),
        "../input/test",
        "../input/test/test",
        "/kaggle/input/test",
        "/kaggle/input/test/test",
    ]
)

for name, p in [
    ("TRAIN_CSV", TRAIN_CSV),
    ("SAMPLE_SUB", SAMPLE_SUB),
    ("TRAIN_DIR", TRAIN_DIR),
    ("TEST_DIR", TEST_DIR),
]:
    print(name, "->", p)
    if p is None:
        raise FileNotFoundError(f"Missing required path for {name}")



## === cell 2
train_y = pd.read_csv(TRAIN_CSV)
test_y = pd.read_csv(SAMPLE_SUB)

print(train_y.head())
print(test_y.head())
print("Train size:", train_y.shape, "Test sample size:", test_y.shape)




## === cell 3
def read_image_rgb32(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (32, 32):
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    return img


train_ids = train_y["id"].tolist()
train_x_list = []
bad_train = 0
for fid in train_ids:
    p = os.path.join(TRAIN_DIR, fid)
    img = read_image_rgb32(p)
    if img is None:
        bad_train += 1
        img = np.zeros((32, 32, 3), dtype=np.uint8)
    train_x_list.append(img)

test_ids = test_y["id"].tolist()
test_x_list = []
bad_test = 0
for fid in test_ids:
    p = os.path.join(TEST_DIR, fid)
    img = read_image_rgb32(p)
    if img is None:
        bad_test += 1
        img = np.zeros((32, 32, 3), dtype=np.uint8)
    test_x_list.append(img)

print("Unreadable train images:", bad_train, "Unreadable test images:", bad_test)

train_x = np.array(train_x_list, dtype=np.float32) / 255.0
test = np.array(test_x_list, dtype=np.float32) / 255.0

filename = train_ids
filenamey = test_ids

print("train_x:", train_x.shape, train_x.dtype, "test:", test.shape, test.dtype)



## === cell 4
len(filenamey)



## === cell 5
train_Y = train_y["has_cactus"].astype(int).values
train_Y = np.array(pd.get_dummies(train_Y))  # shape (N,2) with columns [0,1]
print(
    "train_Y shape:",
    train_Y.shape,
    "class balance:",
    train_y["has_cactus"].value_counts().to_dict(),
)



## === cell 6
train_Y.shape



## === cell 7
keepprob = 0.5
tf.compat.v1.reset_default_graph()


def max_pool_2x2(x_):
    return tf.nn.max_pool2d(x_, ksize=2, strides=2, padding="SAME")


x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3], name="x_image")
y = tf.compat.v1.placeholder(tf.float32, [None, 2], name="y")

W_conv1 = tf.compat.v1.get_variable(
    "W1",
    shape=[3, 3, 3, 12],
    initializer=tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_conv1 = tf.compat.v1.get_variable(
    "b1", shape=[12], initializer=tf.compat.v1.constant_initializer(0.1)
)
h_conv1 = tf.nn.relu(
    tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1
)

W_conv1_2 = tf.compat.v1.get_variable(
    "W1_2",
    shape=[3, 3, 12, 12],
    initializer=tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_conv1_2 = tf.compat.v1.get_variable(
    "b1_2", shape=[12], initializer=tf.compat.v1.constant_initializer(0.1)
)
h_conv1_2 = tf.nn.relu(
    tf.nn.conv2d(h_conv1, W_conv1_2, strides=[1, 1, 1, 1], padding="SAME") + b_conv1_2
)

h_pool1 = max_pool_2x2(h_conv1_2)

W_conv2 = tf.compat.v1.get_variable(
    "W2_1",
    shape=[3, 3, 12, 24],
    initializer=tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_conv2 = tf.compat.v1.get_variable(
    "b2_1", shape=[24], initializer=tf.compat.v1.constant_initializer(0.1)
)
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2
)

W_conv2_2 = tf.compat.v1.get_variable(
    "W2_2",
    shape=[3, 3, 24, 24],
    initializer=tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_conv2_2 = tf.compat.v1.get_variable(
    "b2_2", shape=[24], initializer=tf.compat.v1.constant_initializer(0.1)
)
h_conv2_2 = tf.nn.relu(
    tf.nn.conv2d(h_conv2, W_conv2_2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2_2
)

h_pool2 = max_pool_2x2(h_conv2_2)

W_fc1 = tf.compat.v1.get_variable(
    "Wf1_2",
    shape=[8 * 8 * 24, 128],
    initializer=tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_fc1 = tf.compat.v1.get_variable(
    "bf2", shape=[128], initializer=tf.compat.v1.constant_initializer(0.1)
)
h_pool_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 24])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, rate=1.0 - keep_prob)

W_fc2 = tf.compat.v1.get_variable(
    "Wf3_2",
    shape=[128, 84],
    initializer=tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.1),
)
b_fc2 = tf.compat.v1.get_variable(
    "bf3", shape=[84], initializer=tf.compat.v1.constant_initializer(0.1)
)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

W_fc3 = tf.compat.v1.get_variable(
    "Wf4_2",
    shape=[84, 2],
    initializer=tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.05),
)
b_fc3 = tf.compat.v1.get_variable(
    "bf4", shape=[2], initializer=tf.compat.v1.constant_initializer(0.05)
)

y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
)
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(
    cross_entropy
)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.compat.v1.add_to_collection("y_conv", y_conv)
tf.compat.v1.add_to_collection("cross_entropy", cross_entropy)

probs = tf.nn.softmax(y_conv, axis=1)[:, 1]



## === cell 8
mybatch = 50
iterations = 10


def train_model(mybatch, iterations, m, n, keepprob):
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        a = 0.5
        best_test_probs = None

        N = len(m)
        for j in range(4, 6):
            print("fold:%d" % (j))
            start = j * 2500
            end = min((j + 1) * 2500, N)

            x2 = m[start:end]
            y2 = n[start:end]

            idx_train = np.append(np.array(range(0, start)), np.array(range(end, N)))
            x1 = m[idx_train]
            y1 = n[idx_train]

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
                    best_test_probs = sess.run(
                        probs, feed_dict={x: test, keep_prob: 1.0}
                    )

        if best_test_probs is None:
            best_test_probs = sess.run(probs, feed_dict={x: test, keep_prob: 1.0})

        return best_test_probs




## === cell 9
predict = train_model(mybatch, iterations, train_x, train_Y, keepprob)
print(
    "Pred shape:",
    predict.shape,
    "min/max:",
    float(np.min(predict)),
    float(np.max(predict)),
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3111120575.py in <cell line: 0>()
----> 1 predict = train_model(mybatch, iterations, train_x, train_Y, keepprob)
      2 print(
      3     "Pred shape:",
      4     predict.shape,
      5     "min/max:",

/tmp/ipykernel_11/4196494168.py in train_model(mybatch, iterations, m, n, keepprob)
     24             # original used 17500 constant; use N to avoid out-of-range (runtime fix)
     25             idx_train = np.append(np.array(range(0, start)), np.array(range(end, N)))
---> 26             x1 = m[idx_train]
     27             y1 = n[idx_train]
     28 

IndexError: arrays used as indices must be of integer (or boolean) type

## === cell 10
sub = pd.DataFrame({"id": filenamey, "has_cactus": predict.astype(float)})
sub = test_y[["id"]].merge(sub, how="left", on="id")

sub["has_cactus"] = sub["has_cactus"].fillna(0.5).clip(0.0, 1.0)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/560158823.py in <cell line: 0>()
      1 # Build submission exactly as required; ensure ids are in sample_submission order and file ends with .csv
----> 2 sub = pd.DataFrame({"id": filenamey, "has_cactus": predict.astype(float)})
      3 sub = test_y[["id"]].merge(sub, how="left", on="id")
      4 
      5 # If any missing (shouldn't), fill with 0.5 to keep valid probabilities

NameError: name 'predict' is not defined
