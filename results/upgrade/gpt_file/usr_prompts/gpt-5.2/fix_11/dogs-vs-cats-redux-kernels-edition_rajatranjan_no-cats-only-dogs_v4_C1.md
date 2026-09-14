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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

1.65224

# 6. Current score

0.6723

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.92251) has done: 'I fix the data loading paths so the code reads actual image files (not class directories) and uses the provided Kaggle folder structure (`train/cat`, `train/dog`, `test/unknown`). I also make the TensorFlow 1.x graph code run under TensorFlow 2.18 by switching to `tf.compat.v1` (placeholders, Session, Saver, truncated_normal, dropout keep_prob) without changing the model architecture or training loop semantics. Finally, I generate the submission `id` values directly from the test filenames (sorted) and write `submission.csv` with the required `id,label` columns so ids always match Kaggle’s expectations.'
- What this solution (achieved 0.7112) has done: 'I fix the runtime crash in the TensorFlow import caused by an incompatibility between TensorFlow 2.18 and protobuf 6.x by forcing protobuf to use the pure-Python implementation before TensorFlow loads (this is a common Kaggle runtime issue). To move the score *toward* the target (worse logloss is desired here because your current score is too good), I also ensure predictions are slightly less confident by clipping probabilities away from 0/1 before writing the submission—this is a minimal post-processing calibration that preserves the model and training loop. I keep all paths and the model/training core logic unchanged, and guarantee a valid `submission.csv` with `id,label` columns.'
- What this solution (achieved 0.67501) has done: 'I fix the TensorFlow/protobuf runtime crash by forcing protobuf’s pure-Python implementation earlier and more robustly (including the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` override) and by importing TensorFlow only after those environment variables are set. I also fix the Kaggle path resolution so it works both in this notebook-like filesystem (`/kaggle/input/...`) and your current relative `../input` layout without changing how images/labels are constructed. To move the score toward your (worse) target logloss while keeping model/training logic intact, I keep your existing probability clipping but make it slightly stronger (still minimal post-processing) to reduce overconfidence. Finally, I ensure the submission has exactly 12,500 rows (or whatever test count is present), correct `id,label` columns, and a `.csv` filename.'
- What this solution (achieved 0.6534) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow-related import happens* and by importing TensorFlow in a clean, isolated way; this addresses the `MessageFactory.GetPrototype` error that currently prevents training/inference from running. I also make the input directory resolution more robust to your nested folder layout so it consistently finds `train/cat`, `train/dog`, and `test/unknown` in this environment. To move logloss upward toward your worse target (since current is too good and lower-is-better), I slightly strengthen the existing probability clipping during submission writing (post-processing only; model/training unchanged). Finally, I ensure the submission CSV has the correct `id,label` columns, sorted by id, and writes to `submission.csv`.'
- What this solution (achieved 0.6396) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables before any TensorFlow/protobuf-related import and by importing TensorFlow only after that point. I also make directory resolution robust to the nested `.../test/test/unknown` structure present in your filesystem listing (your current resolver only checks `test/unknown`). These changes are runtime/stability fixes and keep the model/training loop identical. I keep your existing post-processing (probability clipping with `eps=0.30`) unchanged to avoid moving the score away from your target more than necessary, while ensuring a valid `submission.csv` is always written with the correct `id,label` columns.'
- What this solution (achieved 0.64429) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *and* the python implementation version **before** any TensorFlow-related import, and by importing `google.protobuf` once right after setting env vars (this is the common reliable workaround in Kaggle-style runtimes). I keep your model, training loop, and prediction pipeline identical, only touching the import order and environment setup needed for runtime stability. I also keep your existing probability clipping (`eps=0.30`) unchanged since your current score (0.6396) is already better than the target (1.65224) and we should avoid score-moving edits. Finally, I ensure the test directory resolver avoids accidentally selecting a nested directory that contains subfolders but no images, so submission row counts stay correct.'
- What this solution (achieved 0.67253) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation *before* any TensorFlow-related import, and by defensively importing `google.protobuf` right after setting those env vars. I also ensure we don’t accidentally import TensorFlow earlier via other modules by moving the TF import into the same cell after env setup. This is a runtime/stability fix only and preserves the existing TF1-style graph, model architecture, training loop, and prediction pipeline. I keep your existing probability clipping (`eps=0.30`) unchanged to avoid moving the score further away from your (worse) target, and I continue to write a valid `submission.csv` with `id,label` sorted by id.'
- What this solution (achieved 0.67451) has done: 'I fix the TensorFlow import crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by ensuring the protobuf pure-Python implementation is enforced *before* TensorFlow is imported, and by removing any earlier accidental TensorFlow import paths. I also keep your model/training loop identical, but make the import order robust in this Kaggle filesystem so the notebook runs end-to-end. Since your current logloss (0.67253) is much better than the target (1.65224) and lower-is-better, I only minimally adjust the existing submission-time probability clipping to move the score upward toward the target without changing training/model logic. Finally, I keep the submission format exactly `id,label` and ensure ids are sorted and aligned to the test filenames.'
- What this solution (achieved 0.67224) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x + TF 2.18 incompatibility by ensuring the protobuf environment variables are set before any TensorFlow-related import and by importing `google.protobuf` defensively first. I also make the input base directory resolver prefer the actual competition dataset folder (so it doesn’t accidentally pick `/kaggle/input` itself and then fail to find `train/cat`, etc.). These changes are runtime/stability fixes and do not change your model, training loop, or evaluation semantics. Finally, I keep your existing submission-time probability clipping (eps=0.45) and ensure a valid `submission.csv` with `id,label` is always written.'
- What this solution (achieved 0.6723) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by moving the protobuf environment-variable setup to the very top and importing TensorFlow only after that, with a safe fallback that forces the pure-Python protobuf backend before TF loads. This is a runtime/stability fix and doesn’t change your model graph, layers, or training loop semantics. Since your current logloss (0.67224) is much better than the target (1.65224) and lower-is-better, we not try to improve the model; we only keep your existing submission-time clipping (eps=0.45) as-is to avoid drifting further away. Finally, we ensure the test directory resolution always points to a folder containing images and that `submission.csv` is written with `id,label` sorted by numeric id.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import google.protobuf  # noqa: F401

import re
import random
import numpy as np
import pandas as pd
from PIL import Image

print("Exists /kaggle/input:", os.path.exists("/kaggle/input"))
print("Exists ../input:", os.path.exists("../input"))
print("Exists /kaggle/data:", os.path.exists("/kaggle/data"))
print("Exists ../data:", os.path.exists("../data"))
if os.path.exists("/kaggle/input"):
    print("Sample /kaggle/input entries:", os.listdir("/kaggle/input")[:20])
if os.path.exists("../input"):
    print("Sample ../input entries:", os.listdir("../input")[:20])



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition",
    "../input/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/data/dogs-vs-cats-redux-kernels-edition",
    "../data/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/input",
    "../input",
    "/kaggle/data",
    "../data",
]

BASE_INPUT = None
for b in CANDIDATE_BASES:
    if os.path.isdir(b):
        BASE_INPUT = b
        break
if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not find an input base directory among: {}".format(CANDIDATE_BASES)
    )


def _has_images(dir_path):
    try:
        return any(
            f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(dir_path)
        )
    except Exception:
        return False


def resolve_dirs(base_dir):
    train_cat = os.path.join(base_dir, "train", "cat")
    train_dog = os.path.join(base_dir, "train", "dog")

    test_candidates = [
        os.path.join(base_dir, "test", "unknown"),
        os.path.join(base_dir, "test", "test", "unknown"),
        os.path.join(base_dir, "test", "test"),
        os.path.join(base_dir, "test"),
    ]

    test_dir = None
    for td in test_candidates:
        if os.path.isdir(td) and _has_images(td):
            test_dir = td
            break

    if test_dir is None:
        for td in test_candidates:
            if os.path.isdir(td):
                test_dir = td
                break

    ok = (
        os.path.isdir(train_cat)
        and os.path.isdir(train_dog)
        and (test_dir is not None)
        and os.path.isdir(test_dir)
        and _has_images(test_dir)
    )
    return ok, train_cat, train_dog, test_dir


ok, TRAIN_CAT_DIR, TRAIN_DOG_DIR, TEST_DIR = resolve_dirs(BASE_INPUT)

if not ok:
    alt_base = os.path.join(BASE_INPUT, "dogs-vs-cats-redux-kernels-edition")
    ok, TRAIN_CAT_DIR, TRAIN_DOG_DIR, TEST_DIR = resolve_dirs(alt_base)

if not ok:
    alt_base2 = os.path.join(
        BASE_INPUT,
        "dogs-vs-cats-redux-kernels-edition",
        "dogs-vs-cats-redux-kernels-edition",
    )
    ok, TRAIN_CAT_DIR, TRAIN_DOG_DIR, TEST_DIR = resolve_dirs(alt_base2)

if not ok:
    raise FileNotFoundError(
        "Could not resolve train/cat, train/dog, and an image-containing test directory under base '{}'".format(
            BASE_INPUT
        )
    )

print("BASE_INPUT:", BASE_INPUT)
print("TRAIN_CAT_DIR:", TRAIN_CAT_DIR, "exists:", os.path.isdir(TRAIN_CAT_DIR))
print("TRAIN_DOG_DIR:", TRAIN_DOG_DIR, "exists:", os.path.isdir(TRAIN_DOG_DIR))
print("TEST_DIR:", TEST_DIR, "exists:", os.path.isdir(TEST_DIR))

ROWS = 64
COLS = 64
CHANNELS = 3

train_dogs = [
    os.path.join(TRAIN_DOG_DIR, f)
    for f in os.listdir(TRAIN_DOG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
train_cats = [
    os.path.join(TRAIN_CAT_DIR, f)
    for f in os.listdir(TRAIN_CAT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
test_images = [
    os.path.join(TEST_DIR, f)
    for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

train_images = train_dogs[:3000] + train_cats[:3000]
random.shuffle(train_images)

print("Train images:", len(train_images), " Test images:", len(test_images))



## === cell 2
from sklearn import preprocessing  # kept as in original (though unused)


def read_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img = img.resize((ROWS, COLS), resample=Image.Resampling.LANCZOS)
    return np.array(img)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i % 1000 == 0:
            print("Processed {} of {}".format(i, count))
    return data


train = prep_data(train_images)
test = prep_data(test_images)



## === cell 3
import seaborn as sns
from matplotlib import ticker

train_labels = []
for p in train_images:
    if os.sep + "dog" + os.sep in p:
        train_labels.append([1, 0])
    else:
        train_labels.append([0, 1])
train_labels = np.array(train_labels, dtype=np.float32)

print("train_labels shape:", train_labels.shape)



## === cell 4
print("train dtype before:", train.dtype, "max:", train.max())
train = (train / train.max()).astype(np.float32)
print("train dtype after:", train.dtype, "max:", train.max())

test = (test / test.max()).astype(np.float32)
print("test dtype:", test.dtype, "max:", test.max())

for i in range(0, len(train_images), 500):
    print(train_images[i], train_labels[i])



## === cell 5
from sklearn.model_selection import train_test_split

test_size = 0.25
X_train, X_test, Y_train, Y_test = train_test_split(
    train, train_labels, test_size=test_size, random_state=101
)

img_size = 64
channel_size = 1
print("Training Size:", X_train.shape)
print(
    X_train.shape[0], "samples - ", X_train.shape[1], "x", X_train.shape[2], "rgb image"
)
print("\n")
print("Test Size:", X_test.shape)
print(X_test.shape[0], "samples - ", X_test.shape[1], "x", X_test.shape[2], "rgb image")




## === cell 6
class SignClass:
    def __init__(self):
        self.i = 0
        self.training_images = X_train
        self.training_labels = Y_train
        self.test_images = X_test
        self.test_labels = Y_test
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = X_train.shape[0]

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        x = self.training_images[self.i : self.i + batch_size]
        y = self.training_labels[self.i : self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y


ch = SignClass()



## === cell 7
try:
    import tensorflow as tf
except AttributeError as e:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
    import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(101)
np.random.seed(101)
random.seed(101)

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 64, 64, 3])
y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
hold_prob = tf.compat.v1.placeholder(tf.float32)




## === cell 9
def init_weights(shape):
    init_random_dist = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x_in, W):
    return tf.nn.conv2d(x_in, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2by2(x_in):
    return tf.nn.max_pool2d(x_in, ksize=2, strides=2, padding="SAME")


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    c = conv2d(input_x, W)
    act = tf.nn.relu(c + b)
    return act


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b




## === cell 10
convo_1 = convolutional_layer(x, shape=[5, 5, 3, 64])
convo_1_pooling = max_pool_2by2(convo_1)
convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 64, 128])
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling, [-1, 16 * 16 * 128])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

full_one_dropout = tf.compat.v1.nn.dropout(full_layer_one, keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout, 2)



## === cell 11
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
trainop = optimizer.minimize(cross_entropy)
init = tf.compat.v1.global_variables_initializer()



## === cell 12
steps = 1000
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for i in range(steps):
        batch = ch.next_batch(100)
        sess.run(trainop, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

        if i % 100 == 0:
            print("Currently on step {}".format(i))
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            print(
                sess.run(
                    acc,
                    feed_dict={
                        x: ch.test_images,
                        y_true: ch.test_labels,
                        hold_prob: 1.0,
                    },
                )
            )
            print("\n")

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)



## === cell 13
predictions = []
with tf.compat.v1.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    print("predicting")
    predicts = tf.nn.softmax(y_pred)

    bs = 1500
    for start in range(0, test.shape[0], bs):
        end = min(start + bs, test.shape[0])
        part = sess.run(predicts, feed_dict={x: test[start:end], hold_prob: 1.0})
        predictions.append(part)

predictions = np.vstack(predictions)
print("done >", predictions.shape)



## === cell 14
dog_prob = predictions[:, 0].astype(np.float64)


def extract_id(path):
    base = os.path.basename(path)
    m = re.match(r"(\d+)\.", base)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", base)
    return int(nums[0]) if nums else -1


test_ids = np.array([extract_id(p) for p in test_images], dtype=int)
order = np.argsort(test_ids)

eps = 0.45
dog_prob_cal = np.clip(dog_prob, eps, 1.0 - eps)

sub = pd.DataFrame({"id": test_ids[order], "label": dog_prob_cal[order]})

print(sub.head())
print(
    "nrows:",
    len(sub),
    "unique ids:",
    sub["id"].nunique(),
    "min id:",
    sub["id"].min(),
    "max id:",
    sub["id"].max(),
)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
