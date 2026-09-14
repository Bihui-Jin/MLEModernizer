# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU-only for stability

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

print("Python:", sys.version)
print("TF:", tf.__version__)
print("Working dir:", os.getcwd())

random.seed(101)
np.random.seed(101)
try:
    tf.compat.v1.set_random_seed(101)
except Exception as e:
    print("Warning: could not set tf graph seed:", repr(e))

try:
    tf.compat.v1.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("TF_INTRA_OP", "0")) or 0
    )
    tf.compat.v1.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get("TF_INTER_OP", "0")) or 0
    )
except Exception as e:
    print("Warning: could not set TF threading config:", repr(e))




## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition",
    "../input/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/input",
    "../input",
    "/kaggle/data/dogs-vs-cats-redux-kernels-edition",
    "../data/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/data",
    "../data",
]

BASE_DIR = None
for cand in BASE_CANDIDATES:
    if os.path.exists(cand):
        if (
            os.path.exists(os.path.join(cand, "sample_submission.csv"))
            or os.path.exists(os.path.join(cand, "train"))
            or os.path.exists(os.path.join(cand, "test"))
        ):
            BASE_DIR = cand
            break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate dataset base directory in known candidates."
    )

print("Using BASE_DIR:", BASE_DIR)

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

ROWS = 64
COLS = 64
CHANNELS = 3


def list_images_recursive(root, exts=(".jpg", ".jpeg", ".png")):
    paths = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir():
                        stack.append(entry.path)
                    elif entry.is_file():
                        fn = entry.name
                        if fn.lower().endswith(exts):
                            paths.append(entry.path)
        except FileNotFoundError:
            continue
    return sorted(paths)


train_dogs = list_images_recursive(os.path.join(TRAIN_DIR, "dog"))
train_cats = list_images_recursive(os.path.join(TRAIN_DIR, "cat"))

sample_path_candidates = [
    os.path.join(BASE_DIR, "sample_submission.csv"),
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv",
    "../input/sample_submission.csv",
]
sample_path = None
for sp in sample_path_candidates:
    if os.path.exists(sp):
        sample_path = sp
        break
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in known locations."
    )

sample = pd.read_csv(sample_path)
required_ids = sample["id"].astype(int).tolist()

test_candidates_dirs = []
for d in [
    os.path.join(TEST_DIR, "unknown"),
    os.path.join(TEST_DIR, "test", "unknown"),
    os.path.join(TEST_DIR, "test"),
    TEST_DIR,
]:
    if os.path.exists(d):
        test_candidates_dirs.append(d)

id_to_path = {}
missing_ids = []
needed_fns = {f"{i}.jpg" for i in required_ids}

for d in test_candidates_dirs:
    try:
        with os.scandir(d) as it:
            for entry in it:
                if entry.is_file():
                    name = entry.name
                    if name in needed_fns:
                        i = int(name[:-4])
                        if i not in id_to_path:
                            id_to_path[i] = entry.path
    except FileNotFoundError:
        continue

for i in required_ids:
    if i not in id_to_path:
        missing_ids.append(i)

if missing_ids:
    raise FileNotFoundError(
        f"Missing {len(missing_ids)} test images required by sample_submission. "
        f"Example missing ids: {missing_ids[:10]}"
    )

test_images = [id_to_path[i] for i in required_ids]

print(
    "Found train_dogs:",
    len(train_dogs),
    "train_cats:",
    len(train_cats),
    "test_images (aligned to sample_submission):",
    len(test_images),
)

train_images = train_dogs[:3000] + train_cats[:3000]
random.shuffle(train_images)




## === cell 2
train_basenames = np.array(
    [os.path.basename(p).lower() for p in train_images], dtype=object
)
is_dog = np.char.startswith(train_basenames.astype(str), "dog.")
needs_parent = ~(
    np.char.startswith(train_basenames.astype(str), "dog.")
    | np.char.startswith(train_basenames.astype(str), "cat.")
)
if np.any(needs_parent):
    parents = np.array(
        [
            os.path.basename(os.path.dirname(p)).lower()
            for p in np.array(train_images, dtype=object)
        ],
        dtype=object,
    )
    is_dog = np.where(needs_parent, parents == "dog", is_dog)

train_labels = np.zeros((len(train_images), 2), dtype=np.float32)
train_labels[is_dog, 0] = 1.0
train_labels[~is_dog, 1] = 1.0




## === cell 3
from sklearn.model_selection import train_test_split

test_size = 0.25
X_train_paths, X_test_paths, Y_train, Y_test = train_test_split(
    np.array(train_images, dtype=object),
    train_labels,
    test_size=test_size,
    random_state=101,
)

print("Training Size:", (len(X_train_paths), ROWS, COLS, CHANNELS))
print(len(X_train_paths), "samples - ", ROWS, "x", COLS, "rgb image")
print("Test Size:", (len(X_test_paths), ROWS, COLS, CHANNELS))
print(len(X_test_paths), "samples - ", ROWS, "x", COLS, "rgb image")




## === cell 4
for i in range(0, len(train_images), 500):
    print(train_images[i], train_labels[i])




## === cell 5
def _tf_load_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(
        img, [ROWS, COLS], method=tf.image.ResizeMethod.LANCZOS3, antialias=True
    )
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_train_ds(image_paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(
        lambda p, y: (_tf_load_and_resize(p), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_image_only_ds(image_paths, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_tf_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


TRAIN_BATCH = 100
PRED_BATCH = 256

train_ds = make_train_ds(X_train_paths, Y_train, batch_size=TRAIN_BATCH)
val_ds = make_train_ds(X_test_paths, Y_test, batch_size=TRAIN_BATCH)
test_ds = make_image_only_ds(np.array(test_images, dtype=object), batch_size=PRED_BATCH)




## === cell 6
class SignClass:
    def __init__(self, train_dataset):
        self.train_dataset = (
            train_dataset.repeat()
        )  # cycle forever like modulo indexing
        self._iter = tf.compat.v1.data.make_one_shot_iterator(self.train_dataset)
        self.next_elem = self._iter.get_next()

    def next_batch_tensors(self):
        return self.next_elem




## === cell 7
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 64, 64, 3])
y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
hold_prob = tf.compat.v1.placeholder(tf.float32)


def init_weights(shape):
    init_random_dist = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x_in, W):
    return tf.nn.conv2d(x_in, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2by2(x_in):
    return tf.compat.v1.nn.max_pool(
        x_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
    )


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


def build_cnn_logits(inp, hold_prob_tensor, scope, reuse=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        convo_1 = convolutional_layer(inp, shape=[5, 5, 3, 64])
        convo_1_pooling = max_pool_2by2(convo_1)

        convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 64, 128])
        convo_2_pooling = max_pool_2by2(convo_2)

        convo_2_flat = tf.reshape(convo_2_pooling, [-1, 16 * 16 * 128])
        full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

        full_one_dropout = tf.compat.v1.nn.dropout(
            full_layer_one, keep_prob=hold_prob_tensor
        )

        logits = normal_full_layer(full_one_dropout, 2)
    return logits


y_pred = build_cnn_logits(x, hold_prob, scope="ds_model", reuse=False)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
trainop = optimizer.minimize(cross_entropy)

matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
acc = tf.reduce_mean(tf.cast(matches, tf.float32))

saver = tf.compat.v1.train.Saver()




## === cell 8
steps = 1000
model_dir = "/kaggle/working/model"
os.makedirs(model_dir, exist_ok=True)
ckpt_prefix = os.path.join(model_dir, "model.ckpt")

ch = SignClass(train_ds)
bx_t, by_t = ch.next_batch_tensors()

y_pred_ds = build_cnn_logits(bx_t, hold_prob, scope="ds_model", reuse=True)

cross_entropy_ds = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=by_t, logits=y_pred_ds)
)
trainop_from_ds = optimizer.minimize(cross_entropy_ds)

acc_from_ds = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(y_pred_ds, 1), tf.argmax(by_t, 1)), tf.float32)
)

saver_ds = tf.compat.v1.train.Saver(
    var_list=tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="ds_model"
    )
)

init_all = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init_all)

    for i in range(steps):
        _, batch_acc = sess.run(
            [trainop_from_ds, acc_from_ds], feed_dict={hold_prob: 0.5}
        )

        if i % 100 == 0:
            print("Currently on step {}".format(i))
            print("Accuracy is:")
            print(batch_acc)
            print("\n")

    save_path = saver_ds.save(sess, ckpt_prefix, global_step=steps)
    print("Model saved in path:", save_path)




## === cell 9
latest_ckpt = tf.train.latest_checkpoint(model_dir)
if latest_ckpt is None:
    raise ValueError(
        f"No checkpoint found in {model_dir}. Expected something like model.ckpt-1000*"
    )

softmax_pred = tf.nn.softmax(y_pred_ds)

test_iter = tf.compat.v1.data.make_one_shot_iterator(test_ds)
test_next = test_iter.get_next()

predictions = []
with tf.compat.v1.Session() as sess:
    saver_ds.restore(sess, latest_ckpt)
    print("Restored checkpoint:", latest_ckpt)
    print("predicting")

    while True:
        try:
            bx = sess.run(test_next)
            preds = sess.run(softmax_pred, feed_dict={hold_prob: 1.0, bx_t: bx})
            predictions.append(preds)
        except tf.errors.OutOfRangeError:
            break

predictions = np.vstack(predictions)
print("done >", predictions.shape)

alpha = 0.40  # unchanged post-processing
dog_prob = predictions[:, 0].astype(np.float64)
dog_prob = alpha * dog_prob + (1.0 - alpha) * 0.5

eps = 1e-7
dog_prob = np.clip(dog_prob, eps, 1 - eps)

merged = pd.DataFrame({"id": required_ids, "label": dog_prob})

out_path = "/kaggle/working/s1.csv"
merged.to_csv(out_path, index=False)
print("Wrote submission:", out_path)
print(merged.head())
print("rows:", len(merged), "cols:", merged.columns.tolist())
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert merged.shape[0] == len(required_ids)
assert list(merged.columns) == ["id", "label"]
