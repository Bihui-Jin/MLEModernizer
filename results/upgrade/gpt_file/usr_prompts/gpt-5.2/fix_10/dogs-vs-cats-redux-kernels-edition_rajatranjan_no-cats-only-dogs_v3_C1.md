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

2.91387

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.04691) has done: 'I fix the data path logic so the script reads actual image files (not class directories), and make test file discovery robust to the nested `test/unknown` folder structure. Then I keep your original TF1-style CNN/training loop intact by running it through `tf.compat.v1` (placeholders, Session, Saver, deprecated ops) so it works under TensorFlow 2.18. Finally, I generate the submission using the official `sample_submission.csv` ids to guarantee id alignment and column names, and map the “dog” probability correctly to the `label` column (with a small epsilon clip for logloss stability).'
- What this solution (achieved 1.14905) has done: 'I remove the protobuf downgrade/restart logic because it is unnecessary in your environment (TensorFlow 2.18 is already installed with protobuf 6.x) and it can prevent the notebook/script from ever reaching submission creation. I also make the test image discovery deterministic and restricted to the required `unknown/*.jpg` folder to guarantee exactly 2500 test images (matching `sample_submission.csv`) and correct id alignment. Finally, I fix a subtle label leakage bug where your current filename-based rule can mislabel cats because the substring `"dog"` occurs inside `"dogs-vs-cats-redux-kernels-edition"` in the full path; this directly harms logloss, and fixing it preserves the same model/training logic while improving correctness.'
- What this solution (achieved 0.98786) has done: 'I fix the TensorFlow/protobuf runtime crash by ensuring TensorFlow is imported before any protobuf-dependent modules and by avoiding the internal `MessageFactory.GetPrototype` path that breaks under protobuf 6.x in some environments. I also add deterministic seeds at the TF graph level for stability, without changing your model or training loop semantics. Finally, I keep your robust dataset path + test discovery logic and ensure the submission is always written as a valid `.csv` with correct `id,label` alignment to `sample_submission.csv`.'
- What this solution (achieved 0.64606) has done: 'I fix the immediate TensorFlow/protobuf crash by switching TF import to the stable `tf.compat.v1` path and setting the needed environment flags *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` failure in this environment. I keep your CNN architecture and TF1-style training loop unchanged, but I intentionally calibrate the final probabilities slightly toward 0.5 (via a small blending step) so the logloss degrades toward your much higher target (since lower is better and you’re currently far better than target). I also make test discovery strictly match the `sample_submission.csv` ids to guarantee 2500 predictions and correct id alignment. The script still run end-to-end and write `/kaggle/working/s1.csv` with `id,label`.'
- What this solution (achieved 0.64919) has done: 'I fix the TensorFlow/protobuf crash by forcing the stable pure-Python protobuf runtime *before* importing TensorFlow and by using `tf.compat.v1` consistently (TF1 graph mode) so your placeholder/Session code runs under TF 2.18. I keep your CNN, training loop, and prediction logic intact, only changing the import/seed cell to prevent the `MessageFactory.GetPrototype` error. I also keep the submission alignment to `sample_submission.csv` ids and ensure the output is written as a proper `.csv` with `id,label`. No score-targeting changes are needed beyond restoring executability (your existing alpha-blend calibration remains unchanged).'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # keep CPU-only for stability

import tensorflow as tf

print("Python:", sys.version)
print("TF:", tf.__version__)
print("Working dir:", os.getcwd())

print("Listing /kaggle/input (if exists):")
for p in ["../input", "/kaggle/input"]:
    if os.path.exists(p):
        print(p, "->", os.listdir(p)[:10])

random.seed(101)
np.random.seed(101)
try:
    tf.random.set_seed(101)
except Exception as e:
    print("Warning: could not set tf random seed:", repr(e))

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

test_unknown = os.path.join(TEST_DIR, "unknown")
test_search_roots = [test_unknown] if os.path.exists(test_unknown) else [TEST_DIR]

id_to_path = {}
for root in test_search_roots:
    for p in list_images_recursive(root):
        bn = os.path.basename(p)
        stem, _ = os.path.splitext(bn)
        if stem.isdigit():
            id_to_path[int(stem)] = p

missing_ids = [i for i in required_ids if i not in id_to_path]
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
tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 64, 64, 3])
y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
hold_prob = tf.compat.v1.placeholder(tf.float32)

try:
    tf.compat.v1.set_random_seed(101)
except Exception as e:
    print("Warning: could not set graph seed:", repr(e))


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


convo_1 = convolutional_layer(x, shape=[5, 5, 3, 64])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 64, 128])
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling, [-1, 16 * 16 * 128])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

full_one_dropout = tf.compat.v1.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout, 2)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
trainop = optimizer.minimize(cross_entropy)

matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
acc = tf.reduce_mean(tf.cast(matches, tf.float32))

init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()




## === cell 8
steps = 1000
model_path = "/kaggle/working/model.ckpt"

ch = SignClass(train_ds)

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for i in range(steps):
        bx, by = sess.run(ch.next_batch_tensors())
        sess.run(trainop, feed_dict={x: bx, y_true: by, hold_prob: 0.5})

        if i % 100 == 0:
            batch_acc = sess.run(acc, feed_dict={x: bx, y_true: by, hold_prob: 1.0})
            print("Currently on step {}".format(i))
            print("Accuracy is:")
            print(batch_acc)
            print("\n")

    save_path = saver.save(sess, model_path)
    print("Model saved in path:", save_path)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2638253594.py in <cell line: 0>()
      4 model_path = "/kaggle/working/model.ckpt"
      5 
----> 6 ch = SignClass(train_ds)
      7 
      8 with tf.compat.v1.Session() as sess:

/tmp/ipykernel_11/3836804793.py in __init__(self, train_dataset)
      4     def __init__(self, train_dataset):
      5         self.train_dataset = (
----> 6             train_dataset.repeat()
      7         )  # cycle forever like modulo indexing
      8         self._iter = tf.compat.v1.data.make_one_shot_iterator(self.train_dataset)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in repeat(self, count, name)
   1375     # pylint: disable=g-import-not-at-top,protected-access,redefined-outer-name
   1376     from tensorflow.python.data.ops import repeat_op
-> 1377     return repeat_op._repeat(self, count, name)
   1378     # pylint: enable=g-import-not-at-top,protected-access,redefined-outer-name
   1379 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/repeat_op.py in _repeat(input_dataset, count, name)
     23 
     24 def _repeat(input_dataset, count, name):  # pylint: disable=unused-private-name
---> 25   return _RepeatDataset(input_dataset, count, name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/repeat_op.py in __init__(self, input_dataset, count, name)
     38           count, dtype=dtypes.int64, name="count")
     39     self._name = name
---> 40     variant_tensor = gen_dataset_ops.repeat_dataset(
     41         input_dataset._variant_tensor,  # pylint: disable=protected-access
     42         count=self._count,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in repeat_dataset(input_dataset, count, output_types, output_shapes, metadata, name)
   6394     metadata = ""
   6395   metadata = _execute.make_str(metadata, "metadata")
-> 6396   _, _, _op, _outputs = _op_def_library._apply_op_helper(
   6397         "RepeatDataset", input_dataset=input_dataset, count=count,
   6398                          output_types=output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
    776   with g.as_default(), ops.name_scope(name) as scope:
    777     if fallback:
--> 778       _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
    779                              keywords, default_type_attr_map, attrs, inputs,
    780                              input_types)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    549                 preferred_dtype=default_dtype)
    550         else:
--> 551           values = ops.convert_to_tensor(
    552               values,
    553               name=input_arg.name,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    207   overload = getattr(value, "__tf_tensor__", None)
    208   if overload is not None:
--> 209     return overload(dtype, name)  #  pylint: disable=not-callable
    210 
    211   for base_type, conversion_func in get(type(value)):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in __tf_tensor__(self, dtype, name)
    619       graph = get_default_graph()
    620       if not graph.building_function:
--> 621         raise RuntimeError(
    622             _add_error_prefix(
    623                 "Attempting to capture an EagerTensor without "

RuntimeError: input_dataset: Attempting to capture an EagerTensor without building a function.

## === cell 9
softmax_pred = tf.nn.softmax(y_pred)

test_iter = tf.compat.v1.data.make_one_shot_iterator(test_ds)
test_next = test_iter.get_next()

predictions = []
with tf.compat.v1.Session() as sess:
    saver.restore(sess, model_path)
    print("predicting")
    while True:
        try:
            bx = sess.run(test_next)
            preds = sess.run(softmax_pred, feed_dict={x: bx, hold_prob: 1.0})
            predictions.append(preds)
        except tf.errors.OutOfRangeError:
            break

predictions = np.vstack(predictions)
print("done >", predictions.shape)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in make_one_shot_iterator(dataset)
   4347     # datasets (e.g. for prefetching) override its behavior.
-> 4348     return dataset._make_one_shot_iterator()  # pylint: disable=protected-access
   4349   except AttributeError:

AttributeError: '_PrefetchDataset' object has no attribute '_make_one_shot_iterator'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2920921931.py in <cell line: 0>()
      2 softmax_pred = tf.nn.softmax(y_pred)
      3 
----> 4 test_iter = tf.compat.v1.data.make_one_shot_iterator(test_ds)
      5 test_next = test_iter.get_next()
      6 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in make_one_shot_iterator(dataset)
   4348     return dataset._make_one_shot_iterator()  # pylint: disable=protected-access
   4349   except AttributeError:
-> 4350     return DatasetV1Adapter(dataset)._make_one_shot_iterator()  # pylint: disable=protected-access
   4351 
   4352 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in _make_one_shot_iterator(self)
   3797     # to capture by value. We must allowlist these ops so that the capturing
   3798     # logic captures the ops instead of raising an exception.
-> 3799     allowlisted_stateful_ops = traverse.obtain_capture_by_value_ops(self)
   3800     graph_level_seed, op_level_seed = core_random_seed.get_seed(None)
   3801 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/traverse.py in obtain_capture_by_value_ops(dataset)
     62             op.type in OP_TYPES_ALLOWLIST)
     63 
---> 64   return _traverse(dataset, capture_by_value)
     65 
     66 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/traverse.py in _traverse(dataset, op_filter_fn)
     30   result = []
     31   bfs_q = queue.Queue()
---> 32   bfs_q.put(dataset._variant_tensor.op)  # pylint: disable=protected-access
     33   visited = []
     34   while not bfs_q.empty():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __getattr__(self, name)
    258         tf.experimental.numpy.experimental_enable_numpy_behavior()
    259       """)
--> 260     self.__getattribute__(name)
    261 
    262   @property

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in op(self)
    574   @property
    575   def op(self) -> NoReturn:
--> 576     raise AttributeError(
    577         "Tensor.op is undefined when eager execution is enabled.")
    578 

AttributeError: Tensor.op is undefined when eager execution is enabled.

## === cell 10
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

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1091220921.py in <cell line: 0>()
      1 alpha = 0.40  # unchanged post-processing
----> 2 dog_prob = predictions[:, 0].astype(np.float64)
      3 dog_prob = alpha * dog_prob + (1.0 - alpha) * 0.5
      4 
      5 eps = 1e-7

NameError: name 'predictions' is not defined
