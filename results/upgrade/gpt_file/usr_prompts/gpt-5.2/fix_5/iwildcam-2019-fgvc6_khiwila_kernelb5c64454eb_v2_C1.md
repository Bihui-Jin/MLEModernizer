# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Label images of animals with their species.

## Metric
Macro F1 score

## Submission Format
```
Id,Predicted
58857ccf-23d2-11e8-a6a3-ec086b02610b,1
591e4006-23d2-11e8-a6a3-ec086b02610b,5
```

The `Id` column corresponds to the test image id. The `Category` is an integer value that indicates the class of the animal, or `0` to represent the absence of an animal.

## Dataset
The training set contains 196,157 images from 138 different locations in Southern California. 

The test set contains 153,730 images from 100 locations in Idaho.

The task is to label each image with one of the following label ids:

```
name, id
empty, 0
deer, 1
moose, 2
squirrel, 3
rodent, 4
small_mammal, 5
elk, 6
pronghorn_antelope, 7
rabbit, 8
bighorn_sheep, 9
fox, 10
coyote, 11
black_bear, 12
raccoon, 13
skunk, 14
wolf, 15
bobcat, 16
cat, 17
dog, 18
opossum, 19
bison, 20
mountain_goat, 21
mountain_lion, 22
```

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (109 lines)
            sample_submission.csv (16878 lines)
            sample_submission.csv.zip (133.7 kB)
            test.csv (16878 lines)
            test.csv.zip (415.9 kB)
            test.zip (160 Bytes)
            test_images.zip (1.8 GB)
            train.csv (179423 lines)
            train.csv.zip (4.7 MB)
            train.zip (162 Bytes)
            train_images.zip (26.1 GB)
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
            test/
                test/
            test_images/
                59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                ... and 16860 other files
                test_images/
            train/
                train/
            train_images/
                598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                ... and 179222 other files
                train_images/
        input/
            description.md (109 lines)
            sample_submission.csv (16878 lines)
            sample_submission.csv.zip (133.7 kB)
            test.csv (16878 lines)
            test.csv.zip (415.9 kB)
            test.zip (160 Bytes)
            test_images.zip (1.8 GB)
            train.csv (179423 lines)
            train.csv.zip (4.7 MB)
            train.zip (162 Bytes)
            train_images.zip (26.1 GB)
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
            test/
                test/
                    test/
            test_images/
                59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                ... and 16860 other files
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                ... and 179222 other files
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
        working/
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
```

-> data/iwildcam-2019-fgvc6/sample_submission.csv has 16877 rows and 3 columns.
The columns are: Unnamed: 0, Id, Category

-> data/iwildcam-2019-fgvc6/test.csv has 16877 rows and 10 columns.
The columns are: date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/iwildcam-2019-fgvc6/train.csv has 179422 rows and 11 columns.
The columns are: category_id, date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/sample_submission.csv has 16877 rows and 3 columns.
The columns are: Unnamed: 0, Id, Category

-> data/test.csv has 16877 rows and 10 columns.
The columns are: date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/train.csv has 179422 rows and 11 columns.
The columns are: category_id, date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_ROOT = "../input/iwildcam-2019-fgvc6"
WORKING_ROOT = "../working"

os.makedirs(WORKING_ROOT, exist_ok=True)

print("Listing ../input:")
print(os.listdir("../input")[:50])
print("\nListing competition folder:")
print(os.listdir(INPUT_ROOT)[:50])



## === cell 1
train = pd.read_csv(f"{INPUT_ROOT}/train.csv")
test = pd.read_csv(f"{INPUT_ROOT}/test.csv")
sample_submission = pd.read_csv(f"{INPUT_ROOT}/sample_submission.csv")

print("train.shape:", train.shape)
print("test.shape:", test.shape)
print("sample_submission.shape:", sample_submission.shape)
print("sample_submission columns:", sample_submission.columns.tolist())

TRAIN_IMG_DIR = f"{INPUT_ROOT}/train_images"
TEST_IMG_DIR = f"{INPUT_ROOT}/test_images"

print("TRAIN_IMG_DIR exists:", os.path.isdir(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.isdir(TEST_IMG_DIR))



## === cell 2
import cv2
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

import tensorflow as tf

os.environ.setdefault("PYTHONHASHSEED", "1")
np.random.seed(1)
tf.random.set_seed(1)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow version:", tf.__version__)



## === cell 3
print(sample_submission.head())

vc = train["category_id"].value_counts()
print("Top category_id counts:\n", vc.head(10))

first_img_path = os.path.join(TRAIN_IMG_DIR, train["file_name"].iloc[0])
img = plt.imread(first_img_path)
print("First train image:", first_img_path, "shape:", img.shape)




## === cell 4
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[np.array(Y).reshape(-1)].T
    return Y




## === cell 5
etiquetas = train["category_id"].values.astype(np.int64)
y_train = np.eye(23, dtype=np.float32)[etiquetas]  # (m,23)

y_train_path = os.path.join(WORKING_ROOT, "y_train.npy")
np.save(y_train_path, y_train)
print("Saved:", y_train_path, "shape:", y_train.shape)



## === cell 6
from concurrent.futures import ThreadPoolExecutor


def _read_resize_one(args):
    fn, img_dir, target_size = args
    p = os.path.join(img_dir, fn)
    im = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)
    return im


def load_images_from_filenames(
    file_names,
    img_dir,
    target_size=(32, 32),
    max_items=None,
    num_workers=None,
    chunksize=512,
):
    """
    Loads RGB images given file_names (from CSV), resizes to target_size, returns uint8 array.
    """
    if max_items is not None:
        file_names = file_names[:max_items]
    n = len(file_names)
    X = np.empty((n, target_size[0], target_size[1], 3), dtype=np.uint8)

    if num_workers is None:
        num_workers = min(16, (os.cpu_count() or 4))

    missing = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        it = ex.map(
            _read_resize_one,
            ((fn, img_dir, target_size) for fn in file_names),
            chunksize=chunksize,
        )
        for i, im in enumerate(it):
            if im is None:
                X[i] = 0
                missing += 1
            else:
                X[i] = im

    if missing:
        print(f"Warning: {missing}/{n} images missing or unreadable in {img_dir}")
    return X


train_files = train["file_name"].tolist()
test_files = test["file_name"].tolist()

x_train_arr_path = os.path.join(WORKING_ROOT, "X_train_32x32.npy")
x_test_arr_path = os.path.join(WORKING_ROOT, "X_test_32x32.npy")

if os.path.exists(x_train_arr_path) and os.path.exists(x_test_arr_path):
    x_train_arr = np.load(x_train_arr_path, mmap_mode="r")
    x_test_arr = np.load(x_test_arr_path, mmap_mode="r")
    print("Loaded cached arrays from ../working with mmap")
else:
    print("Loading and resizing train images to 32x32...")
    x_train_arr = load_images_from_filenames(
        train_files, TRAIN_IMG_DIR, target_size=(32, 32)
    )
    print("Loading and resizing test images to 32x32...")
    x_test_arr = load_images_from_filenames(
        test_files, TEST_IMG_DIR, target_size=(32, 32)
    )
    np.save(x_train_arr_path, x_train_arr)
    np.save(x_test_arr_path, x_test_arr)
    print("Saved cached arrays to ../working")

y_train_arr = np.load(y_train_path, mmap_mode="r")

print("x_train_arr shape:", x_train_arr.shape, x_train_arr.dtype)
print("x_test_arr shape:", x_test_arr.shape, x_test_arr.dtype)
print("y_train_arr shape:", y_train_arr.shape, y_train_arr.dtype)

print(x_train_arr.shape[0], "train samples")
print(x_test_arr.shape[0], "test samples")



## === cell 7
x_train = x_train_arr  # uint8 (mmap)
y_train = y_train_arr.astype(np.float32, copy=False)
x_test = x_test_arr  # uint8 (mmap)

print("x_train.shape:", x_train.shape, x_train.dtype)
print("y_train.shape:", y_train.shape, y_train.dtype)
print("x_test.shape:", x_test.shape, x_test.dtype)
print("First one-hot labels:", y_train[0], y_train[1], y_train[2])



## === cell 8
labels_int = train["category_id"].values.astype(np.int64)

x_train, x_dev, y_train, y_dev, lab_train, lab_dev = train_test_split(
    x_train,
    y_train,
    labels_int,
    test_size=0.4,
    random_state=32,
    stratify=labels_int,
)
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_dev.shape:", x_dev.shape)
print("y_dev.shape:", y_dev.shape)




## === cell 9
def _split_indices(n, test_size, seed):
    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    if isinstance(test_size, float):
        n_test = int(np.ceil(n * test_size))
    else:
        n_test = int(test_size)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


n_test = x_test.shape[0]
idx_a, idx_b = _split_indices(n_test, test_size=0.5, seed=32)
idx_1, idx_2 = _split_indices(len(idx_a), test_size=0.5, seed=32)
idx_3, idx_4 = _split_indices(len(idx_b), test_size=0.5, seed=32)

x_test1 = x_test[idx_a[idx_1]]
x_test2 = x_test[idx_a[idx_2]]
x_test3 = x_test[idx_b[idx_3]]
x_test4 = x_test[idx_b[idx_4]]

print("x_test1.shape:", x_test1.shape)
print("x_test2.shape:", x_test2.shape)
print("x_test3.shape:", x_test3.shape)
print("x_test4.shape:", x_test4.shape)
print(
    "Total test parts:",
    x_test1.shape[0] + x_test2.shape[0] + x_test3.shape[0] + x_test4.shape[0],
)



## === cell 10
print("test id head:", test["id"].head().tolist())
print("sample_submission Id head:", sample_submission["Id"].head().tolist())
print(
    "Id match ratio:",
    (
        np.mean(sample_submission["Id"].values == test["id"].values)
        if len(sample_submission) == len(test)
        else "len mismatch"
    ),
)




## === cell 11
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.keras.Input(shape=(n_H0, n_W0, n_C0), dtype=tf.float32, name="X")
    Y = tf.keras.Input(shape=(n_y,), dtype=tf.float32, name="Y")
    return X, Y


def initialize_parameters():
    return {}


def forward_propagation(X, parameters):
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
        name="conv1",
    )(X)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(8, 8), strides=(8, 8), padding="same", name="pool1"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
        name="conv2",
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(4, 4), strides=(4, 4), padding="same", name="pool2"
    )(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    Z3 = tf.keras.layers.Dense(23, activation=None, name="dense")(x)
    return Z3


def compute_cost(Z3, Y):
    return tf.reduce_mean(tf.keras.losses.mse(Y, Z3))




## === cell 12
def iter_minibatches_indices(m, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    perm = np.random.permutation(m)
    for start in range(0, m, mini_batch_size):
        end = min(start + mini_batch_size, m)
        yield perm[start:end]




## === cell 13
def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    X_test_test,
    X_test_test1,
    X_test_test2,
    X_test_test3,
    X_test_test4,
    learning_rate=0.009,
    num_epochs=20,
    minibatch_size=64,
    print_cost=True,
):
    tf.random.set_seed(1)
    np.random.seed(1)

    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X_in, Y_in = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X_in, parameters)
    net = tf.keras.Model(inputs=X_in, outputs=Z3)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = net(xb, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mse(yb, logits))
        grads = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss

    def make_epoch_dataset(indices):
        ds = tf.data.Dataset.from_tensor_slices(indices)
        ds = ds.batch(minibatch_size, drop_remainder=False)

        def _gather_and_normalize(ix):
            xb = tf.gather(X_train, ix)  # uint8 -> gathered
            xb = tf.cast(xb, tf.float32) * (1.0 / 255.0)
            yb = tf.gather(Y_train, ix)
            return xb, yb

        ds = ds.map(_gather_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    for epoch in range(num_epochs):
        minibatch_cost = tf.constant(0.0, dtype=tf.float32)
        num_minibatches = int(m / minibatch_size) if m >= minibatch_size else 1
        seed = 3 + epoch

        perm = np.random.RandomState(seed).permutation(m).astype(np.int32, copy=False)
        epoch_ds = make_epoch_dataset(perm)

        for xb, yb in epoch_ds:
            temp_cost = train_step(xb, yb)
            minibatch_cost += temp_cost / float(num_minibatches)

        if print_cost and epoch % 5 == 0:
            print("Cost after epoch %i: %f" % (epoch, float(minibatch_cost.numpy())))
        if print_cost and epoch % 1 == 0:
            costs.append(float(minibatch_cost.numpy()))

    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("epochs")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    def accuracy_eval(x_uint8, y, batch_size=4096):
        n = x_uint8.shape[0]
        correct = 0
        total = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = tf.cast(x_uint8[start:end], tf.float32) * (1.0 / 255.0)
            logits = net(xb, training=False)
            pred = tf.argmax(logits, axis=1, output_type=tf.int64)
            true = tf.argmax(y[start:end], axis=1, output_type=tf.int64)
            correct += int(
                tf.reduce_sum(tf.cast(tf.equal(pred, true), tf.int32)).numpy()
            )
            total += end - start
        return float(correct / total)

    if X_train.shape[0] > 25000:
        train_accuracy = accuracy_eval(X_train[:25000], Y_train[:25000])
        number_for_train_accuracy = 25000
    else:
        train_accuracy = accuracy_eval(X_train, Y_train)
        number_for_train_accuracy = X_train.shape[0]

    if X_test.shape[0] > 25000:
        test_accuracy = accuracy_eval(X_test[:25000], Y_test[:25000])
        number_for_test_accuracy = 25000
    else:
        test_accuracy = accuracy_eval(X_test, Y_test)
        number_for_test_accuracy = X_test.shape[0]

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    def predict_classes_batched(x_uint8, batch_size=4096):
        n = x_uint8.shape[0]
        out = np.empty((n,), dtype=np.int64)
        k = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = tf.cast(x_uint8[start:end], tf.float32) * (1.0 / 255.0)
            logits = net(xb, training=False)
            pred = tf.argmax(logits, axis=1, output_type=tf.int64).numpy()
            out[k : k + pred.shape[0]] = pred
            k += pred.shape[0]
        return out

    test_results_arr = predict_classes_batched(X_test_test)

    if test_results_arr.shape[0] != X_test_test.shape[0]:
        raise ValueError(
            f"Prediction length mismatch: got {test_results_arr.shape[0]} expected {X_test_test.shape[0]}"
        )

    submission = pd.DataFrame(
        {
            "Id": test["id"].values[: X_test_test.shape[0]],
            "Predicted": test_results_arr.astype(np.int64, copy=False),
        }
    )
    out_path = os.path.join(WORKING_ROOT, "submission_got_it2.csv")
    submission.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print("Used in training set:", X_train.shape[0], "elements")
    print("Used in validation set:", X_test.shape[0], "elements")
    print("Used in prediction set:", int(test_results_arr.shape[0]), "elements")
    print("Used for train accuracy:", number_for_train_accuracy, "elements")
    print("Used for test accuracy:", number_for_test_accuracy, "elements")
    print(submission.head())
    print("Summary of predictions:")
    print(submission["Predicted"].value_counts().head(10))

    return train_accuracy, test_accuracy, net




## === cell 14
_, _, parameters = model(
    x_train,
    y_train,
    x_dev,
    y_dev,
    x_test,
    x_test1,
    x_test2,
    x_test3,
    x_test4,
    learning_rate=0.0009,
    num_epochs=100,
    minibatch_size=64,
    print_cost=True,
)



## === cell 15
print("Files in ../working:")
print(os.listdir("../working")[:50])
print("Submission exists:", os.path.exists("../working/submission_got_it2.csv"))

sub_path = os.path.join(WORKING_ROOT, "submission_got_it2.csv")
sub = pd.read_csv(sub_path)
print("submission shape:", sub.shape)
print("submission columns:", sub.columns.tolist())
print(sub.head())
