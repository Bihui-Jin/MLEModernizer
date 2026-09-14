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
tqdm==4.67.1

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

0.5

# 6. Current score

0.99989

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99998) has done: 'I fix the TensorFlow/protobuf startup crash and make the TF1-style graph code work under TF2 by explicitly disabling eager execution and using the compat.v1 API consistently. Then I fix the training loop bug where validation accidentally runs the optimizer (corrupting training and making results unstable), and ensure `BATCH`/`config` are defined by unblocking the earlier cells. Finally, I make the prediction + submission generation robust (correct batch sizing, guaranteed length match, and always writing `submission.csv` with the required columns).'
- What this solution (achieved 0.99986) has done: 'I fix the protobuf/TensorFlow import crash by setting the protobuf implementation environment variables before importing TensorFlow and forcing a compatible protobuf runtime API. Then I keep your TF1-graph-in-TF2 approach intact (disable eager + compat.v1) and leave the model/training logic unchanged to avoid materially changing the score (your current score is already far above the 0.5 target). Finally, I make the data root selection a bit more robust (choose the existing path) and ensure the submission is always written as `submission.csv` with the required `id,has_cactus` columns.'
- What this solution (achieved 0.99998) has done: 'I fix the TensorFlow/protobuf startup crash that prevents cell 1+ from running by setting compatible protobuf env vars before importing TensorFlow and forcing the pure-Python protobuf implementation. I keep your TF1-graph-under-TF2 approach intact (disable eager + compat.v1 placeholders/session) and leave the model/training logic unchanged to avoid score-changing behavior since you’re already far above the 0.5 target. I also make the data root selection slightly more robust (fall back to `/kaggle/input/...` and also your provided `/kaggle/data/...` layout) while keeping the same file usage. Finally, the script always write a valid `submission.csv` with `id,has_cactus`.'
- What this solution (achieved 0.99989) has done: 'The crash happens before any training because TensorFlow import triggers a protobuf API mismatch (`MessageFactory.GetPrototype`), which is common when TF and protobuf versions are out of sync. The minimal robust fix in Kaggle is to force the pure-Python protobuf implementation and (if needed) cap protobuf to the legacy API via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus ensuring we don’t import any protobuf-dependent modules before setting env vars. I also add a small fallback: if TensorFlow still fails due to protobuf, we use `tf_keras` (bundled with TF) and keep the same TF1-graph flow; this is score-neutral and just unblocks execution. No model/training logic be changed, and the script still always write a valid `submission.csv` with `id,has_cactus`.'
- What this solution (achieved 0.99984) has done: 'I fix the crash happening at TensorFlow import by forcing the pure‑Python protobuf implementation *before any protobuf/tensorflow-related import* and by explicitly removing any pre-imported `google.protobuf` modules that could have been loaded by the notebook runtime. This is the minimal change needed to unblock the whole pipeline; it is score-neutral because it does not change the model, training loop, or prediction logic. I also keep your current TF1-graph-under-TF2 approach intact (disable eager + compat.v1) and preserve all paths and submission formatting so `submission.csv` is reliably produced.'
- What this solution (achieved 0.99991) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime early and (critically) preventing TensorFlow from using the C++ protobuf implementation (which triggers the `MessageFactory.GetPrototype` error in this environment). I also keep your TF1-graph-in-TF2 approach intact (disable eager + use `tf.compat.v1`), preserving the model/training logic so score behavior remains essentially unchanged (you’re already far above the 0.5 target). Finally, I make data-root selection and submission writing robust so the notebook always finishes and writes a valid `submission.csv` with `id,has_cactus`.'
- What this solution (achieved 0.99991) has done: 'I fix the TensorFlow/protobuf import crash by forcing TensorFlow to use the Python protobuf runtime and by removing any already-imported protobuf modules before importing TensorFlow, but without changing your TF1-graph-in-TF2 training logic. I also make the environment selection more robust by disabling XLA/JIT (a common trigger for protobuf/C++ descriptor issues in some Kaggle images) while keeping your model, training loop, and submission format identical. Since your current AUC is far above the 0.5 target (and within the tolerance band), I not make any score-affecting changes—only stability/compatibility fixes so it runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.99989) has done: 'I fix the root cause of the crash by stopping the forced C++ protobuf runtime (which is missing `_message` here) and instead forcing the pure-Python protobuf implementation *before* importing TensorFlow, while still clearing any pre-imported `google.protobuf` modules. This unblock `tf`/`DATA_ROOT` initialization so all later cells can run without cascading `NameError`s. I keep your TF1-graph-under-TF2 approach (disable eager + `tf.compat.v1`) and the model/training logic unchanged, only making a small safety tweak to data-root selection and ensuring `submission.csv` is always written with the required `id,has_cactus` columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false"
)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import tensorflow as tf

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

tf.compat.v1.disable_eager_execution()

CANDIDATE_ROOTS = [
    "../input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "/kaggle/data/input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification/aerial-cactus-identification",
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.exists(p)), CANDIDATE_ROOTS[0])

TRAIN_IMAGES_PATH = os.path.join(DATA_ROOT, "train")
TEST_IMAGES_PATH = os.path.join(DATA_ROOT, "test")

print("TensorFlow:", tf.__version__)
print("DATA_ROOT:", DATA_ROOT)
print("DATA_ROOT exists:", os.path.exists(DATA_ROOT))
print("TRAIN_IMAGES_PATH exists:", os.path.exists(TRAIN_IMAGES_PATH))
print("TEST_IMAGES_PATH exists:", os.path.exists(TEST_IMAGES_PATH))

if not (os.path.exists(TRAIN_IMAGES_PATH) and os.path.exists(TEST_IMAGES_PATH)):
    raise FileNotFoundError(
        f"Could not find train/test folders under DATA_ROOT={DATA_ROOT}. "
        f"Checked: {TRAIN_IMAGES_PATH} and {TEST_IMAGES_PATH}"
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

train_image_ids = train_df["id"].values
training_labels = train_df["has_cactus"].values.astype(np.int64)

test_image_ids = test_df["id"].values

print(train_df.shape, test_df.shape)
print("Train positives:", int(training_labels.sum()), "/", len(training_labels))




## === cell 2
def get_images(folder_path, image_ids):
    all_images = []
    for image_name in tqdm(
        image_ids, desc=f"Loading from {os.path.basename(folder_path)}"
    ):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if image.shape[:2] != (32, 32):
            image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        all_images.append(image)
    input_images = np.stack(all_images).astype(np.float32)
    return input_images, input_images / 255.0




## === cell 3
all_train_images, normalized_images = get_images(TRAIN_IMAGES_PATH, train_image_ids)
test_images, normalized_test_images = get_images(TEST_IMAGES_PATH, test_image_ids)

print("Train images:", normalized_images.shape, normalized_images.dtype)
print("Test images:", normalized_test_images.shape, normalized_test_images.dtype)



## === cell 4
augs = [np.fliplr, np.flipud, np.rot90]


def augment(images, labels, augs):
    all_images = []
    all_labels = []
    for i, image in tqdm(list(enumerate(images)), total=len(images), desc="Augmenting"):
        all_images.append(image)
        cur_label = int(labels[i])
        all_labels.append(cur_label)

        if cur_label == 1:
            all_images.append(augs[random.randint(0, 2)](image))
            all_labels.append(cur_label)
        else:
            for aug in augs:
                all_images.append(aug(image))
                all_labels.append(cur_label)

    return np.stack(all_images).astype(np.float32), np.array(all_labels, dtype=np.int64)




## === cell 5
normalized_train_images, final_training_labels = augment(
    normalized_images, training_labels, augs
)

NUM_TRAIN_IMAGES = int(0.75 * normalized_train_images.shape[0])
indices = np.random.permutation(normalized_train_images.shape[0])
training_idx, val_idx = indices[:NUM_TRAIN_IMAGES], indices[NUM_TRAIN_IMAGES:]

train_data = normalized_train_images[training_idx, :]
train_labels = final_training_labels[training_idx]

val_data = normalized_train_images[val_idx, :]
val_labels = final_training_labels[val_idx]

print("Aug train images:", normalized_train_images.shape)
print(
    "Train split:",
    train_data.shape,
    train_labels.shape,
    "positives:",
    int(train_labels.sum()),
)
print(
    "Val split:", val_data.shape, val_labels.shape, "positives:", int(val_labels.sum())
)




## === cell 6
def get_weights(shape):
    initializer = tf.compat.v1.keras.initializers.glorot_uniform()
    return tf.Variable(initializer(shape=shape), name="weights")


def get_biases(length):
    return tf.Variable(tf.constant(0.0005, shape=[length]), name="bias")


def conv_layer(input_tensor, in_channels, filter_size, num_filters):
    shape = [filter_size, filter_size, in_channels, num_filters]
    weights = get_weights(shape)
    bias = get_biases(num_filters)
    layer = tf.nn.convolution(
        input_tensor,
        filters=weights,
        strides=[1, 1],
        dilations=[1, 1],
        padding="SAME",
    )
    layer = tf.nn.bias_add(layer, bias)
    new_layer = tf.nn.relu(layer)
    return new_layer, weights


def flatten(input_tensor):
    layer_shape = input_tensor.get_shape()
    total_elements = layer_shape[1:4].num_elements()
    layer = tf.reshape(input_tensor, [-1, total_elements])
    return layer, total_elements


def fc_layer(input_tensor, in_features, out_features):
    weight = get_weights([in_features, out_features])
    bias = get_biases(out_features)
    layer = tf.matmul(input_tensor, weight) + bias
    return layer




## === cell 7
tf.compat.v1.reset_default_graph()

input_image = tf.compat.v1.placeholder(
    name="input", shape=(None, 32, 32, 3), dtype=tf.float32
)
labels = tf.compat.v1.placeholder(name="labels", shape=(None,), dtype=tf.int64)

with tf.compat.v1.variable_scope("block1_conv1"):
    layer_conv1, weights_1 = conv_layer(input_image, 3, 3, 32)
with tf.compat.v1.variable_scope("block1_conv2"):
    layer_conv2, weights_2 = conv_layer(layer_conv1, 32, 3, 32)
with tf.compat.v1.variable_scope("block2_conv1"):
    layer_conv3, weights_3 = conv_layer(layer_conv2, 32, 3, 64)
with tf.compat.v1.variable_scope("block2_conv2"):
    layer_conv4, weights_4 = conv_layer(layer_conv3, 64, 3, 64)
    layer_output_pool = tf.nn.max_pool(
        layer_conv4, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding="VALID"
    )

with tf.compat.v1.variable_scope("block3_conv1"):
    layer_conv4b, weights_4b = conv_layer(layer_output_pool, 64, 3, 128)
with tf.compat.v1.variable_scope("block3_conv2"):
    layer_conv5, weights_5 = conv_layer(layer_conv4b, 128, 3, 128)
with tf.compat.v1.variable_scope("block3_conv3"):
    layer_conv6, weights_6 = conv_layer(layer_conv5, 128, 3, 128)
    layer_output_pool2 = tf.nn.max_pool(
        layer_conv6, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding="VALID"
    )

flattened_layer, in_features = flatten(layer_output_pool2)
fclyr = fc_layer(flattened_layer, in_features, 128)
fc_layer2 = tf.nn.relu(fclyr)

final_layer = fc_layer(fc_layer2, 128, 2)
y_pred = tf.nn.softmax(final_layer, name="y_pred")

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=final_layer, labels=tf.one_hot(labels, 2)
)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

init = tf.compat.v1.global_variables_initializer()



## === cell 8
sess = tf.compat.v1.Session(config=config)

NUM_ITERATIONS = 20
BATCH = 32

sess.run(init)

for i in tqdm(range(NUM_ITERATIONS), desc="Epochs"):
    num_batches = int(train_data.shape[0] / BATCH) + 1
    losses = []
    epoch_predictions = []

    for j in range(num_batches):
        batch_data = train_data[BATCH * j : BATCH * j + BATCH]
        batch_labels = train_labels[BATCH * j : BATCH * j + BATCH].astype(np.int64)
        if batch_data.shape[0] == 0:
            continue
        loss, _, probabilities = sess.run(
            [cost, optimizer, y_pred],
            feed_dict={input_image: batch_data, labels: batch_labels},
        )
        predictions = np.argmax(probabilities, axis=1)
        epoch_predictions.extend(predictions.tolist())
        losses.append(loss)

    epoch_predictions = np.array(epoch_predictions, dtype=np.int64)
    train_loss = float(np.mean(np.array(losses))) if losses else float("nan")
    train_accuracy = (
        np.sum(epoch_predictions == train_labels[: len(epoch_predictions)])
        / len(epoch_predictions)
    ) * 100.0

    num_batches_val = int(val_data.shape[0] / BATCH) + 1
    val_predictions = []
    val_losses = []
    for j in range(num_batches_val):
        batch_data = val_data[BATCH * j : BATCH * j + BATCH]
        batch_labels = val_labels[BATCH * j : BATCH * j + BATCH].astype(np.int64)
        if batch_data.shape[0] == 0:
            continue
        loss, probabilities = sess.run(
            [cost, y_pred],
            feed_dict={input_image: batch_data, labels: batch_labels},
        )
        predictions = np.argmax(probabilities, axis=1)
        val_predictions.extend(predictions.tolist())
        val_losses.append(loss)

    val_predictions = np.array(val_predictions, dtype=np.int64)
    val_accuracy = (
        np.sum(val_predictions == val_labels[: len(val_predictions)])
        / len(val_predictions)
    ) * 100.0
    val_loss = float(np.mean(np.array(val_losses))) if val_losses else float("nan")

    print(
        "Loss after Epoch - %f, Train Accuracy - %f, Val Accuracy - %f, Val Loss - %f"
        % (train_loss, train_accuracy, val_accuracy, val_loss)
    )



## === cell 9
test_pred_proba = []
num_batches_test = int(normalized_test_images.shape[0] / BATCH) + 1

dummy_labels = np.zeros((BATCH,), dtype=np.int64)

for j in range(num_batches_test):
    batch_data = normalized_test_images[BATCH * j : BATCH * j + BATCH]
    if batch_data.shape[0] == 0:
        continue
    batch_dummy = dummy_labels[: batch_data.shape[0]]
    probabilities = sess.run(
        y_pred, feed_dict={input_image: batch_data, labels: batch_dummy}
    )
    test_pred_proba.extend(probabilities[:, 1].tolist())

test_pred_proba = np.array(test_pred_proba, dtype=np.float32)
print(
    "Preds:",
    test_pred_proba.shape,
    "min/max:",
    float(test_pred_proba.min()),
    float(test_pred_proba.max()),
)

assert len(test_pred_proba) == len(
    test_df
), f"Prediction length mismatch: {len(test_pred_proba)} vs {len(test_df)}"



## === cell 10
submission_df = test_df.copy()
submission_df["has_cactus"] = test_pred_proba.astype(np.float32)
submission_df = submission_df[["id", "has_cactus"]]

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print(submission_df.shape)
