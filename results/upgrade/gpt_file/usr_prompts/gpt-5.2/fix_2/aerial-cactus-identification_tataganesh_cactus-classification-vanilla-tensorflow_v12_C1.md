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

0.9906

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

tf.compat.v1.disable_eager_execution()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_CANDIDATES = [
    "../input/aerial-cactus-identification",
    "../input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
]

BASE_PATH = None
for p in BASE_CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
        BASE_PATH = p
        break

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate aerial-cactus-identification dataset under expected ../input or /kaggle/input paths."
    )

TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, "train", "train")
TEST_IMAGES_PATH = os.path.join(BASE_PATH, "test", "test")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

print("BASE_PATH =", BASE_PATH)
print("TRAIN_IMAGES_PATH exists:", os.path.exists(TRAIN_IMAGES_PATH))
print("TEST_IMAGES_PATH exists:", os.path.exists(TEST_IMAGES_PATH))



## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(SAMPLE_SUB_PATH)

train_image_ids = train_df["id"].astype(str).values
training_labels = train_df["has_cactus"].astype(np.int64).values

test_image_ids = test_df["id"].astype(str).values
test_labels = test_df["has_cactus"].astype(np.int64).values

print(train_df.shape, test_df.shape)




## === cell 3
def get_images(folder_path, image_ids):
    """
    Function to read images from disk and normalize them
    """
    all_images = []
    for image_name in tqdm(
        image_ids, desc=f"Loading from {os.path.basename(os.path.dirname(folder_path))}"
    ):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed for: {image_path}")
        all_images.append(image)
    input_images = np.stack(all_images).astype(np.float32)
    return input_images, input_images / 255.0




## === cell 4
all_train_images, normalized_images = get_images(TRAIN_IMAGES_PATH, train_image_ids)
test_images, normalized_test_images = get_images(TEST_IMAGES_PATH, test_image_ids)

print("Train images:", normalized_images.shape, normalized_images.dtype)
print("Test images:", normalized_test_images.shape, normalized_test_images.dtype)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1777179337.py in <cell line: 0>()
----> 1 all_train_images, normalized_images = get_images(TRAIN_IMAGES_PATH, train_image_ids)
      2 test_images, normalized_test_images = get_images(TEST_IMAGES_PATH, test_image_ids)
      3 
      4 print("Train images:", normalized_images.shape, normalized_images.dtype)
      5 print("Test images:", normalized_test_images.shape, normalized_test_images.dtype)

/tmp/ipykernel_11/1651128083.py in get_images(folder_path, image_ids)
     10         image = cv2.imread(image_path)
     11         if image is None:
---> 12             raise FileNotFoundError(f"cv2.imread failed for: {image_path}")
     13         all_images.append(image)
     14     input_images = np.stack(all_images).astype(np.float32)

FileNotFoundError: cv2.imread failed for: ../input/aerial-cactus-identification/train/train/2de8f189f1dce439766637e75df0ee27.jpg

## === cell 5
augs = [
    np.fliplr,
    np.flipud,
    np.rot90,
]  # List of augmentations to be applied to the data


def augment(images, labels, augs):
    """
    Apply data augmentation to all training images
    To tackle class imbalance, apply one of the augmentations ( Randomly chosen )
    to each image having label 1, and apply all transformations to image having
    label 0.
    """
    all_images = []
    all_labels = []
    for i, image in tqdm(list(enumerate(images)), desc="Augmenting"):
        all_images.append(image)
        cur_label = labels[i]
        all_labels.append(cur_label)
        if cur_label == 1:
            all_images.append(augs[random.randint(0, 2)](image))
            all_labels.append(cur_label)
        else:
            for aug in augs:
                all_labels.append(cur_label)
                all_images.append(aug(image))

    return np.stack(all_images).astype(np.float32), np.array(all_labels, dtype=np.int64)




## === cell 6
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

print("After aug:", normalized_train_images.shape, final_training_labels.shape)
print("Split:", train_data.shape, val_data.shape)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1224581734.py in <cell line: 0>()
      1 normalized_train_images, final_training_labels = augment(
----> 2     normalized_images, training_labels, augs
      3 )
      4 
      5 NUM_TRAIN_IMAGES = int(0.75 * normalized_train_images.shape[0])

NameError: name 'normalized_images' is not defined

## === cell 7
def show_images_horizontally(images, labels=None, lookup_label=None, figsize=(15, 3)):
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [None] * images.shape[0]
    fig = plt.figure(figsize=figsize)
    for i in range(images.shape[0]):
        ax = fig.add_subplot(1, images.shape[0], i + 1)
        if lookup_label is not None and labels[i] is not None:
            ax.set_title(lookup_label[int(labels[i])])
        ax.imshow(images[i])
        ax.axis("off")
    plt.show()


print("Label sample:", final_training_labels[:10])




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2564010463.py in <cell line: 0>()
     15 
     16 # Optional sanity check (kept but small to avoid wasting time)
---> 17 print("Label sample:", final_training_labels[:10])
     18 # show_images_horizontally((normalized_train_images[10:15] * 255).astype(np.uint8), final_training_labels[10:15], lookup_label={1:"has_cactus", 0:"no_cactus"})
     19 

NameError: name 'final_training_labels' is not defined

## === cell 8
def get_weights(shape):
    """
    Weights initializer
    """
    initializer = tf.compat.v1.keras.initializers.glorot_uniform()
    return tf.Variable(initializer(shape=shape), dtype=tf.float32)


def get_biases(length):
    """
    Initializing bias
    """
    return tf.Variable(tf.constant(0.0005, shape=[length], dtype=tf.float32))




## === cell 9
def conv_layer(input_tensor, in_channels, filter_size, num_filters):
    """
    Apply convolution operation to the image
    """
    shape = [filter_size, filter_size, in_channels, num_filters]
    weights = get_weights(shape)
    bias = get_biases(num_filters)
    layer = tf.nn.convolution(
        input_tensor,
        weights,
        strides=[1, 1],
        dilations=[1, 1],
        padding="SAME",
    )
    layer = tf.nn.bias_add(layer, bias)
    new_layer = tf.nn.relu(layer)
    return new_layer, weights


def flatten(input_tensor):
    """
    Flattens input tensor
    """
    layer_shape = input_tensor.get_shape()
    total_elements = layer_shape[1:4].num_elements()
    layer = tf.reshape(input_tensor, [-1, total_elements])
    return layer, total_elements


def fc_layer(input_tensor, in_features, out_features):
    """
    Create fully connected layer
    """
    weight = get_weights([in_features, out_features])
    bias = get_biases(out_features)
    layer = tf.matmul(input_tensor, weight) + bias
    return layer




## === cell 10
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
    layer_output_pool = tf.nn.max_pool2d(
        layer_conv4, ksize=4, strides=2, padding="VALID"
    )

with tf.compat.v1.variable_scope("block3_conv1"):
    layer_conv4b, weights_4b = conv_layer(layer_output_pool, 64, 3, 128)
with tf.compat.v1.variable_scope("block3_conv2"):
    layer_conv5, weights_5 = conv_layer(layer_conv4b, 128, 3, 128)
with tf.compat.v1.variable_scope("block3_conv3"):
    layer_conv6, weights_6 = conv_layer(layer_conv5, 128, 3, 128)
    layer_output_pool2 = tf.nn.max_pool2d(
        layer_conv6, ksize=4, strides=2, padding="VALID"
    )

flattened_layer, in_features = flatten(layer_output_pool2)
fclyr = fc_layer(flattened_layer, in_features, 128)
fc_layer2 = tf.nn.relu(fclyr)

final_layer = fc_layer(fc_layer2, 128, 2)
y_pred = tf.nn.softmax(final_layer, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=final_layer, labels=tf.one_hot(labels, 2)
)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
init = tf.compat.v1.global_variables_initializer()



## === cell 11
sess = tf.compat.v1.Session(config=config)

NUM_ITERATIONS = 20
BATCH = 32

sess.run(init)

for i in tqdm(range(NUM_ITERATIONS), desc="Training epochs"):
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
    train_loss = float(np.mean(np.array(losses))) if len(losses) else np.nan
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

    print(
        "EPOCH %d | Loss %.6f | Train Acc %.3f | Val Acc %.3f"
        % (i, train_loss, train_accuracy, val_accuracy)
    )



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1987355516.py in <cell line: 0>()
      7 
      8 for i in tqdm(range(NUM_ITERATIONS), desc="Training epochs"):
----> 9     num_batches = int(train_data.shape[0] / BATCH) + 1
     10     losses = []
     11     epoch_predictions = []

NameError: name 'train_data' is not defined

## === cell 12
test_probs = []
num_batches = int(normalized_test_images.shape[0] / BATCH) + 1

for j in range(num_batches):
    batch_data = normalized_test_images[BATCH * j : BATCH * j + BATCH]
    if batch_data.shape[0] == 0:
        continue
    probabilities = sess.run(y_pred, feed_dict={input_image: batch_data})
    test_probs.extend(probabilities[:, 1].tolist())

test_probs = np.array(test_probs, dtype=np.float32)
print(
    "Preds:",
    test_probs.shape,
    "min/max:",
    float(test_probs.min()),
    float(test_probs.max()),
)

if len(test_probs) != len(test_df):
    raise ValueError(
        f"Prediction length {len(test_probs)} does not match submission length {len(test_df)}"
    )

submission_df = test_df.copy()
submission_df["has_cactus"] = test_probs

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote", submission_path, submission_df.shape)
print(submission_df.head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/246034069.py in <cell line: 0>()
      2 # Use model's softmax probability for class 1.
      3 test_probs = []
----> 4 num_batches = int(normalized_test_images.shape[0] / BATCH) + 1
      5 
      6 for j in range(num_batches):

NameError: name 'normalized_test_images' is not defined
