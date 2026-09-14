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

0.9943

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input/train/"))
import keras
import cv2
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import random
np.random.seed(0)
from tqdm import tqdm_notebook


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_IMAGES_PATH = '../input/train/train'
TEST_IMAGES_PATH = '../input/test/test'


## === cell 2
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/sample_submission.csv')
train_image_ids = train_df['id']
training_labels = train_df['has_cactus']
test_image_ids = test_df['id']
test_labels = test_df['has_cactus']


## === cell 3
def get_images(folder_path, image_ids):
    """
    Function to read images from disk and normalize them
    """
    all_images = list()
    for image_name in tqdm_notebook(image_ids):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        all_images.append(image)
    input_images = np.stack(all_images)
    return input_images, input_images / 255
    


## === cell 4
all_train_images, normalized_images = get_images(TRAIN_IMAGES_PATH, train_image_ids)
test_images, normalized_test_images = get_images(TEST_IMAGES_PATH, test_image_ids)


## === cell 5
augs = [np.fliplr, np.flipud, np.rot90] # List of augmentations to be applied to the data
def augment(images, labels, augs):
    """
    Apply data augmentation to all training images
    To tackle class imbalance, apply one of the augmentations ( Randomly chosen )
    to each image having label 1, and apply all transformations to image having
    label 0.
    """
    all_images = list()
    all_labels = list()
    for i,image in tqdm_notebook(enumerate(images)):
        all_images.append(image)
        cur_label = labels[i]
        all_labels.append(cur_label)
        if cur_label == 1:
            all_images.append(augs[random.randint(0,2)](image))
            all_labels.append(cur_label)
        else:
            for aug in augs:
                all_labels.append(cur_label)
                all_images.append(aug(image))
    
    return np.stack(all_images), np.array(all_labels)


## === cell 6
normalized_train_images, final_training_labels = augment(normalized_images, np.array(training_labels), augs)
NUM_TRAIN_IMAGES = int(0.75 * normalized_train_images.shape[0])
indices = np.random.permutation(normalized_train_images.shape[0])
training_idx, val_idx = indices[:NUM_TRAIN_IMAGES], indices[NUM_TRAIN_IMAGES:]
train_data = normalized_train_images[training_idx,:]
train_labels = np.array(final_training_labels)[training_idx]

val_data = normalized_train_images[val_idx,:]
val_labels = final_training_labels[val_idx]


## === cell 7
def show_images_horizontally(images, labels=[], lookup_label=None,
                            figsize=(15, 30)):
    """
    Utility function to show images
    """

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure, imshow, axis
    print(labels[0])
    fig = figure(figsize=figsize)
    for i in range(images.shape[0]):
        fig.add_subplot(1, images.shape[0], i + 1)
        if lookup_label:
            plt.title(lookup_label[labels[i]])
        imshow(images[i])
        axis('off')
print(final_training_labels[:30])

show_images_horizontally(normalized_train_images[10:20], np.array(final_training_labels[10:20]), lookup_label={1:"has_cactus", 0: "no_cactus"})


## === cell 8
def get_weights(shape):
  """
  Weights initializer
  """
  initializer = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initializer(shape=shape))
  
def get_biases(length):
    """
    Initializing bias
    """
    return tf.Variable(tf.constant(0.0005, shape=[length]))
    


## === cell 9
def conv_layer(input, in_channels, filter_size, num_filters):
    """
    Apply convolution operation to the image
    """
    shape = [filter_size, filter_size, in_channels, num_filters]
    weights = get_weights(shape)
    bias = get_biases(num_filters)
    layer = tf.nn.convolution(input,
    weights,
    strides=[1,1], dilation_rate=[1,1], 
    padding="SAME")
    layer= tf.nn.bias_add(layer, bias)
    new_layer = tf.nn.relu(layer)
    print(new_layer)
    return new_layer, weights


def flatten(input_tensor):
    """
    Flattens input tensor
    """
    layer_shape = input_tensor.get_shape()
    total_elements = layer_shape[1:4].num_elements()
    layer = tf.reshape(input_tensor, [-1, total_elements])
    return layer, total_elements


def fc_layer(input, in_features, out_features):
  """
  Create fully connected layer
  """
  weight = get_weights([in_features, out_features])
  bias = get_biases(out_features)
  layer = tf.matmul(input, weight) + bias
  return layer


## === cell 10
input_image = tf.placeholder(name="input", shape=(None, 32, 32, 3), dtype=tf.float32)
labels = tf.placeholder(name="labels", shape=(None), dtype=tf.int64)


with tf.variable_scope("block1_conv1"):
  layer_conv1, weights_1 = conv_layer(input_image, 3, 3, 32)
with tf.variable_scope("block1_conv2"):
  layer_conv2, weights_2 = conv_layer(layer_conv1, 32, 3, 32)
with tf.variable_scope("block2_conv1"):
  layer_conv3, weights_3 = conv_layer(layer_conv2, 32, 3, 64)
with tf.variable_scope("block2_conv2"):
  layer_conv4, weights_4 = conv_layer(layer_conv3, 64, 3, 64)
  layer_output_pool = tf.nn.max_pool(layer_conv4, ksize = [1,4,4,1], strides = [1,2,2,1], padding = "VALID")

with tf.variable_scope("block3_conv1"):
  layer_conv4, weights_4 = conv_layer(layer_output_pool, 64, 3, 128)
with tf.variable_scope("block3_conv2"):
  layer_conv5, weights_5 = conv_layer(layer_conv4, 128, 3, 128)
with tf.variable_scope("block3_conv3"):
  layer_conv6, weights_6 = conv_layer(layer_conv5, 128, 3, 128)
  layer_output_pool = tf.nn.max_pool(layer_conv6, ksize = [1,4,4,1], strides = [1,2,2,1], padding = "VALID")
print(layer_output_pool, layer_conv4)
flattened_layer, in_features = flatten(layer_output_pool)
fclyr = fc_layer(flattened_layer, in_features, 128)
fc_layer2 = tf.nn.relu(fclyr)

final_layer = fc_layer(fc_layer2, 128, 2)
y_pred = tf.nn.softmax(final_layer)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer,
                                                        labels=tf.one_hot(labels,2))
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
init=tf.global_variables_initializer()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3439437284.py in <cell line: 0>()
----> 1 input_image = tf.placeholder(name="input", shape=(None, 32, 32, 3), dtype=tf.float32)
      2 labels = tf.placeholder(name="labels", shape=(None), dtype=tf.int64)
      3 # input_image = tf.cast(input_image, tf.float32)
      4 
      5 

AttributeError: module 'tensorflow' has no attribute 'placeholder'

## === cell 11
sess = tf.Session(config = config)
NUM_ITERATIONS = 20
BATCH = 32
sess.run(init)
for i in tqdm_notebook(range(NUM_ITERATIONS)):
    num_batches = int(train_data.shape[0] / BATCH) + 1
    losses = list()
    epoch_predictions = list()
    for j in tqdm_notebook(range(num_batches)):
        batch_labels = train_labels[BATCH*j: BATCH*j + BATCH].astype(np.int64)
        batch_data = train_data[BATCH*j: BATCH*j + BATCH]
        loss, _, probabilities = sess.run([cost, optimizer, y_pred], feed_dict={input_image:batch_data, labels:batch_labels})
        predictions = np.argmax(probabilities, axis=1)
        epoch_predictions.extend(predictions)
        losses.append(loss)
    print("EPOCH " + str(i))
    epoch_predictions = np.array(epoch_predictions)
    train_loss = np.mean(np.array(losses))
    train_accuracy = (np.sum(epoch_predictions==train_labels) / train_labels.shape[0]) * 100
    num_batches = int(val_data.shape[0] / BATCH) + 1
    val_predictions = list()
    for j in range(num_batches):
        batch_data = val_data[BATCH*j: BATCH*j + BATCH]
        batch_labels = val_labels[BATCH*j: BATCH*j + BATCH].astype(np.int64)
        loss, _, probabilities = sess.run([cost, optimizer, y_pred], feed_dict={input_image:batch_data, labels:batch_labels})
        predictions = np.argmax(probabilities, axis=1)
        val_predictions.extend(predictions)
        losses.append(loss)
    val_accuracy = (np.sum(val_predictions==val_labels) / val_labels.shape[0]) * 100
    print("Loss after Epoch - %f, Train Accuracy - %f, Val Accuracy - %f" % (train_loss, train_accuracy, val_accuracy))
        
        
        


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/876200846.py in <cell line: 0>()
----> 1 sess = tf.Session(config = config)
      2 NUM_ITERATIONS = 20
      3 BATCH = 32
      4 sess.run(init)
      5 for i in tqdm_notebook(range(NUM_ITERATIONS)):

AttributeError: module 'tensorflow' has no attribute 'Session'

## === cell 12
test_preds = list()
num_batches = int(normalized_test_images.shape[0] / BATCH) + 1
for j in range(num_batches):
    batch_data = normalized_test_images[BATCH*j: BATCH*j + BATCH]
    batch_labels = test_labels[BATCH*j: BATCH*j + BATCH].astype(np.int64)
    probabilities = sess.run(y_pred, feed_dict={input_image:batch_data, labels:batch_labels})
    predictions = np.argmax(probabilities, axis=1)
    test_preds.extend(predictions)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1143523136.py in <cell line: 0>()
      1 test_preds = list()
----> 2 num_batches = int(normalized_test_images.shape[0] / BATCH) + 1
      3 for j in range(num_batches):
      4     batch_data = normalized_test_images[BATCH*j: BATCH*j + BATCH]
      5     batch_labels = test_labels[BATCH*j: BATCH*j + BATCH].astype(np.int64)

NameError: name 'BATCH' is not defined

## === cell 13
submission_df = test_df
submission_df['has_cactus'] = np.array(test_preds)
submission_df.to_csv("submissions.csv",index=False)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1064999847.py in <cell line: 0>()
      1 submission_df = test_df
----> 2 submission_df['has_cactus'] = np.array(test_preds)
      3 submission_df.to_csv("submissions.csv",index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (3325)
