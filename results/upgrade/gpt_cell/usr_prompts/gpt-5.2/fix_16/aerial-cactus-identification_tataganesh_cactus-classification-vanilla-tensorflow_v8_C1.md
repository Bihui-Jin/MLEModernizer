# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            raise AttributeError("MessageFactory has no GetMessageClass/GetPrototype")

        _message_factory.MessageFactory.GetPrototype = _GetPrototype

    if hasattr(_message_factory, "_DEFAULT") and not hasattr(
        _message_factory._DEFAULT, "GetPrototype"
    ):
        _message_factory._DEFAULT.GetPrototype = (
            _message_factory.MessageFactory.GetPrototype.__get__(
                _message_factory._DEFAULT, _message_factory.MessageFactory
            )
        )
except Exception:
    pass

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir("../input/train/"))
import keras
import cv2
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import random

np.random.seed(0)
from tqdm import tqdm_notebook


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
augs = [np.fliplr, np.flipud, np.rot90]
def augment(images, labels, augs):
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
def augment(images, labels):
    augs = [np.fliplr, np.flipud, np.rot90]
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


## === cell 7
normalized_train_images, final_training_labels = augment(
    normalized_images, np.array(training_labels)
)
NUM_TRAIN_IMAGES = int(0.75 * normalized_train_images.shape[0])
indices = np.random.permutation(normalized_train_images.shape[0])
training_idx, val_idx = indices[:NUM_TRAIN_IMAGES], indices[NUM_TRAIN_IMAGES:]
train_data = normalized_train_images[training_idx, :]
train_labels = np.array(final_training_labels)[training_idx]

val_data = normalized_train_images[val_idx, :]
val_labels = final_training_labels[val_idx]


## === cell 8
train_labels.sum()


## === cell 9
def show_images_horizontally(images, labels=[], lookup_label=None,
                            figsize=(15, 7)):

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

show_images_horizontally(normalized_train_images[:20], np.array(final_training_labels[:20]), lookup_label={1:"has_cactus", 0: "no_cactus"})
print(np.array(train_labels)[[1,2,4]])


## === cell 10
tf.compat.v1.reset_default_graph()


## === cell 11
def get_weights(shape):
  initializer = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initializer(shape=shape))
  
def get_biases(length):
  return tf.Variable(tf.constant(0.0005, shape=[length]))
    


## === cell 12
def conv_layer(input, in_channels, filter_size, num_filters):
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


def flatten(input):
  layer_shape = input.get_shape()
  total_elements = layer_shape[1:4].num_elements()
  layer = tf.reshape(input, [-1, total_elements])
  return layer, total_elements


def fc_layer(input, in_features, out_features):
  weight = get_weights([in_features, out_features])
  bias = get_biases(out_features)
  layer = tf.matmul(input, weight) + bias
  return layer


## === cell 13
tf.compat.v1.disable_eager_execution()


def get_weights(shape):
    initializer = tf.compat.v1.keras.initializers.glorot_uniform()
    return tf.Variable(initializer(shape=shape))


def conv_layer(input, in_channels, filter_size, num_filters):
    shape = [filter_size, filter_size, in_channels, num_filters]
    weights = get_weights(shape)
    bias = get_biases(num_filters)
    layer = tf.nn.convolution(
        input,
        weights,
        strides=[1, 1],
        dilations=[1, 1],
        padding="SAME",
    )
    layer = tf.nn.bias_add(layer, bias)
    new_layer = tf.nn.relu(layer)
    print(new_layer)
    return new_layer, weights


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
    layer_conv4, weights_4 = conv_layer(layer_output_pool, 64, 3, 128)
with tf.compat.v1.variable_scope("block3_conv2"):
    layer_conv5, weights_5 = conv_layer(layer_conv4, 128, 3, 128)
with tf.compat.v1.variable_scope("block3_conv3"):
    layer_conv6, weights_6 = conv_layer(layer_conv5, 128, 3, 128)
    layer_output_pool = tf.nn.max_pool(
        layer_conv6, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding="VALID"
    )

print(layer_output_pool, layer_conv4)
flattened_layer, in_features = flatten(layer_output_pool)
fclyr = fc_layer(flattened_layer, in_features, 128)
fc_layer2 = tf.nn.relu(fclyr)

final_layer = fc_layer(fc_layer2, 128, 2)
y_pred = tf.nn.softmax(final_layer)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=final_layer, labels=tf.one_hot(labels, 2)
)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
init = tf.compat.v1.global_variables_initializer()


## === cell 14
sess = tf.compat.v1.Session(config=config)
NUM_ITERATIONS = 20
BATCH = 32
sess.run(init)
for i in tqdm_notebook(range(NUM_ITERATIONS)):

    num_batches = int(train_data.shape[0] / BATCH) + 1
    losses = list()
    epoch_predictions = list()
    for j in tqdm_notebook(range(num_batches)):
        batch_data = train_data[BATCH * j : BATCH * j + BATCH]
        batch_labels = train_labels[BATCH * j : BATCH * j + BATCH].astype(np.int64)
        loss, _, probabilities = sess.run(
            [cost, optimizer, y_pred],
            feed_dict={input_image: batch_data, labels: batch_labels},
        )
        predictions = np.argmax(probabilities, axis=1)
        epoch_predictions.extend(predictions)
        losses.append(loss)
    print("EPOCH " + str(i))
    epoch_predictions = np.array(epoch_predictions)
    train_loss = np.mean(np.array(losses))
    train_accuracy = (
        np.sum(epoch_predictions == train_labels) / train_labels.shape[0]
    ) * 100
    num_batches = int(val_data.shape[0] / BATCH) + 1
    val_predictions = list()
    for j in range(num_batches):
        batch_data = val_data[BATCH * j : BATCH * j + BATCH]
        batch_labels = val_labels[BATCH * j : BATCH * j + BATCH].astype(np.int64)
        loss, _, probabilities = sess.run(
            [cost, optimizer, y_pred],
            feed_dict={input_image: batch_data, labels: batch_labels},
        )
        predictions = np.argmax(probabilities, axis=1)
        val_predictions.extend(predictions)
        losses.append(loss)
    val_accuracy = (np.sum(val_predictions == val_labels) / val_labels.shape[0]) * 100
    print(
        "Loss after Epoch - %f, Train Accuracy - %f, Val Accuracy - %f"
        % (train_loss, train_accuracy, val_accuracy)
    )


## === cell 15
test_preds = list()
num_batches = int(normalized_test_images.shape[0] / BATCH) + 1
for j in range(num_batches):
    batch_data = normalized_test_images[BATCH*j: BATCH*j + BATCH]
    batch_labels = test_labels[BATCH*j: BATCH*j + BATCH].astype(np.int64)
    probabilities = sess.run(y_pred, feed_dict={input_image:batch_data, labels:batch_labels})
    predictions = np.argmax(probabilities, axis=1)
    test_preds.extend(predictions)


## === cell 16
submission_df = test_df
submission_df['has_cactus'] = np.array(test_preds)
submission_df.to_csv("submissions.csv",index=False)


## === cell 17
tf.compat.v1.reset_default_graph()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


## === cell 18
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,3)))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))          
model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


## === cell 19
print(train_data.shape)
print(train_labels.shape)

if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

model.compile(
    loss="categorical_crossentropy",
    optimizer="Adam",
    metrics=["accuracy"],
    run_eagerly=True,
)

history = model.fit(train_data, np.eye(2)[train_labels], epochs=20)


## --- ERROR in cell 19, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/39283666.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0;31m# relies on eager/`tf.function` for tf.data iteration, so we must re-enable eager here.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;32mif[0m [0;32mnot[0m [0mtf[0m[0;34m.[0m[0mexecuting_eagerly[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m     [0mtf[0m[0;34m.[0m[0mcompat[0m[0;34m.[0m[0mv1[0m[0;34m.[0m[0menable_eager_execution[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m model.compile(

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36menable_eager_execution[0;34m(config, device_policy, execution_mode)[0m
[1;32m   4961[0m   [0mlogging[0m[0;34m.[0m[0mvlog[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0;34m"Enabling eager execution"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4962[0m   [0;32mif[0m [0mcontext[0m[0;34m.[0m[0mdefault_execution_mode[0m [0;34m!=[0m [0mcontext[0m[0;34m.[0m[0mEAGER_MODE[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4963[0;31m     return enable_eager_execution_internal(
[0m[1;32m   4964[0m         [0mconfig[0m[0;34m=[0m[0mconfig[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4965[0m         [0mdevice_policy[0m[0;34m=[0m[0mdevice_policy[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36menable_eager_execution_internal[0;34m(config, device_policy, execution_mode, server_def)[0m
[1;32m   5025[0m         _default_graph_stack._global_default_graph is not None)  # pylint: disable=protected-access
[1;32m   5026[0m     [0;32mif[0m [0mgraph_mode_has_been_used[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 5027[0;31m       raise ValueError(
[0m[1;32m   5028[0m           "tf.enable_eager_execution must be called at program startup.")
[1;32m   5029[0m   [0mcontext[0m[0;34m.[0m[0mdefault_execution_mode[0m [0;34m=[0m [0mcontext[0m[0;34m.[0m[0mEAGER_MODE[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: tf.enable_eager_execution must be called at program startup.

## === cell 21
scores = model.evaluate(val_data, np.eye(2)[np.array(val_labels, dtype=np.int64)], verbose=0)
print("Val Accuracy:"+"%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
