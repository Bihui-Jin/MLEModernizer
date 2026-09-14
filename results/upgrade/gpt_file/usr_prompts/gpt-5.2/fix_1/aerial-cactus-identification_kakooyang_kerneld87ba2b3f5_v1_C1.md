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

0.9685

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
print(os.listdir("../input"))

from PIL import Image
import matplotlib.pyplot as plt

%matplotlib inline



## === cell 1
os.listdir('../input/')


## === cell 2
df = pd.read_csv('../input/train.csv')
df.head()


## === cell 3
os.listdir('../input/test/test')


## === cell 4
os.listdir('../input/train/train')


## === cell 5
data_dir = '../input/train/train/'
filename = df['id'][0]
path = os.path.join(data_dir, filename)
path

test_dir = '../input/test/test/'


## === cell 6
image_pil = Image.open(path)
image_pil


## === cell 7
image = np.array(image_pil)
plt.imshow(image)
plt.show()


## === cell 8
has_cactus = df['has_cactus'][0]
has_cactus


## === cell 9
image = np.array(image_pil)
plt.title(has_cactus)
plt.imshow(image)
plt.show()


## === cell 10
np.mean(df['has_cactus']) # cactus가 포함될 비율


## === cell 11
np.sum(df['has_cactus']), len(df['has_cactus'])


## === cell 12
image.shape


## === cell 13
np.min(image), np.max(image)


## === cell 14
heights = []
widths = []
train_arr = []
test_arr = []

for filename in df['id']:
    path = os.path.join(data_dir, filename)
    image_pil = Image.open(path)
    image = np.array(image_pil)
    train_arr.append(image)
    h, w, c = image.shape
    if h not in heights:
        heights.append(h)
    if w not in widths:
        widths.append(w)
    
train_data = np.array(train_arr) # 또는 np.array의 concatenate 이용

train_labels_list = []
for label in df['has_cactus']:
    train_labels_list.append(label)

train_labels = np.array(train_labels_list)
    
for testfilename in os.listdir(test_dir):
    path = os.path.join(test_dir, testfilename)
    image_pil = Image.open(path)
    image = np.array(image_pil)
    test_arr.append(image)
    
test_data = np.array(test_arr)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/1694052661.py in <cell line: 0>()
     26     #print(testfilename)
     27     path = os.path.join(test_dir, testfilename)
---> 28     image_pil = Image.open(path)
     29     image = np.array(image_pil)
     30     test_arr.append(image)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '../input/test/test/test'

## === cell 15
from tqdm import tqdm_notebook


## === cell 16
for filename in tqdm_notebook(df['id']):
    path = os.path.join(data_dir, filename)
    image = Image.open(path)


## === cell 17
plt.imshow(train_data[12])


## === cell 18
train_data.shape


## === cell 19
train_labels


## === cell 20
heights


## === cell 21
widths


## === cell 22
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import clear_output

import tensorflow as tf
from tensorflow.keras import layers
tf.enable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"]="0"


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 23
print('TensorFlow version: {}'.format(tf.__version__))


## === cell 24
train_data = train_data / 255.
train_data = train_data.reshape([-1, 32, 32, 3])
train_data = train_data.astype(np.float32)
train_labels = train_labels.astype(np.int32)
train_labels = train_labels.astype(np.float64)

test_data = test_data / 255.
test_data = test_data.reshape([-1, 32, 32, 3])
test_data = test_data.astype(np.float32)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3930558281.py in <cell line: 0>()
      5 train_labels = train_labels.astype(np.float64)
      6 
----> 7 test_data = test_data / 255.
      8 test_data = test_data.reshape([-1, 32, 32, 3])
      9 test_data = test_data.astype(np.float32)

NameError: name 'test_data' is not defined

## === cell 25
tf.set_random_seed(219)
batch_size = 32
max_epochs = 20

N = len(train_data)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.shuffle(buffer_size = 10000)
train_dataset = train_dataset.batch(batch_size = batch_size)
print(train_dataset)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3169107637.py in <cell line: 0>()
----> 1 tf.set_random_seed(219)
      2 batch_size = 32
      3 max_epochs = 20
      4 
      5 # for train

AttributeError: module 'tensorflow' has no attribute 'set_random_seed'

## === cell 26
class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.l2_decay = 0.001
    self.conv1 = layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(self.l2_decay))
    self.conv1_bn = layers.BatchNormalization()
    self.pool1 = layers.MaxPool2D()
    self.conv2 = layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(self.l2_decay))
    self.conv2_bn = layers.BatchNormalization()
    self.pool2 = layers.MaxPool2D()
    self.flatten = layers.Flatten()
    self.dense1 = layers.Dense(units=1024,
                               kernel_regularizer=tf.keras.regularizers.l2(self.l2_decay))
    self.dense1_bn = layers.BatchNormalization()
    self.drop1 = layers.Dropout(rate=0.6)
    self.dense2 = layers.Dense(units=1, activation='sigmoid',
                               kernel_regularizer=tf.keras.regularizers.l2(self.l2_decay))

  def call(self, inputs, training=False):
    """Run the model."""
    self.conv1_ = self.conv1(inputs)
    self.conv1_bn_ = self.conv1_bn(self.conv1_, training=training)
    self.conv1_ = tf.nn.relu(self.conv1_bn_)
    self.pool1_ = self.pool1(self.conv1_)
    
    self.conv2_ = self.conv2(self.pool1_)
    self.conv2_bn_ = self.conv2_bn(self.conv2_, training=training)
    self.conv2_ = tf.nn.relu(self.conv2_bn_)
    self.pool2_ = self.pool2(self.conv2_)
    
    self.flatten_ = self.flatten(self.pool2_)
    self.dense1_ = self.dense1(self.flatten_)
    self.dense1_bn_ = self.dense1_bn(self.dense1_, training=training)
    self.dense1_ = tf.nn.relu(self.dense1_bn_)
    self.drop1_ = self.drop1(self.dense1_, training=training)
    
    self.predictions_ = self.dense2(self.drop1_)
    
    return self.predictions_


## === cell 27
model = MNISTModel()


## === cell 28
for images, labels in train_dataset.take(1):
  predictions = model(images[0:1])
  print("Predictions: ", predictions.numpy())
  print(labels[0:1])


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1080475135.py in <cell line: 0>()
      1 # without training, just inference a model in eager execution:
----> 2 for images, labels in train_dataset.take(1):
      3   predictions = model(images[0:1])
      4   print("Predictions: ", predictions.numpy())
      5   print(labels[0:1])

NameError: name 'train_dataset' is not defined

## === cell 29
model.summary()


## === cell 30
loss_object = tf.keras.losses.BinaryCrossentropy()
acc_object = tf.keras.metrics.BinaryAccuracy()


## === cell 31
tf.keras.metrics.Mean?


## === cell 32
optimizer = tf.train.AdamOptimizer(1e-4)

mean_ce = tf.keras.metrics.Mean("binary_cross_entropy")
mean_l2 = tf.keras.metrics.Mean("l2_loss")
mean_total_loss = tf.keras.metrics.Mean("total_loss")

cross_entropy_history = []
l2_loss_history = []
total_loss_history = []
accuracy_history = [(0, 0.0)]


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2112079119.py in <cell line: 0>()
      1 # use Adam optimizer
----> 2 optimizer = tf.train.AdamOptimizer(1e-4)
      3 
      4 # record loss for every epoch
      5 mean_ce = tf.keras.metrics.Mean("binary_cross_entropy")

AttributeError: module 'tensorflow._api.v2.train' has no attribute 'AdamOptimizer'

## === cell 33
print("start training!")
global_step = 0
num_batches_per_epoch = int(N / batch_size)

for epoch in range(max_epochs):
  
  for step, (images, labels) in enumerate(train_dataset):
    start_time = time.time()
    
    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      labels=np.reshape(labels,[len(labels),1])
      binary_cross_entropy = loss_object(labels, predictions)
      l2_loss = tf.reduce_sum(model.losses)
      total_loss = binary_cross_entropy + l2_loss
      acc_value = acc_object(labels, predictions)
      
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    global_step += 1
    
    mean_ce(binary_cross_entropy)
    mean_l2(l2_loss)
    mean_total_loss(total_loss)
    
    cross_entropy_history.append((global_step, mean_ce.result().numpy()))
    l2_loss_history.append((global_step, mean_l2.result().numpy()))
    total_loss_history.append((global_step, mean_total_loss.result().numpy()))

    if global_step % 10 == 0:
      clear_output(wait=True)
      epochs = epoch + step / float(num_batches_per_epoch)
      duration = time.time() - start_time
      examples_per_sec = batch_size / float(duration) 
      print("epochs: {:.2f}, step: {}, loss: {:.4g}, accuracy: {:.4g}% ({:.2f} examples/sec; {:.4f} sec/batch)".format(
          epochs, global_step, mean_total_loss.result().numpy(), acc_value.numpy()*100, examples_per_sec, duration))
      
  accuracy_history.append((global_step, acc_value.numpy()*100))

  mean_ce.reset_states()
  mean_l2.reset_states()
  mean_total_loss.reset_states()
  acc_object.reset_states()

print("training done!")


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3746816031.py in <cell line: 0>()
      1 print("start training!")
      2 global_step = 0
----> 3 num_batches_per_epoch = int(N / batch_size)
      4 
      5 for epoch in range(max_epochs):

NameError: name 'N' is not defined

## === cell 34
df.to_csv("test_submission.csv")


## === cell 35
test_data[0].shape


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/481841217.py in <cell line: 0>()
----> 1 test_data[0].shape

NameError: name 'test_data' is not defined

## === cell 36
test_predictions = []

for i in range(test_data.shape[0]):
    tmp_data = np.reshape(test_data[i], [1,32,32,3])
    predictions = model(tmp_data)
    if predictions>0.5:
        test_predictions.append(1)
    else:
        test_predictions.append(0)
            


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2033580882.py in <cell line: 0>()
      1 test_predictions = []
      2 
----> 3 for i in range(test_data.shape[0]):
      4     tmp_data = np.reshape(test_data[i], [1,32,32,3])
      5     predictions = model(tmp_data)

NameError: name 'test_data' is not defined

## === cell 37
test_filenames=os.listdir(test_dir)

test_df = pd.DataFrame({'id': test_filenames, 'has_cactus': test_predictions },columns=['id','has_cactus'])

test_df.to_csv('test_submission.csv', index=False)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/244271970.py in <cell line: 0>()
      1 test_filenames=os.listdir(test_dir)
      2 
----> 3 test_df = pd.DataFrame({'id': test_filenames, 'has_cactus': test_predictions },columns=['id','has_cactus'])
      4 
      5 test_df.to_csv('test_submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    446             # GH10856
    447             # raise ValueError if only scalars in dict
--> 448             index = _extract_index(arrays[~missing])
    449         else:
    450             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
