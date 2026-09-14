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

No external packages required in the script and installed.

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

0.9553

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))

from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

%matplotlib inline



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
os.listdir('../input/')


## === cell 3
df = pd.read_csv('../input/train.csv')
df.head(10)


## === cell 4
os.listdir('../input/test/test')


## === cell 5
os.listdir('../input/train/train')


## === cell 6
data_dir = '../input/train/train/'
filename = df['id'][0]
path = os.path.join(data_dir, filename)
path

test_dir = '../input/test/test/'


## === cell 7
image_pil = Image.open(path)
image_pil


## === cell 8
image = np.array(image_pil)
plt.imshow(image)
plt.show()


## === cell 9
has_cactus = df['has_cactus'][0]
has_cactus


## === cell 10
image = np.array(image_pil)
plt.title(has_cactus)
plt.imshow(image)
plt.show()


## === cell 11
np.mean(df['has_cactus']) # cactus가 포함될 비율


## === cell 12
np.sum(df['has_cactus']), len(df['has_cactus'])


## === cell 13
image.shape


## === cell 14
np.min(image), np.max(image)


## === cell 15
    

def get_data(pathtuple):
    path, label = pathtuple
    image_pil = Image.open(path)
    image = np.array(image_pil)
    image = image/255.0
    label=tf.keras.utils.to_categorical(label,2)

    return image.astype(np.float32), label.astype(np.float32)

    
    


## === cell 16
tf.keras.utils.to_categorical?


## === cell 17
heights = []
widths = []
train_arr = []
test_arr = []

train_filenames = []
index = 0

for filename in df['id']:
    path = os.path.join(data_dir, filename)
    train_filenames.append((path, df['has_cactus'][index]))
    index = index + 1
    
    
for testfilename in os.listdir(test_dir):
    path = os.path.join(test_dir, testfilename)
    image_pil = Image.open(path)
    image = np.array(image_pil)
    image = image/255.0
    test_arr.append(image.astype(np.float32))
    
test_data = np.array(test_arr)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/949370025.py in <cell line: 0>()
     19     #print(testfilename)
     20     path = os.path.join(test_dir, testfilename)
---> 21     image_pil = Image.open(path)
     22     image = np.array(image_pil)
     23     image = image/255.0

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '../input/test/test/test'

## === cell 18


def make_batch(batch_paths):

    batch_images = []
    batch_labels = []

    for pathtuple in batch_paths:
        path, label = pathtuple
        image, label = get_data(pathtuple)
        batch_images.append(image)
        batch_labels.append(label)

    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels)

    return batch_images, batch_labels



## === cell 19
batch_size = 32


## === cell 20
def data_gen(data_paths, is_training=True):
    global_step = 0
    steps_per_epoch = len(data_paths) // batch_size
    while True:
        step = global_step % steps_per_epoch
        if step == 0:
            np.random.shuffle(data_paths)
        images, labels = make_batch(data_paths[step*batch_size:(step+1)*batch_size])
        global_step += 1
        yield images, labels


## === cell 21
generator = data_gen(train_filenames)

for i, (img, lbl) in enumerate(generator):
    if i < 5:
        plt.title(i + lbl[0])
        plt.imshow(img[0])
        plt.show()
    else:
        break
        


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1477751799.py in <cell line: 0>()
      4 for i, (img, lbl) in enumerate(generator):
      5     if i < 5:
----> 6         plt.title(i + lbl[0])
      7         plt.imshow(img[0])
      8         plt.show()

/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py in title(label, fontdict, loc, pad, y, **kwargs)
   3082 @_copy_docstring_and_deprecators(Axes.set_title)
   3083 def title(label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
-> 3084     return gca().set_title(
   3085         label, fontdict=fontdict, loc=loc, pad=pad, y=y, **kwargs)
   3086 

/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_axes.py in set_title(self, label, fontdict, loc, pad, y, **kwargs)
    174             pad = mpl.rcParams['axes.titlepad']
    175         self._set_title_offset_trans(float(pad))
--> 176         title.set_text(label)
    177         title.update(default)
    178         if fontdict is not None:

/usr/local/lib/python3.11/dist-packages/matplotlib/text.py in set_text(self, s)
   1277         if s is None:
   1278             s = ''
-> 1279         if s != self._text:
   1280             self._text = str(s)
   1281             self.stale = True

ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

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
/tmp/ipykernel_11/1123267089.py in <cell line: 0>()
     14 import tensorflow as tf
     15 from tensorflow.keras import layers
---> 16 tf.enable_eager_execution()
     17 
     18 os.environ["CUDA_VISIBLE_DEVICES"]="0"

AttributeError: module 'tensorflow' has no attribute 'enable_eager_execution'

## === cell 23
batch_size = 32
num_epochs = 10
learning_rate = 0.001

num_classes = 2
input_shape = (32, 32, 3)


## === cell 24
inputs = layers.Input(input_shape)
net = layers.Conv2D(64, (3, 3), padding='same')(inputs)
net = layers.Conv2D(64, (3, 3), padding='same')(net)
net = layers.Conv2D(64, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)

net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net)


## === cell 25
model.compile(loss='categorical_crossentropy', 
             optimizer=tf.keras.optimizers.Adam(learning_rate),
             metrics=['accuracy'])


## === cell 27
model.fit_generator?


## === cell 28
steps_per_epoch = len(train_filenames) // batch_size

history=model.fit_generator(generator=data_gen(train_filenames),
                    steps_per_epoch=steps_per_epoch,
                    epochs=num_epochs,
                    verbose=1)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/178142513.py in <cell line: 0>()
      1 steps_per_epoch = len(train_filenames) // batch_size
      2 
----> 3 history=model.fit_generator(generator=data_gen(train_filenames),
      4                     steps_per_epoch=steps_per_epoch,
      5                     epochs=num_epochs,

AttributeError: 'Functional' object has no attribute 'fit_generator'

## === cell 29
print(history.history.keys())


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3885426154.py in <cell line: 0>()
----> 1 print(history.history.keys())

NameError: name 'history' is not defined

## === cell 30
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/118750979.py in <cell line: 0>()
      1 # summarize history for accuracy
----> 2 plt.plot(history.history['acc'])
      3 plt.title('model accuracy')
      4 plt.ylabel('accuracy')
      5 plt.xlabel('epoch')

NameError: name 'history' is not defined

## === cell 32
test_predictions = []

for i in range(test_data.shape[0]):
    predictions = model.predict(np.expand_dims(test_data[i],0))
    test_predictions.append(np.argmax(tf.squeeze(predictions)))


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/160062411.py in <cell line: 0>()
      1 test_predictions = []
      2 
----> 3 for i in range(test_data.shape[0]):
      4 #for i in range(1):
      5     predictions = model.predict(np.expand_dims(test_data[i],0))

NameError: name 'test_data' is not defined

## === cell 34
test_filenames=os.listdir(test_dir)

test_df = pd.DataFrame({'id': test_filenames, 'has_cactus': test_predictions },columns=['id','has_cactus'])

test_df.to_csv('test_submission.csv', index=False)


## --- ERROR in cell 34, traceback:
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
