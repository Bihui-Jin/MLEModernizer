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
numpy==1.26.4
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

0.8983

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from os.path import join

train_img_dir = '../input/train/train'

test_img_dir = '../input/test/test'
test_img_paths = [join(test_img_dir, img) for img in os.listdir(test_img_dir)]


## === cell 1
from tensorflow.python import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd

train_data = pd.read_csv('../input/train.csv')
train_img_paths = [join(train_img_dir, img) for img in train_data['id']]

img_size = 32

def prep_imgs(img_paths, img_height = img_size, img_width = img_size):
    imgs = [load_img(img, target_size = (img_height, img_width)) for img in img_paths]
    img_arr = np.array([img_to_array(img) for img in imgs]) / 255
    return img_arr

X_train = prep_imgs(train_img_paths)
X_train.shape
y_train = train_data['has_cactus'].values


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(20, kernel_size = 2, strides = 2, activation = 'relu', input_shape = (img_size, img_size, 3)))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size = (2,2), strides = 2)) #applying max pooling to convolution layer
model.add(Conv2D(20, kernel_size = 2, strides = 2, activation = 'relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


## === cell 3
model.fit(X_train, y_train, batch_size = 100, epochs = 3, validation_split = 0.2)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4152188702.py in <cell line: 0>()
----> 1 model.fit(X_train, y_train, batch_size = 100, epochs = 3, validation_split = 0.2)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
   1135          training_utils.RespectCompiledTrainableState(self):
   1136       # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
-> 1137       data_handler = data_adapter.get_data_handler(
   1138           x=x,
   1139           y=y,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/engine/data_adapter.py in get_data_handler(*args, **kwargs)
   1396   if getattr(kwargs["model"], "_cluster_coordinator", None):
   1397     return _ClusterCoordinatorDataHandler(*args, **kwargs)
-> 1398   return DataHandler(*args, **kwargs)
   1399 
   1400 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/engine/data_adapter.py in __init__(self, x, y, sample_weight, batch_size, steps_per_epoch, initial_epoch, epochs, shuffle, class_weight, max_queue_size, workers, use_multiprocessing, model, steps_per_execution, distribute)
   1150       self._steps_per_execution_value = steps_per_execution.numpy().item()
   1151 
-> 1152     adapter_cls = select_data_adapter(x, y)
   1153     self._adapter = adapter_cls(
   1154         x,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/engine/data_adapter.py in select_data_adapter(x, y)
    986 def select_data_adapter(x, y):
    987   """Selects a data adapter than can handle a given x and y."""
--> 988   adapter_cls = [cls for cls in ALL_ADAPTER_CLS if cls.can_handle(x, y)]
    989   if not adapter_cls:
    990     # TODO(scottzhu): This should be a less implementation-specific error.

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/engine/data_adapter.py in <listcomp>(.0)
    986 def select_data_adapter(x, y):
    987   """Selects a data adapter than can handle a given x and y."""
--> 988   adapter_cls = [cls for cls in ALL_ADAPTER_CLS if cls.can_handle(x, y)]
    989   if not adapter_cls:
    990     # TODO(scottzhu): This should be a less implementation-specific error.

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/engine/data_adapter.py in can_handle(x, y)
    705   def can_handle(x, y=None):
    706     return (isinstance(x, (data_types.DatasetV1, data_types.DatasetV2)) or
--> 707             _is_distributed_dataset(x))
    708 
    709   def __init__(self,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/keras/engine/data_adapter.py in _is_distributed_dataset(ds)
   1695 
   1696 def _is_distributed_dataset(ds):
-> 1697   return isinstance(ds, input_lib.DistributedDatasetInterface)

AttributeError: module 'tensorflow.python.distribute.input_lib' has no attribute 'DistributedDatasetInterface'

## === cell 4
X_test = prep_imgs(test_img_paths)
preds_temp = model.predict(X_test)
preds = preds_temp.argmax(axis = -1)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/3234063577.py in <cell line: 0>()
----> 1 X_test = prep_imgs(test_img_paths)
      2 preds_temp = model.predict(X_test)
      3 preds = preds_temp.argmax(axis = -1)

/tmp/ipykernel_11/4174667395.py in prep_imgs(img_paths, img_height, img_width)
     10 
     11 def prep_imgs(img_paths, img_height = img_size, img_width = img_size):
---> 12     imgs = [load_img(img, target_size = (img_height, img_width)) for img in img_paths]
     13     img_arr = np.array([img_to_array(img) for img in imgs]) / 255
     14     return img_arr

/tmp/ipykernel_11/4174667395.py in <listcomp>(.0)
     10 
     11 def prep_imgs(img_paths, img_height = img_size, img_width = img_size):
---> 12     imgs = [load_img(img, target_size = (img_height, img_width)) for img in img_paths]
     13     img_arr = np.array([img_to_array(img) for img in imgs]) / 255
     14     return img_arr

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

IsADirectoryError: [Errno 21] Is a directory: '../input/test/test/test'

## === cell 5
output = pd.DataFrame({'id': os.listdir(test_img_dir), 'has_cactus': preds})
output.to_csv('submission.csv', index = False)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3709667741.py in <cell line: 0>()
----> 1 output = pd.DataFrame({'id': os.listdir(test_img_dir), 'has_cactus': preds})
      2 output.to_csv('submission.csv', index = False)

NameError: name 'preds' is not defined
