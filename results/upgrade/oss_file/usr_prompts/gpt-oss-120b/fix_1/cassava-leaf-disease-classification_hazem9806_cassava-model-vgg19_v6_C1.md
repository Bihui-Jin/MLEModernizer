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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.1494409187065578

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
from collections import Counter

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
main_directory = '/kaggle/input/cassava-leaf-disease-classification/'
training_images_path = main_directory + 'train_images'
print('List of Files:\n',os.listdir(main_directory))

## === cell 5
image_label_data = pd.read_csv(main_directory + 'train.csv')
labels = pd.read_json(main_directory + 'label_num_to_disease_map.json', typ='series')

print('Cassava Leaf Disease Classification Labels are:\n',dict(labels))
image_label_data.sample(10)

## === cell 7
print(Counter(image_label_data['label']))
image_label_data['label'].value_counts(normalize = True)

## === cell 9
image_label_data['disease_name'] = image_label_data.label.map(labels)
print(image_label_data)

## === cell 11
from sklearn.model_selection import train_test_split

TEST_PERCENTAGE = 0.05

train_set_splitted, validation_set_splitted = train_test_split(image_label_data, test_size = TEST_PERCENTAGE, random_state = 42,
                             stratify = image_label_data['disease_name'])



## === cell 13

from keras.preprocessing.image import ImageDataGenerator

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
NO_OF_CLASSES = 5
BATCH_SIZE = 20


TrainingImageGenerator = ImageDataGenerator( 
                                            preprocessing_function = tf.keras.applications.vgg19.preprocess_input,
                                            horizontal_flip = True,
                                            vertical_flip = True,
                                            fill_mode = 'nearest'
                                            )

ValidatonImageGenerator = ImageDataGenerator( 
                                            preprocessing_function = tf.keras.applications.vgg19.preprocess_input)


training_dataset = TrainingImageGenerator.flow_from_dataframe(
                                                         train_set_splitted,
                                                         directory = training_images_path,
                                                         seed=9806,
                                                         x_col = 'image_id',
                                                         y_col = 'disease_name',
                                                         target_size = IMAGE_SIZE,
                                                         class_mode = 'categorical',
                                                         interpolation = 'nearest',
                                                         shuffle = True,
                                                         batch_size = BATCH_SIZE)

validation_dataset = ValidatonImageGenerator.flow_from_dataframe(
                                                         validation_set_splitted,
                                                         directory = training_images_path,
                                                         seed=9806,
                                                         x_col = 'image_id',
                                                         y_col = 'disease_name',
                                                         target_size = IMAGE_SIZE,
                                                         class_mode = 'categorical',
                                                         interpolation = 'nearest',
                                                         shuffle = True,
                                                         batch_size = BATCH_SIZE)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2605633425.py in <cell line: 0>()
      1 # By using ImageDataGenerator we can make an on-fly image Augmentation
      2 
----> 3 from keras.preprocessing.image import ImageDataGenerator
      4 
      5 IMAGE_WIDTH = 224

ImportError: cannot import name 'ImageDataGenerator' from 'keras.preprocessing.image' (/usr/local/lib/python3.11/dist-packages/keras/api/preprocessing/image/__init__.py)

## === cell 15
import keras
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19

LOAD_MODEL = True

if not LOAD_MODEL:
    CassaveDisease_model = Sequential(name='Cassava_Neural_Network')
    CassaveDisease_model.add(VGG19(input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                                   include_top = False,
                                   weights = 'imagenet'))
    CassaveDisease_model.add(GlobalAveragePooling2D())
    CassaveDisease_model.add(Flatten())
    CassaveDisease_model.add(Dense(256, activation = 'relu'#, bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)
                                  ))
    CassaveDisease_model.add(BatchNormalization());
    CassaveDisease_model.add(Dense(NO_OF_CLASSES, activation = 'softmax'))

    CassaveDisease_model.summary()
    keras.utils.plot_model(CassaveDisease_model)
    
    Adam_Optimizer = Adam(learning_rate = 0.001)

    CassaveDisease_model.compile(
                                loss = "categorical_crossentropy", 
                                optimizer = Adam_Optimizer, 
                                metrics = ["accuracy"])

else:
    CassaveDisease_model = keras.models.load_model('../input/84-percentage-model/Cassava_best_Model_Reached_best_model_7_Jan_04_acc_is_86.h5')
    print('Model Loaded Successfully!')
    CassaveDisease_model.optimizers = Adam(learning_rate = 0.00002)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3455158422.py in <cell line: 0>()
     33 
     34 else:
---> 35     CassaveDisease_model = keras.models.load_model('../input/84-percentage-model/Cassava_best_Model_Reached_best_model_7_Jan_04_acc_is_86.h5')
     36     print('Model Loaded Successfully!')
     37     CassaveDisease_model.optimizers = Adam(learning_rate = 0.00002)

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/84-percentage-model/Cassava_best_Model_Reached_best_model_7_Jan_04_acc_is_86.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 17
EPOCHES = 0

EPOCHES_TO_WAIT_WITH_NO_IMPORVEMENT = 3

EARLY_STOP = EarlyStopping(monitor='val_accuracy',
                           patience = EPOCHES_TO_WAIT_WITH_NO_IMPORVEMENT,
                           restore_best_weights = True)

BEST_MODEL_REACHED = ModelCheckpoint(filepath = "./Cassava_best_Model_Reached_best_model_8_Jan_03.h5",
                                save_best_only = True,
                                monitor = 'val_loss',
                                mode = 'min')

REDUCE_LR = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 2,
                              min_lr = 1e-6,
                              mode = 'min',
                              verbose = 1)

Trained_Model = CassaveDisease_model.fit(
                        training_dataset,
                        validation_data = validation_dataset, 
                        epochs = EPOCHES,
                        callbacks = [EARLY_STOP,
                                     BEST_MODEL_REACHED,
                                     REDUCE_LR],
                        verbose=1)

CassaveDisease_model.save("./Cassava_best_Model_Reached_8_jan_03.h5")
print("Model Saved Successfully!")

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3475080373.py in <cell line: 0>()
     24                               verbose = 1)
     25 
---> 26 Trained_Model = CassaveDisease_model.fit(
     27                         training_dataset,
     28                         validation_data = validation_dataset,

NameError: name 'CassaveDisease_model' is not defined

## === cell 18
print(Trained_Model.history.keys())

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1739193301.py in <cell line: 0>()
----> 1 print(Trained_Model.history.keys())

NameError: name 'Trained_Model' is not defined

## === cell 19
def Train_Val_Plot(acc,val_acc,loss,val_loss):
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize= (15,10))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ", fontsize=20)

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy', fontsize=15)
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Accuracy', fontsize=15)
    ax1.legend(['training', 'validation'])


    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss', fontsize=15)
    ax2.set_xlabel('Epochs', fontsize=15)
    ax2.set_ylabel('Loss', fontsize=15)
    ax2.legend(['training', 'validation'])
    plt.show()
    



## === cell 20
TEST_DIR = '../input/cassava-leaf-disease-classification/test_images/'
test_images = os.listdir(TEST_DIR)
predictions = []
size = (IMAGE_WIDTH, IMAGE_HEIGHT)
for image in test_images:
    img = Image.open(TEST_DIR + image)
    img = img.resize(size)
    img = np.expand_dims(img, axis=0)
    predictions.extend(CassaveDisease_model.predict(img).argmax(axis = 1))

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3210210895.py in <cell line: 0>()
      2 test_images = os.listdir(TEST_DIR)
      3 predictions = []
----> 4 size = (IMAGE_WIDTH, IMAGE_HEIGHT)
      5 for image in test_images:
      6     img = Image.open(TEST_DIR + image)

NameError: name 'IMAGE_WIDTH' is not defined

## === cell 21
print(predictions)

## === cell 22

sub = pd.DataFrame({'image_id': test_images, 'label': predictions})
display(sub)
sub.to_csv('submission.csv', index = False)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3132964813.py in <cell line: 0>()
      1 # Creating the CSV for final submission
      2 
----> 3 sub = pd.DataFrame({'image_id': test_images, 'label': predictions})
      4 display(sub)
      5 sub.to_csv('submission.csv', index = False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length
