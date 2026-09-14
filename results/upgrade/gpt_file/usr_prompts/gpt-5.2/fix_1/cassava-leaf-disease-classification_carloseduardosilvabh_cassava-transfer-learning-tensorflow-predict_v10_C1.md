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

0.8318223028105167

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
import pandas as pd
import os
import json

import math
from tensorflow import keras
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.applications import EfficientNetB3, Xception, ResNet50V2
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
image = tf.keras.preprocessing.image.load_img(r'../input/cassava-leaf-disease-classification/train_images/1000015157.jpg')
image

## === cell 2
plt.imread('../input/cassava-leaf-disease-classification/train_images/1000015157.jpg').shape

## === cell 3
path = '../input/cassava-leaf-disease-classification'

## === cell 4
train_images = os.listdir(os.path.join(path, "train_images"))
print("Total images for Train: ", len(train_images))

## === cell 5
with open ('../input/cassava-leaf-disease-classification/label_num_to_disease_map.json') as file:
    classes = json.loads(file.read())
    
print(json.dumps(classes,indent=4))

## === cell 6
df_train = pd.read_csv(os.path.join(path, "train.csv"))
df_train.head()

## === cell 7
df_train['class'] = df_train['label'].map({int(i) : c for i, c in classes.items()}) 
df_train.head()

## === cell 8
plt.subplots(figsize=(12,8))
ax  = sns.countplot(x='class', data=df_train)

for p in ax.patches:
        ax.annotate('{:1}'.format(p.get_height()),
                    (p.get_x()+0.3, p.get_height()))
plt.xticks(rotation=90)
ax.set_title("quantities by classes", fontdict={'fontsize':15})
plt.show();

## === cell 11
def plot(images, labels, predictions = None):
    n_cols = min(4, len(images))
    n_rows = math.ceil(len(images) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (21,16))

    if predictions is None:
              
        predictions = [None] * len(labels)


    for i, (x, y_true, y_pred) in enumerate(zip(images, labels, predictions)):
        
        ax = axes.flat[i]
        a = plt.imread(os.path.join(path,"train_images", x ))
        ax.imshow(a)

        
        ax.set_title(f"Class: {y_true}")
             
        if y_pred is not None:
            ax.set_xlabel(f"Pred: {y_pred}", color='blue', fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

## === cell 15
df_0 = df_train[df_train['label']==0]
df_0 = df_0.sample(12)
df_0_id = df_0['image_id'].values
df_0_class = df_0['class'].values

## === cell 16
plot(df_0_id, df_0_class)

## === cell 19
df_1 = df_train[df_train['label']==1]
df_1 = df_1.sample(12)
df_1_id = df_1['image_id'].values
df_1_class = df_1['class'].values

## === cell 20
plot(df_1_id, df_1_class)

## === cell 23
df_2 = df_train[df_train['label']==2]
df_2 = df_2.sample(12)
df_2_id = df_2['image_id'].values
df_2_class = df_2['class'].values

## === cell 24
plot(df_2_id, df_2_class)

## === cell 27
df_3 = df_train[df_train['label']==3]
df_3 = df_3.sample(12)
df_3_id = df_3['image_id'].values
df_3_class = df_3['class'].values

## === cell 28
plot(df_3_id, df_3_class)

## === cell 30
df_4 = df_train[df_train['label']==4]
df_4 = df_4.sample(12)
df_4_id = df_4['image_id'].values
df_4_class = df_4['class'].values

## === cell 31
plot(df_4_id, df_4_class)

## === cell 34
train = df_train.astype({'label':str})
train, test = train_test_split(train, test_size = .2, random_state=42)

## === cell 36
train_datagen = ImageDataGenerator(
                    rotation_range = 45,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True,
                    vertical_flip = True,
                    fill_mode = 'nearest'
)

## === cell 37
img_size = 300
size = (img_size, img_size)

## === cell 38
train_generator = train_datagen.flow_from_dataframe(
                    train,
                    directory = path+"/train_images",
                    x_col = "image_id",
                    y_col = "class",
                    target_size = size,
                    class_mode = "categorical",
                    batch_size = 32,
                    shuffle = True,
                    seed = 42,
                    interpolation = "nearest"
)

## === cell 39
test_generator = train_datagen.flow_from_dataframe(
                    test,
                    directory = path+"/train_images",
                    x_col = "image_id",
                    y_col = "class",
                    target_size = size,
                    class_mode = "categorical",
                    batch_size = 32,
                    shuffle = False,
                    seed = 42,
                    interpolation = "nearest")

## === cell 40
def modelTransf():
    
    model = models.Sequential()
    model.add(EfficientNetB3(input_shape = (img_size, img_size, 3), include_top = False, weights = 'imagenet'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation = 'softmax'))
    
    return model

## === cell 41
model = modelTransf()

## === cell 42
model.summary()

## === cell 43
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(learning_rate = 0.001),
              metrics = ['accuracy'])

## === cell 44
early_stopping = EarlyStopping(monitor = 'val_loss',
                               patience = 10,
                               mode = 'min', 
                               restore_best_weights = True)

checkpoint = ModelCheckpoint('modelB3.hdf5',
                             monitor = 'val_loss',
                             verbose = 1, mode = 'min',
                             save_best_only = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 10,
                              min_lr = 0.001,
                              mode = 'min', 
                              verbose = 1)

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2444859568.py in <cell line: 0>()
      4                                restore_best_weights = True)
      5 
----> 6 checkpoint = ModelCheckpoint('modelB3.hdf5',
      7                              monitor = 'val_loss',
      8                              verbose = 1, mode = 'min',

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=modelB3.hdf5

## === cell 45
step_size_train = train_generator.n // train_generator.batch_size
step_size_test = test_generator.n // test_generator.batch_size

## === cell 46
step_size_train, step_size_test

## === cell 47
history = model.fit(train_generator,
                    validation_data = test_generator,
                    epochs = 30,
                    steps_per_epoch = step_size_train,
                    validation_steps = step_size_test,
                    callbacks = [early_stopping, checkpoint, reduce_lr])

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3816231407.py in <cell line: 0>()
      4                     steps_per_epoch = step_size_train,
      5                     validation_steps = step_size_test,
----> 6                     callbacks = [early_stopping, checkpoint, reduce_lr])

NameError: name 'checkpoint' is not defined

## === cell 49
model_treined = keras.models.load_model('../input/model-trained/E_best_model.hdf5')

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/739306830.py in <cell line: 0>()
----> 1 model_treined = keras.models.load_model('../input/model-trained/E_best_model.hdf5')

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/model-trained/E_best_model.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 50
filenames = test_generator.filenames
len(filenames)

## === cell 51
predictions = model_treined.predict_generator(test_generator, steps = len(filenames))

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1350323913.py in <cell line: 0>()
----> 1 predictions = model_treined.predict_generator(test_generator, steps = len(filenames))

NameError: name 'model_treined' is not defined

## === cell 52
len(predictions)

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3942580936.py in <cell line: 0>()
----> 1 len(predictions)

NameError: name 'predictions' is not defined

## === cell 53
predictions2 = []
for i in range(len(predictions)):
    predictions2.append(np.argmax(predictions[i]))

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1968477174.py in <cell line: 0>()
      1 predictions2 = []
----> 2 for i in range(len(predictions)):
      3     predictions2.append(np.argmax(predictions[i]))

NameError: name 'predictions' is not defined

## === cell 54
accuracy_score(predictions2, test_generator.classes)

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/423583689.py in <cell line: 0>()
----> 1 accuracy_score(predictions2, test_generator.classes)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in wrapper(*args, **kwargs)
    190 
    191             try:
--> 192                 return func(*args, **kwargs)
    193             except InvalidParameterError as e:
    194                 # When the function is just a wrapper around an estimator, we allow

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py in accuracy_score(y_true, y_pred, normalize, sample_weight)
    219 
    220     # Compute accuracy for each possible representation
--> 221     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    222     check_consistent_length(y_true, y_pred, sample_weight)
    223     if y_type.startswith("multilabel"):

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py in _check_targets(y_true, y_pred)
     84     y_pred : array or indicator matrix
     85     """
---> 86     check_consistent_length(y_true, y_pred)
     87     type_true = type_of_target(y_true, input_name="y_true")
     88     type_pred = type_of_target(y_pred, input_name="y_pred")

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    395     uniques = np.unique(lengths)
    396     if len(uniques) > 1:
--> 397         raise ValueError(
    398             "Found input variables with inconsistent numbers of samples: %r"
    399             % [int(l) for l in lengths]

ValueError: Found input variables with inconsistent numbers of samples: [0, 3745]

## === cell 55
cfm = confusion_matrix(predictions2, test_generator.classes)
cfm

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1486163887.py in <cell line: 0>()
----> 1 cfm = confusion_matrix(predictions2, test_generator.classes)
      2 cfm

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py in confusion_matrix(y_true, y_pred, labels, sample_weight, normalize)
    315     (0, 2, 1, 1)
    316     """
--> 317     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    318     if y_type not in ("binary", "multiclass"):
    319         raise ValueError("%s is not supported" % y_type)

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py in _check_targets(y_true, y_pred)
     84     y_pred : array or indicator matrix
     85     """
---> 86     check_consistent_length(y_true, y_pred)
     87     type_true = type_of_target(y_true, input_name="y_true")
     88     type_pred = type_of_target(y_pred, input_name="y_pred")

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    395     uniques = np.unique(lengths)
    396     if len(uniques) > 1:
--> 397         raise ValueError(
    398             "Found input variables with inconsistent numbers of samples: %r"
    399             % [int(l) for l in lengths]

ValueError: Found input variables with inconsistent numbers of samples: [0, 3745]

## === cell 56
sns.heatmap(cfm, annot=True, fmt="d", cmap='viridis');

## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1907751633.py in <cell line: 0>()
----> 1 sns.heatmap(cfm, annot=True, fmt="d", cmap='viridis');

NameError: name 'cfm' is not defined

## === cell 57
pred = pd.DataFrame(predictions2, columns=['class_pred'])
pred['pred'] = pred['class_pred'].map({int(i) : c for i, c in classes.items()})

## === cell 58
image_ids_test = test["image_id"].values
labels_test = test["class"].values
pred_result = pred['pred'].values

## === cell 59
rand_idxs = np.random.permutation(len(test))[:12]

## === cell 60
plot(image_ids_test[rand_idxs], labels_test[rand_idxs], pred_result[rand_idxs])
plt.show()

## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/574526841.py in <cell line: 0>()
----> 1 plot(image_ids_test[rand_idxs], labels_test[rand_idxs], pred_result[rand_idxs])
      2 plt.show()

IndexError: index 2305 is out of bounds for axis 0 with size 0

## === cell 62
submission_file = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
submission_file

## === cell 63
path_test = '../input/cassava-leaf-disease-classification/test_images/'

## === cell 64
test_img = os.listdir(path_test)
predict = []
for image in test_img:
    img = tf.keras.preprocessing.image.load_img(path_test + image)
    img = img.resize((300, 300))
    img = np.expand_dims(img, axis = 0)
    predict.append(np.argmax(model_treined.predict(img)))

## --- ERROR in cell 64, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3299277016.py in <cell line: 0>()
      5     img = img.resize((300, 300))
      6     img = np.expand_dims(img, axis = 0)
----> 7     predict.append(np.argmax(model_treined.predict(img)))

NameError: name 'model_treined' is not defined

## === cell 65
predict

## === cell 66
submission = pd.DataFrame({'image_id': test_img, 'label': predict})
submission

## --- ERROR in cell 66, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3927883511.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({'image_id': test_img, 'label': predict})
      2 submission

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

## === cell 67
submission.to_csv('submission.csv', index = False)

## --- ERROR in cell 67, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2855364721.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv', index = False)

NameError: name 'submission' is not defined
