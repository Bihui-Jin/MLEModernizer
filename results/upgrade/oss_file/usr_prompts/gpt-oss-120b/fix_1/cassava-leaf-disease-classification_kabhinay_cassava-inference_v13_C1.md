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

0.8824418253248716

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
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras import optimizers
from keras import models
from keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import functools
from tqdm import tqdm

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def acc_gambler(y_true,y_pred):
    y_temp = y_pred[:,1:]
    count = tf.constant((0,))
    for i in range(len(y_true)):
        tf.autograph.experimental.set_loop_options(
        shape_invariants=[(count, tf.TensorShape([None]))])
        if tf.math.argmax(y_temp[i]) == tf.math.argmax(y_true[i]) :
            count = tf.math.add(count,1)
    return float(count)/float(len(y_true))

import keras.backend as K

def loss_gambler(label_smoothing=0.0):
    def loss_gamb(y_true,y_pred):
        y_true = tf.math.add(y_true,tf.math.add(tf.math.multiply(label_smoothing/2.0,tf.math.add(1.0,-1*y_true)),tf.math.multiply(-1*label_smoothing/2.0,y_true)))
        y_temp = y_pred[:,1:]
        f0 = y_pred[:,0]
        lamb = tf.math.divide(tf.math.multiply(K.sum(y_temp),K.sum(y_temp)),K.sum(tf.math.multiply(y_temp,y_temp)))
        loss = tf.constant((0.0,))
        for i in range(len(y_true[0])):
            tf.autograph.experimental.set_loop_options(
            shape_invariants=[(loss, tf.TensorShape([None]))])
            temp = tf.constant((0.0,))
            loss = tf.math.add(loss,tf.math.add(temp,(-1.0*(1/float(len(y_true)))*tf.math.multiply(y_true[:,i],K.log(y_temp[:,i]+f0/lamb)))))
        return tf.math.reduce_sum(loss)
    return loss_gamb

## === cell 2
model_v1 = tf.keras.models.load_model('../input/only-xception-with-cropping/saved-model-11-0.879')
model_v2 = tf.keras.models.load_model('../input/gambler-s-loss-cassava/saved-model-05-0.860',custom_objects={'acc_gambler' : acc_gambler,'loss_gamb' : loss_gambler(0.1)})

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1010044026.py in <cell line: 0>()
----> 1 model_v1 = tf.keras.models.load_model('../input/only-xception-with-cropping/saved-model-11-0.879')
      2 model_v2 = tf.keras.models.load_model('../input/gambler-s-loss-cassava/saved-model-05-0.860',custom_objects={'acc_gambler' : acc_gambler,'loss_gamb' : loss_gambler(0.1)})

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    204         )
    205     else:
--> 206         raise ValueError(
    207             f"File format not supported: filepath={filepath}. "
    208             "Keras 3 only supports V3 `.keras` files and "

ValueError: File format not supported: filepath=../input/only-xception-with-cropping/saved-model-11-0.879. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(../input/only-xception-with-cropping/saved-model-11-0.879, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).

## === cell 4
labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

test_datagen_v1 = ImageDataGenerator()


test_dir_v1 = '../input/cassava-leaf-disease-classification/test_images/'

test_v1=pd.DataFrame()
test_v1['image_id']=os.listdir(test_dir_v1)




test_generator_v1 = test_datagen_v1.flow_from_dataframe(
        test_v1,
        directory = test_dir_v1,
        x_col = 'image_id',
        target_size = (448, 448),
        batch_size = 1,
        class_mode = None,
        shuffle=False)

test_generator_v1.reset()

pred_v1 = model_v1.predict_generator(test_generator_v1,verbose=1,steps = len(test_v1))


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/668397091.py in <cell line: 0>()
     26 test_generator_v1.reset()
     27 
---> 28 pred_v1 = model_v1.predict_generator(test_generator_v1,verbose=1,steps = len(test_v1))

NameError: name 'model_v1' is not defined

## === cell 8
labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

test_datagen_v2 = ImageDataGenerator()

test_dir_v2 = '../input/cassava-leaf-disease-classification/test_images/'

test_v2=pd.DataFrame()
test_v2['image_id']=os.listdir('../input/cassava-leaf-disease-classification/test_images/')

test_generator_v2 = test_datagen_v2.flow_from_dataframe(
        test_v2,
        directory = test_dir_v2,
        x_col = 'image_id',
        target_size = (512, 512),
        batch_size = 1,
        class_mode = None,
        shuffle=False)

test_generator_v2.reset()

temp = model_v2.predict_generator(test_generator_v2,verbose=1,steps = len(test_v2))
pred_v2 = temp[:,1:]

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1777044636.py in <cell line: 0>()
     19 test_generator_v2.reset()
     20 
---> 21 temp = model_v2.predict_generator(test_generator_v2,verbose=1,steps = len(test_v2))
     22 pred_v2 = temp[:,1:]

NameError: name 'model_v2' is not defined

## === cell 12
labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
pred_new = 0.5*pred_v1+0.5*pred_v2

predicted_class_indices_new = np.argmax(pred_new,axis=1)

labels = dict((v,k) for k,v in labels.items())
predictions_new = [labels[k] for k in predicted_class_indices_new]

filenames=test_generator_v2.filenames
results_new=pd.DataFrame({"image_id":filenames,
                      "label":predictions_new})

results_new.to_csv('/kaggle/working/submission.csv',index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/850100833.py in <cell line: 0>()
      1 labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
----> 2 pred_new = 0.5*pred_v1+0.5*pred_v2
      3 
      4 predicted_class_indices_new = np.argmax(pred_new,axis=1)
      5 

NameError: name 'pred_v1' is not defined
