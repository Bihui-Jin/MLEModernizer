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

3.13

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

0.8779087337564219

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

import re

from datetime import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

import tensorflow_hub as hub

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.models import Model

from sklearn.preprocessing import LabelEncoder



label_to_disease = pd.read_json('/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json', typ='series')

train_csv = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')



train_csv['disease'] = train_csv['label'].map(label_to_disease)

train_csv['path'] = '/kaggle/input/cassava-leaf-disease-classification/train_images/' + train_csv['image_id']



train_csv['label_encoded'] = LabelEncoder().fit_transform(train_csv['disease'])




train_csv['disease'] = train_csv['disease'].astype(str)

train_csv['label'] = train_csv['label'].astype(str)




train, valid = train_test_split(train_csv, test_size=0.2, stratify=train_csv['label'])






datagen_aug = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=45,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True,

    fill_mode='nearest'

)




train_generator = datagen_aug.flow_from_dataframe(

    dataframe=train,

    x_col='path',

    y_col='disease',

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical',

    shuffle=True

)




datagen_valid = ImageDataGenerator(preprocessing_function=preprocess_input)



valid_generator = datagen_valid.flow_from_dataframe(

    dataframe=valid,

    x_col='path',

    y_col='disease',

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical',

    shuffle=False

)

## === cell 2
def load_and_preprocess_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)  # Assuming the images are JPEGs

    image = tf.image.resize(image, [224, 224])  # Resize to the expected input size

    image = image / 255.0  # Normalize to [0, 1]

    return image, label




train_ds = tf.data.Dataset.from_tensor_slices((train['path'].values, train['label_encoded'].values))

valid_ds = tf.data.Dataset.from_tensor_slices((valid['path'].values, valid['label_encoded'].values))




train_ds = train_ds.map(load_and_preprocess_image).batch(16).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

valid_ds = valid_ds.map(load_and_preprocess_image).batch(16).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)




## === cell 3
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback





import tensorflow as tf

import tensorflow_hub as hub

import numpy as np



from tensorflow.keras.applications import DenseNet169



class EarlyStoppingCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):


        if self.model.stop_training:

            print(f"Early stopping triggered at epoch {epoch + 1}.")



early_stopping = EarlyStopping(

    monitor='val_loss', 

    patience=3, 

    restore_best_weights=True

)



learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(

    monitor='val_loss', 

    patience=2, 

    factor=0.5, 

    min_lr=1e-7, 

    verbose=1

)




## === cell 8
import tensorflow as tf
from tensorflow.keras.layers import Input, TFSMLayer
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os


cropnet_model_path = '/kaggle/input/vanilla_cropnet/tensorflow2/default/1/kaggle/working/kaggle/working/vanilla_cropnet_tf'
layer = TFSMLayer(cropnet_model_path, call_endpoint='serving_default')
input_layer = Input(shape=(224, 224, 3))
output_layer = layer(input_layer)
cropnet_model = Model(inputs=input_layer, outputs=output_layer)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2151038612.py in <cell line: 0>()
     11 #model path
     12 cropnet_model_path = '/kaggle/input/vanilla_cropnet/tensorflow2/default/1/kaggle/working/kaggle/working/vanilla_cropnet_tf'
---> 13 layer = TFSMLayer(cropnet_model_path, call_endpoint='serving_default')
     14 # Wrap TFSMLayer in a new model for prediction
     15 input_layer = Input(shape=(224, 224, 3))

/usr/local/lib/python3.11/dist-packages/keras/src/export/tfsm_layer.py in __init__(self, filepath, call_endpoint, call_training_endpoint, trainable, name, dtype)
     64         super().__init__(trainable=trainable, name=name, dtype=dtype)
     65 
---> 66         self._reloaded_obj = tf.saved_model.load(filepath)
     67 
     68         self.filepath = filepath

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/load.py in load(export_dir, tags, options)
    910   if isinstance(export_dir, os.PathLike):
    911     export_dir = os.fspath(export_dir)
--> 912   result = load_partial(export_dir, None, tags, options)["root"]
    913   return result
    914 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/load.py in load_partial(export_dir, filters, tags, options)
   1014     tags = nest.flatten(tags)
   1015   saved_model_proto, debug_info = (
-> 1016       loader_impl.parse_saved_model_with_debug_info(export_dir))
   1017 
   1018   loader = None

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/loader_impl.py in parse_saved_model_with_debug_info(export_dir)
     57     parsed. Missing graph debug info file is fine.
     58   """
---> 59   saved_model = parse_saved_model(export_dir)
     60 
     61   debug_info_path = file_io.join(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/loader_impl.py in parse_saved_model(export_dir)
    117       raise IOError(f"Cannot parse file {path_to_pbtxt}: {str(e)}.") from e
    118   else:
--> 119     raise IOError(
    120         f"SavedModel file does not exist at: {export_dir}{os.path.sep}"
    121         f"{{{constants.SAVED_MODEL_FILENAME_PBTXT}|"

OSError: SavedModel file does not exist at: /kaggle/input/vanilla_cropnet/tensorflow2/default/1/kaggle/working/kaggle/working/vanilla_cropnet_tf/{saved_model.pbtxt|saved_model.pb}

## === cell 9
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

image_dir = '/kaggle/input/cassava-leaf-disease-classification/test_images'
predictions = []
image_names = []

img_size = (224, 224)

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"): 
        img_path = os.path.join(image_dir, filename)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Normalize image to [0, 1] range
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        pred = cropnet_model(img_array)
        
        print(f"Keys in prediction dictionary for {filename}: {pred.keys()}")
        
        pred_output = list(pred.values())[0]  # Get the first tensor in the dictionary
        
        print(f"Prediction shape for {filename}: {pred_output.shape}")
        
        if len(pred_output.shape) == 1:  # 1D output
            predicted_class = np.argmax(pred_output)
        else:  # 2D output
            predicted_class = np.argmax(pred_output, axis=1)[0]
        
        predictions.append(predicted_class)
        image_names.append(filename)

submission_df = pd.DataFrame({
    'image_id': image_names,
    'label': predictions
})

submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file created: submission.csv")

print(submission_df.head())


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1442087629.py in <cell line: 0>()
     22 
     23         # Predict using the wrapped model
---> 24         pred = cropnet_model(img_array)
     25 
     26         # Inspect available keys in the prediction output

NameError: name 'cropnet_model' is not defined
