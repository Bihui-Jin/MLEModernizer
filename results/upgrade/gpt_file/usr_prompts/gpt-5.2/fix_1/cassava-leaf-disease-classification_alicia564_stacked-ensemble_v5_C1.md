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

0.8961922030825022

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

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # Assuming the images are JPEGs
    image = tf.image.resize(image, [224, 224])  # Resize to the expected input size
    image = image / 255.0  # Normalize to [0, 1]
    return image, label

label_to_disease = pd.read_json('/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json', typ='series')
train_csv = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')

train_csv['disease'] = train_csv['label'].map(label_to_disease)
train_csv['path'] = '/kaggle/input/cassava-leaf-disease-classification/train_images/' + train_csv['image_id']

train_csv['label_encoded'] = LabelEncoder().fit_transform(train_csv['disease'])

train_csv['disease'] = train_csv['disease'].astype(str)
train_csv['label'] = train_csv['label'].astype(str)

train, valid = train_test_split(train_csv, test_size=0.2, stratify=train_csv['label'])

train_ds = tf.data.Dataset.from_tensor_slices((train['path'].values, train['label_encoded'].values))
valid_ds = tf.data.Dataset.from_tensor_slices((valid['path'].values, valid['label_encoded'].values))

train_ds = train_ds.map(load_and_preprocess_image).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.map(load_and_preprocess_image).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from tensorflow.keras.applications import DenseNet169

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    patience=2, 
    factor=0.5, 
    min_lr=1e-6, 
    verbose=1
)

## === cell 3
from tensorflow.keras.layers import Input, TFSMLayer
from tensorflow.keras.models import Model

cropnet_layer = TFSMLayer('/kaggle/input/cropnet_model/tensorflow2/default/1/kaggle/working/model_feature_extraction_tf', call_endpoint='serving_default')
densenet_layer = TFSMLayer('/kaggle/input/densenet_model/tensorflow2/default/1/kaggle/working/kaggle/working/densenet_model_tf', call_endpoint='serving_default')
efficientnet_layer = TFSMLayer('/kaggle/input/efficientnetb4_model/tensorflow2/default/1/kaggle/working/kaggle/working/efficientnet_model_tf', call_endpoint='serving_default')

input_layer = Input(shape=(224, 224, 3))

cropnet_output = cropnet_layer(input_layer)
densenet_output = densenet_layer(input_layer)
efficientnet_output = efficientnet_layer(input_layer)

cropnet_model = Model(inputs=input_layer, outputs=cropnet_output)
densenet_model = Model(inputs=input_layer, outputs=densenet_output)
efficientnet_model = Model(inputs=input_layer, outputs=efficientnet_output)

cropnet_weight = 0.65
densenet_weight = 0.25
efficientnet_weight = 0.1

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3718978740.py in <cell line: 0>()
      3 
      4 # Load models with TFSMLayer
----> 5 cropnet_layer = TFSMLayer('/kaggle/input/cropnet_model/tensorflow2/default/1/kaggle/working/model_feature_extraction_tf', call_endpoint='serving_default')
      6 densenet_layer = TFSMLayer('/kaggle/input/densenet_model/tensorflow2/default/1/kaggle/working/kaggle/working/densenet_model_tf', call_endpoint='serving_default')
      7 efficientnet_layer = TFSMLayer('/kaggle/input/efficientnetb4_model/tensorflow2/default/1/kaggle/working/kaggle/working/efficientnet_model_tf', call_endpoint='serving_default')

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

OSError: SavedModel file does not exist at: /kaggle/input/cropnet_model/tensorflow2/default/1/kaggle/working/model_feature_extraction_tf/{saved_model.pbtxt|saved_model.pb}

## === cell 7
soft_voting_features = []
soft_voting_labels = []

for img_batch, label_batch in train_ds:
    img_batch = np.array(img_batch)  # Convert images to numpy array

    cropnet_probs = cropnet_model.predict(img_batch)['output_0']
    densenet_probs = densenet_model.predict(img_batch)['output_0']
    efficientnet_probs = efficientnet_model.predict(img_batch)['output_0']
    
    soft_voting_probs = (
        cropnet_weight * cropnet_probs + 
        densenet_weight * densenet_probs + 
        efficientnet_weight * efficientnet_probs
    )
    
    soft_voting_features.extend(soft_voting_probs)
    soft_voting_labels.extend(label_batch)

soft_voting_features = np.array(soft_voting_features)
soft_voting_labels = np.array(soft_voting_labels)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/754606590.py in <cell line: 0>()
      8 
      9     # Get probabilities from each model
---> 10     cropnet_probs = cropnet_model.predict(img_batch)['output_0']
     11     densenet_probs = densenet_model.predict(img_batch)['output_0']
     12     efficientnet_probs = efficientnet_model.predict(img_batch)['output_0']

NameError: name 'cropnet_model' is not defined

## === cell 8
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_log_losses = []

train_distributions = []
val_distributions = []

for train_index, val_index in kf.split(soft_voting_features, soft_voting_labels):
    X_train, X_val = soft_voting_features[train_index], soft_voting_features[val_index]
    y_train, y_val = soft_voting_labels[train_index], soft_voting_labels[val_index]
    
    meta_model = LogisticRegression(max_iter=1000, C=0.1, multi_class='multinomial', penalty='l2')
    meta_model.fit(X_train, y_train)
    
    val_preds = meta_model.predict(X_val)
    val_probs = meta_model.predict_proba(X_val)
    
    fold_accuracy = accuracy_score(y_val, val_preds)
    fold_log_loss = log_loss(y_val, val_probs)
    
    fold_accuracies.append(fold_accuracy)
    fold_log_losses.append(fold_log_loss)
    
    train_dist = pd.Series(y_train).value_counts(normalize=True)
    val_dist = pd.Series(y_val).value_counts(normalize=True)
    
    train_distributions.append(train_dist.to_dict())
    val_distributions.append(val_dist.to_dict())




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3361120762.py in <cell line: 0>()
     16 
     17 # Perform cross-validation
---> 18 for train_index, val_index in kf.split(soft_voting_features, soft_voting_labels):
     19     X_train, X_val = soft_voting_features[train_index], soft_voting_features[val_index]
     20     y_train, y_val = soft_voting_labels[train_index], soft_voting_labels[val_index]

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
    769         to an integer.
    770         """
--> 771         y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
    772         return super().split(X, y, groups)
    773 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    929         n_samples = _num_samples(array)
    930         if n_samples < ensure_min_samples:
--> 931             raise ValueError(
    932                 "Found array with %d sample(s) (shape=%s) while a"
    933                 " minimum of %d is required%s."

ValueError: Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.

## === cell 9
num_folds = len(train_distributions)
plt.figure(figsize=(15, 3 * num_folds))

for fold in range(num_folds):
    train_dist = train_distributions[fold]
    val_dist = val_distributions[fold]
    
    df_train = pd.DataFrame(list(train_dist.items()), columns=['Class', 'Proportion'])
    df_val = pd.DataFrame(list(val_dist.items()), columns=['Class', 'Proportion'])
    
    plt.subplot(num_folds, 4, 4 * fold + 1)
    plt.bar(df_train['Class'], df_train['Proportion'], color='blue', alpha=0.6, label='Train')
    plt.ylim(0, 1)
    plt.title(f'Fold {fold + 1} - Train Distribution')
    plt.xlabel('Class')
    plt.ylabel('Proportion')
    plt.legend()
    
    plt.subplot(num_folds, 4, 4 * fold + 2)
    plt.bar(df_val['Class'], df_val['Proportion'], color='red', alpha=0.6, label='Validation')
    plt.ylim(0, 1)
    plt.title(f'Fold {fold + 1} - Validation Distribution')
    plt.xlabel('Class')
    plt.ylabel('Proportion')
    plt.legend()
    
    

plt.tight_layout()
plt.show()

## === cell 10
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(fold_accuracies) + 1), fold_accuracies, marker='o', color='b', label='Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy by Fold')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(fold_log_losses) + 1), fold_log_losses, marker='o', color='r', label='Log Loss')
plt.xlabel('Fold')
plt.ylabel('Log Loss')
plt.title('Cross-Validation Log Loss by Fold')
plt.legend()

plt.tight_layout()
plt.show() 

## === cell 11
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

meta_model_predictions = meta_model.predict(soft_voting_features)

cm = confusion_matrix(soft_voting_labels, meta_model_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Meta-Model on Training Data")
plt.show()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/242908381.py in <cell line: 0>()
      2 
      3 # Predict on the full training set for the confusion matrix
----> 4 meta_model_predictions = meta_model.predict(soft_voting_features)
      5 
      6 # Calculate confusion matrix

NameError: name 'meta_model' is not defined

## === cell 13
meta_models = []
for train_index, val_index in kf.split(soft_voting_features, soft_voting_labels):
    X_train, X_val = soft_voting_features[train_index], soft_voting_features[val_index]
    y_train, y_val = soft_voting_labels[train_index], soft_voting_labels[val_index]

    meta_model = LogisticRegression(max_iter=1000, C=0.1, multi_class='multinomial', penalty='l2')
    meta_model.fit(X_train, y_train)
    meta_models.append(meta_model)  # Store each fold’s trained meta-model


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2834718600.py in <cell line: 0>()
      1 meta_models = []
----> 2 for train_index, val_index in kf.split(soft_voting_features, soft_voting_labels):
      3     X_train, X_val = soft_voting_features[train_index], soft_voting_features[val_index]
      4     y_train, y_val = soft_voting_labels[train_index], soft_voting_labels[val_index]
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
    769         to an integer.
    770         """
--> 771         y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
    772         return super().split(X, y, groups)
    773 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    929         n_samples = _num_samples(array)
    930         if n_samples < ensure_min_samples:
--> 931             raise ValueError(
    932                 "Found array with %d sample(s) (shape=%s) while a"
    933                 " minimum of %d is required%s."

ValueError: Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.

## === cell 15
predictions = []
image_names = []

image_dir = '/kaggle/input/cassava-leaf-disease-classification/test_images'
img_size = (224, 224)

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(image_dir, filename)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        cropnet_probs = cropnet_model.predict(img_array)['output_0']
        densenet_probs = densenet_model.predict(img_array)['output_0']
        efficientnet_probs = efficientnet_model.predict(img_array)['output_0']

        soft_voting_probs = (
            cropnet_weight * cropnet_probs + 
            densenet_weight * densenet_probs + 
            efficientnet_weight * efficientnet_probs
        ).flatten()  # Flatten to 1D array

        fold_preds = [meta_model.predict([soft_voting_probs])[0] for meta_model in meta_models]
        final_pred = max(set(fold_preds), key=fold_preds.count)  # Majority vote

        predictions.append(final_pred)
        image_names.append(filename)

submission_df = pd.DataFrame({
    'image_id': image_names,
    'label': predictions
})

submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file created: submission.csv")

print(submission_df.head())


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2005090414.py in <cell line: 0>()
     16 
     17         # Get probabilities from each model
---> 18         cropnet_probs = cropnet_model.predict(img_array)['output_0']
     19         densenet_probs = densenet_model.predict(img_array)['output_0']
     20         efficientnet_probs = efficientnet_model.predict(img_array)['output_0']

NameError: name 'cropnet_model' is not defined
