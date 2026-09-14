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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.9815668202764976

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import scipy.stats as ss
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import seaborn as sns
import cv2
import albumentations as A

from albumentations.core.composition import Compose
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, train_test_split

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add, Activation
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
m1 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_b3.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m2 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m3 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/xception.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1075349466.py in <cell line: 0>()
----> 1 m1 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_b3.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      2 m2 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      3 m3 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/xception.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

NameError: name 'hub' is not defined

## === cell 4
m4 = tf.keras.models.load_model('../input/paddydocoutputs/model_resnet150.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m5 = tf.keras.models.load_model('../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m6 = tf.keras.models.load_model('../input/notebooka9ca40495e/model_effnet_s.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1475000358.py in <cell line: 0>()
----> 1 m4 = tf.keras.models.load_model('../input/paddydocoutputs/model_resnet150.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      2 m5 = tf.keras.models.load_model('../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      3 m6 = tf.keras.models.load_model('../input/notebooka9ca40495e/model_effnet_s.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

NameError: name 'hub' is not defined

## === cell 6
def random_cut_out(images):
    return tfa.image.random_cutout(images, (32, 32), constant_values=0)

def random_gaus_blur(image):
    if random.choice([True, False]):
        return cv2.GaussianBlur(image, (7,7), 0)
    
    else:
        return image
    
def fancy_pca(image):
    if random.choice([True, False]):
        o_img = image.copy().astype(np.float64)
        image = image/255.0
        img_res = image.reshape(-1, 3)
        img_centered = img_res - np.mean(img_res, axis=0)
        img_cov = np.cov(img_centered, rowvar=False)
        
        eig_val, eig_vec = np.linalg.eigh(img_cov)
        sort_perm = eig_val[::-1].argsort()
        eig_val[::-1].sort()
        eig_vec = eig_vec[:, sort_perm]
        
        m1 = np.column_stack((eig_vec))
        m2 = np.zeros((3,1))
        alpha = np.random.normal(0, 0.1)
        m2[:,0] = alpha * eig_val[:]
        add_vect = np.matrix(m1) * np.matrix(m2)
        
        for idx in range(3):
            o_img[..., idx] += add_vect[idx]
            
        o_img = np.clip(o_img, 0.0, 255.0)
        o_img = o_img.astype(np.uint8)
        
        return o_img
    
    else:
        return image

def random_displacment(image):
    if random.choice([True, False]):
        ax = random.choice([0,1])
        
        if ax == 0:
            slices = np.split(image, 8, axis=ax)
            np.random.shuffle(slices)
            
            return np.row_stack(slices)
        
            
        else:
            slices = np.split(image, 8, axis=ax)
            np.random.shuffle(slices)
            
            return np.column_stack(slices)
        
    else:
        return image

def center_crop_and_random_augmentations_fn(image):
    image = tf.image.random_crop(image, (256, 256, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    
    return image

## === cell 7
generator = ImageDataGenerator(rescale=1 / 255,
                               rotation_range=5,
                               width_shift_range=0.25,
                               height_shift_range=0.25,
                               shear_range=0.2,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               vertical_flip=True,
                               validation_split=0.2,
                               preprocessing_function=center_crop_and_random_augmentations_fn
                              )

train_datagen = generator.flow_from_directory('../input/paddy-disease-classification/train_images/',
                                              target_size=(256, 256),
                                              batch_size=16,
                                              subset='training')

valid_datagen = generator.flow_from_directory('../input/paddy-disease-classification/train_images/',
                                              target_size=(256, 256),
                                              batch_size=16,
                                              subset='validation')

train_datagen_300 = generator.flow_from_directory('../input/paddy-disease-classification/train_images/',
                                              target_size=(300, 300),
                                              batch_size=16,
                                              subset='training')

valid_datagen_300 = generator.flow_from_directory('../input/paddy-disease-classification/train_images/',
                                              target_size=(300, 300),
                                              batch_size=16,
                                              subset='validation')

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3182148989.py in <cell line: 0>()
----> 1 generator = ImageDataGenerator(rescale=1 / 255,
      2                                rotation_range=5,
      3                                width_shift_range=0.25,
      4                                height_shift_range=0.25,
      5                                shear_range=0.2,

NameError: name 'ImageDataGenerator' is not defined

## === cell 8
test_loc = '../input/paddy-disease-classification/test_images'

test_data_256 = ImageDataGenerator(rescale=1.0/255).flow_from_directory(directory=test_loc,
                                                                    target_size=(256, 256),
                                                                    batch_size=16,
                                                                    classes=['.'],
                                                                    shuffle=False,
                                                                   )

test_data_300 = ImageDataGenerator(rescale=1.0/255).flow_from_directory(directory=test_loc,
                                                                    target_size=(300, 300),
                                                                    batch_size=16,
                                                                    classes=['.'],
                                                                    shuffle=False,
                                                                   )

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/27539720.py in <cell line: 0>()
      1 test_loc = '../input/paddy-disease-classification/test_images'
      2 
----> 3 test_data_256 = ImageDataGenerator(rescale=1.0/255).flow_from_directory(directory=test_loc,
      4                                                                     target_size=(256, 256),
      5                                                                     batch_size=16,

NameError: name 'ImageDataGenerator' is not defined

## === cell 9
model_train_test_score = []

for model in [m1,m2,m3,m4,m5,m6]:
    try:
        model_train_test_score.append(model.evaluate(valid_datagen))
    except:
        model_train_test_score.append(model.evaluate(valid_datagen_300))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2861726453.py in <cell line: 0>()
      1 model_train_test_score = []
      2 
----> 3 for model in [m1,m2,m3,m4,m5,m6]:
      4     try:
      5         model_train_test_score.append(model.evaluate(valid_datagen))

NameError: name 'm1' is not defined

## === cell 10
m1_predict_max = np.argmax(m1.predict(test_data_256, verbose=1),axis=1)
m3_predict_max = np.argmax(m3.predict(test_data_256, verbose=1),axis=1)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1883706862.py in <cell line: 0>()
----> 1 m1_predict_max = np.argmax(m1.predict(test_data_256, verbose=1),axis=1)
      2 m3_predict_max = np.argmax(m3.predict(test_data_256, verbose=1),axis=1)

NameError: name 'm1' is not defined

## === cell 12
m1_p = m1.predict(test_data_256, verbose=1)
m2_p = m2.predict(test_data_300, verbose=1)
m4_p = m4.predict(test_data_300, verbose=1)
m5_p = m5.predict(test_data_256, verbose=1)
m6_p = m6.predict(test_data_256, verbose=1)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1976677560.py in <cell line: 0>()
----> 1 m1_p = m1.predict(test_data_256, verbose=1)
      2 m2_p = m2.predict(test_data_300, verbose=1)
      3 m4_p = m4.predict(test_data_300, verbose=1)
      4 m5_p = m5.predict(test_data_256, verbose=1)
      5 m6_p = m6.predict(test_data_256, verbose=1)

NameError: name 'm1' is not defined

## === cell 13
m1_predict_max = np.argmax(m1_p,axis=1)
m1_confidance = np.max(m1_p,axis=1)

m2_predict_max = np.argmax(m2_p,axis=1)
m2_confidance = np.max(m2_p,axis=1)

m4_predict_max = np.argmax(m4_p,axis=1)
m4_confidance = np.max(m4_p,axis=1)

m5_predict_max = np.argmax(m5_p,axis=1)
m5_confidance = np.max(m5_p,axis=1)

m6_predict_max = np.argmax(m6_p,axis=1)
m6_confidance = np.max(m6_p,axis=1)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2965578321.py in <cell line: 0>()
----> 1 m1_predict_max = np.argmax(m1_p,axis=1)
      2 m1_confidance = np.max(m1_p,axis=1)
      3 
      4 m2_predict_max = np.argmax(m2_p,axis=1)
      5 m2_confidance = np.max(m2_p,axis=1)

NameError: name 'm1_p' is not defined

## === cell 14
class_indices = {'bacterial_leaf_blight': 0,
                 'bacterial_leaf_streak': 1,
                 'bacterial_panicle_blight': 2,
                 'blast': 3,
                 'brown_spot': 4,
                 'dead_heart': 5,
                 'downy_mildew': 6,
                 'hispa': 7,
                 'normal': 8,
                 'tungro': 9}

inverse_map = {v:k for k,v in class_indices.items()}

## === cell 15
pred_df = []
files=test_data_300.filenames
models = [m1_predict_max, m2_predict_max, m4_predict_max, m5_predict_max, m6_predict_max]
confidence = [m1_confidance, m2_confidance, m4_confidance, m5_confidance, m6_confidance]

for preds, conf in zip(models, confidence):
    predictions = [inverse_map[k] for k in preds]
    temp = pd.DataFrame({"image_id":files,
                         "label":predictions,
                         "conf":conf})
    temp.image_id = temp.image_id.str.replace('./', '')
    pred_df.append(temp)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4086899639.py in <cell line: 0>()
      1 pred_df = []
----> 2 files=test_data_300.filenames
      3 models = [m1_predict_max, m2_predict_max, m4_predict_max, m5_predict_max, m6_predict_max]
      4 confidence = [m1_confidance, m2_confidance, m4_confidance, m5_confidance, m6_confidance]
      5 

NameError: name 'test_data_300' is not defined

## === cell 16
s_full = pred_df[0]

for df in pred_df[1:]:
    s_full = pd.concat([s_full, df.drop('image_id', axis=1)], axis=1)
    
s_full.set_index('image_id', inplace=True)
s_full

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/231447905.py in <cell line: 0>()
----> 1 s_full = pred_df[0]
      2 
      3 for df in pred_df[1:]:
      4     s_full = pd.concat([s_full, df.drop('image_id', axis=1)], axis=1)
      5 

IndexError: list index out of range

## === cell 17
pred_df = []


for preds, conf in zip(models, confidence):
    predictions = [inverse_map[k] for k in preds]
    temp = pd.DataFrame({"image_id":files,
                         "label":predictions})
    temp.image_id = temp.image_id.str.replace('./', '')
    pred_df.append(temp)
    

s_mode_full = pred_df[0]

for df in pred_df[1:]:
    s_mode_full = pd.concat([s_mode_full, df.drop('image_id', axis=1)], axis=1)
    
s_mode_full.set_index('image_id', inplace=True)
    
s_full_mode = ss.mode(s_mode_full, axis=1)
s_full_mode

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1942204361.py in <cell line: 0>()
      2 
      3 
----> 4 for preds, conf in zip(models, confidence):
      5     predictions = [inverse_map[k] for k in preds]
      6     temp = pd.DataFrame({"image_id":files,

NameError: name 'models' is not defined

## === cell 18
s_full['most'] = s_full_mode[0].reshape(1,-1)[0]
s_full

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4160632301.py in <cell line: 0>()
----> 1 s_full['most'] = s_full_mode[0].reshape(1,-1)[0]
      2 s_full

NameError: name 's_full_mode' is not defined

## === cell 19
s_df = pd.DataFrame()
s_df['label'] = s_full_mode[0].reshape(1,-1)[0]
s_df['count'] = s_full_mode[1]
s_df

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1859464721.py in <cell line: 0>()
      1 s_df = pd.DataFrame()
----> 2 s_df['label'] = s_full_mode[0].reshape(1,-1)[0]
      3 s_df['count'] = s_full_mode[1]
      4 s_df

NameError: name 's_full_mode' is not defined

## === cell 20
s_df.index = s_full.index
s_df

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2394787086.py in <cell line: 0>()
----> 1 s_df.index = s_full.index
      2 s_df

NameError: name 's_full' is not defined

## === cell 21
s_most = s_df[s_df['count']>=3]
s_most

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3456125090.py in <cell line: 0>()
----> 1 s_most = s_df[s_df['count']>=3]
      2 s_most

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'count'

## === cell 22
s_df[s_df['count']<3].index

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1038482098.py in <cell line: 0>()
----> 1 s_df[s_df['count']<3].index

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'count'

## === cell 23
pd.options.display.max_rows = 71
s_full.loc[s_df[s_df['count']<3].index]

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3389174867.py in <cell line: 0>()
      1 pd.options.display.max_rows = 71
----> 2 s_full.loc[s_df[s_df['count']<3].index]

NameError: name 's_full' is not defined

## === cell 24
less = pd.DataFrame({'image_id':['200107.jpg', '200554.jpg', '200996.jpg', '201312.jpg', '202197.jpg',
                                 '202267.jpg', '202379.jpg', '202423.jpg', '202661.jpg'],
                     'label':['brown_spot', 'normal', 'dead_heart', 'normal', 'hispa', 'blast', 'bacterial_leaf_streak', 'brown_spot', 'tungro'],
                     'count':[1,1,1,1,1,1,1,1,1]})
less.set_index('image_id', inplace=True)
less

## === cell 25
s_final = pd.concat([s_most, less])
s_final = s_final.drop('count', axis=1)
s_final

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3562944395.py in <cell line: 0>()
----> 1 s_final = pd.concat([s_most, less])
      2 s_final = s_final.drop('count', axis=1)
      3 s_final

NameError: name 's_most' is not defined

## === cell 26
s_final.to_csv('model_submission_v12.csv', index=True)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1932727396.py in <cell line: 0>()
----> 1 s_final.to_csv('model_submission_v12.csv', index=True)

NameError: name 's_final' is not defined

## === cell 27
if not os.path.isdir('ensemble_estimators'):
    os.mkdir('ensemble_estimators')
    
m1.save('ensemble_estimators/effnet_b3.hdf5')
m2.save('ensemble_estimators/effnet_v2_m.hdf5')
m3.save('ensemble_estimators/xception.hdf5')
m4.save('ensemble_estimators/resnet_150.hdf5')
m5.save('ensemble_estimators/resinc_v2.hdf5')
m6.save('ensemble_estimators/effnet_v2_m_na.hdf5')

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3306835710.py in <cell line: 0>()
      2     os.mkdir('ensemble_estimators')
      3 
----> 4 m1.save('ensemble_estimators/effnet_b3.hdf5')
      5 m2.save('ensemble_estimators/effnet_v2_m.hdf5')
      6 m3.save('ensemble_estimators/xception.hdf5')

NameError: name 'm1' is not defined
