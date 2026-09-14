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

0.9827188940092166

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
m1 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/resnet_150.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m2 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3282576381.py in <cell line: 0>()
----> 1 m1 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/resnet_150.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      2 m2 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

NameError: name 'hub' is not defined

## === cell 4
m3 = tf.keras.models.load_model('../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m4 = tf.keras.models.load_model('../input/notebooka9ca40495e/model_effnet_s.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m5 = tf.keras.models.load_model('../input/paddy-doctor-training/model_effnet_b4.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/663280605.py in <cell line: 0>()
----> 1 m3 = tf.keras.models.load_model('../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      2 m4 = tf.keras.models.load_model('../input/notebooka9ca40495e/model_effnet_s.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      3 m5 = tf.keras.models.load_model('../input/paddy-doctor-training/model_effnet_b4.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

NameError: name 'hub' is not defined

## === cell 5
m6 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m_na.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
m7 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/resinc_v2.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2454730957.py in <cell line: 0>()
----> 1 m6 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m_na.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
      2 m7 = tf.keras.models.load_model('../input/paddy-doc-ensemble-models/ensemble_estimators/resinc_v2.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

NameError: name 'hub' is not defined

## === cell 7
def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x = []
        anchors_y = []

        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])

            if (rv not in anchors_x):
                anchors_x.append(rv)

        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])

            if (rv not in anchors_y):
                anchors_y.append(rv)

        for x, y in zip(anchors_x, anchors_y):
            image[x:x+patch_size, y:y+patch_size, :] = 0

        return image
    
    else:
        return image

def random_gaus_blur(image):
    if random.choice([True, False]):
        return cv2.GaussianBlur(image, (7,7), 0)
    
    else:
        return image

def random_displacment(image):
    if random.choice([True, False]):
        ax = random.choice([0,1])
        
        if ax == 0:
            slices = np.split(image, 6, axis=ax)
            np.random.shuffle(slices)
            
            return np.row_stack(slices)
        
            
        else:
            slices = np.split(image, 6, axis=ax)
            np.random.shuffle(slices)
            
            return np.column_stack(slices)
        
    else:
        return image

def center_crop_and_random_augmentations_fn(image):
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = random_cutout(image, 16, 16)
    image = random_displacment(image)
    image = random_gaus_blur(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1).numpy()
    
    return image

def test_time_augmentation_fn_1(image):
    image = tf.image.random_crop(image, (256, 256, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    
    return image

def test_time_augmentation_fn_2(image):
    image = tf.image.random_crop(image, (384, 384, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    
    return image

def test_time_augmentation_fn_3(image):
    image = tf.image.random_crop(image, (300, 300, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    
    return image

## === cell 8
generator_1 = ImageDataGenerator(rescale=1.0/255,
                               horizontal_flip=True,
                               vertical_flip=True,
                               preprocessing_function=test_time_augmentation_fn_1)

generator_2 = ImageDataGenerator(rescale=1.0/255,
                               featurewise_center=True,
                               featurewise_std_normalization=True,
                               horizontal_flip=True,
                               vertical_flip=True,
                               preprocessing_function=test_time_augmentation_fn_2)

generator_3 = ImageDataGenerator(rescale=1.0/255,
                               horizontal_flip=True,
                               vertical_flip=True,
                               preprocessing_function=test_time_augmentation_fn_3)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/610967615.py in <cell line: 0>()
----> 1 generator_1 = ImageDataGenerator(rescale=1.0/255,
      2                                horizontal_flip=True,
      3                                vertical_flip=True,
      4                                preprocessing_function=test_time_augmentation_fn_1)
      5 

NameError: name 'ImageDataGenerator' is not defined

## === cell 9
to_gen_fit = []
files_to_fit = ['../input/paddy-disease-classification/train_images/bacterial_leaf_blight/100049.jpg',
                '../input/paddy-disease-classification/train_images/bacterial_leaf_streak/100042.jpg',
                '../input/paddy-disease-classification/train_images/bacterial_panicle_blight/100068.jpg',
                '../input/paddy-disease-classification/train_images/blast/100012.jpg',
                '../input/paddy-disease-classification/train_images/brown_spot/100022.jpg',
                '../input/paddy-disease-classification/train_images/dead_heart/100020.jpg',
                '../input/paddy-disease-classification/train_images/downy_mildew/100059.jpg',
                '../input/paddy-disease-classification/train_images/hispa/100139.jpg',
                '../input/paddy-disease-classification/train_images/normal/100111.jpg',
                '../input/paddy-disease-classification/train_images/tungro/100134.jpg',
               ]

for file in files_to_fit:
    to_gen_fit.append(img_to_array(load_img(file), dtype='uint8'))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1015949397.py in <cell line: 0>()
     13 
     14 for file in files_to_fit:
---> 15     to_gen_fit.append(img_to_array(load_img(file), dtype='uint8'))

NameError: name 'img_to_array' is not defined

## === cell 10
generator_2.fit(to_gen_fit)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/426103849.py in <cell line: 0>()
----> 1 generator_2.fit(to_gen_fit)

NameError: name 'generator_2' is not defined

## === cell 11
test_loc = '../input/paddy-disease-classification/test_images'

test_data_256 = generator_1.flow_from_directory(directory=test_loc,
                                              target_size=(256, 256),
                                              batch_size=32,
                                              classes=['.'],
                                              shuffle=False,
                                             )

test_data_300 = generator_3.flow_from_directory(directory=test_loc,
                                              target_size=(300, 300),
                                              batch_size=16,
                                              classes=['.'],
                                              shuffle=False,
                                             )

test_data_384 = generator_2.flow_from_directory(directory=test_loc,
                                              target_size=(384, 384),
                                              batch_size=16,
                                              classes=['.'],
                                              shuffle=False,
                                             )

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2011292270.py in <cell line: 0>()
      1 test_loc = '../input/paddy-disease-classification/test_images'
      2 
----> 3 test_data_256 = generator_1.flow_from_directory(directory=test_loc,
      4                                               target_size=(256, 256),
      5                                               batch_size=32,

NameError: name 'generator_1' is not defined

## === cell 12
len(test_data_256.next()[0]), len(test_data_384.next()[0]), len(test_data_300.next()[0])

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4055326262.py in <cell line: 0>()
----> 1 len(test_data_256.next()[0]), len(test_data_384.next()[0]), len(test_data_300.next()[0])

NameError: name 'test_data_256' is not defined

## === cell 14
test_encodings = []

for m in [m1,m3,m4, m6, m7]:
    enc = np.zeros((3469, 10))
    
    for _ in range(5):
        enc_ = m.predict(test_data_256, verbose=1)
        enc += enc_
        
    enc /= 5
    test_encodings.append(enc)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2452727994.py in <cell line: 0>()
      1 test_encodings = []
      2 
----> 3 for m in [m1,m3,m4, m6, m7]:
      4     enc = np.zeros((3469, 10))
      5 

NameError: name 'm1' is not defined

## === cell 15
for m in [m6,m7]:
    enc = np.zeros((3469, 10))
    
    for _ in range(5):
        enc_ = m.predict(test_data_256, verbose=1)
        enc += enc_
        
    enc /= 5
    test_encodings.append(enc)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/920837550.py in <cell line: 0>()
----> 1 for m in [m6,m7]:
      2     enc = np.zeros((3469, 10))
      3 
      4     for _ in range(5):
      5         enc_ = m.predict(test_data_256, verbose=1)

NameError: name 'm6' is not defined

## === cell 16
test_encodings2 = np.zeros((3469, 10))

for _ in range(5):
    encodings2 = m2.predict(test_data_300, verbose=1)
    test_encodings2 += encodings2
    
test_encodings2 /= 5
test_encodings.append(test_encodings2)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1632201977.py in <cell line: 0>()
      2 
      3 for _ in range(5):
----> 4     encodings2 = m2.predict(test_data_300, verbose=1)
      5     test_encodings2 += encodings2
      6 

NameError: name 'm2' is not defined

## === cell 17
test_encodings5 = np.zeros((3469, 10))

for _ in range(5):
    encodings5 = m5.predict(test_data_384, verbose=1)
    test_encodings5 += encodings5
    
test_encodings5 /= 5
test_encodings.append(test_encodings5)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/946481735.py in <cell line: 0>()
      2 
      3 for _ in range(5):
----> 4     encodings5 = m5.predict(test_data_384, verbose=1)
      5     test_encodings5 += encodings5
      6 

NameError: name 'm5' is not defined

## === cell 18
test_encodings

## === cell 19
predict_max = [np.argmax(test_enc, axis=1) for test_enc in test_encodings]
predict_max

## === cell 20
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

## === cell 21
predictions = []

for enc in predict_max:
    predictions.append([inverse_map[k] for k in enc])

## === cell 22
files=test_data_256.filenames

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1658739243.py in <cell line: 0>()
----> 1 files=test_data_256.filenames

NameError: name 'test_data_256' is not defined

## === cell 23
s_full = pd.DataFrame({"image_id":files})
s_full.image_id = s_full.image_id.str.replace('./', '')

for pres in predictions:
    temp = pd.DataFrame({"image_id":files,
                          "label":pres})
    temp.image_id = temp.image_id.str.replace('./', '')
    

    s_full = pd.merge(s_full, temp, on='image_id')
    
s_full = s_full.set_index('image_id')
s_full

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1005585544.py in <cell line: 0>()
----> 1 s_full = pd.DataFrame({"image_id":files})
      2 s_full.image_id = s_full.image_id.str.replace('./', '')
      3 
      4 for pres in predictions:
      5     temp = pd.DataFrame({"image_id":files,

NameError: name 'files' is not defined

## === cell 24
s_full_mode = ss.mode(s_full, axis=1)

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3552784536.py in <cell line: 0>()
----> 1 s_full_mode = ss.mode(s_full, axis=1)

NameError: name 's_full' is not defined

## === cell 25
s_full_mode[0].reshape(1,-1)[0]

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/508765162.py in <cell line: 0>()
----> 1 s_full_mode[0].reshape(1,-1)[0]

NameError: name 's_full_mode' is not defined

## === cell 26
final = pd.DataFrame()
final.index = s_full.index
final['label'] = s_full_mode[0].reshape(1,-1)[0]
final

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/35001160.py in <cell line: 0>()
      1 final = pd.DataFrame()
----> 2 final.index = s_full.index
      3 final['label'] = s_full_mode[0].reshape(1,-1)[0]
      4 final

NameError: name 's_full' is not defined

## === cell 27
final.value_counts()

## === cell 28
final.to_csv('model_submission_v22.csv', index=True)

## --- ERROR in outputing the csv:
Invalid submission: Expected columns {'label', 'image_id'}, but got {'Unnamed: 0'}
