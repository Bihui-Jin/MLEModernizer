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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8878958060904764

# 6. Current score

-0.01823

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
!pip install ../input/efficientnet/efficientnet-master/efficientnet-master

## === cell 2
import cv2
import numpy as np # linear algebra

import pandas as pd
import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import Input
from keras.models import Model
from keras.utils import *
from keras.layers import *

from tensorflow import set_random_seed
import matplotlib.pyplot as plt

set_random_seed(2)
np.random.seed(0)

import os
import gc

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
from efficientnet.keras import *

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2751819990.py in <cell line: 0>()
----> 1 from efficientnet.keras import *

ModuleNotFoundError: No module named 'efficientnet'

## === cell 4
'''
    Preprocessing using Ben Graham's method (Last competition's winner) 
    https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
'''
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): 
            return img 
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
def load_ben_color(image, IMG_SIZE, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

## === cell 5
'''
    Define model
'''
def output_relu(x):
    return K.relu(x, max_value=4)

def get_model(version, IMG_SIZE): 
    base_model = 0
    if version == 0:
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    elif version == 1:
        base_model = EfficientNetB1(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    elif version == 2:
        base_model = EfficientNetB2(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    elif version == 3:
        base_model = EfficientNetB3(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    elif version == 4:
        base_model = EfficientNetB4(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    else:
        return None 
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation=output_relu, kernel_initializer='he_normal')(x)
    model = Model(inputs=base_model.input, outputs=x)

    return model

## === cell 6
'''
    Initialize model
'''
model = get_model(3, 300)
model.load_weights('../input/pretrained-weights/model_b3.h5')

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4217787257.py in <cell line: 0>()
      2     Initialize model
      3 '''
----> 4 model = get_model(3, 300)
      5 model.load_weights('../input/pretrained-weights/model_b3.h5')

/tmp/ipykernel_11/3112753539.py in get_model(version, IMG_SIZE)
     14         base_model = EfficientNetB2(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
     15     elif version == 3:
---> 16         base_model = EfficientNetB3(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
     17     elif version == 4:
     18         base_model = EfficientNetB4(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))

NameError: name 'EfficientNetB3' is not defined

## === cell 7
'''
    Optimized Rounder
    https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa
    Objective: Minimizes mse between predictions and ground-truth labels
'''

import scipy as sp
from functools import partial
from sklearn import metrics
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _mse_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = metrics.mean_squared_error(y, X_p)
        return ll

    def fit(self, X, y):
        loss_partial = partial(self._mse_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

## === cell 9
optR = OptimizedRounder()
coefficients = [0.5, 1.5, 2.5, 3.5]
print(coefficients)

## === cell 10
test_csv = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
id_code = test_csv['id_code']
test_prediction = np.empty(len(id_code), dtype='float32')
for i in range(len(id_code)):
    img = cv2.imread('../input/aptos2019-blindness-detection/test_images/{}.png'.format(id_code[i]))
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = img.astype('float32') / 255.
    X = np.array([img])
    
    pred = model.predict(X)
    pred = np.mean(pred)
    test_prediction[i] = pred

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3589078931.py in <cell line: 0>()
     10     X = np.array([img])
     11 
---> 12     pred = model.predict(X)
     13     pred = np.mean(pred)
     14     test_prediction[i] = pred

NameError: name 'model' is not defined

## === cell 11
test_prediction = optR.predict(test_prediction, coefficients).astype('uint8')
test_csv['diagnosis'] = test_prediction
test_csv.to_csv("submission.csv", index=False)
unique, counts = np.unique(test_prediction, return_counts=True)
tmp = dict(zip(unique, counts))
print(tmp)
