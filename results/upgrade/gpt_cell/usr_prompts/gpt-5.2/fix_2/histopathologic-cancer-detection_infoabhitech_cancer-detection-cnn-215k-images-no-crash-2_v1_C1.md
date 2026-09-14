# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
import tf_keras as keras
from tf_keras.preprocessing import image
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten
from tf_keras.layers import Conv2D, MaxPooling2D

from skimage.io import imread
from skimage.io import imshow

import os


## === cell 2
train = pd.read_csv("../input/train_labels.csv")


## === cell 3
train.head()


## === cell 4
print("Number of training smaples -->" ,len(train))


## === cell 5

def train_func_image_file(x):
    folder = '../input/train/'
    path = folder + x + '.tif'
    return path


## === cell 6

train['path'] = train['id'].apply(train_func_image_file)


## === cell 7
print(train['path'][0])


## === cell 8

train['image'] = train['path'][0:215000].map(imread)


## === cell 9
print(imshow(train['image'][1]))


## === cell 10

def crop(x):
    return x[24:72, 24:72]


## === cell 11

train['image_crop'] = train['image'][0:215000].map(crop)


## === cell 12
print("Cropped image" ,imshow(train['image_crop'][1]))


## === cell 13
print("Dimension of image --->" ,train['image'][0].shape)


## === cell 14
print("Dimension of crop image --->" ,train['image_crop'][0].shape)


## === cell 15
train = train.drop(['path'], axis=1)


## === cell 16
train = train.drop(['image'], axis=1)


## === cell 17

import gc; 
gc.collect()


## === cell 18

x_train = np.stack(list(train.image_crop.iloc[0:215000]), axis = 0)


## === cell 19
train = train.drop(['image_crop'], axis=1)


## === cell 20
import gc; 
gc.collect()


## === cell 21
x_train = x_train.astype('float32')


## === cell 22

x_train /= 255


## === cell 23

num_classes = 2


## === cell 24

y_train = train['label'][0:215000]


## === cell 25
y_train = keras.utils.to_categorical(y_train, num_classes)


## === cell 26
del train


## === cell 27
import gc; 
gc.collect()


## === cell 28

img_rows, img_cols = 48, 48

input_shape = (img_rows, img_cols, 3)

batch_size = 128
epochs = 3


## === cell 29

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))


## === cell 30
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


## === cell 31

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)


## === cell 32
del x_train


## === cell 33
import gc; 
gc.collect()


## === cell 34

image_file = []
for file in os.listdir("../input/test/"):
    image_file.append(file)


## === cell 35

test = pd.DataFrame(image_file,columns=['file'])


## === cell 36
test.head()


## === cell 37

def test_func_image_file(x):
    folder = '../input/test/'
    path = folder + x
    return path


## === cell 38
test['path'] = test['file'].apply(test_func_image_file)


## === cell 39

test['image'] = test['path'][0:].map(imread)


## === cell 40
test['image_crop'] = test['image'][0:].map(crop)


## === cell 41
test = test.drop(['image'], axis=1)


## === cell 42
x_test = np.stack(list(test.image_crop.iloc[0:]), axis = 0)


## === cell 43
test = test.drop(['image_crop'], axis=1)


## === cell 44
import gc; 
gc.collect()


## === cell 45
x_test = x_test.astype('float32')


## === cell 46
x_test /= 255


## === cell 47
test['id'] = test['file'].apply(lambda x: os.path.splitext(x)[0])


## === cell 48
predictions = model.predict_classes(x_test)


## === cell 49
test['label'] = pd.Series(predictions)


## === cell 50
print("Cancer Detected - True Positive --> ",len(test['label'][test['label']==1]))


## === cell 51
print("NO Cancer Detected - True Negative --> ",len(test['label'][test['label']==0]))


## === cell 52
test = test.drop(['file','path'], axis=1)


## === cell 53
test.head()


## === cell 54
test.to_csv("submission.csv", columns = test.columns, index=False)
