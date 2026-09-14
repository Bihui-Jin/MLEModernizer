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
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

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

0.9982

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
print(os.listdir("../input"))



## === cell 1
import os
import sys
import numpy as np
import pandas as pd
import zipfile
import keras
from PIL import Image
from keras_applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tqdm import tqdm
from sklearn.model_selection import train_test_split


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
print(os.listdir("../input/train/train")[0])
path = os.path.join('../input/train/train', os.listdir("../input/train/train")[0])
print(path)
print(load_img(path))


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3915284895.py in <cell line: 0>()
      2 path = os.path.join('../input/train/train', os.listdir("../input/train/train")[0])
      3 print(path)
----> 4 print(load_img(path))

NameError: name 'load_img' is not defined

## === cell 3
class DataHandler():
    def __init__(self, csv_name):
        self.dataset_path = '../input'
        print(os.listdir(self.dataset_path))
        self.train_data = pd.read_csv(os.path.join(self.dataset_path, csv_name))

    def get_from_columns(self, *args):
        for column in args:
            yield self.train_data[column]

    def get_only_data(self, folname):
        fol_path = os.path.join(self.dataset_path, folname)
        img_paths = list(map(lambda fname: os.path.join(fname), os.listdir(fol_path)))
        return img_paths

    def get_data_label(self, folname='train/train', val_rate=0.2):
        fol_path = os.path.join(self.dataset_path, folname)
        params = self.train_data.columns.values
        datas = []
        labels = []
        if folname.split('/')[0] == 'train':
            for idx ,row in tqdm(self.train_data.iterrows()):
                for data_or_label, param in enumerate(params):
                    if data_or_label == 0:
                        img_name = row[param]
                        img_data = img_to_array(load_img(os.path.join(fol_path, img_name)))
                        datas.append(img_data)
                    else:
                        labels.append(float(row[param]))
        else:
            img_names = os.listdir(fol_path)
            self.test_names = img_names
            for img_name in tqdm(img_names):
                img_data = img_to_array(load_img(os.path.join(fol_path, img_name)))
                datas.append(img_data)
            datas = np.asarray(datas).astype(np.float)
            datas /= 255
            print(len(datas))
            return datas

        datas = np.asarray(datas).astype(np.float)
        datas /= 255
        labels = np.asarray(labels).astype(np.float)
        datas, val_datas, labels, val_labels = train_test_split(datas, labels, test_size=val_rate)
        return (datas, labels, val_datas, val_labels)

    def get_shape(self, folname='train/train'):
        from random import randint
        fol_path = os.path.join(self.dataset_path, folname)
        sample_img_path = os.path.join(fol_path ,os.listdir(fol_path)[randint(0,20)])
        img_bin = img_to_array(load_img(sample_img_path)) # return ndarray
        return img_bin.shape


## === cell 4
class Train():
    def __init__(self, use_imgnt=False):
        pass

    def build_model(self, img_shape, do_cb=True):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=img_shape))
        self.model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.1))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(5, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.1))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        if do_cb == True:
            self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1)
            self.early_stopping = EarlyStopping(monitor='val_loss', verbose=1, mode='auto')
        return self.model.summary

    def train(self, x_train, y_train, x_test, y_test, epochs=20, batch_size=32):
        hist = self.model.fit(
            x=x_train, y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(x_test, y_test),
        )
        return hist
    
    def predict(self, test):
        pred = self.model.predict(
            test
        )
        return pred


## === cell 5
data_handler = DataHandler('train.csv')
datagen = data_handler.get_from_columns('id', 'has_cactus')
img_shape = data_handler.get_shape()
test = data_handler.get_data_label('test/test')
test_names = data_handler.test_names
print(len(test), len(test_names))
x_train, y_train, x_test, y_test = data_handler.get_data_label('train/train')
model = Train()
summary = model.build_model(img_shape=img_shape)
summary()
hist = model.train(x_train, y_train, x_test, y_test)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1466222436.py in <cell line: 0>()
      1 data_handler = DataHandler('train.csv')
      2 datagen = data_handler.get_from_columns('id', 'has_cactus')
----> 3 img_shape = data_handler.get_shape()
      4 test = data_handler.get_data_label('test/test')
      5 test_names = data_handler.test_names

/tmp/ipykernel_11/543977294.py in get_shape(self, folname)
     55         fol_path = os.path.join(self.dataset_path, folname)
     56         sample_img_path = os.path.join(fol_path ,os.listdir(fol_path)[randint(0,20)])
---> 57         img_bin = img_to_array(load_img(sample_img_path)) # return ndarray
     58         return img_bin.shape

NameError: name 'img_to_array' is not defined

## === cell 6
predict = model.predict(test)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/669041201.py in <cell line: 0>()
----> 1 predict = model.predict(test)

NameError: name 'model' is not defined

## === cell 7
submission_df = pd.DataFrame(predict, columns=['has_cactus'])
submission_df['id'] = ''
cols = submission_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
submission_df = submission_df[cols]
for i, name in enumerate(test_names):
    submission_df.set_value(i, 'id', name)
submission_df.head(10)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/606525111.py in <cell line: 0>()
----> 1 submission_df = pd.DataFrame(predict, columns=['has_cactus'])
      2 submission_df['id'] = ''
      3 cols = submission_df.columns.tolist()
      4 cols = cols[-1:] + cols[:-1]
      5 submission_df = submission_df[cols]

NameError: name 'predict' is not defined

## === cell 8
submission_df.to_csv('submission.csv', index=False)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4111713007.py in <cell line: 0>()
----> 1 submission_df.to_csv('submission.csv', index=False)

NameError: name 'submission_df' is not defined
