# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        input/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        working/
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
```

-> data/leaf-classification/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/leaf-classification/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/leaf-classification/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> data/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

class Data_Clean(object):
    def __init__(self):
        self.numerical_data, self.num_test_data = self.read_numerical_data()
        self.id, self.species, self.num_train, self.test_id, self.test_num = self.split_numerical_data()

    def split_numerical_data(self):
        id = self.numerical_data.pop('id')
        species = self.numerical_data.pop('species')
        species = LabelEncoder().fit(species).transform(species)
        species = to_categorical(species, num_classes=99)
        self.numerical_data = StandardScaler().fit(self.numerical_data).transform(self.numerical_data)

        test_id = self.num_test_data.pop('id')
        num_test_data = StandardScaler().fit(self.num_test_data).transform(self.num_test_data)
        return id, species, self.numerical_data, test_id, num_test_data

    def read_numerical_data(self):
        root = "../input"
        data = pd.read_csv('%s/train.csv' %root)
        test_data = pd.read_csv('%s/test.csv' %(root))
        return data, test_data

    def run(self):
        return self.species, self.num_train, self.test_num, self.test_id

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

class CNN(object):
    def __init__(self, num_train, num_vali, species_train, species_vali, num_test, test_id):
        self.num_train = num_train
        self.num_vali = num_vali
        self.species_train = species_train
        self.species_vali = species_vali
        self.num_test = num_test
        self.test_id = test_id

    def define_CNN(self):
        model = Sequential()  #0.9798
        model.add(Dense(512, activation='relu', input_dim=192))
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(99, activation='softmax'))
        self.model = model


    def RMSprop(self, batch_size, epoch):
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
        self.model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        self.model.fit(self.num_train, self.species_train, batch_size=batch_size, epochs=epoch,
                       validation_data=(self.num_vali, self.species_vali), verbose=2, callbacks=[learning_rate_reduction])

    def make_predict(self):
        ypred_prob = self.model.predict(self.num_test)
        root = "../input"
        labels = sorted(pd.read_csv(os.path.join(root, 'train.csv')).species.unique())
        ypred = pd.DataFrame(ypred_prob, index = self.test_id, columns=labels)
        print(ypred.head(2))
        fp = open('submit.csv', 'w')
        fp.write(ypred.to_csv())

    def run(self):
        self.define_CNN()
        self.RMSprop(batch_size = 128, epoch=100)
        self.make_predict()


if __name__ == '__main__':
    data_clean = Data_Clean()
    species, num, num_test,  test_id = data_clean.run()
    species_train, species_vali, num_train, num_vali = train_test_split(species, num, test_size=0.1, random_state=26)

    cnn = CNN(num_train, num_vali,species_train, species_vali, num_test, test_id)
    cnn.run()


## --- ERROR in cell 1, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'
