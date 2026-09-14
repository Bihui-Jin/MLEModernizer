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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _major(v):
    try:
        return int(v.split(".")[0])
    except Exception:
        return None


if _pb_ver is None or (_major(_pb_ver) is not None and _major(_pb_ver) >= 5):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
    )
    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            del sys.modules[m]

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns

get_ipython().run_line_magic("matplotlib", "inline")

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.utils import to_categorical


class _NPUtilsShim:
    to_categorical = staticmethod(to_categorical)


np_utils = _NPUtilsShim()

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator,
)
from keras.optimizers import RMSprop


import glob


print(os.listdir("../input"))


## === cell 1

def read_pix(jpg_dir):
    filenames = glob.glob(os.path.join(jpg_dir, '*.jpg'))
    img_array = np.zeros((len(filenames), 32, 32, 3), dtype=int)  # images as np.array
    img_index = []  # image filenames in correct order
    for idx, filename in enumerate(filenames):
        im_tmp = matplotlib.image.imread(filename)
        img_array[idx, :, :, :] = np.array(im_tmp, dtype=int)
        img_index.append(os.path.basename(filename))
    return img_array, img_index

def prepare_data(img_array, img_index, train_response):
    
    y_train_series, y_test_series = train_test_split(train_response, 
                                                     shuffle=True,
                                                     random_state=12)
    y_train = y_train_series.values[:, np.newaxis]
    y_test = y_test_series.values[:, np.newaxis]

    train_index = [img_index.index(idx) for idx in y_train_series.index]
    test_index = [img_index.index(idx) for idx in y_test_series.index]
    x_train = img_array[train_index, :, :, :]
    x_test = img_array[test_index, :, :, :]

    return x_train, y_train, x_test, y_test, y_train_series, y_test_series

def simple_oversample(x, y, oversample_class, oversample_factor=3):
    new_x = x.copy()
    new_y = y.copy()
    oversample_mask = np.ravel(y == oversample_class)
    oversample_x = x[oversample_mask, :, :, :]
    oversample_y = y[oversample_mask, :]
    for i in range(oversample_factor):
        new_x = np.concatenate([new_x, oversample_x], axis=0)
        new_y = np.concatenate([new_y, oversample_y], axis=0)
    return new_x, new_y


## === cell 2
train_labels = pd.read_csv('../input/train.csv')  # image labels in training data
sample_submission = pd.read_csv('../input/sample_submission.csv')  # example submission

train_img_array, train_img_index = read_pix('../input/train/train')
test_img_array, test_img_index = read_pix('../input/test/test')
    
train_response = train_labels['has_cactus']
train_response.index = train_labels['id']
train_response = train_response.loc[train_img_index]


## === cell 3
print('Image Labels')
print(train_labels.head(2))
print('\n')

print('Image Labels (Series)')
print(train_response.head(2))
print('\n')

print('Submission Example')
print(sample_submission.head(2))


## === cell 4
print('Some examples')

np.random.seed(27)
inspect = np.random.randint(low=0, high=train_img_array.shape[0], size=9)
for i in range(9):
    pic_id = inspect[i]
    plt.subplot(330 + 1 + i)
    plt.imshow(train_img_array[pic_id].astype(int))
plt.show()

print('Image Labels')
train_response.loc[np.array(train_img_index)[inspect]]


## === cell 5
count_classes = pd.crosstab(train_response, columns='count')
count_classes.index = ['no cactus', 'cactus']
count_classes.plot(kind='bar', legend=False)
plt.xticks(rotation=0)
plt.title('Number of instances in training data')
plt.show()
ratio = count_classes.loc['cactus', 'count'] / count_classes.loc['no cactus', 'count']
print('Ratio (cactus vs. no cactus): {:.2f}'.format(ratio))


## === cell 6
x_train, y_train, x_test, y_test, y_train_series, y_test_series = prepare_data(train_img_array, train_img_index, train_response)

count_classes_train = pd.crosstab(y_train_series, columns='count')
count_classes_train.index = ['no cactus', 'cactus']
count_classes_train.plot(kind='bar', legend=None)
plt.xticks(rotation=0)
plt.title('Number of instances in train data')
plt.show()
ratio = count_classes_train.loc['cactus', 'count'] / count_classes_train.loc['no cactus', 'count']
print('Ratio (cactus vs. no cactus): {:.2f}'.format(ratio))


## === cell 7
print('Some images from the training set')

np.random.seed(26)
inspect = np.random.randint(low=0, high=y_train.shape[0], size=9)
for i in range(9):
    pic_id = inspect[i]
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[pic_id].astype(int))
plt.show()

print('Accompanying labels')

y_train[inspect]


## === cell 8
batch_size = 128
num_classes = 2
epochs = 12

datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
)

x_train, y_train, x_test, y_test, y_train_series, y_test_series = prepare_data(train_img_array, train_img_index, train_response)
x_train, y_train = simple_oversample(x_train, y_train, oversample_class=0, oversample_factor=2)
x_test, y_test = simple_oversample(x_test, y_test, oversample_class=0, oversample_factor=2)

print("Number of 'no cactus' samples in y_train: {}".format((y_train == 0).sum()))
print("Number of 'cactus' samples in y_train: {}".format((y_train == 1).sum()))


## === cell 9
print('Some images from the oversampled training set')

np.random.seed(26)
inspect = np.random.randint(low=0, high=y_train.shape[0], size=9)
for i in range(9):
    pic_id = inspect[i]
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[pic_id].astype(int))
plt.show()

print('Accompanying labels')

y_train[inspect]


## === cell 10
x_train = x_train / 255
x_test = x_test / 255


train_generator = datagen.flow(x_train, y_train)
test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow(x_test, y_test)


## === cell 11
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    return model


## === cell 14
def elaborate_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    return model


## === cell 15
def elaborate_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    return model


## === cell 16
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1910609265.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# summarize history for accuracy[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mplt[0m[0;34m.[0m[0mplot[0m[0;34m([0m[0mhist[0m[0;34m.[0m[0mhistory[0m[0;34m[[0m[0;34m'acc'[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mplt[0m[0;34m.[0m[0mplot[0m[0;34m([0m[0mhist[0m[0;34m.[0m[0mhistory[0m[0;34m[[0m[0;34m'val_acc'[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mplt[0m[0;34m.[0m[0mtitle[0m[0;34m([0m[0;34m'model accuracy'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mplt[0m[0;34m.[0m[0mylabel[0m[0;34m([0m[0;34m'accuracy'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'hist' is not defined

## === cell 17
probability = model.predict_proba(test_img_array/255)
res = pd.DataFrame({
    'id': test_img_index,
    'has_cactus': probability.ravel(),
})
res.to_csv('submission.csv', index=False)
