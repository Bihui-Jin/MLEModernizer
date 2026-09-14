# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.87157

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os



## === cell 1
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _major(ver):
    try:
        return int(str(ver).split(".", 1)[0])
    except Exception:
        return None


if _pb_ver is None or (_major(_pb_ver) is not None and _major(_pb_ver) >= 5):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    import importlib
    import google.protobuf

    importlib.reload(google.protobuf)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt


## === cell 2
train_csv = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
train_csv.head()


## === cell 3
test_csv = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
test_csv.head()


## === cell 4
image = plt.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_0.jpg')
plt.imshow(image)
print(image.shape)


## === cell 5
def image_resize(img, size = (None, None), ratio=3):
    if size[0] is None:
        resize_ratio = ratio
        resize_height = int(img.shape[0]/resize_ratio)
        resize_width = int(img.shape[1]/resize_ratio)
        print(f"height: {resize_height}, width: {resize_width}")
    else:
        resize_height = size[0]
        resize_width = size[1]

    img_resize = tf.image.resize(img, [resize_height,resize_width]).numpy()
    img_resize = img_resize.astype(np.uint8)
    return(img_resize)


## === cell 6
plt.figure(1, figsize=(10,10))
plt.subplot(221)
plt.imshow(image_resize(image, ratio = 3))

plt.subplot(222)
plt.imshow(image_resize(image, ratio = 4))
plt.show()

plt.subplot(223)
plt.imshow(image_resize(image, ratio = 5))

plt.subplot(224)
plt.imshow(image_resize(image, ratio = 6))
plt.show()


## === cell 7
img_height = 227
img_width = 341
plt.imshow(image_resize(image, size=(img_height, img_width)))


## === cell 8
train_resized = []

for img_id in train_csv['image_id'].to_list():
    image = plt.imread(f'/kaggle/input/plant-pathology-2020-fgvc7/images/{img_id}.jpg')
    train_resized.append(image_resize(image, (img_height, img_width)))

print(len(train_resized))

test_resized = []

for img_id in test_csv['image_id'].to_list():
    image = plt.imread(f'/kaggle/input/plant-pathology-2020-fgvc7/images/{img_id}.jpg')
    test_resized.append(image_resize(image, (img_height, img_width)))

print(len(test_resized))


## === cell 9
x_train = np.ndarray(shape = (len(train_resized), img_height, img_width, 3), dtype=np.float32)
x_test = np.ndarray(shape = (len(test_resized), img_height, img_width, 3), dtype=np.float32)

for i in range(len(train_resized)):
    x_train[i] = img_to_array(train_resized[i])

for i in range(len(test_resized)):
    x_test[i] = img_to_array(test_resized[i])

x_train = x_train/255
x_test = x_test/255

print(x_train.shape)
print(x_test.shape)


## === cell 10
y_train = train_csv.iloc[:,1:]
y_train.head()


## === cell 11
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.applications import InceptionResNetV2

resnet = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')

model = Sequential()
model.add(resnet)
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.layers[0].trainable = False

model.summary()


## === cell 12
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics='accuracy')


## === cell 13
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size = 0.2)

print(train_x.shape)
print(train_y.shape)

print(val_x.shape)
print(val_y.shape)


## === cell 14
datagen = ImageDataGenerator(rotation_range=25,
                             shear_range=.20,
                             zoom_range=.20,
                             width_shift_range=.20,
                             height_shift_range=.20,
                             horizontal_flip=True,
                             vertical_flip=True
                            )


batch_size = 24
datagen_without_aug = ImageDataGenerator()

train_datagen = datagen_without_aug.flow(train_x, train_y, batch_size=batch_size)

val_datagen = datagen_without_aug.flow(val_x, val_y, batch_size=batch_size)


## === cell 15
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = model.fit_generator(train_datagen, 
                              epochs=6,
                              steps_per_epoch=train_x.shape[0]//batch_size,
                              validation_data = val_datagen,
                              validation_steps = val_x.shape[0]//batch_size,
                              callbacks = [callback]
                   )


## === cell 16
history_df = pd.DataFrame(history.history)
history_df.head()


## === cell 17
plt.plot(history_df.index, history_df['accuracy'])
plt.plot(history_df.index, history_df['val_accuracy'])
plt.show()


## === cell 18
plt.plot(history_df.index, history_df['loss'])
plt.plot(history_df.index, history_df['val_loss'])
plt.show()


## === cell 19
y_preds = model.predict(x_test)


## === cell 20
y_preds


## === cell 21
res = pd.DataFrame()
res['image_id'] = test_csv['image_id']
res['healthy'] = y_preds[:, 0]
res['multiple_diseases'] = y_preds[:, 1]
res['rust'] = y_preds[:, 2]
res['scab'] = y_preds[:, 3]
res.to_csv('submission.csv', index=False)
