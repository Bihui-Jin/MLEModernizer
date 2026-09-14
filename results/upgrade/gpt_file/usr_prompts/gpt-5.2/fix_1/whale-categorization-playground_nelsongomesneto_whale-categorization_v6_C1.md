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
Predict the individual whale species in images.

## Metric
Mean Average Precision @ 5 (MAP@5).

## Submission Format
For each `Image` in the test set, you may predict up to 5 labels for the whale `Id`. Whales that are not predicted to be one of the labels in the training data should be labeled as `new_whale`. The file should contain a header and have the following format:

```
Image,Id
00029b3a.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
0003c693.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
...
```

## Dataset
This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an `Id`. The challenge is to predict the whale `Id` of images in the test set. What makes this such a challenge is that there are only a few examples for each of 3,000+ whale Ids.

- **train.zip** - a folder containing the training images
- **train.csv** - maps the training `Image` to the appropriate whale `Id`. Whales that are not predicted to have a label identified in the training data should be labeled as `new_whale`.
- **test.zip** - a folder containing the test images to predict the whale `Id`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        input/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        working/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
```

-> data/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> input/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> input/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> (stopped after 10 files for performance)

# 5. Target score

0.00128

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import os
import time
from heapq import heappush, heappop
import tensorflow as tf
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plot
import matplotlib.image as mpimage
import seaborn as sn
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
folder = "../input/whale-categorization-playground/"


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1

idDict = {}
train = pd.read_csv(folder + "train.csv")
for i in train.iterrows():
    if not train.loc[i[0]][1] in idDict:
        idDict[train.loc[i[0]][1]] = len(idDict)
    train.loc[i[0]][0] = folder + "train/train/" + train.loc[i[0]][0]
    train.loc[i[0]][1] = idDict[train.loc[i[0]][1]]
test = os.listdir(folder + "test/test")
for i in range(len(test)):
    test[i] = folder + "test/test/" + test[i]

x_train, y_train = train.iloc[:, 0], train.iloc[:, 1]

width, height, batchSize, iterations = 150, 150, 10000, 1


## === cell 2

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), padding="Same", activation="relu", input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
   
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="Same", activation="relu", input_shape=(75, 75, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Dropout(0.2),
    
        tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding="Same", activation="relu", input_shape=(25, 25, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(5, 5)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4251, activation="softmax")
    ])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


## === cell 3

for k in range(iterations):
    indexes = list(range(len(x_train)))
    shuffle(indexes)
    i, iterationStartTime = 0, time.time()
    while i < len(indexes):
        batchStartTime = time.time()
        x, y = [], []
        for j in range(i, min(len(indexes), i + batchSize)):
            image = tf.keras.preprocessing.image.load_img(x_train[indexes[j]], target_size=(width, height))
            image = tf.keras.preprocessing.image.img_to_array(image)
            x += [image]
            ans = np.zeros(4251)
            ans[y_train[indexes[j]]] = 1
            y += [ans]
        x, y = np.array(x), np.array(y)
        model.fit(x, y, epochs=5, verbose=True, shuffle=True)
        i += batchSize
        print("\tbatch: %Lg%% - %Lg seconds" % (100 * i / len(indexes), time.time() - batchStartTime))
    print("iteration: %Lg%% - %Lg seconds" % (100 * (k+1) / 100, time.time() - iterationStartTime))


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2616171250.py in <cell line: 0>()
      9         x, y = [], []
     10         for j in range(i, min(len(indexes), i + batchSize)):
---> 11             image = tf.keras.preprocessing.image.load_img(x_train[indexes[j]], target_size=(width, height))
     12             image = tf.keras.preprocessing.image.img_to_array(image)
     13             x += [image]

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

FileNotFoundError: [Errno 2] No such file or directory: '../input/whale-categorization-playground/train/train/ccc570bb.jpg'

## === cell 4

y_final = []
pos = 0
for i in test:
    image = tf.keras.preprocessing.image.load_img(i, target_size=(width, height))
    image = tf.keras.preprocessing.image.img_to_array(image)
    x = np.array([image])
    y = model.predict(x)
    ymap, at = [], 0
    for j in sorted(idDict, key=lambda x: x[1]):
        heappush(ymap, [y[0][at], j])
        if len(ymap) > 5:
            heappop(ymap)
        at += 1
    now = []
    for j in range(5):
        now += [heappop(ymap)[1]]
    y_final += [now]
    if (pos % 1000 == 0):
        print(100 * pos / len(test))
    pos += 1


## === cell 5

imageId, y_sub = [], []
for i in range(len(test)):
    imageId += [test[i].split('/')[-1]]
    y_sub += [' '.join(y_final[i])]
submission = pd.DataFrame({"Image": imageId, "Id": y_sub})
submission.to_csv("submission.csv", index=False)


## === cell 6
submission


## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
