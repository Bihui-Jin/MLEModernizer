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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.13602

# 6. Current score

0.15616

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.15616) has done: 'Diagnosis: The crash happens while importing `tflearn`, inside `tflearn/data_utils.py`, because it references `PIL.Image.ANTIALIAS`, which was removed in newer Pillow versions (replaced by `Image.Resampling.LANCZOS`). Since this happens at import time, the model code never runs.  
Patch summary: Add a tiny compatibility shim in cell 8 that defines `PIL.Image.ANTIALIAS` (and related legacy constants) when missing, mapping them to the modern `Image.Resampling.*` equivalents before importing `tflearn`. This keeps tflearn’s core logic unchanged and only prevents the import-time AttributeError.  
Updated cells: Only cell 8 is modified.  
Compatibility notes for cell k+1: The `model` variable is still created exactly as before (same name/type), so `model.save(MODEL_NAME)` in the next cell work unchanged.  
Assumptions: Pillow is installed (it is, since `PIL.Image` is importable) and provides `Image.Resampling` (true for modern Pillow); if not, we fall back to `Image.LANCZOS`.'

# 9. Code solution

## === cell 0
import numpy as np
import os
import cv2  
import pandas as pd
from tqdm import tqdm
from random import shuffle
LR = 1e-3
MODEL_NAME = 'plantclassfication-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which save


## === cell 1
data_dir = '../input/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
IMG_SIZE = 50


## === cell 2
CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
NUM_CATEGORIES = len(CATEGORIES)
print (NUM_CATEGORIES)


## === cell 3
def label_img(word_label):                       
    if word_label == 'Black-grass': return [1,0,0,0,0,0,0,0,0,0,0,0]
    elif word_label == 'Charlock': return [0,1,0,0,0,0,0,0,0,0,0,0]
    elif word_label == 'Cleavers': return [0,0,1,0,0,0,0,0,0,0,0,0]
    elif word_label == 'Common Chickweed': return [0,0,0,1,0,0,0,0,0,0,0,0]
    elif word_label == 'Common wheat': return [0,0,0,0,1,0,0,0,0,0,0,0]
    elif word_label == 'Fat Hen': return [0,0,0,0,0,1,0,0,0,0,0,0]
    elif word_label == 'Loose Silky-bent': return [0,0,0,0,0,0,1,0,0,0,0,0]
    elif word_label == 'Maize': return [0,0,0,0,0,0,0,1,0,0,0,0]
    elif word_label == 'Scentless Mayweed': return [0,0,0,0,0,0,0,0,1,0,0,0]
    elif word_label == 'Shepherds Purse': return [0,0,0,0,0,0,0,0,0,1,0,0]
    elif word_label == 'Small-flowered Cranesbill': return [0,0,0,0,0,0,0,0,0,0,1,0]
    elif word_label == 'Sugar beet': return [0,0,0,0,0,0,0,0,0,0,0,1] 


## === cell 4
def create_train_data():
    train = []
    for category_id, category in enumerate(CATEGORIES):
        for img in tqdm(os.listdir(os.path.join(train_dir, category))):
            label=label_img(category)
            path=os.path.join(train_dir,category,img)
            img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            train.append([np.array(img),np.array(label)])
    shuffle(train)
    return train


## === cell 5
train_data = create_train_data()


## === cell 6
def create_test_data():
    test = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir,img)
        img_num = img
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        test.append([np.array(img), img_num])
        
    shuffle(test)
    return test


## === cell 7
def create_test_data():
    test = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir, img)
        if not os.path.isfile(path):
            continue  # skip directories such as a nested "test" folder
        img_num = img
        img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_arr is None:
            continue  # skip unreadable/non-image files to avoid OpenCV resize assertion
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        test.append([np.array(img_arr), img_num])

    shuffle(test)
    return test


test_data = create_test_data()


## === cell 8
import os

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<4"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tflearn"])

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from PIL import Image

    if not hasattr(Image, "ANTIALIAS"):
        _resampling = getattr(Image, "Resampling", None)
        if _resampling is not None:
            Image.ANTIALIAS = _resampling.LANCZOS
            Image.BILINEAR = _resampling.BILINEAR
            Image.BICUBIC = _resampling.BICUBIC
            Image.NEAREST = _resampling.NEAREST
        else:
            Image.ANTIALIAS = getattr(Image, "LANCZOS", 1)
except Exception:
    pass

import tensorflow as tf

try:
    from tensorflow.python.util import (
        nest as _tf_nest,
    )  # internal module used by tflearn

    if not hasattr(_tf_nest, "is_sequence"):
        _tf_nest.is_sequence = getattr(_tf_nest, "is_nested", None) or tf.nest.is_nested
except Exception:
    pass

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

tf.compat.v1.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name="input")

convnet = conv_2d(convnet, 32, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation="relu")
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation="relu")
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 12, activation="softmax")
convnet = regression(
    convnet,
    optimizer="adam",
    learning_rate=LR,
    loss="categorical_crossentropy",
    name="targets",
)

model = tflearn.DNN(convnet, tensorboard_dir="log")

if os.path.exists("{}.meta".format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print("model loaded!")

train = train_data[:-2200]
test = train_data[-2200:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit(
    {"input": X},
    {"targets": Y},
    n_epoch=5,
    validation_set=({"input": test_x}, {"targets": test_y}),
    snapshot_step=500,
    show_metric=True,
    run_id=MODEL_NAME,
)


## === cell 9
model.save(MODEL_NAME)


## === cell 10
import matplotlib.pyplot as plt

test_data = create_test_data()
fig=plt.figure()
for num,data in enumerate(test_data[:12]): 
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 0: str_label='Black-grass'
    elif np.argmax(model_out) == 1: str_label='Charlock'
    elif np.argmax(model_out) == 2: str_label='Cleavers'
    elif np.argmax(model_out) == 3: str_label='Common Chickweed'
    elif np.argmax(model_out) == 4: str_label='Common wheat'
    elif np.argmax(model_out) == 5: str_label='Fat Hen'
    elif np.argmax(model_out) == 6: str_label='Loose Silky-bent'
    elif np.argmax(model_out) == 7: str_label='Maize'
    elif np.argmax(model_out) == 8: str_label='Scentless Mayweed'
    elif np.argmax(model_out) == 9: str_label='Shepherds Purse'
    elif np.argmax(model_out) == 10: str_label='Small-flowered Cranesbill'
    elif np.argmax(model_out) == 11: str_label='Sugar beet'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


## === cell 11
import pandas as pd
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head(2)


## === cell 12
test_data = create_test_data()
with open('sample_submission.csv','w') as f:
    f.write('file,species\n')
    for data in test_data:
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 0: str_label='Black-grass'
        elif np.argmax(model_out) == 1: str_label='Charlock'
        elif np.argmax(model_out) == 2: str_label='Cleavers'
        elif np.argmax(model_out) == 3: str_label='Common Chickweed'
        elif np.argmax(model_out) == 4: str_label='Common wheat'
        elif np.argmax(model_out) == 5: str_label='Fat Hen'
        elif np.argmax(model_out) == 6: str_label='Loose Silky-bent'
        elif np.argmax(model_out) == 7: str_label='Maize'
        elif np.argmax(model_out) == 8: str_label='Scentless Mayweed'
        elif np.argmax(model_out) == 9: str_label='Shepherds Purse'
        elif np.argmax(model_out) == 10: str_label='Small-flowered Cranesbill'
        elif np.argmax(model_out) == 11: str_label='Sugar beet'
        file = img_num
        species = str_label
        row = file + "," + species + "\n"
        f.write(row)


## === cell 13
import pandas as pd
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission.head(100)
