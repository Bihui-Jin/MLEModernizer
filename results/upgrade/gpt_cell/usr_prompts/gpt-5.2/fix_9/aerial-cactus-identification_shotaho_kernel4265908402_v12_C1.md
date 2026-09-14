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
sklearn-pandas==2.2.0
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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("./"))



## === cell 1
class ImageDataGenerator_self(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.images = []
        self.labels = []

    def flow_from_directory(self, directory, classes, batch_size=32):
        classes = {v: i for i, v in enumerate(sorted(classes))}
        while True:
            for path in pathlib.Path(directory).iterdir():
                with Image.open(path) as f:
                    self.images.append(np.asarray(f.convert('RGB'), dtype=np.float32))
                _, y = path.stem.split('_')
                self.labels.append(to_categorical(classes[y], len(classes)))

                if len(self.images) == batch_size:
                    inputs = np.asarray(self.images, dtype=np.float32)
                    targets = np.asarray(self.labels, dtype=np.float32)
                    self.reset()
                    yield inputs, targets

    def flow_from_dir2(self, data_dir, data_list, label_train, classes, batch_size=32):
        label_train = pd.read_csv(label_csv,index_col=0)
        while True:
            for img_path in data_list:
                img = cv2.imread(data_dir + "/" + img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                self.images.append(img)

                y = label_train.loc[img_path,"has_cactus"]

                self.labels.append(keras.utils.to_categorical(classes[y], len(classes)))

                if len(self.images) == batch_size:
                    inputs = np.asarray(self.images, dtype=np.float32)
                    targets = np.asarray(self.labels, dtype=np.float32)
                    self.reset()
                    yield inputs, targets


## === cell 2
import csv, os
import numpy as np
import pandas as pd


train_order = os.listdir("../input/train/train")
pd_train_order = pd.DataFrame(train_order, columns=["id"])
label_train = pd.read_csv("../input/train.csv")
df_train_label = pd.merge(pd_train_order, label_train)


## === cell 3
import random
data_dir="../input/train/train"
data_list=os.listdir(data_dir)
label_csv="../input/train.csv"
classes=["0","1"]
batch_size = 10
val_ratio = 0.2

train_list = []
val_list = []

for data in data_list:
    if random.random() > val_ratio:
        train_list.append(data)
    else:
        val_list.append(data)
print(len(train_list))
print(len(val_list))

train_datagen=ImageDataGenerator_self()
test_datagen=ImageDataGenerator_self()


## === cell 4
import os as _os

_os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
_os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys as _sys

for _m in list(_sys.modules.keys()):
    if _m.startswith(("google.protobuf", "tensorflow", "tf_keras")):
        _sys.modules.pop(_m, None)

from matplotlib import pyplot as plt
from tf_keras.applications.vgg16 import VGG16
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.models import Sequential, Model
from tf_keras.layers import (
    Input,
    Activation,
    Dropout,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
)
from tf_keras import optimizers
from tf_keras.utils import plot_model
from tf_keras.callbacks import ModelCheckpoint
import cv2

img_height, img_width, img_channel = cv2.imread(data_dir + "/" + data_list[0]).shape
base_model = VGG16(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(img_width, img_height, img_channel),
)

n_categories = 2

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
prediction = Dense(n_categories, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=prediction)

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
os.mkdir("/output")
fpath = '/output/weights.{epoch:03d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
modelCheckpoint = ModelCheckpoint(filepath = fpath,
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='min',
                                  period=1)

model.fit_generator(
    generator=train_datagen.flow_from_dir2(data_dir=data_dir, data_list=train_list, label_train=label_train, classes=classes,batch_size=batch_size),
    steps_per_epoch=int(len(data_list) / batch_size),
    epochs=1,
    verbose=2,
    validation_data=test_datagen.flow_from_dir2(data_dir=data_dir, data_list=val_list, label_train=label_train, classes=classes,batch_size=batch_size),
    validation_steps=int(len(val_list) / batch_size),
    callbacks=[modelCheckpoint]
    )
