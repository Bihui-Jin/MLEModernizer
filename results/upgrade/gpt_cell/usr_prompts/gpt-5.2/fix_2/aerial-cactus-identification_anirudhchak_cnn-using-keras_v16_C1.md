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

0.9979

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from packaging import version
    import google.protobuf as _protobuf

    _pb_ver = getattr(_protobuf, "__version__", "0")
    if version.parse(_pb_ver) >= version.parse("5"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception:
    pass

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import cv2
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from keras.optimizers import Adam
from keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)

print(os.listdir("../input"))

seed = 4529
np.random.seed(seed)


## === cell 1
base_dir = os.path.join("..", "input") # set base directory
train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
train_dir = os.path.join(base_dir, "train/train")
test_dir = os.path.join(base_dir, "test/test")

print(train_df.head())


## === cell 2
%load_ext tensorboard.notebook
%tensorboard --logdir logs


## === cell 4
train_df['has_cactus'] = train_df['has_cactus'].astype(str)

batch_size = 64
train_size = 15750
validation_size = 1750

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.1)

data_args = {
    "dataframe": train_df,
    "directory": train_dir,
    "x_col": 'id',
    "y_col": 'has_cactus',
    "shuffle": True,
    "target_size": (32, 32),
    "batch_size": batch_size,
    "class_mode": 'binary'
}

train_generator = datagen.flow_from_dataframe(**data_args, subset='training')
validation_generator = datagen.flow_from_dataframe(**data_args, subset='validation')


## === cell 5
model = Sequential([
    Conv2D(64, (3,3), padding='same', activation="relu", input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(64, (3,3), padding='same', activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),
    Conv2D(128, (3,3), padding='same', activation="relu"),
    BatchNormalization(),
    Conv2D(128, (3,3), padding='same', activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),
    Conv2D(256, (3,3), padding='same', activation="relu"),
    BatchNormalization(),
    Conv2D(256, (3,3), padding='same', activation="relu"),
    BatchNormalization(),
    Conv2D(256, (3,3), padding='same', activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),
    Conv2D(256, (3,3), padding='same', activation="relu"),
    BatchNormalization(),
    Conv2D(256, (3,3), padding='same', activation="relu"),
    BatchNormalization(),
    Conv2D(256, (3,3), padding='same', activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),
    GlobalAveragePooling2D(),
    Dense(units=256, activation='relu'),
    Dropout(0.5),
    Dense(units=256, activation='relu'),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=Adam(lr=0.001), 
                 loss='binary_crossentropy',
                 metrics=['acc'])
model.summary()


## === cell 6
ckpt_path = 'aerial_cactus_detection.hdf5'

earlystop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, restore_best_weights=False)
reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
modelckpt_cb = ModelCheckpoint(ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tb = TensorBoard()

callbacks = [earlystop, reducelr, modelckpt_cb, tb]


## === cell 7
history = model.fit_generator(train_generator,
              validation_data=validation_generator,
              steps_per_epoch=train_size//batch_size,
              validation_steps=validation_size//batch_size,
              epochs=30, verbose=1, 
              shuffle=True,
              callbacks=callbacks)


## === cell 8
epochs = [i for i in range(1, len(history.history['loss'])+1)]

plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")
plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")
plt.legend(loc='best')
plt.title('loss')
plt.xlabel('epoch')
plt.show()

plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")
plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")
plt.legend(loc='best')
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()


## === cell 9
test_df = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
print(test_df.head())
test_images = []
images = test_df['id'].values

for image_id in images:
    test_images.append(cv2.imread(os.path.join(test_dir, image_id)))
    
test_images = np.asarray(test_images)
test_images = test_images / 255.0
print("Number of Test set images: " + str(len(test_images)))


## === cell 10
pred = model.predict(test_images)
test_df['has_cactus'] = pred
test_df.to_csv('aerial-cactus-submission.csv', index = False)
