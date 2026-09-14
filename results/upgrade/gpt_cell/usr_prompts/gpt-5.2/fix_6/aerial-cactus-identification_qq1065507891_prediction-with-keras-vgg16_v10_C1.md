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
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def to_categorical(y, num_classes=None, dtype="float32"):
    try:
        import tensorflow as tf  # defer TF import; use public API if it works

        return tf.keras.utils.to_categorical(y, num_classes=num_classes, dtype=dtype)
    except Exception:
        y = np.array(y, dtype="int64").ravel()
        if num_classes is None:
            num_classes = int(np.max(y)) + 1 if y.size else 0
        out = np.zeros((y.shape[0], num_classes), dtype=dtype)
        if y.size and num_classes:
            out[np.arange(y.shape[0]), y] = 1
        return out


print(os.listdir("../input"))


## === cell 1
dir = "../input/train/train"
def process_picture():
    image_files = []
    data = pd.read_csv('../input/train.csv')
    images = data['id'].values
    labels = []
    for files in images:
        labels.append(data[data['id'] == files]['has_cactus'].values[0])
        files = os.path.join(dir, files)
        image_files.append(files)
    return image_files, labels


def get_images_lables():
    images_files, labels = process_picture()
    images = []
    
    for index, file in enumerate(images_files):
        image = cv2.imread(file)
        images.append(image)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels,
                                                                           test_size=0.2, random_state=7,
                                                                           shuffle=True)
    train_images = np.array(train_images) / 255
    test_images = np.array(test_images) / 255

    print(train_images.shape)
    
    return train_images, test_images, train_labels, test_labels


## === cell 2

train_images, test_images, train_labels, test_labels = get_images_lables()
class_weight = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)


def _to_categorical_np(y, num_classes=2, dtype="float32"):
    y = np.asarray(y, dtype="int64").ravel()
    out = np.zeros((y.shape[0], num_classes), dtype=dtype)
    if y.size:
        out[np.arange(y.shape[0]), y] = 1
    return out


train_labels = _to_categorical_np(train_labels, 2)
test_labels = _to_categorical_np(test_labels, 2)
print(class_weight)


## === cell 3
import os as _os

_os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
_os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
def bulid_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(BatchNormalization())
    add_model.add(Dense(256, activation='relu', name='FC1'))
    add_model.add(BatchNormalization())
    add_model.add(Dropout(0.5))
    add_model.add(Dense(128, activation='relu', name='FC2'))
    add_model.add(BatchNormalization())
    add_model.add(Dense(2, activation='softmax', name='softmax'))
    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.summary()
    for layer in model.layers:
        layer.trainable = False
    model.trainable = True
    for layer in model.layers:
        trainable = ('block5' in layer.name or 'block4' in layer.name)
        layer.trainable = trainable
    return model


def train(batch_size=64, nb_epoch=500):
    model = bulid_model()
    optimizer = Adam(1e-5)

    model.compile(optimizer=optimizer, metrics=['accuracy'], loss='categorical_crossentropy')

    callback=[callbacks.EarlyStopping(monitor='val_acc', patience=20, mode='auto', restore_best_weights=True),
         callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')]

    train_datagen = ImageDataGenerator(rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=[0.9, 1.5],
                                       vertical_flip=True,
                                       horizontal_flip=True)
    train_datagen.fit(train_images)
    history = model.fit_generator(train_datagen.flow(train_images, train_labels, batch_size=batch_size),
                                  steps_per_epoch=train_images.shape[0] // batch_size,
                                  epochs=nb_epoch,
                                  validation_data=(test_images, test_labels),
                                  class_weight=class_weight,
                                  callbacks=callback)
    score = model.evaluate(test_images, test_labels)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    model.save('./test.h5')
    return model, history
