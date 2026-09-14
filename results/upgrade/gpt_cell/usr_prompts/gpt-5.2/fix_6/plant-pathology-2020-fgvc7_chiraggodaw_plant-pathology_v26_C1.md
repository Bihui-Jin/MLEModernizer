# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
from __future__ import absolute_import, division, print_function, unicode_literals

import os, gc, math
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.utils import to_categorical

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
np.random.seed(42)
tf.random.set_seed(42)

gc.collect()



## === cell 1
train_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

file_name = train_df["image_id"].to_numpy()
train_y = train_df[["healthy", "multiple_diseases", "rust", "scab"]].to_numpy(
    dtype=np.int64
)

N = len(file_name)
H, W = 273, 410

train_X = np.empty((N, H, W, 3), dtype=np.uint8)

base_img_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
for idx, img_id in enumerate(file_name):
    img = cv2.imread(base_img_dir + img_id + ".jpg", cv2.IMREAD_COLOR)
    train_X[idx] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

gc.collect()



## === cell 2
try:
    cv2.destroyAllWindows()
except Exception:
    pass

training_y = train_y.argmax(axis=1).astype(np.int64)
gc.collect()



## === cell 3
gc.collect()

training_X = train_X[0:1120]
train_y_split = training_y[0:1120]
test_X = train_X[1120:1821]
test_Y = training_y[1120:1821]



## === cell 4
gc.collect()

train_datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

model = tf.keras.Sequential(
    [
        tf.keras.applications.Xception(
            weights="imagenet", include_top=False, input_shape=(273, 410, 3)
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adamax(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 5
y_binary = to_categorical(training_y)
y_binary_train = to_categorical(train_y_split)
y_binary_test = to_categorical(test_Y)



## === cell 6
gc.collect()
annealer = ReduceLROnPlateau(
    monitor="accuracy", factor=0.5, patience=5, verbose=1, min_lr=1e-5
)
checkpoint = ModelCheckpoint("model.h5", verbose=1, save_best_only=True)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.99:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True




## === cell 7
gc.collect()

batch_size = 16
train_flow = train_datagen.flow(
    training_X, y_binary_train, batch_size=batch_size, shuffle=True
)
steps_per_epoch = math.ceil(len(training_X) / batch_size)

history = model.fit(
    train_flow,
    steps_per_epoch=steps_per_epoch,
    validation_data=(test_X, y_binary_test),
    epochs=200,
    callbacks=[annealer],
)



## === cell 8
acc = history.history["accuracy"]
val_acc = history.history.get("val_accuracy", [])
loss = history.history["loss"]
val_loss = history.history.get("val_loss", [])

epochs = range(len(acc))

plt.plot(epochs, acc, "r", label="Training accuracy")
plt.title("Training accuracy")
plt.legend(loc=0)
plt.figure()
plt.show()



## === cell 9
plt.plot(epochs, loss, "r", label="Training Loss")
plt.title("Training Loss")
plt.legend(loc=0)
plt.figure()
plt.show()



## === cell 10
test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
test_ids = test_df["image_id"].to_numpy()
Nt = len(test_ids)

test = np.empty((Nt, H, W, 3), dtype=np.uint8)
for idx, img_id in enumerate(test_ids):
    img = cv2.imread(base_img_dir + img_id + ".jpg", cv2.IMREAD_COLOR)
    test[idx] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

gc.collect()



## === cell 11
results = model.predict(test, batch_size=32, verbose=0)



## === cell 12
df = pd.DataFrame(results, columns=["healthy", "multiple_diseases", "rust", "scab"])



## === cell 13
df.insert(0, "image_id", test_ids, False)



## === cell 14
df.to_csv("submission.csv", index=False)



## === cell 15
df
