# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

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
pillow==11.3.0
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
)

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

try:
    import IPython

    IPython.Application.instance().kernel.do_shutdown(True)
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

tf.get_logger().setLevel("ERROR")

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
except Exception:
    pass

tf.config.optimizer.set_jit(True)
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

tf.config.experimental.enable_op_determinism()



## === cell 1
train_df = pd.read_csv("../input/paddy-disease-classification/train.csv")
train_df.head()



## === cell 2
train_images_path = "../input/paddy-disease-classification/train_images/"
all_images = {
    row.image_id: os.path.join(train_images_path, row.label, row.image_id)
    for row in train_df.itertuples(index=False)
}



## === cell 3
categories = sorted(os.listdir(train_images_path))



## === cell 4
label_dict = {category: idx for idx, category in enumerate(categories)}
label_dict




## === cell 5
def get_label(label):
    return label_dict[label]




## === cell 6
num_label = {
    0: "tungro",
    1: "hispa",
    2: "downy_mildew",
    3: "bacterial_leaf_streak",
    4: "bacterial_leaf_blight",
    5: "brown_spot",
    6: "blast",
    7: "normal",
    8: "dead_heart",
    9: "bacterial_panicle_blight",
}




## === cell 7
def get_name(x):
    return num_label[x]




## === cell 8
num_label



## === cell 9
img_size = 128



## === cell 10
from concurrent.futures import ThreadPoolExecutor


def _load_resize(path_size):
    path, size = path_size
    img_array = cv2.imread(path)  # BGR (unchanged)
    if img_array is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.resize(img_array, (size, size), interpolation=cv2.INTER_LINEAR)


n = len(train_df)

X = np.empty((n, img_size, img_size, 3), dtype=np.uint8)

paths = [all_images[img_id] for img_id in train_df["image_id"].to_numpy()]
labels_str = train_df["label"].to_numpy()

label_index = pd.Index(categories)
y = label_index.get_indexer(labels_str).astype(np.int64, copy=False)

sizes = np.full(n, img_size, dtype=np.int32)

max_workers = min(32, (os.cpu_count() or 8))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, arr in enumerate(ex.map(_load_resize, zip(paths, sizes), chunksize=64)):
        X[i] = arr

idx = np.arange(n)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
X = X[idx]
y = y[idx]

training0 = [X[0], y[0]]
training0



## === cell 11
X = X.reshape(-1, img_size, img_size, 3)



## === cell 12
X = X.astype("float32", copy=False)
X /= 255.0

from keras.utils import np_utils

Y = np_utils.to_categorical(y, 10)
print(Y[100])
print(Y.shape)



## === cell 13
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=y
)



## === cell 14
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam



## === cell 15
model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8192, activation="relu"),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)



## === cell 16
model.summary()



## === cell 17
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 18
AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.deterministic = True

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .with_options(options)
    .batch(128, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
)
valid_ds = (
    tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    .with_options(options)
    .batch(128, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
)



## === cell 19
history = model.fit(train_ds, epochs=30, validation_data=valid_ds)



## === cell 20
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Training Data")
plt.plot(history.history["val_accuracy"], label="Validation Data")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.show()



## === cell 21
model.evaluate(X_valid, y_valid)



## === cell 22
submission_df = pd.read_csv(
    "../input/paddy-disease-classification/sample_submission.csv"
)
submission_df



## === cell 23
test_path = "../input/paddy-disease-classification/test_images/"

test_ids = submission_df["image_id"].to_numpy()
m = len(test_ids)
X_test_u8 = np.empty((m, img_size, img_size, 3), dtype=np.uint8)

test_paths = [os.path.join(test_path, img_id) for img_id in test_ids]
test_sizes = np.full(m, img_size, dtype=np.int32)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, arr in enumerate(
        ex.map(_load_resize, zip(test_paths, test_sizes), chunksize=128)
    ):
        X_test_u8[i] = arr

test_images0 = X_test_u8[0]
test_images0



## === cell 24
X_test = X_test_u8.reshape(-1, img_size, img_size, 3)



## === cell 25
X_test[0].shape



## === cell 26
X_test = X_test.astype("float32", copy=False)
X_test /= 255.0



## === cell 27
y_pred = model.predict(X_test, batch_size=128, verbose=0)



## === cell 28
y_pred



## === cell 29
labels = np.argmax(y_pred, axis=1).tolist()
labels



## === cell 30
submission_df["label"] = labels



## === cell 31
submission_df["label"] = submission_df["label"].apply(get_name)



## === cell 32
submission_df



## === cell 33
submission_df.to_csv("submission.csv", index=False)



## === cell 34
model.save("paddy_classification.h5")
