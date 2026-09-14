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

# 5. Target score

0.89976

# 6. Current score

0.13797

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.13797) has done: 'The fix updates the imports to use `tensorflow.keras` (removing the broken legacy Keras imports), corrects the train‑validation split to stratify on the integer labels, and aligns dataset creation with the proper variables. Minor tweaks to the training call (more epochs) help the model actually learn, and the script now reliably writes a correctly‑formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import random
import matplotlib.pyplot as plt



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("../input/paddy-disease-classification/train.csv")
train_df.head()



## === cell 2
all_images = {}
train_images_path = "../input/paddy-disease-classification/train_images/"
for category in os.listdir(train_images_path):
    for img in os.listdir(os.path.join(train_images_path, category)):
        all_images[img] = os.path.join(train_images_path, category, img)



## === cell 3
categories = os.listdir(train_images_path)



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
img_size = 128



## === cell 9
training = []
for i in range(len(train_df)):
    img_path = all_images[train_df.iloc[i, 0]]
    img_array = cv2.imread(img_path)
    if img_array is None:
        continue
    new_array = cv2.resize(img_array, (img_size, img_size))
    label = get_label(train_df.iloc[i, 1])
    training.append([new_array, label])

random.shuffle(training)



## === cell 10
X = []
y = []
for features, label in training:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)



## === cell 11
X = X.astype("float32")
X /= 255.0
Y = tf.keras.utils.to_categorical(y, num_classes=10)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/2053037193.py in <cell line: 0>()
      1 X = X.astype("float32")
      2 X /= 255.0
----> 3 Y = tf.keras.utils.to_categorical(y, num_classes=10)
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/numerical_utils.py in to_categorical(x, num_classes)
     97     batch_size = x.shape[0]
     98     categorical = np.zeros((batch_size, num_classes))
---> 99     categorical[np.arange(batch_size), x] = 1
    100     output_shape = input_shape + (num_classes,)
    101     categorical = np.reshape(categorical, output_shape)

IndexError: index 10 is out of bounds for axis 1 with size 10

## === cell 12
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=y
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/157339037.py in <cell line: 0>()
      2 
      3 X_train, X_valid, y_train, y_valid = train_test_split(
----> 4     X, Y, test_size=0.2, random_state=42, stratify=y
      5 )
      6 

NameError: name 'Y' is not defined

## === cell 13
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    InputLayer,
)



## === cell 14
model = tf.keras.Sequential(
    [
        InputLayer(input_shape=(img_size, img_size, 3)),
        Conv2D(16, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(8192, activation="relu"),
        Dense(1024, activation="relu"),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)



## === cell 15
model.summary()



## === cell 16
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 17
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/658069914.py in <cell line: 0>()
----> 1 train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
      2 valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
      3 

NameError: name 'X_train' is not defined

## === cell 18
history = model.fit(train_ds.batch(128), epochs=50, validation_data=valid_ds.batch(128))



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3267512009.py in <cell line: 0>()
----> 1 history = model.fit(train_ds.batch(128), epochs=50, validation_data=valid_ds.batch(128))
      2 

NameError: name 'train_ds' is not defined

## === cell 19
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2607293666.py in <cell line: 0>()
----> 1 plt.plot(history.history["accuracy"], label="Training")
      2 plt.plot(history.history["val_accuracy"], label="Validation")
      3 plt.xlabel("Epochs")
      4 plt.ylabel("Accuracy")
      5 plt.legend()

NameError: name 'history' is not defined

## === cell 20
model.evaluate(X_valid, y_valid)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/400878630.py in <cell line: 0>()
----> 1 model.evaluate(X_valid, y_valid)
      2 

NameError: name 'X_valid' is not defined

## === cell 21
submission_df = pd.read_csv(
    "../input/paddy-disease-classification/sample_submission.csv"
)



## === cell 22
test_path = "../input/paddy-disease-classification/test_images/"
test_images = []
for i in range(len(submission_df)):
    img_path = os.path.join(test_path, submission_df.iloc[i, 0])
    img_array = cv2.imread(img_path)
    if img_array is None:
        img_array = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    new_array = cv2.resize(img_array, (img_size, img_size))
    test_images.append(new_array)



## === cell 23
X_test = np.array(test_images).reshape(-1, img_size, img_size, 3)
X_test = X_test.astype("float32")
X_test /= 255.0



## === cell 24
y_pred = model.predict(X_test)



## === cell 25
labels = [np.argmax(p) for p in y_pred]
submission_df["label"] = [get_name(l) for l in labels]



## === cell 26
submission_df.to_csv("submission.csv", index=False)



## === cell 27
model.save("paddy_classification.h5")
