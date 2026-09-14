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

0.06341

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.06341) has done: 'I fix the environment/runtime issues first (protobuf/keras import incompatibilities) so the notebook runs end-to-end. Then I correct label mapping logic so the numeric-to-class conversion matches the actual folder/class order used during training (the current hardcoded `num_label` is very likely wrong, causing the extremely low score). I also replace deprecated Keras utilities (`np_utils`) and legacy layer imports with supported `tf.keras` equivalents, and ensure `train_test_split(stratify=...)` uses 1D class indices. Finally, I keep the same CNN architecture/training loop while making the submission creation deterministic and aligned with the competition format.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import cv2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



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

categories = sorted(
    [
        d
        for d in os.listdir(train_images_path)
        if os.path.isdir(os.path.join(train_images_path, d))
    ]
)

for category in categories:
    cat_dir = os.path.join(train_images_path, category)
    for img in os.listdir(cat_dir):
        all_images[img] = os.path.join(cat_dir, img)

len(all_images), categories[:5]



## === cell 3
label_dict = {cat: i for i, cat in enumerate(categories)}
label_dict




## === cell 4
def get_label(label: str) -> int:
    return label_dict[label]


num_label = {i: cat for cat, i in label_dict.items()}
num_label




## === cell 5
def get_name(x: int) -> str:
    return num_label[int(x)]




## === cell 6
img_size = 128

training = []
missing = 0
for i in range(len(train_df)):
    img_id = train_df.iloc[i, 0]
    label_str = train_df.iloc[i, 1]
    img_path = all_images.get(img_id, None)
    if img_path is None:
        missing += 1
        continue
    img_array = cv2.imread(img_path)
    if img_array is None:
        missing += 1
        continue
    new_array = cv2.resize(
        img_array, (img_size, img_size), interpolation=cv2.INTER_AREA
    )
    label = get_label(label_str)
    training.append([new_array, label])

missing, len(training), training[0][0].shape, training[0][1]



## === cell 7
random.shuffle(training)

X = []
y = []
for features, label in training:
    X.append(features)
    y.append(label)

X = np.array(X, dtype=np.float32).reshape(-1, img_size, img_size, 3)
y = np.array(y, dtype=np.int64)

X /= 255.0
Y = tf.keras.utils.to_categorical(y, num_classes=10)

Y.shape, Y[0]



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/218379483.py in <cell line: 0>()
     11 
     12 X /= 255.0
---> 13 Y = tf.keras.utils.to_categorical(y, num_classes=10)
     14 
     15 Y.shape, Y[0]

/usr/local/lib/python3.11/dist-packages/keras/src/utils/numerical_utils.py in to_categorical(x, num_classes)
     97     batch_size = x.shape[0]
     98     categorical = np.zeros((batch_size, num_classes))
---> 99     categorical[np.arange(batch_size), x] = 1
    100     output_shape = input_shape + (num_classes,)
    101     categorical = np.reshape(categorical, output_shape)

IndexError: index 10 is out of bounds for axis 1 with size 10

## === cell 8
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, Y, test_size=0.2, random_state=SEED, stratify=y
)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1380717720.py in <cell line: 0>()
      3 # stratify must be 1D class indices, not one-hot
      4 X_train, X_valid, y_train, y_valid = train_test_split(
----> 5     X, Y, test_size=0.2, random_state=SEED, stratify=y
      6 )
      7 

NameError: name 'Y' is not defined

## === cell 9
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(img_size, img_size, 3)),
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

model.summary()



## === cell 10
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(
    2048, seed=SEED, reshuffle_each_iteration=True
)
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

history = model.fit(
    train_ds.batch(128).prefetch(tf.data.AUTOTUNE),
    epochs=30,
    validation_data=valid_ds.batch(128).prefetch(tf.data.AUTOTUNE),
    verbose=2,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1777559786.py in <cell line: 0>()
      1 model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
      2 
----> 3 train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(
      4     2048, seed=SEED, reshuffle_each_iteration=True
      5 )

NameError: name 'X_train' is not defined

## === cell 11
val_loss, val_acc = model.evaluate(X_valid, y_valid, verbose=0)
val_loss, val_acc



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3290465486.py in <cell line: 0>()
      1 # Optional: quick sanity check evaluation
----> 2 val_loss, val_acc = model.evaluate(X_valid, y_valid, verbose=0)
      3 val_loss, val_acc
      4 

NameError: name 'X_valid' is not defined

## === cell 12
submission_df = pd.read_csv(
    "../input/paddy-disease-classification/sample_submission.csv"
)
submission_df.head()



## === cell 13
test_path = "../input/paddy-disease-classification/test_images/"

test_images = []
bad_test = 0
for i in range(len(submission_df)):
    img_name = submission_df.iloc[i, 0]
    img_array = cv2.imread(os.path.join(test_path, img_name))
    if img_array is None:
        bad_test += 1
        img_array = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    new_array = cv2.resize(
        img_array, (img_size, img_size), interpolation=cv2.INTER_AREA
    )
    test_images.append(new_array)

bad_test, len(test_images), test_images[0].shape



## === cell 14
X_test = np.array(test_images, dtype=np.float32).reshape(-1, img_size, img_size, 3)
X_test /= 255.0
X_test.shape



## === cell 15
y_pred = model.predict(X_test, batch_size=128, verbose=0)
labels = np.argmax(y_pred, axis=1).astype(int)

submission_df["label"] = [get_name(x) for x in labels]
submission_df.head()



## === cell 16
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

model.save("paddy_classification.h5")

submission_path, submission_df.shape
