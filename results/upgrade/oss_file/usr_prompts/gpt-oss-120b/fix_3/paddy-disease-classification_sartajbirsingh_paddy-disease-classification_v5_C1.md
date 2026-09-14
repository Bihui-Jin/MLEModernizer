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

0.80015

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.13797) has done: 'The fix updates the imports to use `tensorflow.keras` (removing the broken legacy Keras imports), corrects the train‑validation split to stratify on the integer labels, and aligns dataset creation with the proper variables. Minor tweaks to the training call (more epochs) help the model actually learn, and the script now reliably writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.80015) has done: 'The fix adds an environment setting to avoid the protobuf import error, builds the label dictionary directly from the training CSV (ensuring only the 10 real classes are mapped), creates a reverse mapping for converting predictions back to class names, and adjusts the data‑split and one‑hot encoding steps accordingly. These changes resolve the import crash, the out‑of‑bounds label index, and the subsequent variable‑name errors, allowing the notebook to run end‑to‑end and write a correctly formatted `submission.csv` while preserving the original model architecture.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("../input/paddy-disease-classification/train.csv")
train_df.head()



## === cell 2
train_images_path = "../input/paddy-disease-classification/train_images/"
all_images = {}
for category in os.listdir(train_images_path):
    category_path = os.path.join(train_images_path, category)
    if not os.path.isdir(category_path):
        continue
    for img in os.listdir(category_path):
        all_images[img] = os.path.join(category_path, img)



## === cell 3
label_names = sorted(train_df["label"].unique())
label_dict = {name: idx for idx, name in enumerate(label_names)}
inv_label_dict = {idx: name for name, idx in label_dict.items()}




## === cell 4
def get_label(label_str):
    return label_dict[label_str]


def get_name(label_idx):
    return inv_label_dict[label_idx]




## === cell 5
img_size = 128



## === cell 6
training = []
for i in range(len(train_df)):
    img_name = train_df.iloc[i, 0]
    img_path = all_images.get(img_name, None)
    if img_path is None:
        continue
    img_array = cv2.imread(img_path)
    if img_array is None:
        continue
    img_resized = cv2.resize(img_array, (img_size, img_size))
    label = get_label(train_df.iloc[i, 1])
    training.append([img_resized, label])

random.shuffle(training)



## === cell 7
X = np.array([item[0] for item in training], dtype=np.float32).reshape(
    -1, img_size, img_size, 3
)
y = np.array([item[1] for item in training], dtype=np.int32)



## === cell 8
X = X / 255.0
Y = tf.keras.utils.to_categorical(y, num_classes=10)



## === cell 9
X_train, X_valid, y_train, y_valid = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=y
)



## === cell 10
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



## === cell 11
model.summary()



## === cell 12
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 13
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(128)
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(128)



## === cell 14
history = model.fit(train_ds, epochs=30, validation_data=valid_ds)



## === cell 15
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



## === cell 16
model.evaluate(X_valid, y_valid)



## === cell 17
submission_df = pd.read_csv(
    "../input/paddy-disease-classification/sample_submission.csv"
)



## === cell 18
test_path = "../input/paddy-disease-classification/test_images/"
test_images = []
for i in range(len(submission_df)):
    img_path = os.path.join(test_path, submission_df.iloc[i, 0])
    img_array = cv2.imread(img_path)
    if img_array is None:
        img_array = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img_resized = cv2.resize(img_array, (img_size, img_size))
    test_images.append(img_resized)



## === cell 19
X_test = np.array(test_images, dtype=np.float32).reshape(-1, img_size, img_size, 3)
X_test = X_test / 255.0



## === cell 20
y_pred = model.predict(X_test)



## === cell 21
labels = np.argmax(y_pred, axis=1)
submission_df["label"] = [get_name(l) for l in labels]



## === cell 22
submission_df.to_csv("submission.csv", index=False)



## === cell 23
model.save("paddy_classification.h5")
