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

No external packages required in the script and installed.

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

0.9827188940092166

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory

BASE_PATH = os.path.join("data", "paddy-disease-classification")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUBMISSION = os.path.join(BASE_PATH, "sample_submission.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="training",
)

val_ds = image_dataset_from_directory(
    TRAIN_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    validation_split=0.2,
    subset="validation",
)

class_names = train_ds.class_names
num_classes = len(class_names)

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/690968307.py in <cell line: 0>()
      1 # Load training images using the folder structure (sub‑folders are class names)
----> 2 train_ds = image_dataset_from_directory(
      3     TRAIN_DIR,
      4     label_mode="categorical",
      5     image_size=IMG_SIZE,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_dataset_utils.py in image_dataset_from_directory(directory, labels, label_mode, class_names, color_mode, batch_size, image_size, shuffle, seed, validation_split, subset, interpolation, follow_links, crop_to_aspect_ratio, pad_to_aspect_ratio, data_format, verbose)
    230     if seed is None:
    231         seed = np.random.randint(1e6)
--> 232     image_paths, labels, class_names = dataset_utils.index_directory(
    233         directory,
    234         labels,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py in index_directory(directory, labels, formats, class_names, shuffle, seed, follow_links, verbose)
    528     if labels == "inferred":
    529         subdirs = []
--> 530         for subdir in sorted(tf.io.gfile.listdir(directory)):
    531             if tf.io.gfile.isdir(tf.io.gfile.join(directory, subdir)):
    532                 if not subdir.startswith("."):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/lib/io/file_io.py in list_directory_v2(path)
    766   """
    767   if not is_directory(path):
--> 768     raise errors.NotFoundError(
    769         node_def=None,
    770         op=None,

NotFoundError: Could not find directory data/paddy-disease-classification/train_images

## === cell 2
base_model = EfficientNetB0(
    include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet"
)
base_model.trainable = False  # Freeze the base for quick training

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1392580118.py in <cell line: 0>()
     10 x = layers.GlobalAveragePooling2D()(x)
     11 x = layers.Dropout(0.2)(x)
---> 12 outputs = layers.Dense(num_classes, activation="softmax")(x)
     13 
     14 model = models.Model(inputs, outputs)

NameError: name 'num_classes' is not defined

## === cell 3
EPOCHS = 5
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/441300082.py in <cell line: 0>()
      1 # Train the model (few epochs – enough for a decent baseline)
      2 EPOCHS = 5
----> 3 model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
      4 

NameError: name 'model' is not defined

## === cell 4
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
test_paths = [os.path.join(TEST_DIR, f) for f in test_files]


def preprocess_path(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


test_images = np.stack([preprocess_path(p) for p in test_paths])
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(BATCH_SIZE)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1305724516.py in <cell line: 0>()
      1 # Prepare test dataset (flat folder of images)
----> 2 test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
      3 test_paths = [os.path.join(TEST_DIR, f) for f in test_files]
      4 
      5 

FileNotFoundError: [Errno 2] No such file or directory: 'data/paddy-disease-classification/test_images'

## === cell 5
pred_probs = model.predict(test_dataset, verbose=0)
pred_indices = np.argmax(pred_probs, axis=1)
pred_labels = [class_names[i] for i in pred_indices]

submission = pd.DataFrame({"image_id": test_files, "label": pred_labels})

submission_path = "model_submission_v22.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2384378932.py in <cell line: 0>()
      1 # Predict labels
----> 2 pred_probs = model.predict(test_dataset, verbose=0)
      3 pred_indices = np.argmax(pred_probs, axis=1)
      4 pred_labels = [class_names[i] for i in pred_indices]
      5 

NameError: name 'model' is not defined
