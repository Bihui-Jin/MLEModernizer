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

0.97926267281106

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18025) has done: 'The fix replaces the missing‑model loading with a lightweight, end‑to‑end training pipeline: it builds a small transfer‑learning model (MobileNetV2), trains it on the provided training images, predicts on the test set, and writes a correctly formatted `submission.csv`. All previous cells that tried to load unavailable HDF5 models are removed, and the new cells handle data loading, label mapping, model definition, training, inference, and CSV export.'
- What this solution (achieved 0.57802) has done: 'I fixed the script by removing the failing TensorFlow import and the empty data generators that couldn’t locate images in the nested class folders. Instead, I added a simple image‑loading routine using Pillow, encoded the labels, and trained a lightweight RandomForest classifier on resized 64×64 RGB pixels. The test images are processed the same way, predictions are mapped back to the original label strings, and a correctly formatted `submission.csv` is written. This resolves all runtime errors and yields a valid submission file.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from sklearn.metrics import accuracy_score

tf.random.set_seed(42)
np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_INPUT = "/kaggle/input/paddy-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUBMISSION = os.path.join(BASE_INPUT, "sample_submission.csv")

IMG_SIZE = (128, 128)  # a modest upscale for better features
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE



## === cell 2
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_IMG_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=0.1,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_IMG_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=0.1,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

class_names = train_ds.class_names  # alphabetical order
idx_to_label = {idx: name for idx, name in enumerate(class_names)}
label_to_idx = {name: idx for idx, name in enumerate(class_names)}



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2924355802.py in <cell line: 0>()
     27 
     28 # Mapping between integer indices and original label strings
---> 29 class_names = train_ds.class_names  # alphabetical order
     30 idx_to_label = {idx: name for idx, name in enumerate(class_names)}
     31 label_to_idx = {name: idx for idx, name in enumerate(class_names)}

AttributeError: '_PrefetchDataset' object has no attribute 'class_names'

## === cell 3
base_model = applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False  # freeze base

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation="softmax"),
    ]
)

model.compile(
    optimizer=optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3648886516.py in <cell line: 0>()
     10         layers.GlobalAveragePooling2D(),
     11         layers.Dropout(0.2),
---> 12         layers.Dense(len(class_names), activation="softmax"),
     13     ]
     14 )

NameError: name 'class_names' is not defined

## === cell 4
EPOCHS = 5  # few epochs are enough for a decent baseline
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

val_acc = history.history["val_accuracy"][-1]
print(f"Validation accuracy: {val_acc:.4f}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3064679002.py in <cell line: 0>()
      1 EPOCHS = 5  # few epochs are enough for a decent baseline
----> 2 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
      3 
      4 # Compute validation accuracy for reporting
      5 val_acc = history.history["val_accuracy"][-1]

NameError: name 'model' is not defined

## === cell 5
test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")])
test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_files]


def preprocess_path(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


test_images = np.stack([preprocess_path(p) for p in test_paths], axis=0)



## === cell 6
pred_probs = model.predict(test_images, batch_size=BATCH_SIZE, verbose=0)
pred_idx = np.argmax(pred_probs, axis=1)
pred_labels = [idx_to_label[i] for i in pred_idx]

submission = pd.DataFrame({"image_id": test_files, "label": pred_labels})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3790571291.py in <cell line: 0>()
----> 1 pred_probs = model.predict(test_images, batch_size=BATCH_SIZE, verbose=0)
      2 pred_idx = np.argmax(pred_probs, axis=1)
      3 pred_labels = [idx_to_label[i] for i in pred_idx]
      4 
      5 submission = pd.DataFrame({"image_id": test_files, "label": pred_labels})

NameError: name 'model' is not defined
