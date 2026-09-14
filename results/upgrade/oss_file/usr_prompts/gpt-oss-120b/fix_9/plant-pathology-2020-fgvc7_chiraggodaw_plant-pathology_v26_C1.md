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

# 5. Target score

0.83563

# 6. Current score

0.71255

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.94457) has done: 'The changes remove the problematic imports, stop using unsupported OpenCV GUI calls, correctly split the training data, replace the deprecated `fit_generator` with `model.fit`, load the test images using the ids from `test.csv`, and build a submission DataFrame that matches the required column order and size.'
- What this solution (achieved 0.98806) has done: 'I set the protobuf implementation to pure Python before importing TensorFlow to avoid the `MessageFactory` attribute error, then keep the rest of the pipeline unchanged. This fixes the runtime failure while preserving the original model and training logic, so the resulting submission file is produced and the score remains around the current high value.'
- What this solution (achieved 0.96658) has done: 'The script already runs end‑to‑end and produces a valid `submission.csv`. The only necessary adjustment is to keep the protobuf‑environment setting before any TensorFlow import (already done) and to add a small safeguard that the generated submission contains exactly the rows of the test set, preventing mismatched lengths. No changes to the model or training logic are introduced, preserving the high current score while guaranteeing a correct submission file.'
- What this solution (achieved 0.97806) has done: 'The fix moves the protobuf environment variable to the very first statements, adds a safeguard that loads the best saved model before inference, and keeps the original logic unchanged so the high score is preserved while guaranteeing a correct `submission.csv` file.'
- What this solution (achieved 0.98812) has done: 'The fix moves the protobuf‑environment setting to the very first cell (executed before any library import) to prevent the `MessageFactory` AttributeError, and corrects the learning‑rate scheduler to monitor validation accuracy (matching the checkpoint). No core modeling logic is changed, so the high score is retained while guaranteeing a valid `submission.csv`.'
- What this solution (achieved 0.98338) has done: 'I moved the protobuf environment setting to the very first lines before any library import, merged it with the import block, and removed the now‑empty placeholder cell. This guarantees TensorFlow loads without the `MessageFactory` error while keeping the original model, training, and prediction logic unchanged, so the high current score is preserved and a valid `submission.csv` is written.'
- What this solution (achieved 0.51778) has done: 'The fix moves the protobuf environment setting to the very first lines and clears any previously‑loaded protobuf modules before importing TensorFlow, guaranteeing the `MessageFactory` error is avoided while keeping the original model and training logic unchanged. No changes to the model or metric handling are made, so the high current score is retained and a valid `submission.csv` is created.'
- What this solution (achieved 0.71255) has done: 'The fix changes the problem from a single‑label classification (using argmax and softmax) to the required multi‑label setting. Labels are kept as four independent binary columns, the model now uses a sigmoid output with binary_crossentropy loss, and the validation metric is AUC (which aligns with the competition’s ROC‑AUC score). The training/validation split still uses a stratified argmax label only for balancing, while the actual targets remain multi‑label. This correction restores proper metric calculation and raises the score toward the target while keeping the original architecture and data pipeline.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
for mod in list(sys.modules.keys()):
    if mod.startswith("google.protobuf"):
        del sys.modules[mod]

import gc
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_images = []
train_labels = []  # multi‑label binary vectors
strat_labels = []  # single int for stratified split only
for _, row in train_df.iterrows():
    img_path = f"/kaggle/input/plant-pathology-2020-fgvc7/images/{row['image_id']}.jpg"
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (410, 273), interpolation=cv2.INTER_AREA)
    train_images.append(img)

    label_vec = row[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(
        np.float32
    )
    train_labels.append(label_vec)

    strat_labels.append(int(np.argmax(label_vec)))

train_X = np.array(train_images, dtype=np.float32)
train_y = np.array(train_labels, dtype=np.float32)
strat_y = np.array(strat_labels, dtype=np.int32)

X_train, X_val, y_train, y_val = train_test_split(
    train_X, train_y, test_size=0.2, stratify=strat_y, random_state=42
)



## === cell 3
reduce_lr = ReduceLROnPlateau(
    monitor="val_auc", factor=0.5, patience=5, verbose=1, min_lr=1e-5
)
checkpoint = ModelCheckpoint(
    "best_model.h5", monitor="val_auc", save_best_only=True, verbose=1
)


class EarlyStopAUC(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs and logs.get("auc") and logs["auc"] > 0.99:
            print("\nReached >99% AUC, stopping training.")
            self.model.stop_training = True


early_stop = EarlyStopAUC()



## === cell 4
model = tf.keras.Sequential(
    [
        tf.keras.applications.Xception(
            weights="imagenet", include_top=False, input_shape=(273, 410, 3)
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adamax(),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

train_datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

batch_size = 16
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=(X_val, y_val),
    epochs=20,
    callbacks=[reduce_lr, checkpoint, early_stop],
    verbose=2,
)

if os.path.exists("best_model.h5"):
    model.load_weights("best_model.h5")



## === cell 5
test_images = []
test_ids = []
for img_id in test_df["image_id"]:
    img_path = f"/kaggle/input/plant-pathology-2020-fgvc7/images/{img_id}.jpg"
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((273, 410, 3), dtype=np.uint8)
    img = cv2.resize(img, (410, 273), interpolation=cv2.INTER_AREA)
    test_images.append(img)
    test_ids.append(img_id)

test_X = np.array(test_images, dtype=np.float32)



## === cell 6
preds = model.predict(test_X, batch_size=32, verbose=0)  # (num_test, 4)



## === cell 7
submission = pd.DataFrame(
    preds, columns=["healthy", "multiple_diseases", "rust", "scab"]
)
submission.insert(0, "image_id", test_ids)

assert len(submission) == len(test_df), "Submission row count mismatch."

submission.to_csv("submission.csv", index=False)



## === cell 8
submission.head()
