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

3.12

# 3. Installed packages

geopandas==0.14.4
imbalanced-learn==0.13.0
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

0.8831

# 6. Current score

0.46087

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.48979) has done: 'I fixed the import error by setting the protobuf implementation before loading TensorFlow, streamlined the data pipeline to avoid the oversampling steps that were failing, and rebuilt a simple multi‑label model (EfficientNetB0 base + global pooling + dense layers) that trains on the four disease columns directly. The script now creates TensorFlow datasets, trains the model, generates predictions for the test set, and writes a correctly formatted `Submission.csv` file. All previous cells that caused NameErrors have been replaced with this functional flow while keeping the original architecture ideas.'
- What this solution (achieved 0.49924) has done: 'I fix the import‑error environment setup, correct the train/validation split (remove invalid stratify on a multilabel array), unfreeze the EfficientNet base so the model can learn visual features, and increase the training epochs to give the network more capacity to improve the ROC‑AUC score. These changes keep the original architecture and workflow while addressing the bugs that prevented execution and boosting performance toward the target metric.'
- What this solution (achieved 0.59142) has done: 'We set the protobuf environment variable before any imports, drop the TensorFlow‑based model (which raised an import error), and replace it with a lightweight scikit‑learn One‑Vs‑Rest logistic‑regression pipeline that works on resized, flattened images.  The new pipeline keeps the original data split, trains on the four disease columns, reports the validation ROC‑AUC (now ≈ 0.60 +, an improvement over the previous 0.50), and writes a correctly formatted `Submission.csv` file.  All other cells are preserved; only the model‑training and prediction sections are changed.'
- What this solution (achieved 0.46087) has done: 'The changes replace the simple logistic‑regression on flattened pixels with a lightweight EfficientNet‑B0 based CNN, keeping the same data splits and file handling but adding a more powerful visual model and proper TensorFlow preprocessing. This is expected to raise the mean ROC‑AUC from ~0.59 toward the target 0.8831 while still producing the required `Submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score




## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
IMG_DIR = os.path.join(BASE_PATH, "images")




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["image_path"] = IMG_DIR + "/" + train_df["image_id"] + ".jpg"
test_df["image_path"] = IMG_DIR + "/" + test_df["image_id"] + ".jpg"




## === cell 3
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
y = train_df[TARGET_COLS].values.astype(np.float32)
X_paths = train_df["image_path"].values




## === cell 4
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    X_paths, y, test_size=0.2, random_state=42
)




## === cell 5
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def decode_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1]
    return image, label


def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, seed=42)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(X_train_paths, y_train, shuffle=True)
val_ds = make_dataset(X_val_paths, y_val, shuffle=False)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
outputs = tf.keras.layers.Dense(len(TARGET_COLS), activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy"
)




## === cell 7
model.fit(train_ds, epochs=12, validation_data=val_ds, verbose=2)




## === cell 8
val_images, _ = next(iter(val_ds.unbatch().batch(len(X_val_paths))))
val_preds = model.predict(val_images, batch_size=BATCH_SIZE)

auc_scores = [
    roc_auc_score(y_val[:, i], val_preds[:, i]) for i in range(len(TARGET_COLS))
]
mean_auc = np.mean(auc_scores)
print(f"Validation AUC per class: {auc_scores}")
print(f"Mean Validation AUC: {mean_auc:.5f}")




## === cell 9
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy"
)
model.fit(train_ds, epochs=4, validation_data=val_ds, verbose=2)




## === cell 10
test_ds = make_dataset(
    test_df["image_path"].values,
    np.zeros((len(test_df), len(TARGET_COLS))),
    shuffle=False,
)
test_preds = model.predict(test_ds, verbose=0)

submission = pd.DataFrame(test_preds, columns=TARGET_COLS)
submission.insert(0, "image_id", test_df["image_id"].values)
submission[TARGET_COLS] = submission[TARGET_COLS].clip(0, 1).round(4)

submission_path = "Submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")
