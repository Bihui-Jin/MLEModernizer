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
tqdm==4.67.1

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

0.86442

# 6. Current score

0.56575

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.65584) has done: 'Outline:
I set the protobuf implementation environment variable before importing TensorFlow to avoid the `MessageFactory` error.  
All `Conv2D` layers use `padding='same'` so the spatial dimensions stay valid after repeated pooling, fixing the negative size error during model building and training.  
The rest of the pipeline remains unchanged, ensuring the script runs end‑to‑end and produces a correctly formatted `submission.csv`.'
- What this solution (achieved 0.56575) has done: 'I added a robust import guard for TensorFlow: if the protobuf‑related error occurs, the script falls back to a pure scikit‑learn solution that trains a separate RandomForest classifier for each label on flattened image pixels. This keeps the original image preprocessing, ensures a valid `submission.csv` is written, and provides a stronger baseline than the previous constant prediction, moving the AUC toward the target. All other logic and file paths remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import tqdm
import cv2

use_tf = False
try:
    import tensorflow as tf
    from tensorflow import keras

    use_tf = True
    print("TensorFlow imported successfully.")
except Exception as e:
    print("TensorFlow import failed, will use sklearn fallback. Error:", e)

DATA_ROOT = "../input/plant-pathology-2020-fgvc7"
train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
y_train = train[target_cols].values.astype("float32")  # already 0/1



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
img_size = 128  # reasonable size for quick training


def read_img(fname):
    path = os.path.join(DATA_ROOT, "images", fname)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image {path} not found")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_to_square(img, size=img_size):
    h, w = img.shape[:2]
    if h > w:
        pad = (h - w) // 2
        img = cv2.copyMakeBorder(
            img, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=0
        )
    elif w > h:
        pad = (w - h) // 2
        img = cv2.copyMakeBorder(
            img, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=0
        )
    img = cv2.resize(img, (size, size))
    return img.astype("float32") / 255.0


train_imgs = np.zeros((train.shape[0], img_size, img_size, 3), dtype="float32")
for i, fid in enumerate(tqdm.tqdm(train["image_id"])):
    img = read_img(f"{fid}.jpg")
    train_imgs[i] = resize_to_square(img)



## === cell 2
test_imgs = np.zeros((test.shape[0], img_size, img_size, 3), dtype="float32")
for i, fid in enumerate(tqdm.tqdm(test["image_id"])):
    img = read_img(f"{fid}.jpg")
    test_imgs[i] = resize_to_square(img)



## === cell 3
if use_tf:
    img_input = keras.layers.Input(shape=(img_size, img_size, 3))
    x = keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(img_input)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(4, activation="sigmoid")(x)  # 4 labels
    model = keras.models.Model(inputs=img_input, outputs=output)

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    model.fit(
        train_imgs,
        y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
    )
else:
    from sklearn.ensemble import RandomForestClassifier

    X_flat = train_imgs.reshape(train_imgs.shape[0], -1)
    rf_models = []
    for col in range(y_train.shape[1]):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
        )
        rf.fit(X_flat, y_train[:, col])
        rf_models.append(rf)



## === cell 4
if use_tf:
    test_preds = model.predict(test_imgs, batch_size=128, verbose=1)
else:
    X_test_flat = test_imgs.reshape(test_imgs.shape[0], -1)
    pred_list = [rf.predict_proba(X_test_flat)[:, 1] for rf in rf_models]
    test_preds = np.stack(pred_list, axis=1)



## === cell 5
submission = pd.DataFrame(test_preds, columns=target_cols)
submission.insert(0, "image_id", test["image_id"])
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
