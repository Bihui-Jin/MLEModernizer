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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9691

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
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print(os.listdir("../input"))




## === cell 2
train_df = pd.read_csv("../input/train.csv")
train_df.head()




## === cell 3
train_df["has_cactus"].value_counts()




## === cell 4
def datagen(df=train_df, img_path="../input/train/train/"):
    """Load images and labels into NumPy arrays."""
    n = len(df)
    X = np.empty((n, 32, 32, 3), dtype=np.uint8)
    y = np.empty(n, dtype=np.float32)

    for i, (img_id, label) in enumerate(df.values):
        img = cv2.imread(os.path.join(img_path, img_id))
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X[i] = img
        y[i] = label

    perm = np.random.permutation(n)
    return X[perm], y[perm]




## === cell 5
X, Y = datagen()
print("Training data shape:", X.shape, Y.shape)




## === cell 6
fig, axs = plt.subplots(1, 5, figsize=(20, 4))
for ax, img, lbl in zip(axs, X[10:15], Y[10:15]):
    ax.set_title("cactus" if lbl == 1 else "no cactus")
    ax.imshow(img)
    ax.axis("off")
plt.show()




## === cell 7
model = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            64, 5, activation="relu", padding="same", input_shape=(32, 32, 3)
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, 5, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=["accuracy"],
)




## === cell 8
model.fit(
    X,
    Y,
    batch_size=256,
    epochs=30,  # reduced epochs for quicker turnaround while retaining performance
    verbose=1,
    validation_split=0.2,
)




## === cell 9
test_imgs = os.listdir("../input/test/test/")
print("Number of test images:", len(test_imgs))
print("First few:", test_imgs[:5])




## === cell 10
def test_pred(img_list=test_imgs, img_path="../input/test/test/"):
    """Generate predictions for the test set."""
    results = []
    for img_name in img_list:
        img = cv2.imread(os.path.join(img_path, img_name))
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)  # shape (1, 32, 32, 3)
        prob = model.predict(img, batch_size=1)[0][0]  # probability of class 1
        prob = float(np.clip(prob, 0.005, 0.995))  # safety clipping as in original code
        results.append([img_name, prob])
    return results




## === cell 11
predictions = test_pred()
pred_df = pd.DataFrame(predictions, columns=["id", "has_cactus"])
pred_df.head()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/3509130965.py in <cell line: 0>()
      1 # Generate predictions DataFrame
----> 2 predictions = test_pred()
      3 pred_df = pd.DataFrame(predictions, columns=["id", "has_cactus"])
      4 pred_df.head()
      5 

/tmp/ipykernel_55/1385214285.py in test_pred(img_list, img_path)
      4     for img_name in img_list:
      5         img = cv2.imread(os.path.join(img_path, img_name))
----> 6         img = cv2.resize(img, (32, 32))
      7         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      8         img = np.expand_dims(img, axis=0)  # shape (1, 32, 32, 3)

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4208: error: (-215:Assertion failed) !ssize.empty() in function 'resize'


## === cell 12
submission_path = "submissions.csv"
pred_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3448573146.py in <cell line: 0>()
      1 # Write submission file
      2 submission_path = "submissions.csv"
----> 3 pred_df.to_csv(submission_path, index=False)
      4 print(f"Submission written to {submission_path}")

NameError: name 'pred_df' is not defined
