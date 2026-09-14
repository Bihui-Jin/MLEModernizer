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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tqdm==4.67.1

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

0.6488

# 6. Current score

0.91355

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.91355) has done: 'The fixes address two critical problems:  
1) The TensorFlow import caused a protobuf‑related `AttributeError`. We replace it with the pure Keras API, which works with the installed `keras` package.  
2) The test image path and submission creation were mismatched, resulting in a row count error. We point to the correct `aerial-cactus-identification` directories and build the submission using the ids from the provided `sample_submission.csv`, guaranteeing the exact number of rows.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import cv2
from tqdm.auto import tqdm
from keras import layers, models, metrics
from sklearn.utils import class_weight

print("input folders:", os.listdir("../input/"))
print("Keras version:", models.__package__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("../input/aerial-cactus-identification/train.csv")
train_dir = "../input/aerial-cactus-identification/train/"



## === cell 2
X_tr = []
Y_tr = []
for img_id in tqdm(train_df["id"].values):
    img_path = os.path.join(train_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:  # safety fallback
        img = np.zeros((32, 32, 3), dtype=np.uint8)
    X_tr.append(img)
    Y_tr.append(train_df.loc[train_df["id"] == img_id, "has_cactus"].values[0])
X_tr = np.asarray(X_tr, dtype="float32") / 255.0
Y_tr = np.asarray(Y_tr)



## === cell 3
shape = X_tr.shape[1:4]  # (height, width, channels)



## === cell 4
model = models.Sequential()
model.add(layers.Conv2D(64, (5, 5), input_shape=shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(64, (5, 5)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(128, (5, 5)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(128, (5, 5)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(256, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(256, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(512, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Conv2D(512, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Flatten())
model.add(layers.Dense(100))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Dense(1, activation="sigmoid"))



## === cell 5
model.summary()



## === cell 6
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[metrics.AUC(name="auc")],
)



## === cell 7
classes = np.unique(Y_tr)
weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=classes, y=Y_tr
)
class_weights = dict(zip(classes, weights))

model.fit(
    X_tr,
    Y_tr,
    epochs=6,
    batch_size=500,
    validation_split=0.1,
    class_weight=class_weights,
    verbose=1,
)



## === cell 8
sample_sub = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
test_ids = sample_sub["id"].tolist()
test_dir = "../input/aerial-cactus-identification/test/"

X_tst = []
for img_id in tqdm(test_ids):
    img_path = os.path.join(test_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
    X_tst.append(img)
X_tst = np.asarray(X_tst, dtype="float32") / 255.0



## === cell 9
test_predictions = model.predict(X_tst).reshape(-1)



## === cell 10
sub_df = pd.DataFrame({"id": test_ids, "has_cactus": test_predictions})
sub_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with", len(sub_df), "rows")
