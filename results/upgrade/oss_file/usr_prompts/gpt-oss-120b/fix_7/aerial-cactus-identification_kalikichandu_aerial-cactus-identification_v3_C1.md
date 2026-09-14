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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8193

# 6. Current score

0.99882

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99919) has done: 'I fixed the import errors by switching to `tensorflow.keras`, corrected the `ModelCheckpoint` filename and monitoring metric, removed the undefined callback reference, replaced the nonexistent `predict_proba` with `predict`, stored real probability values instead of integers, pointed the test‑image loop to the proper directory while skipping sub‑folders, and saved the submission with the correct number of rows and column names. These changes make the notebook run end‑to‑end and produce a valid `Submission.csv` while preserving the original model architecture.'
- What this solution (achieved 0.99907) has done: 'I replace all TensorFlow‑specific Keras imports with the standalone `keras` package imports, which resolves the `MessageFactory` error caused by missing TensorFlow in the environment. The core model architecture, training loop and submission logic remain unchanged, so the script run end‑to‑end and produce a valid `Submission.csv` while keeping the already high AUC score.'
- What this solution (achieved 0.99924) has done: 'I replace the standalone `keras` imports with the TensorFlow‑backed `tensorflow.keras` equivalents (which avoid the protobuf `MessageFactory` error) and adjust the corresponding load_model import. The rest of the logic stays unchanged, preserving the high AUC while fixing the runtime crash and ensuring a proper `Submission.csv` is written.'
- What this solution (achieved 0.99923) has done: 'I replace the TensorFlow‑based imports with the standalone `keras` package to avoid the protobuf `MessageFactory` error that stops the script. All other logic, model architecture, training, and submission steps remain unchanged, so the high AUC score is preserved while the notebook runs end‑to‑end and creates a valid `Submission.csv`.'
- What this solution (achieved 0.99888) has done: 'I replace the failing standalone `keras` imports with the TensorFlow‑backed `tensorflow.keras` equivalents, which resolves the `MessageFactory` protobuf error and lets the notebook run end‑to‑end. All other logic (model architecture, training, prediction, and CSV creation) remains unchanged, preserving the high AUC while ensuring a valid `Submission.csv` is written. The cells are renumbered starting from 1 as required.'
- What this solution (achieved 0.99882) has done: 'The script’s imports caused a protobuf MessageFactory error because the environment lacks TensorFlow; switching to the standalone `keras` package resolves this and lets the notebook run end‑to‑end while keeping the original model architecture and high AUC. No other logic changes are needed, so the model’s performance stays essentially the same (still above the target).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

print(os.listdir("../input"))




## === cell 1
from keras.preprocessing import image
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalMaxPool2D,
    Dropout,
    Dense,
    Flatten,
)
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
output_dir = "../input/aerial-cactus-identification/model_output/CNN"
seed = 7
np.random.seed(seed)




## === cell 3
train_df = pd.read_csv("../input/train.csv")
train_df.head()




## === cell 4
train_image = []
train_path = "../input/train/train"

for idx in tqdm(range(len(train_df))):
    img_path = os.path.join(train_path, train_df["id"][idx])
    img = image.load_img(img_path, target_size=(32, 32))
    img = image.img_to_array(img) / 255.0
    train_image.append(img)

X = np.array(train_image)




## === cell 5
X.shape




## === cell 6
plt.imshow(X[1])




## === cell 7
y = train_df["has_cactus"].values.reshape(-1, 1)




## === cell 8
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




## === cell 9
model = Sequential()
model.add(
    Conv2D(filters=512, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3))
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.summary()




## === cell 10
modelcheckpoint = ModelCheckpoint(
    filepath="weights.best.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)




## === cell 11
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(
    X_train,
    y_train,
    epochs=80,
    validation_data=(X_test, y_test),
    batch_size=32,
    shuffle=True,
    callbacks=[modelcheckpoint],
    verbose=2,
)




## === cell 12
model = load_model("weights.best.keras")




## === cell 13
y_hat = model.predict(X_test, batch_size=32).ravel()
auc_score = roc_auc_score(y_test, y_hat) * 100.0
print(f"AUC: {auc_score:.4f}")




## === cell 14
pred = {}


def predictions(image_path, image_name):
    img = image.load_img(image_path, target_size=(32, 32))
    img = image.img_to_array(img) / 255.0
    prob = model.predict(img.reshape(1, 32, 32, 3), batch_size=1).ravel()[0]
    pred[image_name] = prob




## === cell 15
test_dir = "../input/test"
files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
for file in tqdm(files):
    predictions(os.path.join(test_dir, file), file)




## === cell 16
pred_df = pd.DataFrame(list(pred.items()), columns=["id", "has_cactus"])
print(pred_df.shape)
print(pred_df.head())




## === cell 17
pred_df.to_csv("Submission.csv", index=False)
