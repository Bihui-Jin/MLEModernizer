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

0.5055

# 6. Current score

0.69578

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.69578) has done: 'I fix the environment/import crash by switching to `tf_keras` (the Kaggle image has keras 3 + tf_keras 2.18, and your current `keras.preprocessing` import triggers a protobuf/MessageFactory error). I also fix pathing to use the actual extracted dataset folders under `../input/aerial-cactus-identification/`, ensure class_weight/tqdm imports are available, and update callback monitors to valid Keras metric names so training/checkpointing runs. Finally, I generate predictions in the exact `sample_submission.csv` order and output floating probabilities (not `int`), which fixes the “same number of rows” submission error and is metric-correct for AUC.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tf_keras as keras
from tf_keras.preprocessing import image
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.models import Sequential
from tf_keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Dense,
    Flatten,
    BatchNormalization,
)
from tf_keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight

print("Listing ../input:", os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "../input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

output_dir = os.path.join(BASE_DIR, "model_output", "CNN")
os.makedirs(output_dir, exist_ok=True)

seed = 7
np.random.seed(seed)

print("BASE_DIR exists:", os.path.exists(BASE_DIR))
print(
    "TRAIN_DIR exists:",
    os.path.exists(TRAIN_DIR),
    "num_files:",
    len(os.listdir(TRAIN_DIR)) if os.path.exists(TRAIN_DIR) else 0,
)
print(
    "TEST_DIR exists:",
    os.path.exists(TEST_DIR),
    "num_files:",
    len(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else 0,
)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df.head()



## === cell 3
cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"].values,
)
class_weights = {0: float(cw[0]), 1: float(cw[1])}
print("class_weights:", class_weights)



## === cell 4
train_image = []
for i in tqdm(range(len(train_df)), desc="Loading train images"):
    img = image.load_img(
        os.path.join(TRAIN_DIR, train_df["id"].iloc[i]), target_size=(32, 32)
    )
    img = image.img_to_array(img).astype("float32") / 255.0
    train_image.append(img)

X = np.array(train_image, dtype="float32")
print("X:", X.shape, X.dtype)



## === cell 5
X.shape



## === cell 6
plt.imshow(X[1])
plt.axis("off")
plt.show()



## === cell 7
y = train_df["has_cactus"].values.astype("float32").reshape(-1, 1)
y.shape



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)



## === cell 9
X_train.shape, X_val.shape, y_train.shape, y_val.shape



## === cell 10
img_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    rotation_range=40,
    brightness_range=(0.5, 1.0),
    height_shift_range=0.2,
    width_shift_range=0.2,
)
val_datagen = ImageDataGenerator()

train_generator = img_gen.flow(X_train, y_train, batch_size=32, shuffle=True)
validation_generator = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)



## === cell 11
model = Sequential()
model.add(
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3))
)
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.summary()



## === cell 12
best_path = os.path.join(output_dir, "weights.best.hdf5")
callbacks = [
    ModelCheckpoint(
        filepath=best_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        mode="auto",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", mode="auto", patience=3, min_lr=0.0001, verbose=1
    ),
]



## === cell 13
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_generator,
    epochs=80,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2,
)




## === cell 14
def predict_one(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    arr = image.img_to_array(img).astype("float32") / 255.0
    proba = model.predict(arr.reshape(1, 32, 32, 3), verbose=0)[0, 0]
    return float(proba)




## === cell 15
if os.path.exists(best_path):
    model.load_weights(best_path)
    print("Loaded best weights:", best_path)
else:
    print("Best weights not found; using final epoch weights.")



## === cell 16
y_hat = model.predict(X_val, batch_size=128, verbose=0).reshape(-1)
auc = roc_auc_score(y_val.reshape(-1), y_hat)
print("Validation ROC AUC:", auc)



## === cell 17
sub = pd.read_csv(SAMPLE_SUB)
test_ids = sub["id"].tolist()

preds = []
for fname in tqdm(test_ids, desc="Predicting test"):
    preds.append(predict_one(os.path.join(TEST_DIR, fname)))



## === cell 18
pred_df = pd.DataFrame({"id": test_ids, "has_cactus": preds})
pred_df.shape, pred_df.head()



## === cell 19
out_path = "Submission.csv"
pred_df.to_csv(out_path, index=False)
print(
    "Wrote submission:", out_path, "rows:", len(pred_df), "cols:", list(pred_df.columns)
)
