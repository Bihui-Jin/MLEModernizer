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

0.9753

# 6. Current score

None

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tf_keras as keras
from tf_keras import layers
from tf_keras.callbacks import ModelCheckpoint
from tf_keras.preprocessing import image

from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print(os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "../input/aerial-cactus-identification/model_output/CNN"
seed = 7
np.random.seed(seed)

try:
    keras.utils.set_random_seed(seed)
except Exception:
    pass



## === cell 2
BASE = "../input/aerial-cactus-identification"
if not os.path.exists(BASE):
    BASE = "../input"

TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train", "train")
TEST_DIR = os.path.join(BASE, "test", "test")

if not os.path.exists(TRAIN_DIR):
    TRAIN_DIR = "../input/train/train"
if not os.path.exists(TEST_DIR):
    TEST_DIR = "../input/test/test"
if not os.path.exists(TRAIN_CSV):
    TRAIN_CSV = "../input/train.csv"
if not os.path.exists(SAMPLE_SUB):
    SAMPLE_SUB = "../input/sample_submission.csv"

train_df = pd.read_csv(TRAIN_CSV)
train_df.head()



## === cell 3
train_image = []
for i in tqdm(range(len(train_df))):
    img = image.load_img(
        os.path.join(TRAIN_DIR, train_df["id"].iloc[i]), target_size=(32, 32)
    )
    img = image.img_to_array(img)
    img = img / 255.0
    train_image.append(img)

X = np.array(train_image, dtype=np.float32)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3889287062.py in <cell line: 0>()
      2 train_image = []
      3 for i in tqdm(range(len(train_df))):
----> 4     img = image.load_img(
      5         os.path.join(TRAIN_DIR, train_df["id"].iloc[i]), target_size=(32, 32)
      6     )

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/image_utils.py in load_img(path, grayscale, color_mode, target_size, interpolation, keep_aspect_ratio)
    420         if isinstance(path, pathlib.Path):
    421             path = str(path.resolve())
--> 422         with open(path, "rb") as f:
    423             img = pil_image.open(io.BytesIO(f.read()))
    424     else:

FileNotFoundError: [Errno 2] No such file or directory: '../input/aerial-cactus-identification/train/train/2de8f189f1dce439766637e75df0ee27.jpg'

## === cell 4
X.shape



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3772821318.py in <cell line: 0>()
----> 1 X.shape
      2 

NameError: name 'X' is not defined

## === cell 5
plt.imshow(X[1])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3712634548.py in <cell line: 0>()
----> 1 plt.imshow(X[1])
      2 

NameError: name 'X' is not defined

## === cell 6
y = train_df["has_cactus"].values.astype(np.float32).reshape(-1, 1)
y.shape



## === cell 7
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/932402461.py in <cell line: 0>()
      1 X_train, X_test, y_train, y_test = train_test_split(
----> 2     X, y, random_state=42, test_size=0.2, stratify=y
      3 )
      4 

NameError: name 'X' is not defined

## === cell 8
X_train.shape, X_test.shape, y_train.shape, y_test.shape



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3539845274.py in <cell line: 0>()
----> 1 X_train.shape, X_test.shape, y_train.shape, y_test.shape
      2 

NameError: name 'X_train' is not defined

## === cell 9
model = keras.Sequential()
model.add(
    layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()



## === cell 10
ckpt_path = "weights.best.weights.h5"
modelcheckpoint = ModelCheckpoint(
    filepath=ckpt_path,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    mode="max",
    verbose=1,
)



## === cell 11
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train,
    y_train,
    epochs=80,
    validation_data=(X_test, y_test),
    batch_size=32,
    shuffle=True,
    callbacks=[modelcheckpoint],
)

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2581051443.py in <cell line: 0>()
      1 model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
      2 history = model.fit(
----> 3     X_train,
      4     y_train,
      5     epochs=80,

NameError: name 'X_train' is not defined

## === cell 12
pred = {}


def predictions(imagepath, imagename):
    img = image.load_img(imagepath, target_size=(32, 32))
    img = image.img_to_array(img) / 255.0
    proba = model.predict(img.reshape(1, 32, 32, 3), verbose=0)
    pred.update({imagename: float(proba[0][0])})




## === cell 13
if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)



## === cell 14
y_hat = model.predict(X_test, verbose=0)
get_auc = roc_auc_score(y_test, y_hat) * 100.0
print(get_auc)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/482032895.py in <cell line: 0>()
      1 # Fix: Sequential has no predict_proba; use predict.
----> 2 y_hat = model.predict(X_test, verbose=0)
      3 get_auc = roc_auc_score(y_test, y_hat) * 100.0
      4 print(get_auc)
      5 

NameError: name 'X_test' is not defined

## === cell 15
files = sorted(
    [
        f
        for f in os.listdir(TEST_DIR)
        if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(TEST_DIR, f))
    ]
)
for file in tqdm(files):
    predictions(os.path.join(TEST_DIR, file), file)



## === cell 16
sample = pd.read_csv(SAMPLE_SUB)
pred_series = pd.Series(pred, name="has_cactus")

sub = sample.copy()
sub["has_cactus"] = sub["id"].map(pred_series)

if sub["has_cactus"].isna().any():
    sub["has_cactus"] = sub["has_cactus"].fillna(
        float(np.nanmean(sub["has_cactus"].values))
    )

sub.shape, sub.head()



## === cell 17
sub.to_csv("submission.csv", index=False)
print("Wrote:", "submission.csv", "rows:", len(sub), "cols:", list(sub.columns))
