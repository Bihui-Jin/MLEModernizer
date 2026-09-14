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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.4988

# 6. Current score

0.66743

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.66743) has done: 'I fix the environment-breaking Keras import issue by forcing Keras 3 to use the TensorFlow backend (and avoiding the protobuf-related crash) while keeping your model architecture and training logic the same. Then I correct the dataset paths: your current `train_dir`/`test_dir` point to non-existent nested folders, which caused the image loading to fail and cascaded into later errors. I also add a small safeguard so prediction/submission still runs even if something unexpected happens during training, and ensure the submission is aligned to `sample_submission.csv` order and always written as `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

import keras
from keras import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.applications import EfficientNetB3

print("Keras version:", keras.__version__)
print("KERAS_BACKEND:", os.environ.get("KERAS_BACKEND"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "/kaggle/input/aerial-cactus-identification"

train_dir = os.path.join(BASE, "train")
test_dir = os.path.join(BASE, "test")
train_csv_path = os.path.join(BASE, "train.csv")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")

assert os.path.exists(train_dir), f"Missing train_dir: {train_dir}"
assert os.path.exists(test_dir), f"Missing test_dir: {test_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv: {sample_sub_path}"

train_df = pd.read_csv(train_csv_path)
print("Train rows:", len(train_df), "cols:", list(train_df.columns))
train_df.head()



## === cell 2
example_path = os.path.join(train_dir, train_df["id"].iloc[0])
im = cv2.imread(example_path)
print("Example image path:", example_path)
print("Image shape:", None if im is None else im.shape)

if im is not None:
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()



## === cell 3
eff_net = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3),
)
eff_net.trainable = False

model = Sequential()
model.add(eff_net)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=1e-5),
    metrics=["accuracy"],
)

model.summary()



## === cell 4
X_tr = []
Y_tr = []

label_map = dict(zip(train_df["id"].values, train_df["has_cactus"].values))

img_ids = train_df["id"].values
for img_id in tqdm(img_ids, desc="Loading train images"):
    img_path = os.path.join(train_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    X_tr.append(img)
    Y_tr.append(label_map[img_id])

X_tr = np.asarray(X_tr, dtype=np.float32) / 255.0
Y_tr = np.asarray(Y_tr, dtype=np.float32)

print(
    "Train X shape:", X_tr.shape, "Y shape:", Y_tr.shape, "Y mean:", float(Y_tr.mean())
)



## === cell 5
batch_size = 32
nb_epoch = 10

history = model.fit(
    X_tr,
    Y_tr,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_split=0.1,
    shuffle=True,
    verbose=2,
)



## === cell 6
with open("history.json", "w") as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
print(history_df.tail())

if "loss" in history_df and "val_loss" in history_df:
    history_df[["loss", "val_loss"]].plot(title="Loss")
    plt.show()

acc_key = (
    "accuracy"
    if "accuracy" in history_df.columns
    else ("acc" if "acc" in history_df.columns else None)
)
val_acc_key = (
    "val_accuracy"
    if "val_accuracy" in history_df.columns
    else ("val_acc" if "val_acc" in history_df.columns else None)
)
if acc_key and val_acc_key:
    history_df[[acc_key, val_acc_key]].plot(title="Accuracy")
    plt.show()



## === cell 7
test_ids = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
print("Test images found:", len(test_ids))

X_tst = []
for img_id in tqdm(test_ids, desc="Loading test images"):
    img_path = os.path.join(test_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    X_tst.append(img)

X_tst = np.asarray(X_tst, dtype=np.float32) / 255.0
print("Test X shape:", X_tst.shape)



## === cell 8
if X_tst.shape[0] == 0:
    raise RuntimeError("No test images were loaded; cannot predict.")

test_predictions = model.predict(X_tst, batch_size=256, verbose=1).reshape(-1)

print(
    "Pred shape:",
    test_predictions.shape,
    "min/max:",
    float(test_predictions.min()),
    float(test_predictions.max()),
)



## === cell 9
sub_df = pd.DataFrame(
    {"id": test_ids, "has_cactus": test_predictions.astype(np.float32)}
)

sample_sub = pd.read_csv(sample_sub_path)
if "id" in sample_sub.columns and set(sample_sub["id"]) == set(sub_df["id"]):
    sub_df = sample_sub[["id"]].merge(sub_df, on="id", how="left")
else:
    sub_df = sub_df.sort_values("id").reset_index(drop=True)

sub_df.head()



## === cell 10
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub_df), "cols:", list(sub_df.columns))
print(sub_df.describe(include="all"))
