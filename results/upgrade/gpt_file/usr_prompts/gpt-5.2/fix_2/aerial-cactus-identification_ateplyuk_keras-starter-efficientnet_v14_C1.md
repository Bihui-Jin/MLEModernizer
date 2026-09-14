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

0.9853

# 6. Current score

0.70975

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.70975) has done: 'I remove the internet `pip install` and switch the backbone import to the built-in `tf_keras.applications.EfficientNetB3`, which matches the original EfficientNetB3 intent and avoids the missing symbol error. I also fix the Keras/protobuf crash by using `tf_keras` consistently instead of `keras==3` APIs in this TF1/TF2-style notebook. Then I correct the dataset paths to the provided `/kaggle/input/aerial-cactus-identification/...` folders and ensure test images are loaded in a deterministic sorted order and resized to (32,32) to prevent the inhomogeneous array error. Finally, I output probabilities (no hard thresholding) in `submission.csv` with the exact `id,has_cactus` columns required for AUC.'

# 9. Code solution

## === cell 0

import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import tf_keras as keras
from tf_keras import optimizers
from tf_keras.models import Model
from tf_keras.layers import Dropout, Flatten, Dense
from tf_keras.applications import EfficientNetB3

np.random.seed(42)
try:
    keras.utils.set_random_seed(42)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "/kaggle/input/aerial-cactus-identification"
train_dir = os.path.join(BASE, "train")
test_dir = os.path.join(BASE, "test")
train_csv = os.path.join(BASE, "train.csv")
sample_sub_csv = os.path.join(BASE, "sample_submission.csv")

assert os.path.isdir(train_dir), f"Missing train_dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing test_dir: {test_dir}"
assert os.path.isfile(train_csv), f"Missing train.csv: {train_csv}"
assert os.path.isfile(
    sample_sub_csv
), f"Missing sample_submission.csv: {sample_sub_csv}"

train_df = pd.read_csv(train_csv)
train_df.head()



## === cell 2
im_path = os.path.join(train_dir, train_df["id"].iloc[0])
im = cv2.imread(im_path)
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(2, 2))
plt.imshow(im_rgb)
plt.axis("off")
plt.show()



## === cell 3
eff_net = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
eff_net.trainable = False

x = eff_net.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=eff_net.input, outputs=predictions)

model.compile(
    optimizer=optimizers.RMSprop(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 4
X_tr = []
Y_tr = []
imges = train_df["id"].values

for img_id in tqdm(imges, desc="Loading train"):
    img_path = os.path.join(train_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {img_path}")
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X_tr.append(img)
    Y_tr.append(int(train_df.loc[train_df["id"] == img_id, "has_cactus"].values[0]))

X_tr = np.asarray(X_tr, dtype="float32") / 255.0
Y_tr = np.asarray(Y_tr, dtype="float32")

X_tr.shape, Y_tr.shape, Y_tr.mean()



## === cell 5
batch_size = 256
nb_epoch = 25



## === cell 6
history = model.fit(
    X_tr,
    Y_tr,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_split=0.1,
    shuffle=True,
    verbose=2,
)



## === cell 7
with open("history.json", "w") as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)

plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
history_df[["loss", "val_loss"]].plot(ax=plt.gca())
plt.title("Loss")

plt.subplot(1, 2, 2)
acc_cols = [c for c in ["accuracy", "val_accuracy"] if c in history_df.columns]
if acc_cols:
    history_df[acc_cols].plot(ax=plt.gca())
plt.title("Accuracy")
plt.tight_layout()
plt.show()



## === cell 8
test_ids = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])

X_tst = []
for img_id in tqdm(test_ids, desc="Loading test"):
    img_path = os.path.join(test_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {img_path}")
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X_tst.append(img)

X_tst = np.asarray(X_tst, dtype="float32") / 255.0
X_tst.shape



## === cell 9
test_predictions = model.predict(X_tst, batch_size=512, verbose=1).reshape(-1)

test_predictions[:5], test_predictions.shape



## === cell 10
sub_df = pd.DataFrame({"id": test_ids, "has_cactus": test_predictions.astype(float)})

sample_sub = pd.read_csv(sample_sub_csv)
if set(sample_sub["id"]) == set(sub_df["id"]):
    sub_df = sample_sub[["id"]].merge(sub_df, on="id", how="left")

sub_df.head()



## === cell 11
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {sub_df.shape} and columns {list(sub_df.columns)}")
print(sub_df.isna().sum())
