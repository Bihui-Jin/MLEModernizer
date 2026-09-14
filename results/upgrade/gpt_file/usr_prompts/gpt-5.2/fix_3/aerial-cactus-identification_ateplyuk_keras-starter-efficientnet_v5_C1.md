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

0.496

# 6. Current score

0.70137

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.70066) has done: 'I remove the internet `pip install` and switch the EfficientNet import to the built-in `tf_keras.applications` version so the model can be constructed in this offline Kaggle environment. I fix the Keras 3 optimizer argument (`lr` → `learning_rate`) so `compile()` works and training can start. I correct the dataset paths to the provided `/kaggle/input/aerial-cactus-identification/...` folders and ensure test images are read in a deterministic order and resized to 32×32 to avoid the inhomogeneous-shape error. Finally, I produce a valid `submission.csv` with probabilities (no thresholding) aligned to `sample_submission.csv` ids.'
- What this solution (achieved 0.70137) has done: 'I fix the runtime crash happening before training by addressing the `MessageFactory.GetPrototype` error, which is caused by an incompatible protobuf implementation being imported in this environment when using `tf_keras`. The minimal robust fix is to force the pure-Python protobuf backend before importing `tf_keras`, which avoids the C++ message factory path that triggers this attribute error. I keep the model/training/prediction logic unchanged and only adjust the import order and environment variable so the notebook runs end-to-end. This is score-neutral (it just restores executability) and still write a valid `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import tf_keras as tfk
from tf_keras import layers
from tf_keras.models import Sequential
from tf_keras.optimizers import Adam

EfficientNetB3 = tfk.applications.EfficientNetB3

np.random.seed(42)
tfk.utils.set_random_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "/kaggle/input/aerial-cactus-identification"
train_dir = os.path.join(BASE_DIR, "train")
test_dir = os.path.join(BASE_DIR, "test")

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"

train_df = pd.read_csv(train_csv_path)
train_df.head()



## === cell 2
im_path = os.path.join(train_dir, train_df["id"].iloc[0])
im = cv2.imread(im_path)
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(2, 2))
plt.imshow(im_rgb)
plt.axis("off")



## === cell 3
eff_net = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
eff_net.trainable = False

model = Sequential()
model.add(eff_net)
model.add(layers.Flatten())
model.add(layers.Dense(1024))
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation("sigmoid"))



## === cell 4
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=1e-5),
    metrics=["accuracy"],
)

model.summary()



## === cell 5
X_tr = np.zeros((len(train_df), 32, 32, 3), dtype=np.float32)
Y_tr = train_df["has_cactus"].values.astype(np.float32)

for i, img_id in enumerate(tqdm(train_df["id"].values, desc="Loading train images")):
    img_path = os.path.join(train_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {img_path}")
    if img.shape[:2] != (32, 32):
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    X_tr[i] = img.astype(np.float32) / 255.0

X_tr.shape, Y_tr.shape



## === cell 6
batch_size = 96
nb_epoch = 25



## === cell 7
history = model.fit(
    X_tr,
    Y_tr,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_split=0.1,
    shuffle=True,
    verbose=2,
)



## === cell 8
with open("history.json", "w") as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)

plt.figure(figsize=(8, 3))
history_df[["loss", "val_loss"]].plot(ax=plt.gca(), title="Loss")
plt.show()

acc_cols = [c for c in ["accuracy", "val_accuracy"] if c in history_df.columns]
if len(acc_cols) == 2:
    plt.figure(figsize=(8, 3))
    history_df[acc_cols].plot(ax=plt.gca(), title="Accuracy")
    plt.show()



## === cell 9
sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["id"].values

X_tst = np.zeros((len(test_ids), 32, 32, 3), dtype=np.float32)
for i, img_id in enumerate(tqdm(test_ids, desc="Loading test images")):
    img_path = os.path.join(test_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {img_path}")
    if img.shape[:2] != (32, 32):
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    X_tst[i] = img.astype(np.float32) / 255.0

X_tst.shape



## === cell 10
test_predictions = model.predict(X_tst, batch_size=256, verbose=1).reshape(-1)
test_predictions[:5], test_predictions.shape



## === cell 11
sub_df = pd.DataFrame(
    {"id": test_ids, "has_cactus": test_predictions.astype(np.float32)}
)
sub_df.head()



## === cell 12
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {sub_df.shape}")
print(sub_df.head())
