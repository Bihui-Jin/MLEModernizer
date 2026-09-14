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

No external packages required in the script and installed.

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

0.9881666666666666

# 6. Current score

0.85209

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.85209) has done: 'I remove the broken dependency on a non-existent `../input/firstaa/submission.csv` and instead build predictions from the provided `sample_submission.csv` + `test/` images, producing a valid `submission.csv`. I also fix the Keras import/runtime crash by using `tf.keras` (compatible in Kaggle) while keeping the same intended ResNet50 transfer-learning approach. Missing variables (`train`, `X_test`, `model`, `history`) are resolved by adding the required data loading, preprocessing, model construction, training, and inference steps in the original cell order. Finally, I keep predictions as probabilities (no hard thresholding) because the metric is ROC AUC and thresholding would usually hurt score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from skimage.io import imread

INPUT_ROOT_CANDIDATES = ["../input", "/kaggle/input"]
INPUT_ROOT = next(
    (p for p in INPUT_ROOT_CANDIDATES if os.path.exists(p)), INPUT_ROOT_CANDIDATES[0]
)

print("INPUT_ROOT:", INPUT_ROOT)
print("Top-level input dirs/files:", os.listdir(INPUT_ROOT)[:50])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



## === cell 1
COMP_DIR = os.path.join(INPUT_ROOT, "aerial-cactus-identification")
if not os.path.exists(COMP_DIR):
    COMP_DIR = INPUT_ROOT

sample_path = os.path.join(COMP_DIR, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = os.path.join(INPUT_ROOT, "sample_submission.csv")

test = pd.read_csv(sample_path)
print("sample_submission shape:", test.shape)
test.head()



## === cell 2
assert "id" in test.columns and "has_cactus" in test.columns



## === cell 3
train_path = os.path.join(COMP_DIR, "train.csv")
if not os.path.exists(train_path):
    train_path = os.path.join(INPUT_ROOT, "train.csv")

train = pd.read_csv(train_path)
print("train shape:", train.shape)
len(train)



## === cell 4
train_dir = os.path.join(COMP_DIR, "train")
test_dir = os.path.join(COMP_DIR, "test")

if not os.path.exists(train_dir):
    train_dir = os.path.join(INPUT_ROOT, "train")
if not os.path.exists(test_dir):
    test_dir = os.path.join(INPUT_ROOT, "test")

print("train_dir exists:", os.path.exists(train_dir), train_dir)
print("test_dir exists:", os.path.exists(test_dir), test_dir)



## === cell 5
y_train = np.array(train.has_cactus).astype(np.float32)

IMG_SIZE = 32  # dataset thumbnails are 32x32
N_TRAIN = len(train)

X_train = np.zeros((N_TRAIN, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

for i, img_id in enumerate(train["id"].values):
    img_path = os.path.join(train_dir, img_id)
    img = imread(img_path)
    if img.ndim == 2:  # grayscale safety
        img = np.stack([img, img, img], axis=-1)
    X_train[i] = img.astype(np.float32)

X_train /= 255.0

print("X_train:", X_train.shape, X_train.dtype, "y_train:", y_train.shape)



## === cell 6
N_TEST = len(test)
X_test = np.zeros((N_TEST, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

for i, img_id in enumerate(test["id"].values):
    img_path = os.path.join(test_dir, img_id)
    img = imread(img_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    X_test[i] = img.astype(np.float32)

X_test /= 255.0
print("X_test:", X_test.shape, X_test.dtype)



## === cell 7
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
print("Train split:", X_tr.shape, "Val split:", X_val.shape)



## === cell 8
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
for layer in base_model.layers:
    layer.trainable = False



## === cell 10
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inp, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
out = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=out)

model.summary()



## === cell 11
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

red = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
    verbose=1,
    mode="auto",
    min_lr=0.0,
)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=50, verbose=1, restore_best_weights=True
)



## === cell 12
from tensorflow.keras.optimizers import Adam

model.compile(
    loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"]
)



## === cell 13
BATCH_SIZE = 64
EPOCHS = 25

history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[red, early_stopping],
    verbose=2,
)



## === cell 14
plt.figure(figsize=(8, 4))
plt.plot(history.history.get("loss", []), label="train")
plt.plot(history.history.get("val_loss", []), label="val")
plt.legend()
plt.tight_layout()
plt.show()



## === cell 15
pre = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)
print("pred stats:", float(pre.min()), float(pre.max()), float(pre.mean()))



## === cell 16
test["has_cactus"] = pre.astype(np.float32)
test.head()



## === cell 17
out_path = "submission.csv"
test[["id", "has_cactus"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", test.shape)



## === cell 18
assert os.path.exists(out_path)
assert list(test.columns) == ["id", "has_cactus"]
print(test.head())
