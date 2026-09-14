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

3.8

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
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.9933

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, shutil, zipfile as zip

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential, load_model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
os.makedirs("../working/train", exist_ok=True)
os.makedirs("../working/test", exist_ok=True)

with zip.ZipFile("../input/aerial-cactus-identification/train.zip", "r") as z:
    z.extractall("../working/train")
with zip.ZipFile("../input/aerial-cactus-identification/test.zip", "r") as z:
    z.extractall("../working/test")

TRAIN_DATA_PATH = "../working/train"
TEST_DATA_PATH = "../working/test"

print("train samples:", len(os.listdir(TRAIN_DATA_PATH)))
print("test samples :", len(os.listdir(TEST_DATA_PATH)))



## === cell 2
df_train = pd.read_csv("../input/aerial-cactus-identification/train.csv")
df_test = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
print(df_train.shape, df_test.shape)




## === cell 3
def plot_roc_auc(truelabel, pred):
    fpr, tpr, _ = roc_curve(truelabel, pred)
    roc_auc = auc(fpr, ttp)
    print("AUC:", roc_auc)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.6f})")
    plt.fill_between(fpr, tpr, alpha=0.3, color="yellow")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()
    return roc_auc




## === cell 4
fig, ax = plt.subplots(4, 8, figsize=(12, 6))
for i in range(32):
    img_path = os.path.join(TRAIN_DATA_PATH, df_train.iloc[i]["id"])
    ax[i // 8, i % 8].imshow(np.array(Image.open(img_path)))
    ax[i // 8, i % 8].axis("off")
    ax[i // 8, i % 8].set_title(f"{df_train.iloc[i]['has_cactus']}")
plt.tight_layout()



## === cell 5
sns.countplot(x="has_cactus", data=df_train)
plt.title("Class distribution")
plt.show()
ratio = len(df_train[df_train.has_cactus == 0]) / len(
    df_train[df_train.has_cactus == 1]
)
print("imbalance ratio (0/1):", ratio)



## === cell 6
train_imgs = []
for img_id in df_train["id"]:
    img_path = os.path.join(TRAIN_DATA_PATH, img_id)
    train_imgs.append(np.array(Image.open(img_path)))
X_train = np.stack(train_imgs, axis=0).astype("float32") / 255.0
y_train = df_train["has_cactus"].values

test_imgs = []
for img_id in df_test["id"]:
    img_path = os.path.join(TEST_DATA_PATH, img_id)
    test_imgs.append(np.array(Image.open(img_path)))
X_test = np.stack(test_imgs, axis=0).astype("float32") / 255.0

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)




## === cell 7
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(scale=False))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(scale=False))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(scale=False))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model




## === cell 8
neg, pos = len(df_train[df_train.has_cactus == 0]), len(
    df_train[df_train.has_cactus == 1]
)
weight_for_0 = (1 / neg) * (neg + pos) / 2.0
weight_for_1 = (1 / pos) * (neg + pos) / 2.0
class_weights = {0: weight_for_0, 1: weight_for_1}
print("class_weights:", class_weights)



## === cell 9
BATCH_SIZE = 64
EPOCHS = 10  # increased modestly to improve AUC without over‑fitting
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof_pred = np.zeros(len(df_train))
sub_pred = np.zeros(len(df_test))
checkpoint_path = "tmp_checkpoint.h5"

histories = []

for fold_id, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Fold {fold_id} ---")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    datagen = ImageDataGenerator(height_shift_range=0.1, horizontal_flip=True)
    tf.keras.backend.clear_session()
    model = build_model(X_train.shape[1:])

    history = model.fit(
        datagen.flow(X_tr, y_tr, batch_size=BATCH_SIZE),
        steps_per_epoch=int(np.ceil(len(X_tr) / BATCH_SIZE)),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.7, patience=5, verbose=1, min_delta=5e-5
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, verbose=1
            ),
        ],
        verbose=2,
    )
    histories.append(history)

    model.load_weights(checkpoint_path)
    oof_pred[val_idx] = model.predict(X_val, batch_size=BATCH_SIZE).flatten()
    sub_pred += model.predict(X_test, batch_size=BATCH_SIZE).flatten() / skf.n_splits

    fold_auc = roc_auc_score(y_val, oof_pred[val_idx])
    print(f"Fold {fold_id} AUC: {fold_auc:.6f}")
    plot_roc_auc(y_val, oof_pred[val_idx])

if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

sub_pred = np.clip(sub_pred, 0.0, 1.0)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2061092707.py in <cell line: 0>()
      1 BATCH_SIZE = 64
      2 EPOCHS = 10  # increased modestly to improve AUC without over‑fitting
----> 3 skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
      4 
      5 oof_pred = np.zeros(len(df_train))

NameError: name 'StratifiedKFold' is not defined

## === cell 10
print("Submission prediction range:", sub_pred.min(), sub_pred.max())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1862014230.py in <cell line: 0>()
----> 1 print("Submission prediction range:", sub_pred.min(), sub_pred.max())
      2 

NameError: name 'sub_pred' is not defined

## === cell 11
plt.hist(sub_pred, bins=50)
plt.title("Distribution of test predictions")
plt.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/953212737.py in <cell line: 0>()
----> 1 plt.hist(sub_pred, bins=50)
      2 plt.title("Distribution of test predictions")
      3 plt.show()
      4 

NameError: name 'sub_pred' is not defined

## === cell 12
ambiguous_idx = np.where((sub_pred > 0.20) & (sub_pred < 0.80))[0]
print("Number of ambiguous predictions:", len(ambiguous_idx))
print("First few ambiguous indices:", ambiguous_idx[:10])



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1609971806.py in <cell line: 0>()
----> 1 ambiguous_idx = np.where((sub_pred > 0.20) & (sub_pred < 0.80))[0]
      2 print("Number of ambiguous predictions:", len(ambiguous_idx))
      3 print("First few ambiguous indices:", ambiguous_idx[:10])
      4 

NameError: name 'sub_pred' is not defined

## === cell 13
submission = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
submission["has_cactus"] = sub_pred
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)

shutil.rmtree("../working/train")
shutil.rmtree("../working/test")
print("Cleanup done.")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1283857567.py in <cell line: 0>()
      1 submission = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
----> 2 submission["has_cactus"] = sub_pred
      3 submission.to_csv("submission.csv", index=False)
      4 print("Saved submission.csv with shape:", submission.shape)
      5 

NameError: name 'sub_pred' is not defined
