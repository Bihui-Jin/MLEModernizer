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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the environment-breaking imports by switching from standalone `keras` to `tf_keras` (available here) to avoid the protobuf `MessageFactory` crash and to restore `ImageDataGenerator`. I also make the unzip/paths consistent with Kaggle’s `/kaggle/input` and `/kaggle/working` layout so `os`, `plt`, `sns`, and `Image` are always defined before use, eliminating the `NameError`s. I keep your model architecture and training loop intact, only updating deprecated/removed API calls (`fit_generator` → `fit`) and ensuring the checkpoint filename is valid. Finally, I guarantee a correct `submission.csv` (id-aligned, right columns) is written to the working directory.'

# 9. Code solution

## === cell 0

import os
import zipfile as zip
import shutil

import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

import tf_keras as keras
from tf_keras.models import Model, Sequential, load_model
from tf_keras.layers import Activation, BatchNormalization, Conv2D, Dropout
from tf_keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D
from tf_keras.preprocessing.image import ImageDataGenerator

import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

np.random.seed(42)
keras.utils.set_random_seed(42)

print("Imports OK")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
INPUT_ROOT = "/kaggle/input/aerial-cactus-identification"
WORKING_ROOT = "/kaggle/working"

if not os.path.exists(INPUT_ROOT):
    INPUT_ROOT = "../input/aerial-cactus-identification"
    WORKING_ROOT = "../working"

print("INPUT_ROOT:", INPUT_ROOT)
print("WORKING_ROOT:", WORKING_ROOT)

os.makedirs(WORKING_ROOT, exist_ok=True)

train_zip = os.path.join(INPUT_ROOT, "train.zip")
test_zip = os.path.join(INPUT_ROOT, "test.zip")

train_dir = os.path.join(WORKING_ROOT, "train")
test_dir = os.path.join(WORKING_ROOT, "test")

if not os.path.isdir(train_dir):
    with zip.ZipFile(train_zip, "r") as zf:
        zf.extractall(WORKING_ROOT)

if not os.path.isdir(test_dir):
    with zip.ZipFile(test_zip, "r") as zf:
        zf.extractall(WORKING_ROOT)

print("Train images:", len(os.listdir(train_dir)))
print("Test images:", len(os.listdir(test_dir)))
print("Example train file:", os.listdir(train_dir)[0])
print("Example test file:", os.listdir(test_dir)[0])



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/396501857.py in <cell line: 0>()
     28         zf.extractall(WORKING_ROOT)
     29 
---> 30 print("Train images:", len(os.listdir(train_dir)))
     31 print("Test images:", len(os.listdir(test_dir)))
     32 print("Example train file:", os.listdir(train_dir)[0])

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train'

## === cell 2
TRAIN_DATA_PATH = train_dir
TEST_DATA_PATH = test_dir

exemplo = os.path.join(TRAIN_DATA_PATH, os.listdir(TRAIN_DATA_PATH)[0])
print("Example path:", exemplo)
print("Example shape:", np.array(Image.open(exemplo)).shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1884698915.py in <cell line: 0>()
      2 TEST_DATA_PATH = test_dir
      3 
----> 4 exemplo = os.path.join(TRAIN_DATA_PATH, os.listdir(TRAIN_DATA_PATH)[0])
      5 print("Example path:", exemplo)
      6 print("Example shape:", np.array(Image.open(exemplo)).shape)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train'

## === cell 3
df_train = pd.read_csv(os.path.join(INPUT_ROOT, "train.csv"))
df_test = pd.read_csv(os.path.join(INPUT_ROOT, "sample_submission.csv"))

print(df_train.shape, df_test.shape)
display(df_train.head(5))




## === cell 4
def plot_roc_auc(truelabel, pred):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(truelabel, pred)
    auc = sklearn.metrics.auc(fpr, tpr)
    print("AUC:", auc)

    plt.plot(fpr, tpr, label="ROC curve (auc = %.6f)" % auc)
    plt.fill_between(x=fpr, y1=tpr, facecolor="yellow", alpha=0.5)
    plt.legend()
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.show()
    return auc




## === cell 5
fig, ax = plt.subplots(4, 8, figsize=(12, 6))
for i in range(32):
    ax[i // 8][i % 8].tick_params(
        labelbottom=False, labelleft=False, bottom=False, left=False
    )
    target = df_train.iloc[i]["has_cactus"]
    ax[i // 8][i % 8].set_title(f"{i} -> {target}")
    ax[i // 8][i % 8].imshow(
        np.array(Image.open(os.path.join(TRAIN_DATA_PATH, df_train.iloc[i]["id"])))
    )
plt.tight_layout()
plt.show()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3238173771.py in <cell line: 0>()
      8     ax[i // 8][i % 8].set_title(f"{i} -> {target}")
      9     ax[i // 8][i % 8].imshow(
---> 10         np.array(Image.open(os.path.join(TRAIN_DATA_PATH, df_train.iloc[i]["id"])))
     11     )
     12 plt.tight_layout()

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/2de8f189f1dce439766637e75df0ee27.jpg'

## === cell 6
sns.countplot(x=df_train.has_cactus)
plt.show()
print(
    "target 0:1->",
    len(df_train[df_train.has_cactus == 0]) / len(df_train[df_train.has_cactus == 1]),
)



## === cell 7
tmp = []
for i in range(len(df_train)):
    tmp.append(
        np.array(Image.open(os.path.join(TRAIN_DATA_PATH, df_train.iloc[i]["id"])))
    )
X_train = np.array(tmp, dtype=np.float32) / 255.0
y_train = df_train["has_cactus"].astype(np.float32).values
del tmp

tmp = []
for i in range(len(df_test)):
    tmp.append(
        np.array(Image.open(os.path.join(TEST_DATA_PATH, df_test.iloc[i]["id"])))
    )
X_test = np.array(tmp, dtype=np.float32) / 255.0
del tmp

print(X_train.shape, y_train.shape)
print(X_test.shape)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/356429928.py in <cell line: 0>()
      3 for i in range(len(df_train)):
      4     tmp.append(
----> 5         np.array(Image.open(os.path.join(TRAIN_DATA_PATH, df_train.iloc[i]["id"])))
      6     )
      7 X_train = np.array(tmp, dtype=np.float32) / 255.0

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/2de8f189f1dce439766637e75df0ee27.jpg'

## === cell 8
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(input_shape)))
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

    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model




## === cell 9
print(
    1.0
    / len(df_train[df_train.has_cactus == 0])
    * len(df_train[df_train.has_cactus == 1])
)



## === cell 10
histories = []
oof_pred = np.zeros(len(df_train), dtype=np.float32)
sub_pred = np.zeros(len(df_test), dtype=np.float32)

class_weights = {}
weights = [3.010082493125573, 1.0]
for i in range(2):
    class_weights[i] = weights[i]
print("class_weights:", class_weights)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
BATCH_SIZE = 64
EPOCHS = 5

for fold_id, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    print(f"fold id: {fold_id}")
    X_tr, y_tr = X_train[train_index], y_train[train_index]
    X_val, y_val = X_train[val_index], y_train[val_index]

    checkpoint_name = os.path.join(WORKING_ROOT, f"checkpoint_fold{fold_id}.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_name,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.7, patience=5, verbose=1, min_delta=0.00005
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=False
        ),
    ]

    datagen = ImageDataGenerator(height_shift_range=0.1, horizontal_flip=True)

    keras.backend.clear_session()
    model = build_model(X_train.shape[1:])
    model.summary()

    histories.append(
        model.fit(
            datagen.flow(X_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True),
            steps_per_epoch=int(np.ceil(len(X_tr) / BATCH_SIZE)),
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2,
        )
    )

    model = load_model(checkpoint_name)
    oof_pred[val_index] = (
        model.predict(X_val, batch_size=256, verbose=0).flatten().astype(np.float32)
    )
    sub_pred += (
        model.predict(X_test, batch_size=256, verbose=0).flatten().astype(np.float32)
        / skf.n_splits
    )

    fold_auc = roc_auc_score(y_val, oof_pred[val_index])
    print("Fold AUC:", fold_auc)
    plot_roc_auc(y_val, oof_pred[val_index])

sub_pred = np.clip(sub_pred, 0.0, 1.0)

overall_auc = roc_auc_score(y_train, oof_pred)
print("OOF AUC:", overall_auc)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1471211388.py in <cell line: 0>()
     15 EPOCHS = 5
     16 
---> 17 for fold_id, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
     18     print(f"fold id: {fold_id}")
     19     X_tr, y_tr = X_train[train_index], y_train[train_index]

NameError: name 'X_train' is not defined

## === cell 11
print(sub_pred.min(), sub_pred.max())



## === cell 12
plt.hist(sub_pred, bins=50)
plt.title("Submission prediction distribution")
plt.show()

plt.title("count (between 0.80 - 0.20)")
plt.hist(sub_pred[(sub_pred < 0.80) & (sub_pred > 0.20)], bins=100)
plt.show()

plt.title("count (between 0.70 - 0.30)")
plt.hist(sub_pred[(sub_pred < 0.70) & (sub_pred > 0.30)], bins=100)
plt.show()

plt.title("count (between 0.60 - 0.40)")
plt.hist(sub_pred[(sub_pred < 0.60) & (sub_pred > 0.40)], bins=100)
plt.show()



## === cell 13
print("Ambiguous image index:", np.where((sub_pred < 0.80) & (sub_pred > 0.20))[0][:32])



## === cell 14
submission = pd.read_csv(os.path.join(INPUT_ROOT, "sample_submission.csv"))
submission["has_cactus"] = sub_pred.astype(float)

submission = submission[["id", "has_cactus"]]
submission.to_csv("submission.csv", index=False)

print(submission.head(10))
print("Wrote submission.csv with shape:", submission.shape)

if os.path.isdir(train_dir):
    shutil.rmtree(train_dir, ignore_errors=True)
if os.path.isdir(test_dir):
    shutil.rmtree(test_dir, ignore_errors=True)

print("Working dir contents:", os.listdir(WORKING_ROOT))
print("Local cwd contents:", os.listdir("."))
