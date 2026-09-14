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

3.9

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

0.984

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
WORK_DIR = "/kaggle/working"
INPUT_DIR = "/kaggle/input/aerial-cactus-identification"

train_zip = os.path.join(INPUT_DIR, "train.zip")
test_zip = os.path.join(INPUT_DIR, "test.zip")
train_csv_path = os.path.join(INPUT_DIR, "train.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

extract_root = os.path.join(WORK_DIR, "extracted_cactus")
train_extract_dir = os.path.join(extract_root, "train")
test_extract_dir = os.path.join(extract_root, "test")


def _ensure_extracted(zip_path: str, extract_root_dir: str):
    """Extract zip into extract_root_dir if not already extracted."""
    os.makedirs(extract_root_dir, exist_ok=True)
    marker = os.path.join(extract_root_dir, ".extracted_ok")
    if os.path.exists(marker) and len(os.listdir(extract_root_dir)) > 1:
        return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_root_dir)
    with open(marker, "w") as f:
        f.write("ok")


def _find_image_dir(root: str, target_basename: str) -> str:
    """
    Bugfix: reliably locate the folder that actually contains images for the given split.
    The original resolver could return the wrong folder (e.g., train_dir -> test folder).
    """
    candidate = os.path.join(root, target_basename)
    if os.path.isdir(candidate) and any(
        f.lower().endswith(".jpg") for f in os.listdir(candidate)
    ):
        return candidate

    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == target_basename and any(
            f.lower().endswith(".jpg") for f in filenames
        ):
            return dirpath

    raise RuntimeError(
        f"Could not find '{target_basename}' image directory under: {root}"
    )


_ensure_extracted(train_zip, extract_root)
_ensure_extracted(test_zip, extract_root)

train_dir = _find_image_dir(extract_root, "train")
test_dir = _find_image_dir(extract_root, "test")

print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)

train_df = pd.read_csv(train_csv_path)
path_ids = train_df["id"].astype(str).tolist()
labels = train_df["has_cactus"].astype(int).tolist()


def _load_image_32_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (32, 32):
            im = im.resize((32, 32), resample=Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


x_train_0, x_train_1, y_train_0, y_train_1 = [], [], [], []
missing_train = 0

for img_id, y in zip(path_ids, labels):
    img_path = os.path.join(train_dir, img_id)
    if not os.path.exists(img_path):
        missing_train += 1
        continue
    data_img = _load_image_32_rgb(img_path)
    if int(y) == 0:
        x_train_0.append(data_img)
        y_train_0.append(y)
    else:
        x_train_1.append(data_img)
        y_train_1.append(y)

if missing_train:
    print(
        f"Warning: missing {missing_train} training images (unexpected). Proceeding with found images."
    )

taille = min(len(x_train_0), len(x_train_1))
x_train = np.array(x_train_0[:taille] + x_train_1[:taille], dtype=np.uint8)
y_train = np.array(y_train_0[:taille] + y_train_1[:taille], dtype=np.int64)

if x_train.shape[0] == 0:
    raise RuntimeError(
        f"No training images were loaded. Check extraction and paths. "
        f"Resolved train_dir={train_dir} (exists={os.path.isdir(train_dir)})"
    )

sub_df = pd.read_csv(sample_sub_path)
test_ids = sub_df["id"].astype(str).tolist()

x_test = np.empty((len(test_ids), 32, 32, 3), dtype=np.uint8)
missing = 0
for i, fname in enumerate(test_ids):
    img_path = os.path.join(test_dir, fname)
    if not os.path.exists(img_path):
        missing += 1
        continue
    x_test[i] = _load_image_32_rgb(img_path)

if missing > 0:
    raise RuntimeError(
        f"Missing {missing} test images under resolved test_dir={test_dir}. "
        f"Check extraction and paths."
    )

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes=2)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.30, random_state=42, shuffle=True
)

print(
    "Train:",
    x_train.shape,
    y_train.shape,
    "\nVal  :",
    x_val.shape,
    y_val.shape,
    "\nTest :",
    x_test.shape,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_12/898266804.py in <cell line: 0>()
     53 _ensure_extracted(test_zip, extract_root)
     54 
---> 55 train_dir = _find_image_dir(extract_root, "train")
     56 test_dir = _find_image_dir(extract_root, "test")
     57 

/tmp/ipykernel_12/898266804.py in _find_image_dir(root, target_basename)
     44             return dirpath
     45 
---> 46     raise RuntimeError(
     47         f"Could not find '{target_basename}' image directory under: {root}"
     48     )

RuntimeError: Could not find 'train' image directory under: /kaggle/working/extracted_cactus

## === cell 2
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)



## === cell 3
model = Sequential()

model.add(
    Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:], activation="relu")
)
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(2, activation="softmax"))
model.summary()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/960831720.py in <cell line: 0>()
      2 
      3 model.add(
----> 4     Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:], activation="relu")
      5 )
      6 model.add(Conv2D(32, (3, 3), activation="relu"))

NameError: name 'x_train' is not defined

## === cell 4
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=0.001,
)

ckpt_path = os.path.join(WORK_DIR, "model.keras")
checkpointer = ModelCheckpoint(filepath=ckpt_path, verbose=1, save_best_only=True)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=250),
    epochs=70,
    validation_data=(x_val, y_val),
    callbacks=[reduce_lr, checkpointer],
    verbose=2,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/96778291.py in <cell line: 0>()
     14 
     15 history = model.fit(
---> 16     datagen.flow(x_train, y_train, batch_size=250),
     17     epochs=70,
     18     validation_data=(x_val, y_val),

NameError: name 'x_train' is not defined

## === cell 5
def plot_history(history_obj):
    plt.plot(history_obj.history["accuracy"])
    plt.plot(history_obj.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()

    plt.plot(history_obj.history["loss"])
    plt.plot(history_obj.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


plot_history(history)

if os.path.exists(ckpt_path):
    best_model = keras.models.load_model(ckpt_path)
else:
    best_model = model

val_metrics = best_model.evaluate(x_val, y_val, verbose=0)
print("Validation:", dict(zip(best_model.metrics_names, val_metrics)))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2185945821.py in <cell line: 0>()
     17 
     18 
---> 19 plot_history(history)
     20 
     21 if os.path.exists(ckpt_path):

NameError: name 'history' is not defined

## === cell 6
proba = best_model.predict(x_test, batch_size=512, verbose=0)
has_cactus_proba = proba[:, 1].astype(float)

submission = pd.DataFrame({"id": test_ids, "has_cactus": has_cactus_proba})
submission_path = os.path.join(WORK_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Submission shape:", submission.shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3402329388.py in <cell line: 0>()
----> 1 proba = best_model.predict(x_test, batch_size=512, verbose=0)
      2 has_cactus_proba = proba[:, 1].astype(float)
      3 
      4 submission = pd.DataFrame({"id": test_ids, "has_cactus": has_cactus_proba})
      5 submission_path = os.path.join(WORK_DIR, "submission.csv")

NameError: name 'best_model' is not defined

## === cell 7
print("Working dir files:", sorted(os.listdir(WORK_DIR))[:50])

if os.path.isdir(extract_root):
    shutil.rmtree(extract_root, ignore_errors=True)
print("Cleanup done. Kept:", submission_path)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/972691405.py in <cell line: 0>()
      5 if os.path.isdir(extract_root):
      6     shutil.rmtree(extract_root, ignore_errors=True)
----> 7 print("Cleanup done. Kept:", submission_path)

NameError: name 'submission_path' is not defined
