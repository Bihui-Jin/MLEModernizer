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
protobuf==6.33.0
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

0.592

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = "unknown"


def _major(ver):
    try:
        return int(str(ver).split(".")[0])
    except Exception:
        return None


if _major(_pb_ver) is None or _major(_pb_ver) >= 5:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])



## === cell 1
import pandas as pd

BASE = "../input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")


def _resolve_dir(base, *candidates):
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.isdir(p):
            return p
    return os.path.join(base, candidates[0])


TRAIN_DIR = _resolve_dir(BASE, "train/train", "train")
TEST_DIR = _resolve_dir(BASE, "test/test", "test")

train_ds = pd.read_csv(TRAIN_CSV, dtype={"id": str})
train_ds["has_cactus"] = train_ds["has_cactus"].astype(str)

print("Resolved TRAIN_DIR:", TRAIN_DIR)
print("Resolved TEST_DIR:", TEST_DIR)
print(train_ds.head())



## === cell 2
train_dir = TRAIN_DIR
print("train_dir:", train_dir)
print("total training images:", len(os.listdir(train_dir)))

train_files = os.listdir(train_dir)
print(train_files[:10])

if len(train_files) == 0:
    raise RuntimeError(
        f"No training images found in {train_dir}. Check TRAIN_DIR resolution."
    )



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/41158516.py in <cell line: 0>()
      8 # Sanity check to prevent zero-length generators later
      9 if len(train_files) == 0:
---> 10     raise RuntimeError(
     11         f"No training images found in {train_dir}. Check TRAIN_DIR resolution."
     12     )

RuntimeError: No training images found in ../input/aerial-cactus-identification/train/train. Check TRAIN_DIR resolution.

## === cell 3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2
next_img = [
    os.path.join(train_dir, fname) for fname in train_files[pic_index - 2 : pic_index]
]

for img_path in next_img:
    print(img_path)
    img = mpimg.imread(img_path)
    print(img.shape)
    plt.imshow(img)
    plt.axis("off")
    plt.show()



## === cell 4
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)
np.random.seed(42)

TRAINING_DIR = TRAIN_DIR

training_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.25,
)

train_generator = training_datagen.flow_from_dataframe(
    dataframe=train_ds,
    directory=TRAINING_DIR,
    shuffle=True,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    class_mode="categorical",
    subset="training",
    batch_size=32,
)

validation_generator = training_datagen.flow_from_dataframe(
    dataframe=train_ds,
    directory=TRAINING_DIR,
    shuffle=True,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    class_mode="categorical",
    subset="validation",
    batch_size=32,
)

if train_generator.samples == 0:
    raise RuntimeError(
        "train_generator has 0 samples. Check TRAINING_DIR and that filenames in train.csv exist."
    )
if validation_generator.samples == 0:
    raise RuntimeError(
        "validation_generator has 0 samples. Check TRAINING_DIR and validation_split."
    )

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

model.summary()
model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

steps_per_epoch = int(np.ceil(train_generator.samples / train_generator.batch_size))
val_steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))

history = model.fit(
    train_generator,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=val_steps,
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1405038028.py in <cell line: 0>()
     47 # Bugfix guard: fail early if misconfigured paths create empty datasets
     48 if train_generator.samples == 0:
---> 49     raise RuntimeError(
     50         "train_generator has 0 samples. Check TRAINING_DIR and that filenames in train.csv exist."
     51     )

RuntimeError: train_generator has 0 samples. Check TRAINING_DIR and that filenames in train.csv exist.

## === cell 5
import matplotlib.pyplot as plt

acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

epochs = range(len(acc))

plt.plot(epochs, acc, "r", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, "r", label="Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and validation Loss")
plt.legend(loc=0)
plt.figure()
plt.show()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3298707243.py in <cell line: 0>()
      1 import matplotlib.pyplot as plt
      2 
----> 3 acc = history.history.get("accuracy", [])
      4 val_acc = history.history.get("val_accuracy", [])
      5 loss = history.history.get("loss", [])

NameError: name 'history' is not defined

## === cell 6
testdf = pd.read_csv(SAMPLE_SUB, dtype={"id": str})
testdf["has_cactus"] = testdf["has_cactus"].astype(str)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=TEST_DIR,
    target_size=(32, 32),
    x_col="id",
    y_col=None,
    class_mode=None,
    shuffle=False,
    batch_size=32,
)

if test_generator.samples == 0:
    raise RuntimeError(
        "test_generator has 0 samples. Check TEST_DIR and sample_submission.csv filenames."
    )

test_generator.reset()
pred = model.predict(test_generator, verbose=1)

class_indices = train_generator.class_indices  # e.g. {'0': 0, '1': 1}
pos_index = class_indices.get("1", 1)
has_cactus_prob = pred[:, pos_index].astype(np.float64)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/4129277173.py in <cell line: 0>()
     18 
     19 if test_generator.samples == 0:
---> 20     raise RuntimeError(
     21         "test_generator has 0 samples. Check TEST_DIR and sample_submission.csv filenames."
     22     )

RuntimeError: test_generator has 0 samples. Check TEST_DIR and sample_submission.csv filenames.

## === cell 7
results = pd.DataFrame({"id": testdf["id"].values, "has_cactus": has_cactus_prob})
results["has_cactus"] = results["has_cactus"].clip(0.0, 1.0)

results.to_csv("submission.csv", index=False)
print(results.head())
print("Wrote submission.csv with shape:", results.shape)
print("submission.csv columns:", results.columns.tolist())
if results.columns.tolist() != ["id", "has_cactus"]:
    raise RuntimeError("Submission columns are incorrect.")
if results.isna().any().any():
    raise RuntimeError("Submission contains NaNs.")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4104456357.py in <cell line: 0>()
----> 1 results = pd.DataFrame({"id": testdf["id"].values, "has_cactus": has_cactus_prob})
      2 results["has_cactus"] = results["has_cactus"].clip(0.0, 1.0)
      3 
      4 # Ensure correct filename + extension for Kaggle submission
      5 results.to_csv("submission.csv", index=False)

NameError: name 'has_cactus_prob' is not defined
