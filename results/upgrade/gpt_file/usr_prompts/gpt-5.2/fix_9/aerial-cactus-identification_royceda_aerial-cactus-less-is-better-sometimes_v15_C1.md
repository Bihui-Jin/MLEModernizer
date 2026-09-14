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

0.9057

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the environment/runtime issues caused by mixing `keras` (Keras 3) APIs with TensorFlow/Keras generators by switching to `tf.keras` equivalents that still provide `ImageDataGenerator`. I also remove the hard GPU requirement (so it runs whether GPU is available or not), correct the unzip/copy commands, and make paths robust so `train/` and `test/` are found. Finally, I ensure the submission is created from `sample_submission.csv` so it has the correct `id` column/order and writes a valid `submission.csv` with `id,has_cactus`.'
- What this solution (achieved 0.5) has done: 'I fix the dataset path detection so it correctly finds `train.csv`, `train.zip`, `test.zip`, and `sample_submission.csv` in your provided filesystem (the current candidate path points one level too deep). I also address the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves this common Kaggle runtime issue. Then I keep your exact model/training logic intact, but ensure all dependent variables are defined by making the data-prep cell succeed so later cells don’t cascade into `NameError`. Finally, I guarantee a valid `/kaggle/working/submission.csv` in the required `id,has_cactus` format and aligned to `sample_submission.csv` order.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import shutil
import zipfile

WORKDIR = "/kaggle/working"

CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/input",
]


def _is_dataset_root(p: str) -> bool:
    return (
        os.path.isdir(p)
        and os.path.exists(os.path.join(p, "train.csv"))
        and os.path.exists(os.path.join(p, "train.zip"))
        and os.path.exists(os.path.join(p, "test.zip"))
        and os.path.exists(os.path.join(p, "sample_submission.csv"))
    )


INPUT_DIR = next((p for p in CANDIDATES if _is_dataset_root(p)), None)
if INPUT_DIR is None:
    found = None
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if _is_dataset_root(dirpath):
            found = dirpath
            break
    INPUT_DIR = found

if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not find dataset root with required files under /kaggle/input. "
        f"Tried candidates: {CANDIDATES}"
    )

train_csv_src = os.path.join(INPUT_DIR, "train.csv")
train_zip_src = os.path.join(INPUT_DIR, "train.zip")
test_zip_src = os.path.join(INPUT_DIR, "test.zip")
sample_sub_src = os.path.join(INPUT_DIR, "sample_submission.csv")

for p in [train_csv_src, train_zip_src, test_zip_src, sample_sub_src]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")

print("Using INPUT_DIR:", INPUT_DIR)

os.makedirs(WORKDIR, exist_ok=True)

EXTRACT_DIR = os.path.join(WORKDIR, "_cactus_extracted")
if os.path.isdir(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

shutil.copy(train_csv_src, os.path.join(WORKDIR, "train.csv"))

with zipfile.ZipFile(train_zip_src, "r") as z:
    z.extractall(EXTRACT_DIR)
with zipfile.ZipFile(test_zip_src, "r") as z:
    z.extractall(EXTRACT_DIR)

print("Top-level EXTRACT_DIR contents:", sorted(os.listdir(EXTRACT_DIR))[:30])


def _find_dir_with_jpgs(root: str) -> str:
    """
    Locate the directory under `root` that contains the most .jpg files.
    Handles possible zip structures like:
      EXTRACT_DIR/train/*.jpg
      EXTRACT_DIR/train/train/*.jpg
      EXTRACT_DIR/aerial-cactus-identification/train/*.jpg
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"Root not found: {root}")

    best = None
    best_count = -1
    for dirpath, _, filenames in os.walk(root):
        cnt = sum(1 for f in filenames if f.lower().endswith(".jpg"))
        if cnt > best_count:
            best_count = cnt
            best = dirpath

    if best is None or best_count <= 0:
        raise FileNotFoundError(f"Could not find any .jpg files under {root}")

    return best


TRAIN_DIR = _find_dir_with_jpgs(os.path.join(EXTRACT_DIR, "train"))
TEST_DIR = _find_dir_with_jpgs(os.path.join(EXTRACT_DIR, "test"))

print(
    "Resolved TRAIN_DIR:",
    TRAIN_DIR,
    "sample:",
    sorted([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")])[:3],
)
print(
    "Resolved TEST_DIR :",
    TEST_DIR,
    "sample:",
    sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])[:3],
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4467694.py in <cell line: 0>()
     94 
     95 
---> 96 TRAIN_DIR = _find_dir_with_jpgs(os.path.join(EXTRACT_DIR, "train"))
     97 TEST_DIR = _find_dir_with_jpgs(os.path.join(EXTRACT_DIR, "test"))
     98 

/tmp/ipykernel_11/4467694.py in _find_dir_with_jpgs(root)
     78     """
     79     if not os.path.exists(root):
---> 80         raise FileNotFoundError(f"Root not found: {root}")
     81 
     82     best = None

FileNotFoundError: Root not found: /kaggle/working/_cactus_extracted/train

## === cell 2
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import matplotlib.pyplot as plt
import tensorflow as tf

print("tf version :", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

tf.random.set_seed(42)
np.random.seed(42)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
df = pd.read_csv("/kaggle/working/train.csv")
print(df.sample(3, random_state=42))
df.has_cactus.value_counts().plot.bar()
plt.grid(True)
plt.show()



## === cell 4
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

filename = df.id.iloc[10]
print(filename)
image = load_img(os.path.join(TRAIN_DIR, filename))
plt.imshow(image)
plt.axis("off")
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2268777273.py in <cell line: 0>()
      3 filename = df.id.iloc[10]
      4 print(filename)
----> 5 image = load_img(os.path.join(TRAIN_DIR, filename))
      6 plt.imshow(image)
      7 plt.axis("off")

NameError: name 'TRAIN_DIR' is not defined

## === cell 5
from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["has_cactus"]
)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)



## === cell 6
train_datagen = ImageDataGenerator(
    rotation_range=45,
    rescale=1.0 / 255,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)



## === cell 7
BATCH_SIZE = 64
IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (32, 32, 3)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
    seed=42,
)

validation_generator = valid_datagen.flow_from_dataframe(
    dataframe=validate_df,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False,
)

print("train steps:", len(train_generator), "valid steps:", len(validation_generator))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/62181986.py in <cell line: 0>()
      5 train_generator = train_datagen.flow_from_dataframe(
      6     dataframe=train_df,
----> 7     directory=TRAIN_DIR,
      8     x_col="id",
      9     y_col="has_cactus",

NameError: name 'TRAIN_DIR' is not defined

## === cell 8
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    AveragePooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential(
    [
        Conv2D(
            filters=64,
            kernel_size=(4, 4),
            strides=(1, 1),
            activation="relu",
            input_shape=INPUT_SHAPE,
            padding="same",
        ),
        BatchNormalization(),
        AveragePooling2D(pool_size=(3, 3)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dropout(0.45),
        Dense(1, activation="sigmoid"),
    ]
)

earlystop = EarlyStopping(patience=4, restore_best_weights=True)
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.summary()



## === cell 9
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[earlystop],
    verbose=2,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/731527035.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_generator,
      3     epochs=30,
      4     validation_data=validation_generator,
      5     callbacks=[earlystop],

NameError: name 'train_generator' is not defined

## === cell 10
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/206874453.py in <cell line: 0>()
----> 1 pd.DataFrame(history.history).plot()
      2 plt.grid(True)
      3 plt.show()
      4 

NameError: name 'history' is not defined

## === cell 11
sample_sub_path = sample_sub_src
sub = pd.read_csv(sample_sub_path)
print(sub.head(), "rows:", len(sub))

test_files = set([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
missing = [i for i in sub["id"].tolist() if i not in test_files]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} ids from sample_submission.csv were not found in TEST_DIR={TEST_DIR}. "
        f"Example missing: {missing[:5]}"
    )



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1068570203.py in <cell line: 0>()
      4 print(sub.head(), "rows:", len(sub))
      5 
----> 6 test_files = set([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
      7 missing = [i for i in sub["id"].tolist() if i not in test_files]
      8 if missing:

NameError: name 'TEST_DIR' is not defined

## === cell 12
test_gen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_gen.flow_from_dataframe(
    sub,
    directory=TEST_DIR,
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("test steps:", len(test_generator))

pred = model.predict(test_generator, steps=len(test_generator), verbose=0).reshape(-1)
pred = pred[: len(sub)]

if len(pred) != len(sub):
    raise RuntimeError(f"Prediction length {len(pred)} != submission length {len(sub)}")

sub["has_cactus"] = pred.astype(np.float32)
print(sub.head(), "rows:", len(sub))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3032699947.py in <cell line: 0>()
      2 test_generator = test_gen.flow_from_dataframe(
      3     sub,
----> 4     directory=TEST_DIR,
      5     x_col="id",
      6     y_col=None,

NameError: name 'TEST_DIR' is not defined

## === cell 13
assert list(sub.columns) == ["id", "has_cactus"]
assert sub["id"].isna().sum() == 0
assert sub["has_cactus"].isna().sum() == 0
assert len(sub) == pd.read_csv(sample_sub_path).shape[0]

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub))



## === cell 14
print(sub["has_cactus"].describe())



## === cell 15
pass
