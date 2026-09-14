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

0.913

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import subprocess
import os

subprocess.run(
    "cp -f /kaggle/input/aerial-cactus-identification/train.csv /kaggle/working/train.csv",
    shell=True,
    check=True,
)

subprocess.run(
    "unzip -o /kaggle/input/aerial-cactus-identification/train.zip -d /kaggle/working",
    shell=True,
    check=True,
)
subprocess.run(
    "unzip -o /kaggle/input/aerial-cactus-identification/test.zip -d /kaggle/working",
    shell=True,
    check=True,
)

print(
    "Working dir contents:",
    sorted(
        [
            p
            for p in os.listdir("/kaggle/working")
            if p in ["train", "test", "train.csv"]
        ]
    ),
)
print(
    "train exists:",
    os.path.isdir("/kaggle/working/train"),
    "n_files:",
    (
        len(os.listdir("/kaggle/working/train"))
        if os.path.isdir("/kaggle/working/train")
        else 0
    ),
)
print(
    "test  exists:",
    os.path.isdir("/kaggle/working/test"),
    "n_files:",
    (
        len(os.listdir("/kaggle/working/test"))
        if os.path.isdir("/kaggle/working/test")
        else 0
    ),
)



## === cell 2
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split

print("tf version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
df = pd.read_csv("/kaggle/working/train.csv")
display(df.sample(3))
df.has_cactus.value_counts().plot.bar()
plt.show()



## === cell 4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img

filename = df.id.iloc[10]
print("Example file:", filename)

train_img_path = os.path.join("/kaggle/working/train", filename)
if not os.path.isfile(train_img_path):
    raise FileNotFoundError(f"Expected training image missing at: {train_img_path}")

image = load_img(train_img_path)
plt.imshow(image)
plt.axis("off")
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1050771504.py in <cell line: 0>()
      8 if not os.path.isfile(train_img_path):
      9     # Fix: provide a clearer error if unzip path is wrong.
---> 10     raise FileNotFoundError(f"Expected training image missing at: {train_img_path}")
     11 
     12 image = load_img(train_img_path)

FileNotFoundError: Expected training image missing at: /kaggle/working/train/17acf1cbe632db5f8447872f571581ca.jpg

## === cell 5
train_df, validate_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["has_cactus"]
)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)



## === cell 6
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

BATCH_SIZE = 2**10
IMAGE_SIZE = (32, 32)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="/kaggle/working/train",
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
    seed=42,
    validate_filenames=False,
)

validation_generator = valid_datagen.flow_from_dataframe(
    dataframe=validate_df,
    directory="/kaggle/working/train",
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False,
    validate_filenames=False,
)

print("train_generator batches:", len(train_generator))
print("validation_generator batches:", len(validation_generator))
if len(train_generator) == 0 or len(validation_generator) == 0:
    raise ValueError(
        "One of the generators has length 0; check extracted folders and filenames."
    )



## === cell 7
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
        Conv2D(128, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        BatchNormalization(),
        AveragePooling2D(pool_size=(3, 3)),
        Conv2D(256, (2, 2), activation="relu"),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(64, (2, 2), activation="relu"),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),
    ]
)

earlystop = EarlyStopping(patience=5, restore_best_weights=True)
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
callbacks = [earlystop]

model.summary()



## === cell 8
history = model.fit(
    train_generator, epochs=4, validation_data=validation_generator, callbacks=callbacks
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3945836856.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_generator, epochs=4, validation_data=validation_generator, callbacks=callbacks
      3 )
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/e08037abe1e49ea0a7083f9ea224e65a.jpg'

## === cell 9
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/206874453.py in <cell line: 0>()
----> 1 pd.DataFrame(history.history).plot()
      2 plt.grid(True)
      3 plt.show()
      4 

NameError: name 'history' is not defined

## === cell 10
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)

test_dir = "/kaggle/working/test"
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Missing test directory at {test_dir}")

test_df = sample_sub[["id"]].copy()

test_gen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_gen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validate_filenames=False,
)

pred = model.predict(test_generator, verbose=1)
test_df["has_cactus"] = pred.reshape(-1)

display(test_df.sample(5))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1416700894.py in <cell line: 0>()
      5 test_dir = "/kaggle/working/test"
      6 if not os.path.isdir(test_dir):
----> 7     raise FileNotFoundError(f"Missing test directory at {test_dir}")
      8 
      9 test_df = sample_sub[["id"]].copy()

FileNotFoundError: Missing test directory at /kaggle/working/test

## === cell 11
pred



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/576945016.py in <cell line: 0>()
----> 1 pred
      2 

NameError: name 'pred' is not defined

## === cell 12
np.transpose(pred)[0]



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/769546132.py in <cell line: 0>()
----> 1 np.transpose(pred)[0]
      2 

NameError: name 'pred' is not defined

## === cell 13
test_df.has_cactus.max()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3020247713.py in <cell line: 0>()
----> 1 test_df.has_cactus.max()
      2 

NameError: name 'test_df' is not defined

## === cell 14
submission = test_df[["id", "has_cactus"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1962030155.py in <cell line: 0>()
----> 1 submission = test_df[["id", "has_cactus"]].copy()
      2 submission.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", submission.shape)
      4 print(submission.head())
      5 

NameError: name 'test_df' is not defined

## === cell 15
import os

print("Files in /kaggle/working:", sorted(os.listdir("/kaggle/working"))[:50])



## === cell 16
submission.head()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2023294942.py in <cell line: 0>()
----> 1 submission.head()
      2 

NameError: name 'submission' is not defined

## === cell 17
submission.has_cactus.describe()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1555286387.py in <cell line: 0>()
----> 1 submission.has_cactus.describe()
      2 

NameError: name 'submission' is not defined

## === cell 18
import shutil
import os

for p in ["/kaggle/working/train", "/kaggle/working/test", "/kaggle/working/train.csv"]:
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    elif os.path.isfile(p):
        try:
            os.remove(p)
        except OSError:
            pass
print(
    "Cleanup done. submission.csv remains:",
    os.path.isfile("/kaggle/working/submission.csv"),
)
