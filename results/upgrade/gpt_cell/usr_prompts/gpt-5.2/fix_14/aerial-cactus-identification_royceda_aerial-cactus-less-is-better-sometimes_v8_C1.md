# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.913

# 6. Current score

0.40078

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.40078) has done: 'Diagnosis: The crash happens when assigning predictions to `df["has_cactus"]` because `df` is built from `os.listdir(test_img_dir)` and includes an extra entry (commonly a nested `test/` directory or other non-image file), so `len(df)` becomes 3326 while the generator (which silently drops invalid/non-image rows) yields only 3325 predictions. This creates a length mismatch during the column assignment.  
Patch summary: In cell 11, filter the listed test entries to files only and restrict to image extensions before creating the dataframe, ensuring the dataframe length matches what `flow_from_dataframe` and `model.predict` process. Also flatten `pred` safely to a 1D array before assignment to avoid shape quirks.  
Updated cells: Only cell 11 is changed.  
Compatibility notes for cell k+1: `pred` remains the raw output of `model.predict(test_generator)` and is still available for cell 12 unchanged in meaning; only `df` construction is corrected and `df["has_cactus"]` assignment now succeeds.  
Assumptions: Test images are `.jpg` (as in the provided directory listing) and any extra entry causing the off-by-one is a directory or non-image file that should not be submitted/predicted.'

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
! cp -rf /kaggle/input/aerial-cactus-identification/train.csv -d /kaggle/working
! unzip -o /kaggle/input/aerial-cactus-identification/train.zip -d /kaggle/working
! unzip /kaggle/input/aerial-cactus-identification/test.zip -d /kaggle/working


## === cell 2
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

try:
    from google.protobuf import message_factory as _message_factory

    MF = getattr(_message_factory, "MessageFactory", None)
    if MF is not None:
        try:
            has_get_prototype = hasattr(MF, "GetPrototype")
        except Exception:
            has_get_prototype = False

        if not has_get_prototype:
            try:

                def _GetPrototype(self, descriptor):
                    if hasattr(self, "GetMessageClass"):
                        return self.GetMessageClass(descriptor)
                    raise AttributeError("MessageFactory has no GetMessageClass")

                setattr(MF, "GetPrototype", _GetPrototype)
            except Exception:
                pass
except Exception:
    pass

import tensorflow as tf
import keras

from keras.datasets import mnist
from sklearn.model_selection import train_test_split

print("tf version : ", tf.__version__)

device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    raise SystemError("GPU device not found")

print("Found GPU at: {}".format(device_name))


## === cell 3
df = pd.read_csv('train.csv')
df.sample(3)
df.has_cactus.value_counts().plot.bar()


## === cell 4
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical

candidate_train_dirs = [
    "./train",
    "/kaggle/working/train",
    "/kaggle/working/aerial-cactus-identification/train",
]
train_img_dir = next((d for d in candidate_train_dirs if os.path.isdir(d)), None)
if train_img_dir is None:
    raise FileNotFoundError(
        f"Could not find training image directory. Checked: {candidate_train_dirs}"
    )

filename = df.id.iloc[10]
print(filename)
image_path = os.path.join(train_img_dir, filename)
image = load_img(image_path)

plt.imshow(image)


## === cell 5
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


## === cell 6
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./32,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1

)




## === cell 7
BATCH_SIZE = 128
IMAGE_SIZE = (32,32)

INPUT_SHAPE=(32, 32, 3)
BATCH_SIZE=2**10

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df, 
    directory="./train",
    x_col='id',
    y_col='has_cactus',
    target_size=IMAGE_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode="raw"
)


validation_generator = train_datagen.flow_from_dataframe(
    dataframe=validate_df, 
    directory="./train",
    x_col='id',
    y_col='has_cactus',
    target_size=IMAGE_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode="raw"
)


## === cell 8
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, AveragePooling2D


model = Sequential([
                    Conv2D(128, (3,3), activation='relu', input_shape=(32, 32, 3)),
                    BatchNormalization(),
                    AveragePooling2D( pool_size=(3, 3)), 

                    Conv2D(256, (2, 2), activation='relu'),
                    BatchNormalization(),
                    AveragePooling2D( pool_size=(2, 2)), 
    
                    Conv2D(64, (2, 2), activation='relu'),
                    BatchNormalization(),
                    AveragePooling2D( pool_size=(2, 2)), 
                    Dropout(0.2),

                    Flatten(),
                    Dense(128, activation='relu'),
                    Dropout(0.4),
                    Dense(1, activation='sigmoid')
])


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=5)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
callbacks = [earlystop]


## === cell 9
BATCH_SIZE = 128
IMAGE_SIZE = (32, 32)

INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 2**10

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_img_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=validate_df,
    directory=train_img_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
)


## === cell 10
if "history" in globals() and hasattr(history, "history"):
    pd.DataFrame(history.history).plot()


## === cell 11
candidate_test_dirs = [
    "./test",
    "/kaggle/working/test",
    "/kaggle/working/aerial-cactus-identification/test",
]
test_img_dir = next((d for d in candidate_test_dirs if os.path.isdir(d)), None)
if test_img_dir is None:
    raise FileNotFoundError(
        f"Could not find test image directory. Checked: {candidate_test_dirs}"
    )

test_ids = []
for name in os.listdir(test_img_dir):
    full_path = os.path.join(test_img_dir, name)
    if os.path.isfile(full_path) and name.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):
        test_ids.append(name)

df = pd.DataFrame()
df["id"] = test_ids
df.head()

from keras.preprocessing import image_dataset_from_directory

test_gen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_gen.flow_from_dataframe(
    df,
    test_img_dir,
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

pred = model.predict(test_generator)

df["has_cactus"] = np.asarray(pred).reshape(-1)
df.sample(5)


## === cell 12
pred


## === cell 13
np.transpose(pred)[0]


## === cell 14
df.has_cactus.max()


## === cell 15
submission = df.copy()
submission.to_csv('submission.csv', index=False)


## === cell 16
! ls ../


## === cell 17
submission.head()


## === cell 18
submission.has_cactus.describe()


## === cell 19
! rm -rf train test train.csv
