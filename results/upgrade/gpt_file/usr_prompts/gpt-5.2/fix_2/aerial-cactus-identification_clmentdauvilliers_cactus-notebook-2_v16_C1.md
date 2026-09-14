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
protobuf==6.33.0
scikit-image==0.25.2
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

0.951

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import random

random.seed(42)
np.random.seed(42)

DATA_PATH = "/kaggle/input/aerial-cactus-identification/"
TRAIN_ZIP = os.path.join(DATA_PATH, "train.zip")
TEST_ZIP = os.path.join(DATA_PATH, "test.zip")
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
SAMPLE_SUB = os.path.join(DATA_PATH, "sample_submission.csv")

TRAIN_EXTRACT_DIR = "./training/"
TEST_EXTRACT_DIR = "./test/"

os.makedirs(TRAIN_EXTRACT_DIR, exist_ok=True)
os.makedirs(TEST_EXTRACT_DIR, exist_ok=True)

print("Listing a few input files for sanity:")
for dirname, _, filenames in os.walk("/kaggle/input/aerial-cactus-identification/"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
files_dataframe = pd.read_csv(TRAIN_CSV, dtype={"id": str, "has_cactus": str})
files_dataframe.head()



## === cell 3
if not os.path.exists(os.path.join(TRAIN_EXTRACT_DIR, "train")):
    with ZipFile(TRAIN_ZIP, "r") as zipper:
        zipper.extractall(TRAIN_EXTRACT_DIR)

if not os.path.exists(os.path.join(TEST_EXTRACT_DIR, "test")):
    with ZipFile(TEST_ZIP, "r") as zipper:
        zipper.extractall(TEST_EXTRACT_DIR)

training_files = "train/" + files_dataframe["id"]
print("Training sample:")
print(training_files.head(2))

print("Train dir exists:", os.path.exists(os.path.join(TRAIN_EXTRACT_DIR, "train")))
print("Test dir exists:", os.path.exists(os.path.join(TEST_EXTRACT_DIR, "test")))



## === cell 4
import matplotlib.pyplot as plt

class_reparts = files_dataframe["has_cactus"].value_counts()
ax = class_reparts.plot.bar()



## === cell 5
total_samples = files_dataframe["has_cactus"].size
print("Total number of samples: ", total_samples)
has_cactus_weight = total_samples / (2 * class_reparts["1"])
no_cactus_weight = total_samples / (2 * class_reparts["0"])
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights: ", class_weights)



## === cell 6
from matplotlib.image import imread

plt.figure(figsize=(36, 12))
base_train_dir = os.path.join(TRAIN_EXTRACT_DIR, "train")
for i, k in enumerate(np.random.randint(0, len(files_dataframe), size=(20,))):
    p = os.path.join(base_train_dir, files_dataframe["id"].iloc[k])
    if os.path.exists(p):
        plt.subplot(4, 5, i + 1)
        plt.imshow(imread(p))
        plt.title("Label :" + str(files_dataframe["has_cactus"].iloc[k]))



## === cell 7
import skimage.exposure as exposure


def preprocess(img):
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale


plt.figure(figsize=(12, 4))
p0 = os.path.join(base_train_dir, files_dataframe["id"].iloc[0])
if os.path.exists(p0):
    plt.subplot(121)
    img = imread(p0)
    plt.imshow(img)
    plt.title("Before histogram equalization")

    plt.subplot(122)
    img2 = preprocess(img)
    plt.imshow(img2)
    plt.title("After histogram equalization")



## === cell 8
generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=45,
    validation_split=0.1,
    shear_range=10,
    preprocessing_function=preprocess,
)

noPreprocessGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=45,
    validation_split=0.1,
)

noAugmentationGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=False,
    horizontal_flip=False,
    preprocessing_function=preprocess,
)



## === cell 9
training_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=base_train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
)

validation_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=base_train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=True,
)

noproc_training_generator = noPreprocessGenerator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=base_train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="training",
    shuffle=True,
)

noproc_validation_generator = noPreprocessGenerator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=base_train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=32,
    subset="validation",
    shuffle=True,
)



## === cell 10
import keras
from keras import layers, models

plt.figure(figsize=(36, 12))
imgs, labels = next(validation_generator)
plotindx = 1
for img, label in zip(imgs, labels):
    plt.subplot(7, 5, plotindx)
    plt.imshow(img)
    labl = 0
    if label[0] == 0:
        labl = 1
    plt.title("Label :" + str(labl))
    plotindx += 1



## === cell 11
model = models.Sequential()

model.add(
    layers.Conv2D(
        32, (5, 5), padding="valid", activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 12
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

CKPT_PATH = "/tmp/checkpoint.keras"
save_best_model = ModelCheckpoint(
    CKPT_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)



## === cell 13
history = model.fit(
    training_generator,
    validation_data=validation_generator,
    steps_per_epoch=training_generator.n // training_generator.batch_size,
    verbose=1,
    epochs=4,
    class_weight=class_weights,
    callbacks=[reduce_lr, save_best_model],
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2960783181.py in <cell line: 0>()
      1 # BUGFIX: fit_generator removed; use fit().
----> 2 history = model.fit(
      3     training_generator,
      4     validation_data=validation_generator,
      5     steps_per_epoch=training_generator.n // training_generator.batch_size,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 14
test_dir = os.path.join(TEST_EXTRACT_DIR, "test")
test_generator = noAugmentationGenerator.flow_from_directory(
    directory=TEST_EXTRACT_DIR,
    classes=[
        "test"
    ],  # ensures filenames are "test/<id>.jpg" and order matches directory listing
    class_mode=None,
    target_size=(32, 32),
    batch_size=32,
    shuffle=False,
)



## === cell 15
from keras.models import load_model

model = load_model(CKPT_PATH)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/530027645.py in <cell line: 0>()
      1 from keras.models import load_model
      2 
----> 3 model = load_model(CKPT_PATH)
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/tmp/checkpoint.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 16
proba = model.predict(test_generator, verbose=1)
if proba.ndim == 2 and proba.shape[1] == 2:
    idx_pos = training_generator.class_indices.get("1", 1)
    has_cactus = proba[:, idx_pos]
else:
    has_cactus = proba.reshape(-1)

output = pd.DataFrame({"id": test_generator.filenames, "has_cactus": has_cactus})
output["id"] = output["id"].apply(lambda s: os.path.basename(s))
output.to_csv("submission.csv", index=False)

print(output.head())
print("Wrote submission.csv with shape:", output.shape)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3408175304.py in <cell line: 0>()
      2 # With flow_from_dataframe + categorical, class_indices sorts string labels, typically {'0':0,'1':1}.
      3 # We map the "1" class index to extract correct probability column.
----> 4 proba = model.predict(test_generator, verbose=1)
      5 if proba.ndim == 2 and proba.shape[1] == 2:
      6     idx_pos = training_generator.class_indices.get("1", 1)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 17
import shutil

try:
    shutil.rmtree("test")
except OSError:
    print("Test files already erased or not present")
try:
    shutil.rmtree("training")
except OSError:
    print("Training files already erased or not present")
