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

0.9576

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, math, zipfile, shutil
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, utils

np.random.seed(1)
tf_random_seed = 1
utils.set_random_seed(tf_random_seed)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(
    "../input/aerial-cactus-identification/train.csv",
    dtype={"id": str, "has_cactus": int},
)
test_df = pd.read_csv(
    "../input/aerial-cactus-identification/sample_submission.csv", dtype=str
)



## === cell 2
zipfile.ZipFile("/kaggle/input/aerial-cactus-identification/train.zip").extractall(
    "/kaggle/working/"
)
zipfile.ZipFile("/kaggle/input/aerial-cactus-identification/test.zip").extractall(
    "/kaggle/working/"
)

possible_train = [
    os.path.join("/kaggle/working", d)
    for d in os.listdir("/kaggle/working")
    if os.path.isdir(os.path.join("/kaggle/working", d)) and ("train" in d.lower())
]
train_path = possible_train[0] if possible_train else "/kaggle/working/train/"

possible_test = [
    os.path.join("/kaggle/working", d)
    for d in os.listdir("/kaggle/working")
    if os.path.isdir(os.path.join("/kaggle/working", d)) and ("test" in d.lower())
]
test_path = possible_test[0] if possible_test else "/kaggle/working/test/"

print("Training images path:", train_path, "exists:", os.path.isdir(train_path))
print("Testing images path :", test_path, "exists:", os.path.isdir(test_path))
print("Training images count:", len(os.listdir(train_path)))
print("Testing images count :", len(os.listdir(test_path)))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1076219103.py in <cell line: 0>()
     24 print("Training images path:", train_path, "exists:", os.path.isdir(train_path))
     25 print("Testing images path :", test_path, "exists:", os.path.isdir(test_path))
---> 26 print("Training images count:", len(os.listdir(train_path)))
     27 print("Testing images count :", len(os.listdir(test_path)))
     28 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/'

## === cell 3
cmap = plt.get_cmap("Blues")
colors = [cmap(i) for i in np.linspace(0, 0.7, train_df["has_cactus"].nunique())]
train_df["has_cactus"].value_counts().plot(
    kind="pie", figsize=(6, 6), autopct="%1.2f%%", shadow=True, colors=colors
)
plt.title("Class distribution")
plt.ylabel("")
plt.show()



## === cell 4
train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.20)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

batch_size = 64

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_path,
    x_col="id",
    y_col="has_cactus",
    subset="training",
    batch_size=batch_size,
    shuffle=True,
    class_mode="binary",
    target_size=(32, 32),
    color_mode="rgb",
)

valid_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_path,
    x_col="id",
    y_col="has_cactus",
    subset="validation",
    batch_size=batch_size,
    shuffle=False,
    class_mode="binary",
    target_size=(32, 32),
    color_mode="rgb",
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_path,
    x_col="id",
    y_col=None,
    batch_size=batch_size,
    shuffle=False,
    class_mode=None,
    target_size=(32, 32),
    color_mode="rgb",
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3746016914.py in <cell line: 0>()
----> 1 train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.20)
      2 test_datagen = ImageDataGenerator(rescale=1.0 / 255)
      3 
      4 batch_size = 64
      5 

NameError: name 'ImageDataGenerator' is not defined

## === cell 5
def show_training_samples(seed=42, n=36):
    np.random.seed(seed)
    imgs, labels = next(train_generator)
    plt.figure(figsize=(14, 14))
    for i in range(min(n, len(imgs))):
        plt.subplot(6, 6, i + 1)
        plt.imshow(imgs[i])
        plt.title("Cactus" if labels[i] == 1 else "No cactus")
        plt.axis("off")
    plt.show()


show_training_samples()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/641393016.py in <cell line: 0>()
     11 
     12 
---> 13 show_training_samples()
     14 

/tmp/ipykernel_55/641393016.py in show_training_samples(seed, n)
      1 def show_training_samples(seed=42, n=36):
      2     np.random.seed(seed)
----> 3     imgs, labels = next(train_generator)
      4     plt.figure(figsize=(14, 14))
      5     for i in range(min(n, len(imgs))):

NameError: name 'train_generator' is not defined

## === cell 6
cnn = Sequential(
    [
        Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        Conv2D(16, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(1, activation="sigmoid"),
    ]
)

cnn.summary()



## === cell 7
opt = optimizers.Adam(learning_rate=1e-4)
cnn.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

history = cnn.fit(
    train_generator, validation_data=valid_generator, epochs=15, verbose=1
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1304740840.py in <cell line: 0>()
----> 1 opt = optimizers.Adam(learning_rate=1e-4)
      2 cnn.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
      3 
      4 history = cnn.fit(
      5     train_generator, validation_data=valid_generator, epochs=15, verbose=1

NameError: name 'optimizers' is not defined

## === cell 8
test_steps = math.ceil(test_generator.samples / batch_size)
test_pred_probs = cnn.predict(test_generator, steps=test_steps, verbose=1).flatten()

test_filenames = test_generator.filenames
submission = pd.DataFrame({"id": test_filenames, "has_cactus": test_pred_probs})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2623085681.py in <cell line: 0>()
----> 1 test_steps = math.ceil(test_generator.samples / batch_size)
      2 test_pred_probs = cnn.predict(test_generator, steps=test_steps, verbose=1).flatten()
      3 
      4 test_filenames = test_generator.filenames
      5 submission = pd.DataFrame({"id": test_filenames, "has_cactus": test_pred_probs})

NameError: name 'test_generator' is not defined

## === cell 9
for p in [train_path, test_path]:
    if os.path.isdir(p):
        shutil.rmtree(p)
