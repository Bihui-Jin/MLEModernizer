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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

0.5001

# 6. Current score

0.99536

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.99536) has done: 'I fix the environment-breaking import error by switching from `tf_keras`/`keras` image utilities to `tensorflow.keras`, which avoids the protobuf `MessageFactory.GetPrototype` crash in this Kaggle runtime. Then I correct the dataset paths (your images are in `.../train` and `.../test`, not `.../train/train`), and repair the train/validation split so the validation generator is non-empty and has both classes. Finally, I make feature extraction work for both train and test by allowing `has_cactus` to be optional (test has no labels) and ensure a properly formatted `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2

from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

print("TensorFlow:", tf.__version__)
print("Listing ../input:", os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "../input/aerial-cactus-identification"

train_dir = os.path.join(BASE_DIR, "train")
test_dir = os.path.join(BASE_DIR, "test")

train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df_test = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))

print("train_dir exists:", os.path.isdir(train_dir), train_dir)
print("test_dir exists:", os.path.isdir(test_dir), test_dir)
print("train.csv shape:", train.shape, "sample_submission shape:", df_test.shape)



## === cell 2
train.head(5)
train.has_cactus = train.has_cactus.astype(str)



## === cell 3
print("out dataset has {} rows and {} columns".format(train.shape[0], train.shape[1]))



## === cell 4
train["has_cactus"].value_counts()



## === cell 5
print("The number of rows in test set is %d" % (len(os.listdir(test_dir))))



## === cell 6
Image(filename=os.path.join(train_dir, train.iloc[0, 0]), width=250, height=250)



## === cell 7
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
batch_size = 150



## === cell 8
train = train.sample(frac=1.0, random_state=42).reset_index(drop=True)

val_size = max(1, int(0.1 * len(train)))
train_df = train.iloc[:-val_size].copy()
val_df = train.iloc[-val_size:].copy()

if val_df["has_cactus"].nunique() < 2:
    train_int = train.copy()
    train_int["has_cactus_int"] = train_int["has_cactus"].astype(int)
    val_df = (
        train_int.groupby("has_cactus_int", group_keys=False)
        .apply(lambda x: x.sample(max(1, int(0.05 * len(x))), random_state=42))
        .reset_index(drop=True)
    )
    train_df = train_int.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.drop(columns=["has_cactus_int"]).copy()
    train_df = train_df.drop(columns=["has_cactus_int"]).copy()

print("Train split:", train_df.shape, "Val split:", val_df.shape)
print("Val class counts:\n", val_df["has_cactus"].value_counts())

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
    batch_size=batch_size,
    target_size=(150, 150),
    shuffle=True,
    seed=42,
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
    batch_size=50,
    target_size=(150, 150),
    shuffle=False,
)



## === cell 9
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))



## === cell 10
model.summary()



## === cell 11
model.compile(
    loss="binary_crossentropy", optimizer=optimizers.RMSprop(), metrics=["acc"]
)



## === cell 12
epochs = 10

steps_per_epoch = min(100, max(1, len(train_generator)))
validation_steps = min(50, max(1, len(validation_generator)))

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
)



## === cell 13
acc = history.history.get("acc", history.history.get("accuracy", []))
epochs_ = range(0, epochs)
plt.plot(epochs_, acc, label="training accuracy")
plt.xlabel("no of epochs")
plt.ylabel("accuracy")

acc_val = history.history.get("val_acc", history.history.get("val_accuracy", []))
plt.scatter(epochs_, acc_val, label="validation accuracy")
plt.title("no of epochs vs accuracy")
plt.legend()
plt.show()



## === cell 14
loss = history.history.get("loss", [])
epochs_ = range(0, epochs)
plt.plot(epochs_, loss, label="training loss")
plt.xlabel("No of epochs")
plt.ylabel("loss")

loss_val = history.history.get("val_loss", [])
plt.scatter(epochs_, loss_val, label="validation loss")
plt.title("no of epochs vs loss")
plt.legend()
plt.show()



## === cell 15
model_vg = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
model_vg.summary()




## === cell 16
def extract_features(directory, samples, df):
    """Extract VGG16 conv features for a dataframe of filenames.
    Works for train (with labels) and test (without labels)."""
    features = np.zeros(shape=(samples, 4, 4, 512), dtype=np.float32)
    labels = np.zeros(shape=(samples,), dtype=np.float32)

    if "has_cactus" in df.columns:
        generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=directory,
            x_col="id",
            y_col="has_cactus",
            class_mode="raw",
            batch_size=batch_size,
            target_size=(150, 150),
            shuffle=False,
        )
        has_labels = True
    else:
        generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=directory,
            x_col="id",
            y_col=None,
            class_mode=None,
            batch_size=batch_size,
            target_size=(150, 150),
            shuffle=False,
        )
        has_labels = False

    i = 0
    filled = 0
    for batch in generator:
        if has_labels:
            input_batch, label_batch = batch
        else:
            input_batch = batch
            label_batch = None

        feature_batch = model_vg.predict(input_batch, verbose=0)
        current_batch = feature_batch.shape[0]

        start = i * batch_size
        end = min(start + current_batch, samples)
        if start >= samples:
            break

        take = end - start
        features[start:end] = feature_batch[:take]
        if has_labels and label_batch is not None:
            labels[start:end] = np.array(label_batch).reshape(-1)[:take]

        i += 1
        filled = end
        if filled >= samples:
            break

    return features, labels


train_int = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
train_int["has_cactus"] = train_int["has_cactus"].astype(int)

n_train_total = min(17500, len(train_int))
features, labels = extract_features(
    train_dir, n_train_total, train_int.iloc[:n_train_total].copy()
)

split_train = min(15001, n_train_total)
split_val_start = min(15000, n_train_total)

train_features = features[:split_train]
train_labels = labels[:split_train]

validation_features = features[split_val_start:]
validation_labels = labels[split_val_start:]

print(
    "train_features:",
    train_features.shape,
    "validation_features:",
    validation_features.shape,
)



## === cell 17
train_features = train_features.reshape((train_features.shape[0], 4 * 4 * 512))
validation_features = validation_features.reshape(
    (validation_features.shape[0], 4 * 4 * 512)
)

n_test = len(df_test)
test_features, _ = extract_features(test_dir, n_test, df_test[["id"]].copy())
test_features = test_features.reshape((test_features.shape[0], 4 * 4 * 512))

print("test_features:", test_features.shape)



## === cell 18
model = models.Sequential()
model.add(
    layers.Dense(
        212,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(0.001),
        input_dim=(4 * 4 * 512),
    )
)
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation="sigmoid"))



## === cell 19
model.compile(
    optimizer=optimizers.RMSprop(), loss="binary_crossentropy", metrics=["acc"]
)



## === cell 20
history = model.fit(
    train_features,
    train_labels,
    epochs=30,
    batch_size=15,
    validation_data=(validation_features, validation_labels),
)



## === cell 21
y_pre = model.predict(test_features, verbose=0).reshape(-1)
y_pre = np.clip(y_pre.astype(np.float64), 0.0, 1.0)
print("Pred stats:", float(y_pre.min()), float(y_pre.max()), float(y_pre.mean()))

sub = pd.DataFrame({"id": df_test["id"].values, "has_cactus": y_pre})
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("submission.csv exists:", os.path.isfile("submission.csv"))
