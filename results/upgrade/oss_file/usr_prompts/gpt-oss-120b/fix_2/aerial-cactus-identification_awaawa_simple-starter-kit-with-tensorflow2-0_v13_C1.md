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
seaborn==0.12.2
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
tqdm==4.67.1

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

0.9916

# 6. Current score

0.56931

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.56931) has done: 'I fix the TensorFlow import error, correct the optimizer argument, ensure the image directories point to the right locations after unzipping, make the plotting code tolerant of missing files, adjust metric names to match Keras output, clean up the prediction routine, and remove the destructive “rm -r *” cell so the notebook runs end‑to‑end and writes a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys, math, tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Dropout,
    GlobalMaxPooling2D,
)
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print(sys.version)
print("tensorflow -> ", tf.__version__)



## === cell 2
input_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
submission = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)

base_dir = "/kaggle/working"
candidate_train = os.path.join(base_dir, "train")
candidate_test = os.path.join(base_dir, "test")
if not os.path.isdir(candidate_train):
    candidate_train = os.path.join(base_dir, "aerial-cactus-identification", "train")
if not os.path.isdir(candidate_test):
    candidate_test = os.path.join(base_dir, "aerial-cactus-identification", "test")
train_dir = candidate_train
test_dir = candidate_test

test_df = submission["id"]



## === cell 3
input_df.head()



## === cell 4
print("shape: ", input_df.shape)
print("ーーーーーーーーーーーーーーーーーーーーー")
print(input_df["has_cactus"].value_counts())



## === cell 5
plt.style.use("default")
sns.set()
sns.set_style("whitegrid")
sns.set_palette("Pastel2")

x = ["has cactus", "hasn't cactus"]
y = input_df.groupby("has_cactus").size()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.pie(y, labels=x, autopct="%1.1f%%")
plt.show()



## === cell 6
fig, ax = plt.subplots(2, 5, figsize=(12, 6))

for i, idx in enumerate(input_df[input_df["has_cactus"] == 1]["id"][-5:]):
    path = os.path.join(train_dir, idx)
    try:
        img = load_img(path)
        ax[0, i].axis("off")
        ax[0, i].set_title("has cactus")
        ax[0, i].imshow(img)
    except FileNotFoundError:
        ax[0, i].axis("off")
        ax[0, i].set_title("missing")
for i, idx in enumerate(input_df[input_df["has_cactus"] == 0]["id"][-5:]):
    path = os.path.join(train_dir, idx)
    try:
        img = load_img(path)
        ax[1, i].axis("off")
        ax[1, i].set_title("hasn't cactus")
        ax[1, i].imshow(img)
    except FileNotFoundError:
        ax[1, i].axis("off")
        ax[1, i].set_title("missing")
plt.show()



## === cell 7
tra_df, val_df = train_test_split(
    input_df,
    test_size=0.25,
    stratify=input_df["has_cactus"],
    shuffle=True,
    random_state=12,
)

tra_df = tra_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

total_tra = tra_df.shape[0]
total_val = val_df.shape[0]

print(f"total_tra: {total_tra}, total_val: {total_val}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3524265489.py in <cell line: 0>()
----> 1 tra_df, val_df = train_test_split(
      2     input_df,
      3     test_size=0.25,
      4     stratify=input_df["has_cactus"],
      5     shuffle=True,

NameError: name 'train_test_split' is not defined

## === cell 8
img_width, img_height = 32, 32
target_size = (img_width, img_height)

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

tra_df["has_cactus"] = tra_df["has_cactus"].astype(str)
val_df["has_cactus"] = val_df["has_cactus"].astype(str)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2125010290.py in <cell line: 0>()
      5 val_datagen = ImageDataGenerator(rescale=1.0 / 255)
      6 
----> 7 tra_df["has_cactus"] = tra_df["has_cactus"].astype(str)
      8 val_df["has_cactus"] = val_df["has_cactus"].astype(str)
      9 

NameError: name 'tra_df' is not defined

## === cell 9
bs = 32
x_col, y_col = "id", "has_cactus"
class_mode = "binary"

tra_gen = train_datagen.flow_from_dataframe(
    tra_df,
    train_dir,
    x_col=x_col,
    y_col=y_col,
    class_mode=class_mode,
    target_size=target_size,
    batch_size=bs,
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    train_dir,
    x_col=x_col,
    y_col=y_col,
    class_mode=class_mode,
    target_size=target_size,
    batch_size=bs,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1245881574.py in <cell line: 0>()
      4 
      5 tra_gen = train_datagen.flow_from_dataframe(
----> 6     tra_df,
      7     train_dir,
      8     x_col=x_col,

NameError: name 'tra_df' is not defined

## === cell 10
input_shape = (img_width, img_height, 3)
optimizer = optimizers.Adam(learning_rate=1e-3)

model = Sequential(
    [
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        GlobalMaxPooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.summary()




## === cell 11
def step_decay(epoch):
    initial_rate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_rate * math.pow(drop, math.floor(epoch / epochs_drop))
    return lrate




## === cell 12
lrate = LearningRateScheduler(step_decay)
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
ep = 30

history = model.fit(
    tra_gen,
    epochs=ep,
    steps_per_epoch=100,
    validation_data=val_gen,
    validation_steps=50,
    callbacks=[lrate, es],
    verbose=2,
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/490960655.py in <cell line: 0>()
      4 
      5 history = model.fit(
----> 6     tra_gen,
      7     epochs=ep,
      8     steps_per_epoch=100,

NameError: name 'tra_gen' is not defined

## === cell 13
if "accuracy" in history.history:
    pd.DataFrame(
        {
            "accuracy": history.history["accuracy"],
            "val_accuracy": history.history.get("val_accuracy"),
        }
    ).plot()
if "loss" in history.history:
    pd.DataFrame(
        {"loss": history.history["loss"], "val_loss": history.history.get("val_loss")}
    ).plot()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1240387000.py in <cell line: 0>()
      1 # Plot training curves using the correct metric keys
----> 2 if "accuracy" in history.history:
      3     pd.DataFrame(
      4         {
      5             "accuracy": history.history["accuracy"],

NameError: name 'history' is not defined

## === cell 14
def predict(model, submission_df):
    preds = np.empty((submission_df.shape[0],))
    for n in tqdm(range(submission_df.shape[0])):
        img_path = os.path.join(test_dir, submission_df.id[n])
        img = load_img(img_path, target_size=target_size)
        arr = np.array(img) / 255.0
        preds[n] = model.predict(arr.reshape((1, img_width, img_height, 3)), verbose=0)[
            0
        ][0]
    submission_df["has_cactus"] = preds
    return submission_df




## === cell 15
df_prediction = predict(model, submission)



## === cell 16
df_prediction.to_csv("submission.csv", index=False, header=True)
