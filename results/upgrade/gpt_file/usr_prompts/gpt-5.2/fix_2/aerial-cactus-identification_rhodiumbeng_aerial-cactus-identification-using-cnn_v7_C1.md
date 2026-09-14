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

0.993

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

import tf_keras as keras
from tf_keras.utils import load_img, img_to_array

print(
    "Listing ../input:",
    os.listdir("../input") if os.path.exists("../input") else "MISSING",
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "../input/aerial-cactus-identification"

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

train_dir = os.path.join(BASE_DIR, "train", "train")
test_dir = os.path.join(BASE_DIR, "test", "test")

if not os.path.isdir(train_dir):
    alt = os.path.join(BASE_DIR, "train")
    if os.path.isdir(alt):
        train_dir = alt
if not os.path.isdir(test_dir):
    alt = os.path.join(BASE_DIR, "test")
    if os.path.isdir(alt):
        test_dir = alt

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(sample_sub_path)

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print("train_dir exists:", os.path.isdir(train_dir), train_dir)
print("test_dir exists:", os.path.isdir(test_dir), test_dir)



## === cell 2
train_df["has_cactus"].value_counts()



## === cell 3
import matplotlib.pyplot as plt

train_path = train_dir + ("" if train_dir.endswith(os.sep) else os.sep)
test_path = test_dir + ("" if test_dir.endswith(os.sep) else os.sep)

try:
    has_cactus = train_df[train_df["has_cactus"] == 1]
    plt.figure(figsize=(10, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(load_img(train_path + has_cactus.iloc[i]["id"]))
        plt.axis("off")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Plot preview skipped due to:", repr(e))




## === cell 4
def prep_cnn_data(df, n_x, n_c, path):
    """
    This function loads the image jpg data into tensors
    """
    tensors = np.zeros((df.shape[0], n_x, n_x, n_c), dtype=np.float32)
    for i in range(df.shape[0]):
        pic = load_img(os.path.join(path, df.iloc[i]["id"]), target_size=(n_x, n_x))
        pic_array = img_to_array(pic)
        tensors[i, :] = pic_array
    tensors = tensors / 255.0
    return tensors




## === cell 5
train_pic_array = prep_cnn_data(train_df, 32, 3, path=train_dir)
train_Y = train_df["has_cactus"].values.astype("float32")

test_pic_array = prep_cnn_data(test_df, 32, 3, path=test_dir)

print(train_pic_array.shape, train_Y.shape)
print(test_pic_array.shape)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/41608751.py in <cell line: 0>()
----> 1 train_pic_array = prep_cnn_data(train_df, 32, 3, path=train_dir)
      2 train_Y = train_df["has_cactus"].values.astype("float32")
      3 
      4 test_pic_array = prep_cnn_data(test_df, 32, 3, path=test_dir)
      5 

/tmp/ipykernel_11/2638257736.py in prep_cnn_data(df, n_x, n_c, path)
      5     tensors = np.zeros((df.shape[0], n_x, n_x, n_c), dtype=np.float32)
      6     for i in range(df.shape[0]):
----> 7         pic = load_img(os.path.join(path, df.iloc[i]["id"]), target_size=(n_x, n_x))
      8         pic_array = img_to_array(pic)
      9         tensors[i, :] = pic_array

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/image_utils.py in load_img(path, grayscale, color_mode, target_size, interpolation, keep_aspect_ratio)
    420         if isinstance(path, pathlib.Path):
    421             path = str(path.resolve())
--> 422         with open(path, "rb") as f:
    423             img = pil_image.open(io.BytesIO(f.read()))
    424     else:

FileNotFoundError: [Errno 2] No such file or directory: '../input/aerial-cactus-identification/train/train/2de8f189f1dce439766637e75df0ee27.jpg'

## === cell 6
from tf_keras.preprocessing.image import ImageDataGenerator

data_augment = ImageDataGenerator(
    zoom_range=0.1, horizontal_flip=True, vertical_flip=True
)



## === cell 7
from tf_keras import models
from tf_keras import layers

model = models.Sequential()
model.add(
    layers.Conv2D(
        32, kernel_size=3, padding="same", activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.Conv2D(32, kernel_size=3, padding="valid", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Dropout(rate=0.4))
model.add(layers.Conv2D(64, kernel_size=5, padding="same", activation="relu"))
model.add(layers.Conv2D(64, kernel_size=5, padding="valid", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Dropout(rate=0.4))
model.add(layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"))
model.add(layers.Conv2D(128, kernel_size=3, padding="valid", activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()



## === cell 8
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 9
X_dev = train_pic_array[:3500]
rem_X_train = train_pic_array[3500:]
Y_dev = train_Y[:3500]
rem_Y_train = train_Y[3500:]

print(X_dev.shape, rem_X_train.shape)
print(Y_dev.shape, rem_Y_train.shape)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3965897471.py in <cell line: 0>()
      1 # Keep original split logic (first 3500 as dev), but ensure shapes/dtypes consistent
----> 2 X_dev = train_pic_array[:3500]
      3 rem_X_train = train_pic_array[3500:]
      4 Y_dev = train_Y[:3500]
      5 rem_Y_train = train_Y[3500:]

NameError: name 'train_pic_array' is not defined

## === cell 10
epochs = 100
batch_size = 512

train_gen = data_augment.flow(
    rem_X_train, rem_Y_train, batch_size=batch_size, shuffle=True
)
steps_per_epoch = int(np.ceil(rem_X_train.shape[0] / batch_size))

history = model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_dev, Y_dev),
    verbose=2,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1807644646.py in <cell line: 0>()
      4 
      5 train_gen = data_augment.flow(
----> 6     rem_X_train, rem_Y_train, batch_size=batch_size, shuffle=True
      7 )
      8 steps_per_epoch = int(np.ceil(rem_X_train.shape[0] / batch_size))

NameError: name 'rem_X_train' is not defined

## === cell 11
loss = history.history["loss"]
dev_loss = history.history["val_loss"]
ep_range = range(1, len(loss) + 1)

from matplotlib import pyplot as plt

plt.plot(ep_range, loss, "bo", label="training loss")
plt.plot(ep_range, dev_loss, "b", label="validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2362088544.py in <cell line: 0>()
      1 # plot and visualise the training and validation losses
----> 2 loss = history.history["loss"]
      3 dev_loss = history.history["val_loss"]
      4 ep_range = range(1, len(loss) + 1)
      5 

NameError: name 'history' is not defined

## === cell 12
pred_dev = model.predict(X_dev, verbose=0)
pred_dev_bin = (pred_dev > 0.5).astype(int)

result = pd.DataFrame(train_Y[:3500], columns=["Y_dev"])
result["Y_pred"] = pred_dev_bin
result["correct"] = result["Y_dev"] - result["Y_pred"]
errors = result[result["correct"] != 0]
error_list = errors.index.to_list()
print("Number of errors is", len(errors))
print("First error indices:", error_list[:20])



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1441709914.py in <cell line: 0>()
      1 # Dev predictions for inspection (keep thresholding only for this diagnostic)
----> 2 pred_dev = model.predict(X_dev, verbose=0)
      3 pred_dev_bin = (pred_dev > 0.5).astype(int)
      4 
      5 result = pd.DataFrame(train_Y[:3500], columns=["Y_dev"])

NameError: name 'X_dev' is not defined

## === cell 13
try:
    max_show = min(len(error_list), 40)
    plt.figure(figsize=(15, 8))
    for i in range(max_show):
        plt.subplot(4, 10, i + 1)
        plt.imshow(load_img(train_path + train_df.iloc[error_list[i]]["id"]))
        plt.title(
            "true={}\npred={}".format(
                int(train_Y[error_list[i]]), int(pred_dev_bin[i])
            ),
            y=1,
        )
        plt.axis("off")
    plt.subplots_adjust(wspace=0.3, hspace=-0.1)
    plt.show()
except Exception as e:
    print("Error visualization skipped due to:", repr(e))



## === cell 14
predictions = model.predict(test_pic_array, verbose=0).reshape(-1)
print(predictions.shape, predictions.min(), predictions.max())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2486790486.py in <cell line: 0>()
      1 # Predict probabilities for AUC-based submission (do NOT threshold).
----> 2 predictions = model.predict(test_pic_array, verbose=0).reshape(-1)
      3 print(predictions.shape, predictions.min(), predictions.max())
      4 

NameError: name 'test_pic_array' is not defined

## === cell 15
sub = test_df.copy()
sub["has_cactus"] = predictions.astype("float32")
sub[["id", "has_cactus"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1938785775.py in <cell line: 0>()
      1 # Fix: submission must be probabilities; keep id order from sample_submission.
      2 sub = test_df.copy()
----> 3 sub["has_cactus"] = predictions.astype("float32")
      4 sub[["id", "has_cactus"]].to_csv("submission.csv", index=False)
      5 print("Wrote submission.csv with shape:", sub.shape)

NameError: name 'predictions' is not defined
