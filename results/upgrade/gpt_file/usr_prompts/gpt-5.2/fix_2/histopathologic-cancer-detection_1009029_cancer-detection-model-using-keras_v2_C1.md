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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.7793147687630658

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from glob import glob
import gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from skimage.io import imread

import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential

print(os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("../input/train_labels.csv")
train_df.head()



## === cell 2
train_df.label.unique()



## === cell 3
distribution = train_df.label.value_counts()
print(distribution)
p = distribution[1] / distribution.sum()
print("Percentage of cancer affected cells are {}".format(p))



## === cell 4
label_counts = train_df["label"].value_counts()
fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
ax1.bar(np.arange(len(label_counts)) + 0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts)) + 0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation=90)



## === cell 5
base_tile_dir = "../input/train/"
df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.tif"))})

df["id"] = df["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])

labels = pd.read_csv("../input/train_labels.csv")
df = df.merge(labels, on="id")
df.head(10)



## === cell 6
df0 = df[df.label == 0].sample(5000, random_state=42)
df1 = df[df.label == 1].sample(5000, random_state=42)
df = pd.concat([df0, df1], ignore_index=True).reset_index(drop=True)
df = df[["path", "id", "label"]]
df.sample(10)



## === cell 7
df["image"] = df["path"].map(imread)
df.sample(3)



## === cell 8
images = [
    (df["image"].iloc[0], df["label"].iloc[0]),
    (df["image"].iloc[1], df["label"].iloc[1]),
    (df["image"].iloc[2], df["label"].iloc[2]),
    (df["image"].iloc[5000], df["label"].iloc[5000]),
    (df["image"].iloc[5001], df["label"].iloc[5001]),
    (df["image"].iloc[5002], df["label"].iloc[5002]),
]

fig, m_axs = plt.subplots(1, len(images), figsize=(20, 2))
for ii, c_ax in enumerate(m_axs):
    c_ax.imshow(images[ii][0])
    c_ax.set_title(images[ii][1])
    c_ax.axis("off")



## === cell 9
input_images = np.stack(list(df.image), axis=0)

input_images = input_images.astype(np.float32) / 255.0

input_images.shape



## === cell 10
x = input_images
y = df["label"].astype(np.int32).values

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.10, random_state=101, stratify=y
)

train_y.shape



## === cell 11
np.random.seed(42)
try:
    tf.random.set_seed(42)
except Exception:
    pass

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=False
)
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

model = Sequential()
model.add(
    Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu",
        input_shape=(96, 96, 3),
    )
)
model.add(Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="elu"))
model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="elu"))
model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="elu"))

model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/937057543.py in <cell line: 0>()
     10     monitor="val_loss", patience=5, restore_best_weights=False
     11 )
---> 12 checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
     13 
     14 model = Sequential()

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=weights.hdf5

## === cell 12
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

epochs = 15
model.fit(
    train_x,
    train_y,
    validation_data=(test_x, test_y),
    epochs=epochs,
    batch_size=80,
    verbose=1,
    callbacks=[early_stopping, checkpointer],
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3224535874.py in <cell line: 0>()
----> 1 model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
      2 
      3 epochs = 15
      4 model.fit(
      5     train_x,

NameError: name 'model' is not defined

## === cell 13
model.load_weights("weights.hdf5")

val_pred = model.predict(test_x, batch_size=256).ravel()

val_auc = roc_auc_score(test_y, val_pred)
val_auc



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4068941466.py in <cell line: 0>()
----> 1 model.load_weights("weights.hdf5")
      2 
      3 # Use vectorized prediction (same model, same outputs) for speed and correctness.
      4 val_pred = model.predict(test_x, batch_size=256).ravel()
      5 

NameError: name 'model' is not defined

## === cell 14
base_tile_dir = "../input/test/"
test_df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.tif"))})
test_df["id"] = test_df["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])

sample_sub = pd.read_csv("../input/sample_submission.csv")
test_df = sample_sub[["id"]].merge(test_df, on="id", how="left")

test_df.head()



## === cell 15
batch_size = 256
n = len(test_df)
preds = np.zeros(n, dtype=np.float32)

for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    batch_paths = test_df.loc[start : end - 1, "path"].values
    batch_imgs = (
        np.stack([imread(p) for p in batch_paths], axis=0).astype(np.float32) / 255.0
    )
    preds[start:end] = model.predict(batch_imgs, batch_size=64).ravel()
    del batch_imgs
    gc.collect()

test_df["label"] = preds
test_df[["id", "label"]].head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1807652286.py in <cell line: 0>()
     11         np.stack([imread(p) for p in batch_paths], axis=0).astype(np.float32) / 255.0
     12     )
---> 13     preds[start:end] = model.predict(batch_imgs, batch_size=64).ravel()
     14     del batch_imgs
     15     gc.collect()

NameError: name 'model' is not defined

## === cell 16
submission = test_df[["id", "label"]].copy()
submission.to_csv("submission.csv", index=False, header=True)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2732699940.py in <cell line: 0>()
      1 # IMPORTANT for ROC-AUC: keep probabilities (do NOT threshold to 0/1).
----> 2 submission = test_df[["id", "label"]].copy()
      3 submission.to_csv("submission.csv", index=False, header=True)
      4 
      5 print("Wrote submission.csv with shape:", submission.shape)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['label'] not in index"
