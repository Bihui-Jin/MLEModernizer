# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
from glob import glob
import gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from skimage.io import imread

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential

print("TF version:", tf.__version__)
print("Listing ../input:", os.listdir("../input")[:20])


def pick_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these paths exist: " + str(paths))


DATA_ROOT = pick_existing(
    "../input/histopathologic-cancer-detection",
    "../input",
    "/kaggle/input/histopathologic-cancer-detection",
    "/kaggle/input",
)

TRAIN_LABELS_CSV = pick_existing(
    os.path.join(DATA_ROOT, "train_labels.csv"),
    "../input/train_labels.csv",
)
SAMPLE_SUB_CSV = pick_existing(
    os.path.join(DATA_ROOT, "sample_submission.csv"),
    "../input/sample_submission.csv",
)

TRAIN_DIR = pick_existing(
    os.path.join(DATA_ROOT, "train"),
    "../input/train",
)
TEST_DIR = pick_existing(
    os.path.join(DATA_ROOT, "test"),
    "../input/test",
)

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_LABELS_CSV:", TRAIN_LABELS_CSV)
print("SAMPLE_SUB_CSV:", SAMPLE_SUB_CSV)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)



## === cell 1
train_df = pd.read_csv(TRAIN_LABELS_CSV)
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
base_tile_dir = TRAIN_DIR
df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.tif"))})

df["id"] = df["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])

labels = pd.read_csv(TRAIN_LABELS_CSV)
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
tf.random.set_seed(42)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=False
)

checkpointer = ModelCheckpoint(filepath="weights.keras", verbose=1, save_best_only=True)

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

model.summary()



## === cell 12
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

epochs = 15
history = model.fit(
    train_x,
    train_y,
    validation_data=(test_x, test_y),
    epochs=epochs,
    batch_size=80,
    verbose=1,
    callbacks=[early_stopping, checkpointer],
)



## === cell 13
model = tf.keras.models.load_model("weights.keras")

val_pred = model.predict(test_x, batch_size=256, verbose=0).ravel()
val_auc = roc_auc_score(test_y, val_pred)
print("Validation ROC-AUC:", val_auc)



## === cell 14
base_tile_dir = TEST_DIR
test_df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.tif"))})
test_df["id"] = test_df["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])

sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_df = sample_sub[["id"]].merge(test_df, on="id", how="left")

missing = test_df["path"].isna().sum()
print("Test rows:", len(test_df), "Missing paths after merge:", missing)
test_df.head()



## === cell 15
batch_size = 256
n = len(test_df)
preds = np.zeros(n, dtype=np.float32)

for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    batch_paths = test_df.loc[start : end - 1, "path"].values

    batch_imgs_list = []
    batch_valid_idx = []
    for i, p in enumerate(batch_paths):
        if isinstance(p, str) and os.path.exists(p):
            batch_imgs_list.append(imread(p))
            batch_valid_idx.append(i)

    if batch_valid_idx:
        batch_imgs = np.stack(batch_imgs_list, axis=0).astype(np.float32) / 255.0
        batch_pred = (
            model.predict(batch_imgs, batch_size=64, verbose=0)
            .ravel()
            .astype(np.float32)
        )
        preds[start:end] = 0.5
        preds[start + np.array(batch_valid_idx)] = batch_pred
        del batch_imgs, batch_pred
    else:
        preds[start:end] = 0.5

    del batch_imgs_list, batch_valid_idx
    gc.collect()

test_df["label"] = preds
test_df[["id", "label"]].head()



## === cell 16
submission = test_df[["id", "label"]].copy()
submission.to_csv("submission.csv", index=False, header=True)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
