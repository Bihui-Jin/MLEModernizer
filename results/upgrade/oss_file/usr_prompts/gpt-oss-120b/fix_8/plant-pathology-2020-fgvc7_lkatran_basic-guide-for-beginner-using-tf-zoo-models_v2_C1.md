# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime as dt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
import concurrent.futures

np.random.seed(42)
tf.random.set_seed(42)
tf.config.optimizer.set_jit(True)  # enable XLA

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")



## === cell 1
submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
)
train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")



## === cell 2
train.head()



## === cell 3
train.loc[:, "healthy":"scab"].sum(axis=0) / (train.shape[0] / 100)



## === cell 4
test.head()



## === cell 5
submission.head()




## === cell 6
def _load_and_preprocess(im_id, base_path):
    """Read, resize, and convert an image to float32."""
    im_path = os.path.join(base_path, im_id + ".jpg")
    img = cv2.imread(im_path)
    img = cv2.resize(img, (224, 224))
    return img.astype("float32")


train_img = []
path = "/kaggle/input/plant-pathology-2020-fgvc7/images"
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    for img in tqdm(
        executor.map(_load_and_preprocess, train["image_id"], [path] * len(train)),
        total=len(train),
    ):
        train_img.append(img)



## === cell 7
test_img = []
path = "/kaggle/input/plant-pathology-2020-fgvc7/images"
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    for img in tqdm(
        executor.map(_load_and_preprocess, test["image_id"], [path] * len(test)),
        total=len(test),
    ):
        test_img.append(img)



## === cell 8
train_label = train.loc[:, "healthy":"scab"]



## === cell 9
train_img = np.array(train_img) / 255.0
test_img = np.array(test_img) / 255.0
train_label = np.array(train_label, dtype="float32")



## === cell 10
print(train_img.shape)
print(test_img.shape)
print(train_label.shape)



## === cell 11
X_train, X_val, y_train, y_val = train_test_split(
    train_img, train_label, test_size=0.1, random_state=42, stratify=train_label
)

batch_size = 128  # increased from 64

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = (
    train_ds.shuffle(buffer_size=len(X_train))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)



## === cell 12
tf.keras.backend.clear_session()  # free any previous graph memory

base_model = DenseNet201(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
)

model = Sequential()
model.add(base_model)
model.add(BatchNormalization())
model.add(Dropout(0.8))
model.add(Dense(128, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

base_model.trainable = True

reduce_learning_rate = ReduceLROnPlateau(
    monitor="val_auc", factor=0.1, patience=2, cooldown=2, min_lr=1e-7, verbose=1
)
early_stopping = EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True)

check_point = ModelCheckpoint(
    filepath="best_model.h5", monitor="val_auc", save_best_only=True, verbose=0
)

callbacks = [reduce_learning_rate, early_stopping, check_point]

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[AUC(name="auc")])



## === cell 13
model.summary()



## === cell 14
start = dt.now()
history = model.fit(
    train_ds,
    epochs=30,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=2,
)
print(f"Training time: {dt.now() - start}. Epochs run: {len(history.epoch)}")



## === cell 15
import gc

del train_img, train_label, X_train, X_val, y_train, y_val
gc.collect()




## === cell 16
def plot_loss(his, title):
    epoch = len(his.epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(epoch), his.history["loss"], label="train_loss")
    plt.plot(np.arange(epoch), his.history["val_loss"], label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()


def plot_auc(his, title):
    epoch = len(his.epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(epoch), his.history["auc"], label="train_auc")
    plt.plot(np.arange(epoch), his.history["val_auc"], label="val_auc")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("AUC")
    plt.legend(loc="lower right")
    plt.show()




## === cell 17
plot_loss(history, "Training & Validation Loss")
plot_auc(history, "Training & Validation AUC")



## === cell 18
y_pred = model.predict(test_img, batch_size=64)



## === cell 19
submission.loc[:, "healthy":"scab"] = y_pred



## === cell 20
submission.to_csv("submission.csv", index=False)
