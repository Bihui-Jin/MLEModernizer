# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os


import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)

print("TensorFlow version:", tf.__version__)
print("Keras backend:", tf.keras.backend.backend())
print("Files in /kaggle/input:", os.listdir("/kaggle/input"))




## === cell 1
base_dir = "/kaggle/input/aerial-cactus-identification"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
print("Train samples:", len(train_df))




## === cell 2
dim_x, dim_y, dim_ch = 32, 32, 3
x_train = []
y_train = []

for img_id in tqdm(train_df["id"].values, desc="Loading train images"):
    img_path = os.path.join(train_dir, img_id)
    img = load_img(img_path, target_size=(dim_x, dim_y))
    arr = img_to_array(img)  # (32,32,3)
    x_train.append(arr)
    y_train.append(train_df.loc[train_df["id"] == img_id, "has_cactus"].values[0])

x_train = np.asarray(x_train, dtype="float32") / 255.0
y_train = np.asarray(y_train, dtype="float32")
print("Loaded training images shape:", x_train.shape)




## === cell 3
nb_valid = int(0.1 * len(x_train))
x_valid = x_train[-nb_valid:]
y_valid = y_train[-nb_valid:]
x_train = x_train[:-nb_valid]
y_train = y_train[:-nb_valid]
print("Train/val sizes:", x_train.shape[0], x_valid.shape[0])




## === cell 4
batch_size = 32
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    shear_range=0.15,
    brightness_range=[0.9, 1.1],
    channel_shift_range=0.12,
    rotation_range=90.0,
    zoom_range=0.2,
    width_shift_range=0.075,
    height_shift_range=0.075,
)

train_generator = train_datagen.flow(
    x=x_train, y=y_train, batch_size=batch_size, shuffle=True
)

test_datagen = ImageDataGenerator()
valid_generator = test_datagen.flow(
    x=x_valid, y=y_valid, batch_size=batch_size, shuffle=False
)




## === cell 5
my_net = VGG16(
    weights="imagenet", include_top=False, input_shape=(dim_x, dim_y, dim_ch)
)
my_net.trainable = True

model = Sequential([my_net, Flatten(), Dropout(0.5), Dense(1, activation="sigmoid")])
model.summary()




## === cell 6
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=1e-4),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)




## === cell 7
early = EarlyStopping(
    monitor="val_auc", patience=5, verbose=1, mode="max", restore_best_weights=True
)
nb_epochs = 20  # increased epochs for better convergence

history = model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=valid_generator,
    validation_steps=len(x_valid) // batch_size,
    epochs=nb_epochs,
    callbacks=[early],
    verbose=2,
)




## === cell 8
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()




## === cell 9
test_imgs = []
x_test = []

for img_id in tqdm(sorted(os.listdir(test_dir)), desc="Loading test images"):
    img_path = os.path.join(test_dir, img_id)
    if not os.path.isfile(img_path):
        continue  # skip any sub‑directories
    img = load_img(img_path, target_size=(dim_x, dim_y))
    arr = img_to_array(img)
    x_test.append(arr)
    test_imgs.append(img_id)

x_test = np.asarray(x_test, dtype="float32") / 255.0
print("Test images shape:", x_test.shape)




## === cell 10
test_predictions = model.predict(x_test, batch_size=batch_size, verbose=1)




## === cell 11
sub_df = pd.DataFrame({"id": test_imgs, "has_cactus": test_predictions.ravel()})
sub_df.to_csv("./submission.csv", index=False)
print("Submission saved to ./submission.csv, rows:", len(sub_df))
