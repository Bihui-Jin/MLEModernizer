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

3.13

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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
pass



## === cell 2
import zipfile

extract_dir = "/kaggle/working"

with zipfile.ZipFile(
    "/kaggle/input/aerial-cactus-identification/train.zip", "r"
) as zip_ref:
    zip_ref.extractall(os.path.join(extract_dir, "train"))

with zipfile.ZipFile(
    "/kaggle/input/aerial-cactus-identification/test.zip", "r"
) as zip_ref:
    zip_ref.extractall(os.path.join(extract_dir, "test"))



## === cell 3
for dirname, _, _ in os.walk("/kaggle/working"):
    print(dirname)



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
train_dir_candidate = "/kaggle/working/train/train"
test_dir_candidate = "/kaggle/working/test/test"

train_dir = (
    train_dir_candidate
    if os.path.isdir(train_dir_candidate)
    else "/kaggle/input/aerial-cactus-identification/train"
)
test_dir = (
    test_dir_candidate
    if os.path.isdir(test_dir_candidate)
    else "/kaggle/input/aerial-cactus-identification/test"
)

train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
train_df.head(20)



## === cell 8
pass




## === cell 9
def count_files(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images: {test_count}")



## === cell 10
pass



## === cell 11
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 12
import matplotlib.pyplot as plt

counts = train_df["has_cactus"].value_counts()

labels = ["Has Cactus (1)", "No Cactus (0)"]
colors = ["lightgreen", "lightcoral"]

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
plt.title("Distribution of Cactus Presence (has_cactus)")
plt.axis("equal")
plt.show()



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
try:
    from tensorflow.keras.applications.efficientnet import (
        EfficientNetB3 as EfficientNetClass,
    )
except Exception:
    from tensorflow.keras.applications.efficientnet import (
        EfficientNetB0 as EfficientNetClass,
    )

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential



## === cell 17
pass



## === cell 18
train_df["has_cactus"] = train_df["has_cactus"].astype(int)



## === cell 19
pass



## === cell 20
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random


def custom_preprocessing(image):
    k = random.randint(0, 3)
    image = np.rot90(image, k)

    if random.random() > 0.5:
        image = np.fliplr(image)

    if random.random() > 0.5:
        image = np.flipud(image)

    factor = random.uniform(0.8, 1.2)
    image = np.clip(image * factor, 0, 255).astype(np.uint32) / 255.0

    return image


train_datagen = ImageDataGenerator(
    validation_split=0.10,
    preprocessing_function=custom_preprocessing,
    rescale=1 / 255.0,
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    subset="training",
    batch_size=256,
    shuffle=True,
    class_mode="raw",  # use raw numeric labels
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    subset="validation",
    batch_size=256,
    shuffle=False,
    class_mode="raw",
)



## === cell 21
pass



## === cell 22
pass



## === cell 23
import cv2

cactus = []
for i in range(12):
    path = os.path.join(train_dir, train_df["id"][i])
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cactus.append(img)

cactus_augmented = [custom_preprocessing(img) for img in cactus]

plt.figure(figsize=(10, 10))
for i in range(len(cactus_augmented)):
    plt.subplot(4, 3, i + 1)
    plt.imshow(cactus_augmented[i])
    plt.title(f"Image {i+1}")
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 24
test_datagen = ImageDataGenerator(rescale=1 / 255.0)

test_filenames = [
    f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))
]
test_df = pd.DataFrame({"id": test_filenames})

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="id",
    y_col=None,
    target_size=(32, 32),
    batch_size=1,
    shuffle=False,
    class_mode=None,
)



## === cell 25
pass



## === cell 26
pass



## === cell 27
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

efficient_net = EfficientNetClass(
    weights="imagenet", input_shape=(32, 32, 3), include_top=False, pooling="max"
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.summary()



## === cell 28
pass



## === cell 29
from tensorflow.keras.metrics import AUC

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auc")],
)



## === cell 30
pass



## === cell 31
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_auc",
    verbose=1,
    save_best_only=True,
    mode="max",
)



## === cell 32
pass



## === cell 33
history = model.fit(
    train_generator,
    epochs=15,  # a few more epochs for better AUC
    validation_data=val_generator,
    callbacks=[checkpoint],
)



## === cell 34
pass



## === cell 35
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
auc = history.history["auc"]
val_auc = history.history["val_auc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label="Training Accuracy")
plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, auc, "go", label="Training AUC")
plt.plot(epochs, val_auc, "g", label="Validation AUC")
plt.title("Training and Validation AUC")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()



## === cell 36
pass



## === cell 37
from tensorflow.keras.models import load_model

if os.path.exists("best_model.h5"):
    best_model = load_model("best_model.h5")
else:
    best_model = model  # fallback to the last trained model

preds = best_model.predict(
    test_generator, steps=len(test_generator.filenames), verbose=1
)



## === cell 38
image_ids = [os.path.basename(path) for path in test_generator.filenames]
predictions = preds.flatten()
submission = pd.DataFrame({"id": image_ids, "has_cactus": predictions})
print(submission.head())



## === cell 39
pass



## === cell 40
submission.to_csv("/kaggle/working/submission.csv", index=False)



## === cell 41
print(os.listdir("/kaggle/working"))
