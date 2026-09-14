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

# 5. Target score

0.7414026666666667

# 6. Current score

0.28664

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.28664) has done: 'I fix the missing `/kaggle/working/train` and `/kaggle/working/test` directories by detecting the actual extracted folder structure after unzipping and pointing `train_dir`/`test_dir` to the correct locations, which resolves the “PyDataset has length 0” and image FileNotFoundErrors. I also fix the TensorFlow import crash (`MessageFactory` / protobuf incompatibility) by using the already-installed `tf_keras`/Keras 2 API fallback that works in this Kaggle image, without changing your model architecture or training loop semantics. Finally, I make the preprocessing output dtype valid for Keras (`float32`), ensure consistent rescaling between train/val/test, and always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
COMP_ROOT = "/kaggle/input/aerial-cactus-identification"
if not os.path.exists(COMP_ROOT):
    COMP_ROOT = "/kaggle/input"

print("COMP_ROOT =", COMP_ROOT)
print("Exists train.zip:", os.path.exists(os.path.join(COMP_ROOT, "train.zip")))
print("Exists test.zip:", os.path.exists(os.path.join(COMP_ROOT, "test.zip")))
print("Exists train.csv:", os.path.exists(os.path.join(COMP_ROOT, "train.csv")))
print(
    "Exists sample_submission.csv:",
    os.path.exists(os.path.join(COMP_ROOT, "sample_submission.csv")),
)



## === cell 2
import zipfile

extract_dir = "/kaggle/working"

train_zip_path = os.path.join(COMP_ROOT, "train.zip")
test_zip_path = os.path.join(COMP_ROOT, "test.zip")

with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction done to:", extract_dir)



## === cell 3
for dirname, _, _ in os.walk("/kaggle/working"):
    if dirname.count(os.sep) <= "/kaggle/working".count(os.sep) + 2:
        print(dirname)



## === cell 4


def find_dir_with_jpgs(base, dirname_candidates):
    for d in dirname_candidates:
        cand = os.path.join(base, d)
        if os.path.isdir(cand):
            try:
                files = os.listdir(cand)
            except Exception:
                continue
            if any(f.lower().endswith(".jpg") for f in files):
                return cand
    return None


train_dir = find_dir_with_jpgs(
    "/kaggle/working", ["train", "aerial-cactus-identification/train"]
) or find_dir_with_jpgs("/kaggle/working/aerial-cactus-identification", ["train"])

test_dir = find_dir_with_jpgs(
    "/kaggle/working", ["test", "aerial-cactus-identification/test"]
) or find_dir_with_jpgs("/kaggle/working/aerial-cactus-identification", ["test"])

if train_dir is None or test_dir is None:
    for root, dirs, files in os.walk("/kaggle/working"):
        if root.count(os.sep) > "/kaggle/working".count(os.sep) + 5:
            continue
        base = os.path.basename(root)
        if base == "train" and any(f.lower().endswith(".jpg") for f in files):
            train_dir = root
        if base == "test" and any(f.lower().endswith(".jpg") for f in files):
            test_dir = root
    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Could not locate extracted train/test image folders. train_dir={train_dir}, test_dir={test_dir}"
        )

print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)

train_csv_path = os.path.join(COMP_ROOT, "train.csv")
sample_sub_path = os.path.join(COMP_ROOT, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
print(train_df.head())
print(
    "train_dir exists:",
    os.path.exists(train_dir),
    "n_files:",
    len(os.listdir(train_dir)) if os.path.exists(train_dir) else None,
)
print(
    "test_dir exists:",
    os.path.exists(test_dir),
    "n_files:",
    len(os.listdir(test_dir)) if os.path.exists(test_dir) else None,
)




## === cell 5
def count_files(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images: {test_count}")



## === cell 6
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 7
import matplotlib.pyplot as plt

counts = train_df["has_cactus"].value_counts()
labels = ["Has Cactus (1)", "No Cactus (0)"]
colors = ["lightgreen", "lightcoral"]

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
plt.title("Distribution of Cactus Presence (has_cactus)")
plt.axis("equal")
plt.show()



## === cell 8
import cv2

cactus = []
idxs = [0, 1, 2, 8, 9, 12, 6, 7, 11, 14, 16, 17]
for i in idxs:
    fp = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(fp)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {fp}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus.append(img)

labels_vis = ["cactus"] * 6 + ["no cactus"] * 6

plt.figure(figsize=[10, 10])
for x in range(12):
    plt.subplot(4, 3, x + 1)
    plt.imshow(cactus[x])
    plt.title(labels_vis[x])
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 9
cactus2 = []
for i in range(12):
    fp = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(fp)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {fp}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus2.append(img)

plt.figure(figsize=(10, 10))
for x in range(12):
    plt.subplot(4, 3, x + 1)
    plt.imshow(cactus2[x])
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 10
import tf_keras as tf
from tf_keras import callbacks
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras.optimizers import Adam
from tf_keras.applications import EfficientNetB3

print("Using tf_keras version:", tf.__version__)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 11
train_df["has_cactus"] = train_df["has_cactus"].astype(str)



## === cell 12
from tf_keras.preprocessing.image import ImageDataGenerator
import random


def custom_preprocessing(image):
    k = random.randint(0, 3)
    image = np.rot90(image, k)

    if random.random() > 0.5:
        image = np.fliplr(image)

    if random.random() > 0.5:
        image = np.flipud(image)

    factor = random.uniform(0.8, 1.2)
    image = np.clip(image * factor, 0, 255).astype(np.float32) / 255.0
    return image


train_datagen = ImageDataGenerator(
    validation_split=0.10,
    preprocessing_function=custom_preprocessing,
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    subset="training",
    batch_size=1024,
    shuffle=True,
    class_mode="binary",
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    subset="validation",
    batch_size=256,
    shuffle=True,
    class_mode="binary",
)

print("train batches:", len(train_generator), "val batches:", len(val_generator))
if len(train_generator) == 0 or len(val_generator) == 0:
    raise ValueError(
        f"Empty generator(s). Check paths and csv ids. len(train)={len(train_generator)} len(val)={len(val_generator)}"
    )



## === cell 13
cactus = []
for i in range(12):
    fp = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(fp)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {fp}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus.append(img)

cactus_augmented = [custom_preprocessing(img) for img in cactus]

plt.figure(figsize=(10, 10))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.imshow(cactus_augmented[i])
    plt.title(f"Image {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()



## === cell 14
sample_sub = pd.read_csv(sample_sub_path)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=sample_sub,
    directory=test_dir,
    x_col="id",
    y_col=None,
    target_size=(32, 32),
    batch_size=32,
    shuffle=False,
    class_mode=None,
)

print("test batches:", len(test_generator), "n_test:", test_generator.n)
if len(test_generator) == 0:
    raise ValueError("Empty test_generator. Check test_dir and sample_submission ids.")



## === cell 15
efficient_net = EfficientNetB3(
    weights="imagenet", input_shape=(32, 32, 3), include_top=False, pooling="max"
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.summary()



## === cell 16
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 17
history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=15,
    validation_data=val_generator,
    validation_steps=7,
)



## === cell 18
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label="Training Accuracy")
plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()



## === cell 19
preds = model.predict(test_generator, steps=len(test_generator), verbose=1)



## === cell 20
predictions = preds.reshape(-1)
submission = sample_sub.copy()
submission["has_cactus"] = predictions[: len(submission)]
submission["has_cactus"] = submission["has_cactus"].astype(float).clip(0.0, 1.0)
print(submission.head())
print(submission.shape)



## === cell 21
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)



## === cell 22
print(os.listdir("/kaggle/working"))
print("submission.csv exists:", os.path.exists("/kaggle/working/submission.csv"))
print(
    "submission.csv size:",
    (
        os.path.getsize("/kaggle/working/submission.csv")
        if os.path.exists("/kaggle/working/submission.csv")
        else None
    ),
)
