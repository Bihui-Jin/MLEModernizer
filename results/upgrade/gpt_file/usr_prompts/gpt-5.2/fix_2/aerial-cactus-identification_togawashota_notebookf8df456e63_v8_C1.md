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
import random

random.seed(42)
np.random.seed(42)

BASE_INPUT = "/kaggle/input/aerial-cactus-identification"
BASE_WORK = "/kaggle/working"



## === cell 2
import zipfile

train_zip = os.path.join(BASE_INPUT, "train.zip")
test_zip = os.path.join(BASE_INPUT, "test.zip")

train_extract_dir = os.path.join(BASE_WORK, "train")
test_extract_dir = os.path.join(BASE_WORK, "test")

os.makedirs(train_extract_dir, exist_ok=True)
os.makedirs(test_extract_dir, exist_ok=True)

with zipfile.ZipFile(train_zip, "r") as zip_ref:
    zip_ref.extractall(train_extract_dir)

with zipfile.ZipFile(test_zip, "r") as zip_ref:
    zip_ref.extractall(test_extract_dir)

print("Extracted train to:", train_extract_dir)
print("Extracted test to:", test_extract_dir)



## === cell 3
for dirname, _, _ in os.walk("/kaggle/working"):
    print(dirname)




## === cell 4
def _find_image_dir(root_dir, expected_count_min=1000):
    candidates = [
        root_dir,
        os.path.join(root_dir, "train"),
        os.path.join(root_dir, "test"),
        os.path.join(root_dir, "train", "train"),
        os.path.join(root_dir, "test", "test"),
    ]
    try:
        for d in os.listdir(root_dir):
            p = os.path.join(root_dir, d)
            if os.path.isdir(p):
                candidates.append(p)
    except FileNotFoundError:
        pass

    def _count_jpgs(p):
        try:
            return sum(
                1
                for f in os.listdir(p)
                if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(p, f))
            )
        except Exception:
            return 0

    best = None
    best_count = -1
    for c in candidates:
        cnt = _count_jpgs(c)
        if cnt > best_count:
            best_count = cnt
            best = c
    if best is None or best_count < expected_count_min:
        raise FileNotFoundError(
            f"Could not find image directory under {root_dir}. Best={best} count={best_count}"
        )
    return best, best_count


train_dir, train_count = _find_image_dir(train_extract_dir, expected_count_min=10000)
test_dir, test_count = _find_image_dir(test_extract_dir, expected_count_min=1000)

print("Resolved train_dir:", train_dir, "count:", train_count)
print("Resolved test_dir:", test_dir, "count:", test_count)



## === cell 5
print(
    "Skipping external pip install; using tensorflow.keras EfficientNet implementation."
)



## === cell 6
import tensorflow as tf
from tensorflow import keras

print("TensorFlow:", tf.__version__)
print("Keras (tf.keras):", keras.__version__)



## === cell 7
train_df = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
train_df.head(20)




## === cell 8
def count_files(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


train_count2 = count_files(train_dir)
test_count2 = count_files(test_dir)

print(f"Train images: {train_count2}")
print(f"Test images: {test_count2}")



## === cell 9
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 10
import matplotlib.pyplot as plt

counts = train_df["has_cactus"].value_counts()

labels = ["Has Cactus (1)", "No Cactus (0)"]
colors = ["lightgreen", "lightcoral"]

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
plt.title("Distribution of Cactus Presence (has_cactus)")
plt.axis("equal")
plt.show()



## === cell 11
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)



## === cell 12
import cv2

cactus = []
idxs = [0, 1, 2, 8, 9, 12, 6, 7, 11, 14, 16, 17]
for i in idxs:
    p = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(p)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus.append(img)

labels_plot = ["cactus"] * 6 + ["no cactus"] * 6

plt.figure(figsize=[10, 10])
for x in range(min(12, len(cactus))):
    plt.subplot(4, 3, x + 1)
    plt.imshow(cactus[x])
    plt.title(labels_plot[x])
    plt.axis("off")

plt.tight_layout()
plt.show()



## === cell 13
cactus2 = []
for i in range(12):
    p = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(p)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus2.append(img)

plt.figure(figsize=(10, 10))
for x in range(min(12, len(cactus2))):
    plt.subplot(4, 3, x + 1)
    plt.imshow(cactus2[x])
    plt.axis("off")

plt.tight_layout()
plt.show()



## === cell 14
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential



## === cell 15
train_df["has_cactus"] = train_df["has_cactus"].astype("str")



## === cell 16
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    batch_size=128,
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
    batch_size=64,
    shuffle=True,
    class_mode="binary",
)



## === cell 17
cactus_vis = []
for i in range(12):
    p = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(p)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus_vis.append(img)

cactus_augmented = [custom_preprocessing(img) for img in cactus_vis]

plt.figure(figsize=(10, 10))
for i in range(min(12, len(cactus_augmented))):
    plt.subplot(4, 3, i + 1)
    plt.imshow(cactus_augmented[i])
    plt.title(f"Image {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()



## === cell 18
test_datagen = ImageDataGenerator(rescale=1 / 255)

test_root = os.path.join(BASE_WORK, "test_flow")
unknown_dir = os.path.join(test_root, "unknown")
os.makedirs(unknown_dir, exist_ok=True)

existing = len([f for f in os.listdir(unknown_dir) if f.lower().endswith(".jpg")])
if existing < test_count2:
    for f in os.listdir(test_dir):
        if not f.lower().endswith(".jpg"):
            continue
        src = os.path.join(test_dir, f)
        dst = os.path.join(unknown_dir, f)
        if os.path.exists(dst):
            continue
        try:
            os.symlink(src, dst)
        except Exception:
            import shutil

            shutil.copy2(src, dst)

test_generator = test_datagen.flow_from_directory(
    directory=test_root,
    target_size=(32, 32),
    batch_size=1,
    shuffle=False,
    class_mode=None,
)



## === cell 19
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

efficient_net = EfficientNetB3(
    weights="imagenet", input_shape=(32, 32, 3), include_top=False, pooling="max"
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.summary()



## === cell 20
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 21
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)

history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    class_weight=class_weights_dict,
)



## === cell 22
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



## === cell 23
preds = model.predict(test_generator, steps=len(test_generator), verbose=1)



## === cell 24
image_ids = [os.path.basename(name) for name in test_generator.filenames]
predictions = preds.flatten()

submission = pd.DataFrame({"id": image_ids, "has_cactus": predictions})

sample_sub = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))
submission = sample_sub[["id"]].merge(submission, on="id", how="left")

submission["has_cactus"] = submission["has_cactus"].astype(np.float32)
submission["has_cactus"] = submission["has_cactus"].fillna(0.5)

print(submission.head(20))
print("Submission shape:", submission.shape)



## === cell 25
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv")



## === cell 26
print(os.listdir("/kaggle/working"))
print("submission.csv exists:", os.path.exists("/kaggle/working/submission.csv"))
