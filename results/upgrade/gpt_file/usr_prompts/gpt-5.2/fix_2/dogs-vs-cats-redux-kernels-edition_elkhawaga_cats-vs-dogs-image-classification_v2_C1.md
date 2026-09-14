# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os, zipfile, shutil, re, random
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow import keras

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## === cell 1
train_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
work_root = "/kaggle/working"
extract_root = os.path.join(work_root, "extracted")

if os.path.exists(extract_root):
    shutil.rmtree(extract_root)
os.makedirs(extract_root, exist_ok=True)

with zipfile.ZipFile(train_zip_path, "r") as z:
    z.extractall(extract_root)

train_dir = None
for root, dirs, files in os.walk(extract_root):
    if any(f.startswith("cat.") and f.endswith(".jpg") for f in files) and any(
        f.startswith("dog.") and f.endswith(".jpg") for f in files
    ):
        train_dir = root
        break

if train_dir is None:
    for root, dirs, files in os.walk(extract_root):
        if any(f.lower().endswith(".jpg") for f in files):
            train_dir = root
            break

if train_dir is None:
    raise FileNotFoundError(
        "Could not locate extracted train images folder from train.zip."
    )

print("Train images dir:", train_dir)
print("Num files in train_dir:", len(os.listdir(train_dir)))
print("First 10 files:", sorted(os.listdir(train_dir))[:10])



## === cell 2
for root, dirs, files in os.walk(extract_root):
    level = root.replace(extract_root, "").count(os.sep)
    if level > 3:
        continue
    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 4 * (level + 1)
    for f in sorted(files)[:5]:
        print(f"{subindent}{f}")



## === cell 3
train_image_paths = []
train_labels = []

for fname in os.listdir(train_dir):
    if not fname.lower().endswith(".jpg"):
        continue
    fpath = os.path.join(train_dir, fname)
    if os.path.getsize(fpath) <= 0:
        continue
    if fname.startswith("cat."):
        train_labels.append(0)
        train_image_paths.append(fpath)
    elif fname.startswith("dog."):
        train_labels.append(1)
        train_image_paths.append(fpath)

print("Loaded paths:", len(train_image_paths), "labels:", len(train_labels))
print("Cats:", train_labels.count(0), "Dogs:", train_labels.count(1))
if len(train_image_paths) == 0:
    raise RuntimeError(
        "No training images were found; check extraction and train_dir detection."
    )



## === cell 4
img_size = (150, 150)

train_images = []
kept_labels = []
for path, y in zip(train_image_paths, train_labels):
    img = cv2.imread(path)  # BGR
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    train_images.append(img)
    kept_labels.append(y)

train_images = np.array(train_images, dtype="float32")
train_labels = np.array(kept_labels, dtype="int32")

print("Train images array:", train_images.shape, train_images.dtype)
print("Train labels array:", train_labels.shape, train_labels.dtype)



## === cell 5
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=SEED, stratify=train_labels
)

print("X_train:", X_train.shape, "X_val:", X_val.shape)



## === cell 6
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## === cell 7
train_generator = train_datagen.flow(
    X_train, y_train, batch_size=32, shuffle=True, seed=SEED
)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)

batch_x, batch_y = next(train_generator)
print(
    "Batch X:", batch_x.shape, batch_x.min(), batch_x.max(), "Batch y:", batch_y.shape
)



## === cell 8
import matplotlib.pyplot as plt

n_show = 16
idx = np.random.choice(len(X_train), min(n_show, len(X_train)), replace=False)

plt.figure(figsize=(10, 10))
for i, index in enumerate(idx):
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_train[index].astype(np.uint8))
    plt.title("Dog" if y_train[index] == 1 else "Cat")
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 9
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential(
    [
        Conv2D(16, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        Conv2D(256, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(256, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 10
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop],
    verbose=1,
)



## === cell 11
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



## === cell 12
test_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"
test_extract_root = os.path.join(extract_root, "test_extract")

if os.path.exists(test_extract_root):
    shutil.rmtree(test_extract_root)
os.makedirs(test_extract_root, exist_ok=True)

with zipfile.ZipFile(test_zip_path, "r") as z:
    z.extractall(test_extract_root)

test_images_dir = None
for root, dirs, files in os.walk(test_extract_root):
    if any(f.lower().endswith(".jpg") for f in files):
        test_images_dir = root
        break

if test_images_dir is None:
    raise FileNotFoundError(
        "Could not locate extracted test images folder from test.zip."
    )

print("Test images dir:", test_images_dir)
print(
    "Num test files:",
    len([f for f in os.listdir(test_images_dir) if f.lower().endswith(".jpg")]),
)
print(
    "First 5 test files:",
    sorted([f for f in os.listdir(test_images_dir) if f.lower().endswith(".jpg")])[:5],
)



## === cell 13
parent_dir = os.path.dirname(test_images_dir)
class_name = os.path.basename(test_images_dir)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    directory=parent_dir,
    classes=[class_name],
    target_size=img_size,
    batch_size=32,
    class_mode=None,
    shuffle=False,
)

ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in test_generator.filenames]
print("Test generator samples:", test_generator.samples, "First 5 ids:", ids[:5])



## === cell 14
preds = model.predict(test_generator, verbose=1).ravel()

eps = 1e-7
preds = np.clip(preds, eps, 1 - eps)

submission = pd.DataFrame({"id": ids, "label": preds})
submission["id"] = submission["id"].astype(int)
submission = submission.sort_values("id").reset_index(drop=True)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head(10))
print(submission.tail(10))
print("Rows:", len(submission), "Cols:", submission.columns.tolist())
