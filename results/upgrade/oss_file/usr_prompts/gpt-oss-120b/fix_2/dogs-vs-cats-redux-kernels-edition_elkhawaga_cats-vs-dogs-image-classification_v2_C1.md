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
import zipfile
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2



## === cell 1
train_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train"
assert os.path.isdir(train_dir), f"Train directory not found: {train_dir}"
print("Training images will be read from:", train_dir)



## === cell 2
train_image_paths = []
train_labels = []

for root, _, files in os.walk(train_dir):
    for fname in files:
        if not fname.lower().endswith(".jpg"):
            continue
        fpath = os.path.join(root, fname)
        if os.path.basename(root) == "cat":
            train_labels.append(0)
            train_image_paths.append(fpath)
        elif os.path.basename(root) == "dog":
            train_labels.append(1)
            train_image_paths.append(fpath)

print("Found", len(train_image_paths), "images.")
print("Cats:", train_labels.count(0), "Dogs:", train_labels.count(1))



## === cell 3
train_images = []
for path in train_image_paths:
    img = cv2.imread(path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    img = cv2.resize(img, (150, 150))
    train_images.append(img)



## === cell 4
train_images = np.array(train_images, dtype="float32") / 255.0
train_labels = np.array(train_labels, dtype="int32")
print("Image array shape:", train_images.shape)



## === cell 5
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)



## === cell 6
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator()



## === cell 9
train_generator = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)



## === cell 10
import matplotlib.pyplot as plt

idx = np.random.choice(len(train_images), 16, replace=False)

plt.figure(figsize=(10, 10))
for i, index in enumerate(idx):
    plt.subplot(4, 4, i + 1)
    plt.imshow(train_images[index])
    plt.title("Dog" if train_labels[index] == 1 else "Cat")
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 11
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

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



## === cell 12
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 13
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)



## === cell 14
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop],
    verbose=2,
)



## === cell 15
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



## === cell 16
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



## === cell 17
test_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test"
test_image_paths = [
    os.path.join(test_dir, fname)
    for fname in os.listdir(test_dir)
    if fname.lower().endswith(".jpg")
]
print("Found", len(test_image_paths), "test images.")

test_images = []
for path in test_image_paths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    test_images.append(img)

test_images = np.array(test_images, dtype="float32") / 255.0



## === cell 18
preds = model.predict(test_images, batch_size=32, verbose=1).ravel()

ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in test_image_paths]



## === cell 19
submission = pd.DataFrame({"id": ids, "label": preds})
submission = submission.sort_values("id")
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv. Preview:")
print(submission.head())
