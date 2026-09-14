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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        input/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
            test/
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
            train/
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        working/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

# 5. Code solution

## === cell 0
import os, random, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



## === cell 1
possible_paths = [
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/dogs-vs-cats-redux-kernels-edition",
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/working",
]
BASE_PATH = next((p for p in possible_paths if os.path.isdir(p)), None)
if BASE_PATH is None:
    raise FileNotFoundError("Dataset folder not found in any known location.")
train_dir = os.path.join(BASE_PATH, "train")
test_dir = os.path.join(BASE_PATH, "test")
print("train_dir exists:", os.path.isdir(train_dir))
print("test_dir exists :", os.path.isdir(test_dir))



## === cell 2
all_train_files = [f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")]
train_dogs = [
    os.path.join(train_dir, f) for f in all_train_files if f.lower().startswith("dog")
]
train_cats = [
    os.path.join(train_dir, f) for f in all_train_files if f.lower().startswith("cat")
]
test_imgs = [
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.lower().endswith(".jpg")
]
print(
    "num dogs :",
    len(train_dogs),
    "num cats :",
    len(train_cats),
    "num test :",
    len(test_imgs),
)



## === cell 3
train_imgs = train_dogs + train_cats



## === cell 4
random.shuffle(train_imgs)  # shuffle training list



## === cell 5
img_size = 150  # target size for the network


def read_and_process_image(list_of_images):
    """
    Returns three lists:
        X      – resized image arrays
        y      – labels (1 for dog, 0 for cat, 0 for test images)
        l_id   – string ids extracted from filenames
    Images that cannot be read are skipped.
    """
    X, y, l_id = [], [], []
    for img_path in list_of_images:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        X.append(img)
        fname = os.path.basename(img_path)
        img_id = os.path.splitext(fname)[0]  # e.g. "dog.10425" or "1332"
        l_id.append(img_id)
        if "dog" in img_path.lower():
            y.append(1)
        elif "cat" in img_path.lower():
            y.append(0)
        else:
            y.append(0)
    return X, y, l_id




## === cell 6
X, y, l_id = read_and_process_image(train_imgs)



## === cell 7
plt.figure(figsize=(20, 4))
columns = 5
for i in range(min(columns, len(X))):
    plt.subplot(1, columns, i + 1)
    plt.imshow(cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))
    plt.axis("off")
plt.show()



## === cell 8
X = np.array(X, dtype=np.float32) / 255.0  # scale now for convenience
y = np.array(y, dtype=np.int32)



## === cell 9
sns.countplot(x=y)
plt.title("Label distribution (0 = cat, 1 = dog)")
plt.show()



## === cell 10
print("Shape of X :", X.shape)
print("Shape of y :", y.shape)



## === cell 11
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=1, stratify=y
)



## === cell 12
del X
gc.collect()



## === cell 13
print("X_train shape:", X_train.shape)
print("X_val   shape:", X_val.shape)



## === cell 14
ntrain = X_train.shape[0]
nval = X_val.shape[0]



## === cell 15
conv_base = models.Sequential(
    [
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
    ]
)



## === cell 16
model = models.Sequential(
    [
        conv_base,
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



## === cell 17
batch_size = 128
train_datagen = ImageDataGenerator(
    rotation_range=30, horizontal_flip=True, fill_mode="nearest"
)
val_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)



## === cell 18
epochs = 30  # more epochs for better learning
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, ntrain // batch_size),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=max(1, nval // batch_size),
    verbose=2,
)



## === cell 19
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.plot(epochs_range, acc, "b", label="Training accuracy")
plt.plot(epochs_range, val_acc, "r", label="Validation accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.show()

plt.plot(epochs_range, loss, "b", label="Training loss")
plt.plot(epochs_range, val_loss, "r", label="Validation loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()



## === cell 20
X_test, _, test_ids = read_and_process_image(test_imgs)
X_test = np.array(X_test, dtype=np.float32) / 255.0
predictions = model.predict(X_test, batch_size=256).flatten()
submission = pd.DataFrame({"id": test_ids, "label": predictions})
submission["id"] = submission["id"].astype(int)
submission = submission.sort_values("id")
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv – first 5 rows:")
print(submission.head())
