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
import os, random, gc, math, sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from tf_keras import models, layers
from tf_keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass  # If TensorFlow is not available or mixed precision unsupported, continue safely




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
random.shuffle(train_imgs)  # shuffle training list




## === cell 4
labels = [1 if "dog" in p.lower() else 0 for p in train_imgs]
df = pd.DataFrame({"filename": train_imgs, "label": labels})

train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    random_state=1,
    stratify=df["label"],
)

print("Training samples :", len(train_df))
print("Validation samples :", len(val_df))




## === cell 5
print("Train dataframe head:")
print(train_df.head())
print("\nValidation dataframe head:")
print(val_df.head())




## === cell 6
ntrain = len(train_df)
nval = len(val_df)




## === cell 7
img_size = 150  # target size for the network
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




## === cell 8
model = models.Sequential(
    [
        conv_base,
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])




## === cell 9
batch_size = 128
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="filename",
    y_col="label",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="raw",
    shuffle=True,
)
val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="filename",
    y_col="label",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="raw",
    shuffle=False,
)




## === cell 10
epochs = 30
history = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(ntrain / batch_size),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=math.ceil(nval / batch_size),
    verbose=2,
    workers=4,  # use multiple processes for data loading
    use_multiprocessing=True,  # enable multiprocessing
)




## === cell 11
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs_range, acc, "b", label="Training accuracy")
plt.plot(epochs_range, val_acc, "r", label="Validation accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs_range, loss, "b", label="Training loss")
plt.plot(epochs_range, val_loss, "r", label="Validation loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()




## === cell 12
test_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in test_imgs]
test_df = pd.DataFrame({"filename": test_imgs, "id": test_ids})

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="filename",
    y_col=None,
    target_size=(img_size, img_size),
    batch_size=256,
    class_mode=None,
    shuffle=False,
)

predictions = model.predict(
    test_generator,
    steps=math.ceil(len(test_df) / 256),
    verbose=0,
).flatten()

submission = pd.DataFrame({"id": test_df["id"], "label": predictions})
submission = submission.sort_values("id")
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv – first 5 rows:")
print(submission.head())
