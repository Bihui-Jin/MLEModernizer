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

3.9

# 3. Installed packages



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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split



## === cell 1
import zipfile

INPUT_ROOT = "../input/dogs-vs-cats-redux-kernels-edition"
WORK_ROOT = "./dogs-vs-cats-data"

os.makedirs(WORK_ROOT, exist_ok=True)

train_zip = os.path.join(INPUT_ROOT, "train.zip")
test_zip = os.path.join(INPUT_ROOT, "test.zip")

train_extract_root = os.path.join(WORK_ROOT, "train")
test_extract_root = os.path.join(WORK_ROOT, "test")

if (not os.path.isdir(train_extract_root)) or (
    len(os.listdir(train_extract_root)) == 0
):
    with zipfile.ZipFile(train_zip, "r") as z:
        z.extractall(WORK_ROOT)

if (not os.path.isdir(test_extract_root)) or (len(os.listdir(test_extract_root)) == 0):
    with zipfile.ZipFile(test_zip, "r") as z:
        z.extractall(WORK_ROOT)


def _find_image_dir(root_dir: str) -> str:
    """
    Robustly find a directory containing .jpg files, supporting common nested layouts after extraction:
      - WORK_ROOT/train/train/*.jpg
      - WORK_ROOT/test/test/*.jpg
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root dir does not exist: {root_dir}")

    jpgs = [f for f in os.listdir(root_dir) if f.lower().endswith(".jpg")]
    if jpgs:
        return root_dir

    for sub in ["train", "test", "unknown"]:
        cand = os.path.join(root_dir, sub)
        if os.path.isdir(cand):
            jpgs = [f for f in os.listdir(cand) if f.lower().endswith(".jpg")]
            if jpgs:
                return cand

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(fn.lower().endswith(".jpg") for fn in filenames):
            return dirpath
        rel = os.path.relpath(dirpath, root_dir)
        if rel != "." and rel.count(os.sep) >= 4:
            dirnames[:] = []

    raise FileNotFoundError(f"Could not find any .jpg files under: {root_dir}")


TRAIN_DIR = _find_image_dir(
    train_extract_root if os.path.isdir(train_extract_root) else WORK_ROOT
)
TEST_DIR = _find_image_dir(
    test_extract_root if os.path.isdir(test_extract_root) else WORK_ROOT
)

print(
    "TRAIN_DIR:",
    TRAIN_DIR,
    "num_files:",
    len([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")]),
)
print(
    "TEST_DIR:",
    TEST_DIR,
    "num_files:",
    len([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]),
)



## === cell 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3
BATCH_SIZE = 256



## === cell 3
filenames = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")]
categories = []
for filename in filenames:
    category = filename.split(".")[0]
    if category == "dog":
        categories.append(1)
    else:
        categories.append(0)



## === cell 4
all_data = pd.DataFrame(
    {
        "filename": filenames,
        "category": categories,
    },
    dtype="str",
)

all_data.head(), all_data.shape



## === cell 5
index = min(357, len(all_data) - 1)
sample_img_filename, sample_img_label = all_data.iloc[index, :]
sample_img_label = int(sample_img_label)
sample_img = plt.imread(os.path.join(TRAIN_DIR, sample_img_filename))
plt.figure(figsize=(3, 3))
plt.imshow(sample_img)
plt.axis("off")
print("Label: {}({})".format(["Cat", "Dog"][sample_img_label], sample_img_label))



## === cell 6
train_data, validation_data = train_test_split(
    all_data, test_size=0.05, shuffle=True, random_state=2
)
train_data = train_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)

train_data.shape, validation_data.shape



## === cell 7
num_train = train_data.shape[0]
num_val = validation_data.shape[0]
num_train, num_val



## === cell 8
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Activation,
)

tf.random.set_seed(2)
np.random.seed(2)



## === cell 9
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory=TRAIN_DIR,
    x_col="filename",
    y_col="category",
    class_mode="binary",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)



## === cell 10
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_dataframe(
    validation_data,
    directory=TRAIN_DIR,
    x_col="filename",
    y_col="category",
    class_mode="binary",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)



## === cell 11
example_generator = train_datagen.flow_from_dataframe(
    train_data.sample(n=1, random_state=0),
    directory=TRAIN_DIR,
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    batch_size=15,
)



## === cell 12
plt.figure(figsize=(12, 12))
example_data = None
for i in range(15):
    example_data = next(example_generator)
    plt.subplot(5, 3, i + 1)
    image = np.squeeze(example_data[0][0])  # first image in batch
    plt.imshow(image)
    plt.axis("off")

label = int(example_data[1][0])
print("Label: {}({})".format(["Cat", "Dog"][label], label))



## === cell 13
pretrained_base = VGG19(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
    pooling=None,
)

for layer in pretrained_base.layers[:5]:
    layer.trainable = True
for layer in pretrained_base.layers[5:]:
    layer.trainable = False

pretrained_base.summary()



## === cell 14
model = Sequential(
    [
        pretrained_base,
        Flatten(),
        Dropout(0.2),
        Dense(512),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Dense(128),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Dense(32),
        BatchNormalization(),
        Activation("relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.summary()



## === cell 15
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 16
steps_per_epoch = max(1, num_train // BATCH_SIZE)
validation_steps = max(1, num_val // BATCH_SIZE)

history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_steps,
)



## === cell 17
test_filenames = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
test_filenames = sorted(test_filenames, key=lambda x: int(os.path.splitext(x)[0]))

test_data = pd.DataFrame({"filename": test_filenames}, dtype="str")
test_data.head(), test_data.shape



## === cell 18
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_dataframe(
    test_data,
    directory=TEST_DIR,
    x_col="filename",
    y_col=None,
    target_size=IMAGE_SIZE,
    class_mode=None,
    shuffle=False,
    batch_size=1,
)



## === cell 19
predictions = model.predict(
    x=test_generator, batch_size=1, steps=test_data.shape[0], verbose=1
)
predictions = np.squeeze(predictions)
predictions.shape



## === cell 20
ids = np.array([int(os.path.splitext(f)[0]) for f in test_filenames], dtype=int)
ids.shape, ids[:5]



## === cell 21
submission = pd.DataFrame({"id": ids, "label": predictions.astype(float)})
submission = submission.sort_values("id").reset_index(drop=True)
submission["label"] = submission["label"].clip(1e-7, 1 - 1e-7)

submission.head()



## === cell 22
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.describe(include="all"))
