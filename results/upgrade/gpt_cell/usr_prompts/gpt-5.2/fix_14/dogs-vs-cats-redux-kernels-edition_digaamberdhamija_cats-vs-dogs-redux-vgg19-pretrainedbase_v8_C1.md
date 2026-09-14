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
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split



## === cell 1
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3
BATCH_SIZE = 256



## === cell 2
possible_train_dirs = [
    "train",
    "../input/dogs-vs-cats-redux-kernels-edition/train",
    "../input/train",
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train",
    "/kaggle/input/train",
    "/kaggle/data/train",
    "/kaggle/data/dogs-vs-cats-redux-kernels-edition/train",
]
train_dir = next((d for d in possible_train_dirs if os.path.isdir(d)), None)
if train_dir is None:
    raise FileNotFoundError(
        "Could not find the training directory. Tried: "
        + ", ".join(possible_train_dirs)
    )

entries = os.listdir(train_dir)
filenames = [f for f in entries if os.path.isfile(os.path.join(train_dir, f))]
if len(filenames) > 0:
    categories = (
        np.where(pd.Series(filenames).str.startswith("dog."), 1, 0).astype(str).tolist()
    )
else:
    categories = []  # handled in cell 5



## === cell 3
all_data = pd.DataFrame(
    {
        "filename": filenames,
        "category": categories,
    },
    dtype="str",
)



## === cell 4
index = 357
if len(all_data) == 0:
    cat_dir = os.path.join(train_dir, "cat")
    dog_dir = os.path.join(train_dir, "dog")

    if os.path.isdir(cat_dir) and os.path.isdir(dog_dir):
        cat_files = [
            f for f in os.listdir(cat_dir) if os.path.isfile(os.path.join(cat_dir, f))
        ]
        dog_files = [
            f for f in os.listdir(dog_dir) if os.path.isfile(os.path.join(dog_dir, f))
        ]

        valid_data = pd.DataFrame(
            {
                "filename": cat_files + dog_files,
                "category": (["0"] * len(cat_files)) + (["1"] * len(dog_files)),
            },
            dtype="str",
        ).reset_index(drop=True)
        in_subfolders = True
    else:
        raise ValueError(f"No image files found under training directory: {train_dir}")
else:
    valid_data = all_data.copy().reset_index(drop=True)
    in_subfolders = False

index = min(index, len(valid_data) - 1)

sample_img_filename, sample_img_label = valid_data.iloc[index, :]
sample_img_label = int(sample_img_label)

if in_subfolders:
    subfolder = "dog" if sample_img_label == 1 else "cat"
    sample_img_path = os.path.join(train_dir, subfolder, sample_img_filename)
else:
    sample_img_path = os.path.join(train_dir, sample_img_filename)

sample_img = plt.imread(sample_img_path)
plt.imshow(sample_img)
print("Label: {}({})".format(["Cat", "Dog"][sample_img_label], sample_img_label))



## === cell 5
train_data, validation_data = train_test_split(
    valid_data, test_size=0.05, shuffle=True, random_state=2
)

train_data = train_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)

train_data.shape, validation_data.shape



## === cell 6
num_train = train_data.shape[0]
num_val = validation_data.shape[0]



## === cell 7
import os as _os

try:
    import google.protobuf  # noqa: F401
    from packaging.version import Version  # noqa: F401
    import protobuf  # noqa: F401
except Exception:
    google = None  # type: ignore

try:
    import google.protobuf as _gp

    _pb_ver = getattr(_gp, "__version__", None)
    if _pb_ver is not None:
        try:
            from packaging.version import Version as _V

            if _V(_pb_ver) >= _V("4.21.0"):
                import sys, subprocess

                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
                )
        except Exception:
            pass
except Exception:
    pass

_os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
_os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Activation,
)



## === cell 8
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

_train_df = train_data
_train_dir = train_dir
if in_subfolders:
    _train_df = _train_df.copy()
    cats_mask = _train_df["category"].astype(int).values == 0
    _train_df.loc[cats_mask, "filename"] = (
        "cat/" + _train_df.loc[cats_mask, "filename"].astype(str).values
    )
    _train_df.loc[~cats_mask, "filename"] = (
        "dog/" + _train_df.loc[~cats_mask, "filename"].astype(str).values
    )

train_generator = train_datagen.flow_from_dataframe(
    _train_df,
    directory=_train_dir,
    x_col="filename",
    y_col="category",
    class_mode="binary",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)



## === cell 9
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

_val_df = validation_data
_val_dir = train_dir
if in_subfolders:
    _val_df = _val_df.copy()
    cats_mask = _val_df["category"].astype(int).values == 0
    _val_df.loc[cats_mask, "filename"] = (
        "cat/" + _val_df.loc[cats_mask, "filename"].astype(str).values
    )
    _val_df.loc[~cats_mask, "filename"] = (
        "dog/" + _val_df.loc[~cats_mask, "filename"].astype(str).values
    )

validation_generator = validation_datagen.flow_from_dataframe(
    _val_df,
    directory=_val_dir,
    x_col="filename",
    y_col="category",
    class_mode="binary",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)



## === cell 10
example_generator = train_datagen.flow_from_dataframe(
    _train_df.sample(n=1, random_state=2),
    directory=_train_dir,
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    batch_size=1,
)



## === cell 11
example_data = next(example_generator)
plt.figure(figsize=(4, 4))
plt.imshow(example_data[0][0])
label = int(np.ravel(example_data[1])[0])
print("Label: {}({})".format(["Cat", "Dog"][label], label))



## === cell 12
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



## === cell 13
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



## === cell 14
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 15
model.fit(
    x=train_generator,
    steps_per_epoch=num_train // BATCH_SIZE,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=num_val // BATCH_SIZE,
)



## === cell 16
possible_test_dirs = [
    "test",
    "../input/dogs-vs-cats-redux-kernels-edition/test",
    "../input/test",
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test",
    "/kaggle/input/test",
    "/kaggle/data/test",
    "/kaggle/data/dogs-vs-cats-redux-kernels-edition/test",
]
test_dir = next((d for d in possible_test_dirs if os.path.isdir(d)), None)
if test_dir is None:
    raise FileNotFoundError(
        "Could not find the test directory. Tried: " + ", ".join(possible_test_dirs)
    )

candidate_subdirs = [
    test_dir,
    os.path.join(test_dir, "test"),
    os.path.join(test_dir, "unknown"),
    os.path.join(test_dir, "test", "unknown"),
]
test_image_dir = next(
    (
        d
        for d in candidate_subdirs
        if os.path.isdir(d)
        and any(os.path.isfile(os.path.join(d, f)) for f in os.listdir(d))
    ),
    None,
)
if test_image_dir is None:
    raise ValueError(f"No image files found under test directory: {test_dir}")

filenames = [
    f
    for f in os.listdir(test_image_dir)
    if os.path.isfile(os.path.join(test_image_dir, f))
]
test_data = pd.DataFrame({"filename": filenames}, dtype="str")



## === cell 17
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_dataframe(
    test_data,
    directory=test_image_dir,
    x_col="filename",
    y_col=None,
    target_size=IMAGE_SIZE,
    class_mode=None,
    shuffle=False,
    batch_size=BATCH_SIZE,
)



## === cell 18
predictions = model.predict(x=test_generator, verbose=1)
predictions.shape



## === cell 19
predictions = np.squeeze(predictions)



## === cell 20
ids = (
    pd.Series(test_data["filename"])
    .str.replace(".jpg", "", regex=False)
    .astype(int)
    .values
)
order = np.argsort(ids)
ids = ids[order]
predictions = predictions[order]
ids.shape



## === cell 21
submission = pd.DataFrame(
    {
        "id": ids,
        "label": predictions,
    }
)



## === cell 22
submission.to_csv("submission.csv", index=False)
