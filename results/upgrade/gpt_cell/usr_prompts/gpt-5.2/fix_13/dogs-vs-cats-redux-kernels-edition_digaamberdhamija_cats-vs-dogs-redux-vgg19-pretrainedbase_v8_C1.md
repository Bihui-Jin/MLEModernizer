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
! unzip "../input/dogs-vs-cats-redux-kernels-edition/train.zip"
! unzip "../input/dogs-vs-cats-redux-kernels-edition/test.zip"


## === cell 1
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


## === cell 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3
BATCH_SIZE = 256


## === cell 3
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

filenames = os.listdir(train_dir)
categories = []
for filename in filenames:
    category = filename.split(".")[0]
    if category == "dog":
        categories.append(1)
    else:
        categories.append(0)


## === cell 4
all_data = pd.DataFrame({
    "filename": filenames,
    "category": categories,
}, dtype = "str")


## === cell 5
index = 357
if len(all_data) == 0:
    raise ValueError(f"No files found in training directory: {train_dir}")

valid_mask = all_data["filename"].apply(
    lambda fn: os.path.isfile(os.path.join(train_dir, fn))
)
valid_data = all_data[valid_mask].reset_index(drop=True)

if len(valid_data) == 0:
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
        raise ValueError(
            f"No image files found directly under training directory: {train_dir}"
        )
else:
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


## === cell 6
train_data, validation_data = train_test_split(all_data, test_size = 0.05, shuffle = True, random_state = 2)

train_data = train_data.reset_index(drop = True)
validation_data = validation_data.reset_index(drop = True)

train_data.shape, validation_data.shape


## === cell 7
num_train = train_data.shape[0]
num_val = validation_data.shape[0]


## === cell 8
import os as _os

try:
    import google.protobuf  # noqa: F401
    from packaging.version import (
        Version,
    )  # packaging is typically available in Kaggle envs
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


## === cell 9
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 15,
    shear_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory = "train/",
    x_col = "filename",
    y_col = "category",
    class_mode = "binary",
    target_size = IMAGE_SIZE,
    batch_size = BATCH_SIZE,
)


## === cell 10
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

_validation_df = validation_data
_validation_dir = "train/"

if (
    "valid_data" in globals()
    and isinstance(valid_data, pd.DataFrame)
    and len(valid_data) > 0
):
    _validation_df = valid_data.sample(frac=0.05, random_state=2).reset_index(drop=True)

    if "in_subfolders" in globals() and in_subfolders and "train_dir" in globals():
        _validation_dir = train_dir
        _validation_df = _validation_df.copy()
        _validation_df["filename"] = _validation_df.apply(
            lambda r: os.path.join(
                "dog" if int(r["category"]) == 1 else "cat", r["filename"]
            ),
            axis=1,
        )

validation_generator = validation_datagen.flow_from_dataframe(
    _validation_df,
    directory=_validation_dir,
    x_col="filename",
    y_col="category",
    class_mode="binary",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)


## === cell 11
example_generator = train_datagen.flow_from_dataframe(
    train_data.sample(n = 1),
    directory = "train/",
    x_col = "filename",
    y_col = "category",
    target_size = IMAGE_SIZE,
    batch_size = 15,
)


## === cell 12
plt.figure(figsize=(12, 12))

last_example_data = None
for i in range(15):
    example_data = next(example_generator)

    while example_data[0].shape[0] == 0:
        example_data = next(example_generator)

    last_example_data = example_data
    plt.subplot(5, 3, i + 1)
    image = example_data[0][0]  # take first image in the batch -> (128, 128, 3)
    plt.imshow(image)

label = int(np.ravel(last_example_data[1])[0])
print("Label: {}({})".format(["Cat", "Dog"][label], label))


## === cell 13
pretrained_base = VGG19(
    include_top = False, 
    weights = "imagenet",
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
    pooling = None,
)

for layer in pretrained_base.layers[:5]:
    layer.trainable = True
for layer in pretrained_base.layers[5:]:
    layer.trainable = False
pretrained_base.summary()


## === cell 14
model = Sequential([
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
    Dense(1, activation = "sigmoid"),
])

model.summary()


## === cell 15
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


## === cell 16
model.fit(
    x = train_generator, 
    steps_per_epoch = num_train // BATCH_SIZE,
    epochs = 20,
    validation_data = validation_generator,
    validation_steps = num_val // BATCH_SIZE,
)


## === cell 17
filenames = os.listdir("test")
test_data = pd.DataFrame({
    "filename": filenames
}, dtype = "str")


## === cell 18
test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_dataframe(
    test_data,
    directory = "test",
    x_col = "filename",
    y_col = None,
    target_size = IMAGE_SIZE,
    class_mode = None,
    shuffle = False,
    batch_size = 1,
)


## === cell 19
predictions = model.predict(x = test_generator, batch_size = 1, steps = test_data.shape[0], verbose = 1)
predictions.shape


## === cell 20
predictions = np.squeeze(predictions)


## === cell 21
ids = np.arange(1, test_data.shape[0] + 1, 1)
ids.shape


## === cell 22
submission = pd.DataFrame({
    "id": ids,
    "label": predictions,
})


## === cell 23
submission.to_csv("submission.csv", index = False)
