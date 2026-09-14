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

3.10

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import shutil

print("TensorFlow:", tf.__version__)



## === cell 2
import zipfile

train_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"


def unzip_to_working(zip_path, dest="/kaggle/working"):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


unzip_to_working(train_zip, "/kaggle/working")
unzip_to_working(test_zip, "/kaggle/working")

candidate_train_dirs = [
    "/kaggle/working/train",
    "/kaggle/working/train/train",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/train",
]
candidate_test_dirs = [
    "/kaggle/working/test",
    "/kaggle/working/test/test",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test/test",
]


def first_existing_with_jpg(paths):
    for p in paths:
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if f.lower().endswith(".jpg")]
            if len(files) > 0:
                return p
    return None


train_dir = first_existing_with_jpg(candidate_train_dirs)
test_dir = first_existing_with_jpg(candidate_test_dirs)

print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)
assert train_dir is not None, "Could not find extracted train directory with jpgs."
assert test_dir is not None, "Could not find extracted test directory with jpgs."



## === cell 3
datasets_train = sorted(
    [f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")]
)
datasets_test = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])

print("n_train:", len(datasets_train), "n_test:", len(datasets_test))
print("train sample:", datasets_train[:5])
print("test sample :", datasets_test[:5])



## === cell 4
labels = []
for imagename in datasets_train:
    if "dog" in imagename:
        labels.append("dog")
    elif "cat" in imagename:
        labels.append("cat")
    else:
        labels.append(None)

dfx = (
    pd.DataFrame({"imagename": datasets_train, "labels": labels})
    .dropna()
    .reset_index(drop=True)
)
dftest = pd.DataFrame({"image": datasets_test}).reset_index(drop=True)

print(dfx.head())
print(dftest.head())




## === cell 5
def show_image(imageadd):
    image = cv2.imread(imageadd)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.title(os.path.basename(imageadd))
    plt.imshow(image)
    plt.axis("off")


show_image(os.path.join(train_dir, dfx.loc[0, "imagename"]))



## === cell 6
idg = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    rotation_range=25,
    validation_split=0.0,  # keep identical semantics: use all data for training, no validation
)

bs = 32

train_idg = idg.flow_from_dataframe(
    dfx,
    directory=train_dir,
    x_col="imagename",
    y_col="labels",
    target_size=(180, 200),
    batch_size=bs,
    class_mode="categorical",
    shuffle=True,
)

print("class_indices:", train_idg.class_indices)



## === cell 7
VGG16transfer = tf.keras.applications.VGG16(
    include_top=False,
    input_shape=(180, 200, 3),
    weights="imagenet",
)
for layer in VGG16transfer.layers:
    layer.trainable = False

flat1 = tf.keras.layers.Flatten()(VGG16transfer.output)
d1 = tf.keras.layers.Dense(32, activation="relu")(flat1)
pred = tf.keras.layers.Dense(2, activation="softmax")(d1)

model1 = tf.keras.Model(inputs=[VGG16transfer.input], outputs=[pred])
model1.summary()



## === cell 8
model1.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0031415),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=["acc"],
)

history = model1.fit(train_idg, epochs=1, batch_size=bs)



## === cell 9
idg2 = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

test_gen = idg2.flow_from_dataframe(
    dftest,
    directory=test_dir,
    x_col="image",
    class_mode=None,
    target_size=(180, 200),
    batch_size=bs,
    shuffle=False,
)

predict = model1.predict(test_gen, batch_size=bs, max_queue_size=1, verbose=1)
print("predict shape:", predict.shape)



## === cell 10
class_indices = train_idg.class_indices
dog_col = class_indices.get("dog", 1)

dog_prob = predict[:, dog_col].astype(np.float64)

ids = dftest["image"].str.replace(".jpg", "", regex=False).astype(int)

result = pd.DataFrame({"id": ids, "label": dog_prob})
result = result.sort_values("id").reset_index(drop=True)

assert list(result.columns) == ["id", "label"]
assert result["id"].is_monotonic_increasing
assert len(result) == len(dftest)

result.to_csv("submission.csv", index=False)
print(result.head())
print("Wrote submission.csv with", len(result), "rows")
