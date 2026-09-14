# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the individual whale species in images.

## Metric
Mean Average Precision @ 5 (MAP@5).

## Submission Format
For each `Image` in the test set, you may predict up to 5 labels for the whale `Id`. Whales that are not predicted to be one of the labels in the training data should be labeled as `new_whale`. The file should contain a header and have the following format:

```
Image,Id
00029b3a.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
0003c693.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
...
```

## Dataset
This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an `Id`. The challenge is to predict the whale `Id` of images in the test set. What makes this such a challenge is that there are only a few examples for each of 3,000+ whale Ids.

- **train.zip** - a folder containing the training images
- **train.csv** - maps the training `Image` to the appropriate whale `Id`. Whales that are not predicted to have a label identified in the training data should be labeled as `new_whale`.
- **test.zip** - a folder containing the test images to predict the whale `Id`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        input/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        working/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
```

-> data/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> input/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> input/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
)

import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pylab as plt
import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
tf.random.set_seed(42)

BASE_INPUT = "../input"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input"

print("Using BASE_INPUT:", BASE_INPUT)
print("Listing input dir:")
print("\n".join(sorted(os.listdir(BASE_INPUT))[:50]))




## === cell 1
def resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


train_csv_path = resolve_path(
    os.path.join(BASE_INPUT, "train.csv"),
    os.path.join(BASE_INPUT, "whale-categorization-playground", "train.csv"),
)
sample_sub_path = resolve_path(
    os.path.join(BASE_INPUT, "sample_submission.csv"),
    os.path.join(
        BASE_INPUT, "whale-categorization-playground", "sample_submission.csv"
    ),
)
train_dir = resolve_path(
    os.path.join(BASE_INPUT, "train"),
    os.path.join(BASE_INPUT, "whale-categorization-playground", "train"),
)
test_dir = resolve_path(
    os.path.join(BASE_INPUT, "test"),
    os.path.join(BASE_INPUT, "whale-categorization-playground", "test"),
)

print("train_csv_path:", train_csv_path)
print("sample_sub_path:", sample_sub_path)
print("train_dir:", train_dir)
print("test_dir:", test_dir)

train_images = sorted(glob(os.path.join(train_dir, "*jpg")))
test_images = sorted(glob(os.path.join(test_dir, "*jpg")))

df = pd.read_csv(train_csv_path)
df["ImagePath"] = df["Image"].map(lambda x: os.path.join(train_dir, x))
ImageToLabelDict = dict(zip(df["ImagePath"], df["Id"]))

print("Train images found:", len(train_images))
print("Test images found:", len(test_images))
print("Train CSV rows:", len(df))



## === cell 2
SIZE = 64


def ImportImage(filename):
    img = Image.open(filename).convert("LA").resize((SIZE, SIZE))
    return np.array(img)[:, :, 0]


train_img = np.array([ImportImage(img) for img in train_images])
x = train_img



## === cell 3
print("%d training images" % x.shape[0])

print("Nbr of samples/class\tNbr of classes")
for index, val in df["Id"].value_counts().value_counts().sort_index().items():
    print("%d\t\t\t%d" % (index, val))



## === cell 4
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class LabelOneHotEncoder:
    def __init__(self):
        self.ohe = OneHotEncoder(sparse=True, handle_unknown="ignore")
        self.le = LabelEncoder()

    def fit_transform(self, x):
        features = self.le.fit_transform(x)
        return self.ohe.fit_transform(features.reshape(-1, 1))

    def transform(self, x):
        features = self.le.transform(x)
        return self.ohe.transform(features.reshape(-1, 1))

    def inverse_labels(self, class_indices):
        class_indices = np.asarray(class_indices, dtype=np.int64).reshape(-1)
        ohe_categories = self.ohe.categories_[0]
        int_labels = ohe_categories[class_indices]
        return self.le.inverse_transform(int_labels.astype(np.int64))


y = list(map(ImageToLabelDict.get, train_images))
lohe = LabelOneHotEncoder()
y_cat = lohe.fit_transform(y)




## === cell 5
def plotImages(images_arr, n_images=4):
    fig, axes = plt.subplots(n_images, n_images, figsize=(12, 12))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        if img.ndim != 2:
            img = img.reshape((SIZE, SIZE))
        ax.imshow(img, cmap="Greys_r")
        ax.set_xticks(())
        ax.set_yticks(())
    plt.tight_layout()


plotImages(x)



## === cell 6
x = x.reshape((-1, SIZE, SIZE, 1))
input_shape = x[0].shape
x_train = x.astype("float32")
y_train = y_cat

image_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
)

image_gen.fit(x_train, augment=True)

augmented_images, _ = next(
    image_gen.flow(x_train, y_train.toarray(), batch_size=4 * 4, shuffle=True)
)
plotImages(augmented_images)



## === cell 7
batch_size = 128
num_classes = y_cat.shape[1]
epochs = 5  # keep original

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print("num_classes:", num_classes)

model = Sequential()
model.add(
    tf.keras.layers.Conv2D(
        48, kernel_size=(3, 3), activation="relu", input_shape=input_shape
    )
)
model.add(tf.keras.layers.Conv2D(48, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(tf.keras.layers.Dropout(0.33))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(24, activation="relu"))
model.add(tf.keras.layers.Dropout(0.33))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)
model.summary()

model.fit(
    image_gen.flow(x_train, y_train.toarray(), batch_size=batch_size, shuffle=True),
    steps_per_epoch=25,
    epochs=epochs,
    verbose=1,
)



## === cell 8
sub_df = pd.read_csv(sample_sub_path)
assert "Image" in sub_df.columns and "Id" in sub_df.columns

test_path_map = {os.path.basename(p): p for p in test_images}

NEW_WHALE_TOKEN = "new_whale"

TOP1_PROB_THRESH = 0.55

TTA_PASSES = 3


def _predict_tta(xt_batch_1):
    probs = None
    for _ in range(TTA_PASSES):
        xa = next(image_gen.flow(xt_batch_1, batch_size=1, shuffle=False))
        xa = image_gen.standardize(xa.astype("float32"))
        p = model.predict(xa, verbose=0)[0].astype("float32")
        probs = p if probs is None else (probs + p)
    probs = probs / float(TTA_PASSES)
    return probs


out_path = "submission.csv"  # must end with .csv for Kaggle
with open(out_path, "w") as f:
    f.write("Image,Id\n")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        for image_name in sub_df["Image"].tolist():
            image_path = test_path_map.get(
                image_name, os.path.join(test_dir, image_name)
            )

            img = ImportImage(image_path)
            xt = img.astype("float32").reshape(1, SIZE, SIZE, 1)

            y_pred = _predict_tta(xt)

            order = np.argsort(y_pred)[::-1]
            top1 = float(y_pred[order[0]])

            uncertain = top1 < TOP1_PROB_THRESH

            predicted_tags = list(lohe.inverse_labels(order))
            final = []

            if uncertain:
                final.append(NEW_WHALE_TOKEN)

            for t in predicted_tags:
                if t not in final:
                    final.append(t)
                if len(final) == 5:
                    break

            if len(final) < 5:
                if NEW_WHALE_TOKEN not in final:
                    final.append(NEW_WHALE_TOKEN)
                for t in predicted_tags:
                    if t not in final:
                        final.append(t)
                    if len(final) == 5:
                        break

            f.write("{},{}\n".format(image_name, " ".join(final[:5])))

print("Wrote submission:", out_path)
print("Submission preview:")
print(pd.read_csv(out_path).head())
print("Submission rows:", len(pd.read_csv(out_path)))
assert len(pd.read_csv(out_path)) == len(sub_df)
assert list(pd.read_csv(out_path).columns) == ["Image", "Id"]

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
