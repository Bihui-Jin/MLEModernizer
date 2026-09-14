# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.07682

# 6. Current score

0.13514

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.87988) has done: 'I fix the data-loading crash by pointing the image scan directly at the competition’s `train/` and `test/` folders (your current `os.walk('../input')` never finds images because the real paths are nested), and I guard `fill_dict()` against empty path lists. Then I resolve the Keras 3 incompatibilities by switching to `tf_keras` (so `ImageDataGenerator` exists), updating the Adam optimizer argument (`learning_rate` instead of `lr`), and replacing deprecated `fit_generator`/`predict_classes` with `fit` and `argmax(model.predict())`. Finally, I ensure the submission is written as a valid `.csv` with the exact `file,species` columns and correct file ordering.'
- What this solution (achieved 0.86937) has done: 'The runtime crash happens before training: `tf_keras.utils.to_categorical` triggers a protobuf `MessageFactory.GetPrototype` incompatibility in this Kaggle image. To keep the exact same core model/training logic and make the notebook run end-to-end, I replace `to_categorical` with a small NumPy one-hot encoder (score-neutral) and keep stratified splitting unchanged. I also add a tiny safety fix to ensure `steps_per_epoch` is never zero (only relevant for very small datasets; score-neutral here). Everything else (data loading, architecture, augmentation, optimizer, epochs, and submission formatting) is preserved so the score should remain essentially the same (and thus not drift further away from your much-lower target).'
- What this solution (achieved 0.13514) has done: 'I fix the runtime crash caused by importing `keras` (Keras 3) in this environment by switching those imports to `tf_keras`, which provides the same APIs used here (Sequential, ImageDataGenerator, callbacks, etc.). This unblocks model creation/training so the pipeline runs end-to-end without changing the model architecture, augmentation, optimizer settings, epochs, or loss. I also add a small safety normalization (`/255.0`) to match the common expectation for image models and improve score legitimately (still the same core logic), and keep file ordering aligned to `sample_submission.csv` to guarantee a valid submission. Finally, the script always write a `.csv` submission with exactly `file,species`.'
- What this solution (achieved 0.13514) has done: 'The crash happens at `import tf_keras` due to a protobuf incompatibility (`MessageFactory.GetPrototype`) in this Kaggle image, so the pipeline never reaches training/inference. To keep the same model/augmentation/training logic but make it run end-to-end, I switch the Keras stack to `tensorflow.keras` (stable in Kaggle) while leaving the architecture, optimizer settings, epochs, datagen, and submission formatting unchanged. Since your current score (0.13514) is already above the target (0.07682) and higher-is-better, I avoid any score-improving changes and focus purely on correctness/stability. The script still write a valid `file,species` submission CSV aligned to `sample_submission.csv`.'
- What this solution (achieved 0.13514) has done: 'I fix the runtime crash that happens when importing TensorFlow/Keras due to a protobuf incompatibility (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before TensorFlow is imported. This is a stability fix that unblocks training/inference end-to-end without changing your model, augmentation, loss, optimizer settings, or submission logic. Since your current score (0.13514) is already above the target (0.07682) and higher-is-better, I avoid any score-improving changes and keep behavior as score-neutral as possible. The script still write a valid `.csv` submission with exactly `file,species` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.13514) has done: 'We fix the runtime crash at the TensorFlow import (`MessageFactory.GetPrototype`) by pinning protobuf to the pure-Python implementation early and ensuring it takes effect before any TensorFlow/Keras/protobuf modules load (including deleting any preloaded `google.protobuf` modules if present). This is a stability-only change that keeps your model architecture, augmentation, training loop, and submission formatting identical, so the score should remain essentially unchanged (and not drift further away from your already-above-target score). We also keep paths and output filename the same and ensure the submission CSV is written with the required `file,species` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.13514) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring the pure-Python protobuf setting is applied before any TensorFlow-related import and by safely purging any already-loaded protobuf modules. This is a stability-only change that unblocks training/inference without changing your model architecture, augmentation, loss, optimizer, epochs, or submission formatting. Because your current score (0.13514) is already above the target (0.07682) and higher-is-better, I not introduce any score-improving changes; the goal is to run end-to-end and keep behavior as score-neutral as possible. The script still write a valid `.csv` submission with exact `file,species` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.13514) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* any TensorFlow/Keras import and purging any already-loaded protobuf modules, then importing TensorFlow in a guarded way. Since your current score (0.13514) is already above the target (0.07682) and higher-is-better, I avoid any score-improving changes and keep the model/training/inference logic unchanged aside from stability fixes. I also ensure the submission is always written as a valid `.csv` with exact `file,species` columns aligned to `sample_submission.csv`. All data paths and the core CNN/augmentation/training loop remain the same.'
- What this solution (achieved 0.13514) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by enforcing the pure-Python protobuf implementation *before* any TensorFlow-related imports and purging any already-loaded protobuf modules, then importing TensorFlow lazily/guarded. This is a stability-only change that unblocks training/inference end-to-end without changing the model architecture, augmentation, optimizer, epochs, or submission logic (so the score should remain essentially unchanged and not drift further away from your already-above-target score). I also add a small path fallback to the known Kaggle dataset locations to prevent “no images found” edge cases. The submission writing stays the same and still produces a valid `.csv` with `file,species` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import numpy as np
import pandas as pd

seed = 42
np.random.seed(seed)

BASE_INPUT = "../input/plant-seedlings-classification"
ALT_BASE_INPUTS = [
    "../input/plant-seedlings-classification/plant-seedlings-classification",
    "../input",
]


def _resolve_dir(base, name):
    p = os.path.join(base, name)
    if os.path.exists(p):
        return p
    return None


train_dir = _resolve_dir(BASE_INPUT, "train")
test_dir = _resolve_dir(BASE_INPUT, "test")

if train_dir is None or test_dir is None:
    for b in ALT_BASE_INPUTS:
        if train_dir is None:
            train_dir = _resolve_dir(b, "train")
        if test_dir is None:
            test_dir = _resolve_dir(b, "test")
        if train_dir is not None and test_dir is not None:
            break

TRAIN_DIR = train_dir if train_dir is not None else os.path.join(BASE_INPUT, "train")
TEST_DIR = test_dir if test_dir is not None else os.path.join(BASE_INPUT, "test")

print("Listing ../input:", os.listdir("../input")[:20])
print("Using TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.exists(TRAIN_DIR))
print("Using TEST_DIR :", TEST_DIR, "exists:", os.path.exists(TEST_DIR))



## === cell 1
CLASS = {
    "Black-grass": 0,
    "Charlock": 1,
    "Cleavers": 2,
    "Common Chickweed": 3,
    "Common wheat": 4,
    "Fat Hen": 5,
    "Loose Silky-bent": 6,
    "Maize": 7,
    "Scentless Mayweed": 8,
    "Shepherds Purse": 9,
    "Small-flowered Cranesbill": 10,
    "Sugar beet": 11,
}
INV_CLASS = {v: k for k, v in CLASS.items()}

dim = 64
nclasses = 12



## === cell 2
sample_sub = pd.read_csv("../input/sample_submission.csv")
print(sample_sub.head(10))
print("sample_submission shape:", sample_sub.shape)



## === cell 3
import imageio
from skimage.transform import resize as imresize
from tqdm import tqdm


def img_reshape(img):
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    img = imresize(img, (dim, dim, 3))
    return img.astype(np.float32)


def img_label(path):
    return str(path.split("/")[-1])


def img_class(path):
    return str(path.split("/")[-2])


def fill_dict(paths, some_dict):
    if not paths:
        return some_dict

    text = "Start fill dict"
    if "train" in paths[0]:
        text = "Start fill train_dict"
    elif "test" in paths[0]:
        text = "Start fill test_dict"

    for p in tqdm(paths, ascii=True, ncols=85, desc=text):
        img = imageio.imread(p)
        img = img_reshape(img)
        some_dict["image"].append(img)
        some_dict["label"].append(img_label(p))
        if "train" in paths[0]:
            some_dict["class"].append(img_class(p))
    return some_dict




## === cell 4
train_path, test_path = [], []
file_ext = []

for root, dirs, files in os.walk(TRAIN_DIR):
    for f in files:
        ext = os.path.splitext(f)[1].lower().lstrip(".")
        if ext and ext not in file_ext:
            file_ext.append(ext)
        if ext == "png":
            train_path.append(os.path.join(root, f))

for root, dirs, files in os.walk(TEST_DIR):
    for f in files:
        ext = os.path.splitext(f)[1].lower().lstrip(".")
        if ext and ext not in file_ext:
            file_ext.append(ext)
        if ext == "png":
            test_path.append(os.path.join(root, f))

train_path = sorted(train_path)
test_path = sorted(test_path)

print("Found train images:", len(train_path))
print("Found test images :", len(test_path))
print("Extensions seen:", sorted(file_ext)[:10], "...")

if len(train_path) == 0 or len(test_path) == 0:
    raise RuntimeError(
        f"No images found. TRAIN_DIR={TRAIN_DIR} ({os.path.exists(TRAIN_DIR)}), "
        f"TEST_DIR={TEST_DIR} ({os.path.exists(TEST_DIR)})."
    )

train_dict = {"image": [], "label": [], "class": []}
test_dict = {"image": [], "label": []}

train_dict = fill_dict(train_path, train_dict)
test_dict = fill_dict(test_path, test_dict)



## === cell 5
print(len(train_dict["image"]), len(train_dict["label"]), len(train_dict["class"]))
print(len(test_dict["image"]), len(test_dict["label"]))
print("Example train classes:", train_dict["class"][:5])
print("Example test labels :", test_dict["label"][:5])



## === cell 6
print("file_ext:", file_ext)
print("train_path[:5]:", train_path[:5])
print("test_path[:5] :", test_path[:5])




## === cell 7
def to_categorical_np(y, num_classes):
    y = np.asarray(y, dtype=np.int64).ravel()
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


xtrain = np.array(train_dict["image"], dtype=np.float32)
_ytrain = np.array([CLASS[l] for l in train_dict["class"]], dtype=np.int64)
ytrain = to_categorical_np(_ytrain, num_classes=nclasses)

xtrain = xtrain / 255.0

print(
    "xtrain:",
    xtrain.shape,
    xtrain.dtype,
    "min/max:",
    float(xtrain.min()),
    float(xtrain.max()),
)
print("ytrain:", ytrain.shape, ytrain.dtype)



## === cell 8
import seaborn as sns

sns.set(style="white", context="notebook", palette="deep")

sns.countplot(x=_ytrain)

print(_ytrain.shape)
print(type(_ytrain))
__ytrain = pd.Series(_ytrain)

vals_class = __ytrain.value_counts()
print(vals_class)

cls_mean = np.mean(vals_class)
cls_std = np.std(vals_class, ddof=1)

print("The mean amount of elements per class is", cls_mean)
print("The standard deviation in the element per class distribution is", cls_std)

if cls_std > cls_mean * (0.6827 / 2):
    print("The standard deviation is high")



## === cell 9
xtest = np.array(test_dict["image"], dtype=np.float32)
label = np.array(test_dict["label"])

xtest = xtest / 255.0

print(
    "xtest:",
    xtest.shape,
    xtest.dtype,
    "min/max:",
    float(xtest.min()),
    float(xtest.max()),
)
print("labels:", label.shape)



## === cell 10
print(xtrain[:1].shape)



## === cell 11
print("xtrain.shape:", xtrain.shape)  # expected ~ (4750, 64, 64, 3)



## === cell 12
print("ytrain.shape:", ytrain.shape)  # expected ~ (4750, 12)



## === cell 13
from sklearn.model_selection import train_test_split

split_pct = 0.05
xtrain, xval, ytrain, yval = train_test_split(
    xtrain,
    ytrain,
    test_size=split_pct,
    random_state=seed,
    stratify=_ytrain,  # stratify should be 1D labels, not one-hot.
)

print(xtrain.shape)
print(xval.shape)
print(ytrain.shape)
print(yval.shape)



## === cell 14
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D

print("tensorflow version:", getattr(tf, "__version__", "unknown"))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 15
model = Sequential()

ksize = 5

model.add(
    Conv2D(
        filters=32,
        kernel_size=(ksize, ksize),
        padding="same",
        activation="relu",
        input_shape=(dim, dim, 3),
    )
)
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(
    Conv2D(filters=64, kernel_size=(ksize, ksize), padding="same", activation="relu")
)
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(
    Conv2D(filters=64, kernel_size=(ksize, ksize), padding="same", activation="relu")
)
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.75))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.75))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.75))
model.add(Dense(nclasses, activation="softmax"))



## === cell 16
model.summary()



## === cell 17
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy",
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=1e-6,
)



## === cell 18
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=30,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

datagen.fit(xtrain)



## === cell 19
epochs = 35
batch_size = 64
print("epochs:", epochs, "batch_size:", batch_size)



## === cell 20
steps_per_epoch = max(1, xtrain.shape[0] // batch_size)

history = model.fit(
    datagen.flow(xtrain, ytrain, batch_size=batch_size, shuffle=True),
    epochs=epochs,
    validation_data=(xval, yval),
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    callbacks=[learning_rate_reduction],
)



## === cell 21
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(history.history["loss"], color="b", label="Training loss")
ax[0].plot(history.history["val_loss"], color="r", label="Validation loss")
ax[0].legend(loc="best", shadow=True)

ax[1].plot(history.history["accuracy"], color="b", label="Training accuracy")
ax[1].plot(history.history["val_accuracy"], color="r", label="Validation accuracy")
ax[1].legend(loc="best", shadow=True)

plt.tight_layout()
plt.show()



## === cell 22
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}" if normalize else str(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


ypred = model.predict(xval, verbose=0)
ypred_classes = np.argmax(ypred, axis=1)
ytrue = np.argmax(yval, axis=1)

confusion_mtx = confusion_matrix(ytrue, ypred_classes)
plot_confusion_matrix(confusion_mtx, classes=list(range(nclasses)))



## === cell 23
INV_CLASS = {
    0: "Black-grass",
    1: "Charlock",
    2: "Cleavers",
    3: "Common Chickweed",
    4: "Common wheat",
    5: "Fat Hen",
    6: "Loose Silky-bent",
    7: "Maize",
    8: "Scentless Mayweed",
    9: "Shepherds Purse",
    10: "Small-flowered Cranesbill",
    11: "Sugar beet",
}



## === cell 24
test_proba = model.predict(xtest, verbose=1)
predictions = np.argmax(test_proba, axis=1)

sub = pd.DataFrame(
    {
        "file": label,
        "species": [INV_CLASS[int(p)] for p in predictions],
    }
)

sub["file"] = sub["file"].astype(str).str.split("/").str[-1]
sub = sub.sort_values("file").reset_index(drop=True)

sub = sample_sub[["file"]].merge(sub, on="file", how="left")

if sub["species"].isna().any():
    mode_class = pd.Series(train_dict["class"]).mode().iloc[0]
    sub["species"] = sub["species"].fillna(mode_class)

print(sub.head(10))
print("submission shape:", sub.shape)
print("NA species:", int(sub["species"].isna().sum()))



## === cell 25
out_path = "plant0708.csv"
sub.to_csv(out_path, index=False, header=True)
print("Wrote submission:", out_path, "rows:", len(sub), "cols:", list(sub.columns))
print(sub.head())
