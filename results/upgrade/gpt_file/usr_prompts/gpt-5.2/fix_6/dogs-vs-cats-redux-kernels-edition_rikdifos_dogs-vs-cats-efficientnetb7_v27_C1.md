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

3.8

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

# 5. Target score

0.70535

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import warnings

warnings.filterwarnings("ignore")

import re, random, time, zipfile, gc, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam

import cv2

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
train_zip_path = os.path.join(PATH, "train.zip")
test_zip_path = os.path.join(PATH, "test.zip")

WORK_DATA_DIR = "/kaggle/working/data"
os.makedirs(WORK_DATA_DIR, exist_ok=True)


def maybe_extract(zip_path, out_dir):
    expected = os.path.join(
        out_dir, os.path.splitext(os.path.basename(zip_path))[0]
    )  # out_dir/train or out_dir/test
    if os.path.isdir(expected) and len(os.listdir(expected)) > 0:
        return expected
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return expected


train_extract_dir = maybe_extract(train_zip_path, WORK_DATA_DIR)
test_extract_dir = maybe_extract(test_zip_path, WORK_DATA_DIR)

print(
    "Extracted top-level folders in WORK_DATA_DIR:",
    sorted(os.listdir(WORK_DATA_DIR))[:50],
)
print("train_extract_dir:", train_extract_dir)
print("test_extract_dir:", test_extract_dir)



## === cell 2
start = time.time()


def find_dir_with_images(root, exts=(".jpg", ".jpeg", ".png")):
    """Find a directory under 'root' that contains the most images."""
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        img_count = sum(fn.lower().endswith(exts) for fn in filenames)
        if img_count > 0:
            candidates.append((img_count, dirpath))
    if not candidates:
        return None
    candidates.sort(reverse=True)  # most images first
    return candidates[0][1]


def list_images_flat(root, exts=(".jpg", ".jpeg", ".png")):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                paths.append(os.path.join(dirpath, fn))
    return paths


train_root = find_dir_with_images(train_extract_dir)
test_root = find_dir_with_images(test_extract_dir)

if train_root is None or test_root is None:
    raise FileNotFoundError(
        f"Could not locate extracted train/test image folders. train_root={train_root}, test_root={test_root}"
    )

train_images = list_images_flat(train_root)
test_images = list_images_flat(test_root)

numeric_like = [
    p
    for p in test_images
    if re.match(r"^\d+\.(jpg|jpeg|png)$", os.path.basename(p).lower())
]
if len(numeric_like) > 0:
    test_images = numeric_like

print("Detected train_root:", train_root)
print("Detected test_root:", test_root)
print("Num train images found:", len(train_images))
print("Num test images found:", len(test_images))
print("Sample train file:", os.path.basename(train_images[0]) if train_images else None)
print("Sample test file:", os.path.basename(test_images[0]) if test_images else None)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3777941817.py in <cell line: 0>()
     30 
     31 if train_root is None or test_root is None:
---> 32     raise FileNotFoundError(
     33         f"Could not locate extracted train/test image folders. train_root={train_root}, test_root={test_root}"
     34     )

FileNotFoundError: Could not locate extracted train/test image folders. train_root=None, test_root=None

## === cell 3
def txt_dig(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [txt_dig(c) for c in re.split(r"(\d+)", text)]


def extract_id_from_path(p):
    base = os.path.basename(p)
    m = re.match(r"(\d+)\.", base)
    return int(m.group(1)) if m else None


train_images.sort(key=natural_keys)

test_ids_from_files = [extract_id_from_path(p) for p in test_images]
if len(test_images) > 0 and all(t is not None for t in test_ids_from_files):
    test_images = [
        p for _, p in sorted(zip(test_ids_from_files, test_images), key=lambda x: x[0])
    ]
else:
    test_images.sort(key=natural_keys)

print("First 5 test basenames:", [os.path.basename(p) for p in test_images[:5]])



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3434451151.py in <cell line: 0>()
     13 
     14 
---> 15 train_images.sort(key=natural_keys)
     16 
     17 test_ids_from_files = [extract_id_from_path(p) for p in test_images]

NameError: name 'train_images' is not defined

## === cell 4
n = len(train_images)
part1 = train_images[: min(1300, n)]
part2 = train_images[min(12500, n) : min(13800, n)]
train_images = part1 + part2

print("Using sampled train images:", len(train_images))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/883194176.py in <cell line: 0>()
----> 1 n = len(train_images)
      2 part1 = train_images[: min(1300, n)]
      3 part2 = train_images[min(12500, n) : min(13800, n)]
      4 train_images = part1 + part2
      5 

NameError: name 'train_images' is not defined

## === cell 5
IMG_WIDTH = 128
IMG_HEIGHT = 128


def read_and_resize(path, w=128, h=128):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img


x = [read_and_resize(p, IMG_WIDTH, IMG_HEIGHT) for p in train_images]
test = [read_and_resize(p, IMG_WIDTH, IMG_HEIGHT) for p in test_images]

x = np.array(x, dtype=np.uint8)
test = np.array(test, dtype=np.uint8)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2734847625.py in <cell line: 0>()
     12 
     13 
---> 14 x = [read_and_resize(p, IMG_WIDTH, IMG_HEIGHT) for p in train_images]
     15 test = [read_and_resize(p, IMG_WIDTH, IMG_HEIGHT) for p in test_images]
     16 

NameError: name 'train_images' is not defined

## === cell 6
print("The shape of train data is {}".format(x.shape))
print("The shape of test data is {}".format(test.shape))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1876675754.py in <cell line: 0>()
----> 1 print("The shape of train data is {}".format(x.shape))
      2 print("The shape of test data is {}".format(test.shape))
      3 

NameError: name 'x' is not defined

## === cell 7
random.seed(558)
plt.rcParams["figure.facecolor"] = "white"
plt.figure(figsize=(10, 4))

for i in range(3):
    sample = random.choice(train_images)
    image = load_img(sample)
    plt.subplot(1, 3, i + 1)
    plt.imshow(image)
    plt.axis("off")
plt.tight_layout()
plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/647587976.py in <cell line: 0>()
      4 
      5 for i in range(3):
----> 6     sample = random.choice(train_images)
      7     image = load_img(sample)
      8     plt.subplot(1, 3, i + 1)

NameError: name 'train_images' is not defined

## === cell 8
plt.rcParams["figure.facecolor"] = "white"
plt.figure(figsize=(10, 4))
idxs = [min(0, len(x) - 1), min(1, len(x) - 1), min(2, len(x) - 1)]
for i, idx in enumerate(idxs):
    plt.subplot(1, 3, i + 1)
    plt.imshow(x[idx])
    plt.axis("off")
plt.tight_layout()
plt.show()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3363161366.py in <cell line: 0>()
      1 plt.rcParams["figure.facecolor"] = "white"
      2 plt.figure(figsize=(10, 4))
----> 3 idxs = [min(0, len(x) - 1), min(1, len(x) - 1), min(2, len(x) - 1)]
      4 for i, idx in enumerate(idxs):
      5     plt.subplot(1, 3, i + 1)

NameError: name 'x' is not defined

## === cell 9
y = []
for p in train_images:
    base = os.path.basename(p).lower()
    parts = os.path.normpath(p).lower().split(os.sep)
    if "dog" in parts or base.startswith("dog"):
        y.append(1)
    elif "cat" in parts or base.startswith("cat"):
        y.append(0)
    else:
        raise ValueError(f"Cannot infer label from path: {p}")

y = np.array(y, dtype=np.int32)
print(
    "Num labels:", len(y), "Pos (dog):", int(y.sum()), "Neg (cat):", int((1 - y).sum())
)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=2020, stratify=y
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2877637926.py in <cell line: 0>()
      1 y = []
----> 2 for p in train_images:
      3     base = os.path.basename(p).lower()
      4     parts = os.path.normpath(p).lower().split(os.sep)
      5     if "dog" in parts or base.startswith("dog"):

NameError: name 'train_images' is not defined

## === cell 10
model = models.Sequential()

efnModel = tf.keras.applications.EfficientNetB7(
    weights="imagenet", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False
)
model.add(efnModel)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation="sigmoid"))

opt1 = RMSprop(learning_rate=0.005, decay=1e-6)
opt2 = Adam(learning_rate=0.006)

model.compile(loss="binary_crossentropy", optimizer=opt2, metrics=["accuracy"])
model.summary()



## === cell 11
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)




## === cell 12
def plot_gened(train_images, seed=320):
    df = pd.DataFrame({"filename": train_images})
    np.random.seed(seed)
    vis_df = df.sample(n=1).reset_index(drop=True)
    vis_df["category"] = "0"
    vis_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    vis_gen0 = vis_gen.flow_from_dataframe(
        vis_df,
        x_col="filename",
        y_col="category",
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=16,
        class_mode="raw",
        shuffle=False,
    )
    plt.rcParams["figure.facecolor"] = "white"
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        X_batch, _ = next(vis_gen0)
        plt.imshow(X_batch[0])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


plot_gened(train_images)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3913674290.py in <cell line: 0>()
     34 
     35 
---> 36 plot_gened(train_images)
     37 

NameError: name 'train_images' is not defined

## === cell 13
BATCH_SIZE = 16

train_flow = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_flow = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

earlystop1 = EarlyStopping(patience=5, restore_best_weights=True)
earlystop2 = ReduceLROnPlateau(
    monitor="val_accuracy", min_lr=0.001, patience=5, mode="max", verbose=1
)

history = model.fit(
    train_flow,
    steps_per_epoch=55,
    epochs=20,
    validation_data=val_flow,
    callbacks=[earlystop1, earlystop2],
    validation_steps=25,
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1103144407.py in <cell line: 0>()
      1 BATCH_SIZE = 16
      2 
----> 3 train_flow = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
      4 val_flow = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
      5 

NameError: name 'x_train' is not defined

## === cell 14
plt.rcParams["figure.facecolor"] = "white"
model_loss = pd.DataFrame(history.history)

ax = model_loss[["accuracy", "val_accuracy"]].plot(ylim=[0.4, 1.0], figsize=(6, 4))
ax.set_title("Accuracy")
plt.show()

ax = model_loss[["loss", "val_loss"]].plot(
    ylim=[0.0, max(1.0, float(model_loss[["loss", "val_loss"]].max().max()))],
    figsize=(6, 4),
)
ax.set_title("Loss")
plt.show()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1195348699.py in <cell line: 0>()
      1 plt.rcParams["figure.facecolor"] = "white"
----> 2 model_loss = pd.DataFrame(history.history)
      3 
      4 ax = model_loss[["accuracy", "val_accuracy"]].plot(ylim=[0.4, 1.0], figsize=(6, 4))
      5 ax.set_title("Accuracy")

NameError: name 'history' is not defined

## === cell 15
val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))
val_preds = model.predict(val_flow, verbose=1, steps=val_steps).ravel()
print(val_steps)
print("Out of Fold log loss is {:.5f}".format(log_loss(y_val, val_preds)))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2566653425.py in <cell line: 0>()
----> 1 val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))
      2 val_preds = model.predict(val_flow, verbose=1, steps=val_steps).ravel()
      3 print(val_steps)
      4 print("Out of Fold log loss is {:.5f}".format(log_loss(y_val, val_preds)))
      5 

NameError: name 'x_val' is not defined

## === cell 16
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_flow = test_datagen.flow(test, batch_size=BATCH_SIZE, shuffle=False)

test_steps = int(np.ceil(len(test) / BATCH_SIZE))
test_pred = model.predict(test_flow, verbose=1, steps=test_steps).ravel()
print(test_steps)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2610218705.py in <cell line: 0>()
      1 test_datagen = ImageDataGenerator(rescale=1.0 / 255)
----> 2 test_flow = test_datagen.flow(test, batch_size=BATCH_SIZE, shuffle=False)
      3 
      4 test_steps = int(np.ceil(len(test) / BATCH_SIZE))
      5 test_pred = model.predict(test_flow, verbose=1, steps=test_steps).ravel()

NameError: name 'test' is not defined

## === cell 17
sample_path_candidates = [
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
sample_path = None
for p in sample_path_candidates:
    if os.path.exists(p):
        sample_path = p
        break
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected input paths."
    )

sample_sub = pd.read_csv(sample_path)
ids_required = sample_sub["id"].tolist()

ids_from_files = [extract_id_from_path(p) for p in test_images]
pred_map = {
    int(i): float(p)
    for i, p in zip(ids_from_files, test_pred[: len(ids_from_files)])
    if i is not None
}

labels = [pred_map.get(int(i), 0.5) for i in ids_required]
submission = pd.DataFrame({"id": ids_required, "label": labels})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print("This program costs {:.2f} seconds".format(time.time() - start))
print(submission.head())



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/935322364.py in <cell line: 0>()
     16 ids_required = sample_sub["id"].tolist()
     17 
---> 18 ids_from_files = [extract_id_from_path(p) for p in test_images]
     19 pred_map = {
     20     int(i): float(p)

NameError: name 'test_images' is not defined

## === cell 18
shutil.rmtree("/kaggle/working/data/", ignore_errors=True)
gc.collect()
print("Cleaned /kaggle/working/data/")
