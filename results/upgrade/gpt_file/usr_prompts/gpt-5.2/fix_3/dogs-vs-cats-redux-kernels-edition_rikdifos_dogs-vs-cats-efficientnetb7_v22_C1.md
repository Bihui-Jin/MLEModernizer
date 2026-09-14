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

0.73669

# 6. Current score

1.56685

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.56685) has done: 'I fix the environment crash caused by the protobuf/TensorFlow “MessageFactory.GetPrototype” incompatibility by forcing TensorFlow to use the pure-Python protobuf implementation before importing TF. Then I fix the wrong test directory detection which accidentally picked up training images (e.g., `cat.0.jpg`) and caused submission ID parsing to fail; this also ensure we predict on the real test set and produce a valid submission. Finally, I make submission ID extraction robust by filtering only numeric filenames and aligning predictions to those IDs, so `submission.csv` is always written with the required `id,label` columns.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2, re, random, time, zipfile, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

SEED = 558
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
train_image_path = os.path.join(PATH, "train.zip")
test_image_path = os.path.join(PATH, "test.zip")

os.makedirs("./data", exist_ok=True)

with zipfile.ZipFile(train_image_path, "r") as z:
    z.extractall("./data")

with zipfile.ZipFile(test_image_path, "r") as z:
    z.extractall("./data")

print("Extraction complete. Top-level ./data contents:", os.listdir("./data")[:10])



## === cell 2
start = time.time()


def find_dir_with_images(root, must_contain=None, exclude_substrings=None):
    """
    Find a directory under `root` that contains images.
    If must_contain is given, require those substrings to appear in filenames (e.g., 'cat.'/'dog.').
    If exclude_substrings is given, skip directories whose path contains any of them.
    Returns the matching directory with the most images.
    """
    best_dir, best_count = None, -1
    for dirpath, dirnames, filenames in os.walk(root):
        if exclude_substrings and any(s in dirpath for s in exclude_substrings):
            continue
        jpgs = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not jpgs:
            continue
        if must_contain:
            ok = True
            for s in must_contain:
                if not any(s in f for f in jpgs):
                    ok = False
                    break
            if not ok:
                continue
        if len(jpgs) > best_count:
            best_dir, best_count = dirpath, len(jpgs)
    return best_dir


TRAIN_DIR = find_dir_with_images("./data", must_contain=["cat.", "dog."])


def is_numeric_stem(filename):
    stem, _ = os.path.splitext(filename)
    return stem.isdigit()


def find_test_dir_numeric(root):
    best_dir, best_count = None, -1
    for dirpath, dirnames, filenames in os.walk(root):
        jpgs = [f for f in filenames if f.lower().endswith(".jpg")]
        if not jpgs:
            continue
        numeric = [f for f in jpgs if is_numeric_stem(f)]
        if len(numeric) > best_count:
            best_dir, best_count = dirpath, len(numeric)
    return best_dir


TEST_DIR = find_test_dir_numeric("./data")

if TRAIN_DIR is None or TEST_DIR is None:
    raise FileNotFoundError(
        f"Could not find extracted image folders. TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}"
    )

print("Detected TRAIN_DIR:", TRAIN_DIR)
print("Detected TEST_DIR :", TEST_DIR)

train_images = [
    os.path.join(TRAIN_DIR, i)
    for i in os.listdir(TRAIN_DIR)
    if i.lower().endswith(".jpg")
]
test_images = [
    os.path.join(TEST_DIR, i)
    for i in os.listdir(TEST_DIR)
    if i.lower().endswith(".jpg") and is_numeric_stem(i)
]

print("Raw counts:", len(train_images), len(test_images))
if len(test_images) == 0:
    raise RuntimeError(f"No numeric test images found in TEST_DIR={TEST_DIR}")




## === cell 3
def txt_dig(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [txt_dig(c) for c in re.split(r"(\d+)", text)]




## === cell 4
train_images.sort(key=natural_keys)
test_images.sort(key=natural_keys)

n = len(train_images)
if n >= 13800:
    train_images = train_images[0:1300] + train_images[12500:13800]
else:
    k = min(2600, n)
    idx = np.linspace(0, n - 1, k).astype(int)
    train_images = [train_images[i] for i in idx.tolist()]

print("Using train sample size:", len(train_images))



## === cell 5
IMG_WIDTH = 128
IMG_HEIGHT = 128

x = []
bad_train = 0
for img in train_images:
    im = cv2.imread(img)
    if im is None:
        bad_train += 1
        continue
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    x.append(cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC))

test = []
bad_test = 0
kept_test_images = []
for img in test_images:
    im = cv2.imread(img)
    if im is None:
        bad_test += 1
        continue
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    test.append(cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC))
    kept_test_images.append(img)

x = np.array(x, dtype=np.uint8)
test = np.array(test, dtype=np.uint8)
test_images = kept_test_images  # keep alignment between test array and filenames

print("Bad reads - train:", bad_train, "test:", bad_test)



## === cell 6
print("The shape of train data is {}".format(x.shape))
print("The shape of test data is {}".format(test.shape))



## === cell 7
if len(train_images) >= 3:
    random.seed(SEED)
    plt.subplots(facecolor="white", figsize=(10, 6))

    for j in range(3):
        sample = random.choice(train_images)
        image = load_img(sample)
        plt.subplot(1, 3, j + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.show()



## === cell 8
plt.subplots(facecolor="white", figsize=(10, 6))
idxs = [0, min(1, len(x) - 1), min(2, len(x) - 1)]
for j, idx in enumerate(idxs):
    plt.subplot(1, 3, j + 1)
    plt.imshow(x[idx])
    plt.axis("off")
plt.show()



## === cell 9
y = []
for i in train_images:
    fn = os.path.basename(i)
    if "dog" in fn:
        y.append(1)
    elif "cat" in fn:
        y.append(0)

x = x[: len(y)]  # in case some images failed to read
y = np.array(y, dtype=np.int32)

print(
    "Labels:",
    len(y),
    "Positives(dog):",
    int(y.sum()),
    "Negatives(cat):",
    int((1 - y).sum()),
)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=2020, stratify=y
)



## === cell 10
model = models.Sequential()

efnModel = tf.keras.applications.EfficientNetB7(
    weights="imagenet", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False
)
model.add(efnModel)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=0.005, decay=1e-6),
    metrics=["accuracy"],
)

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
    )
    plt.rcParams["figure.facecolor"] = "white"
    plt.figure(figsize=(8, 8))
    for i in range(0, 9):
        plt.subplot(3, 3, i + 1)
        for X_batch, Y_batch in vis_gen0:
            image = X_batch[0]
            plt.imshow(image)
            plt.axis("off")
            break
    plt.tight_layout()
    plt.show()


plot_gened(train_images)



## === cell 13
BATCH_SIZE = 16
train_flow = datagen.flow(
    x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
)
val_flow = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

earlystop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    train_flow,
    steps_per_epoch=45,
    epochs=15,
    validation_data=val_flow,
    callbacks=[earlystop],
    validation_steps=int(np.ceil(len(x_val) / BATCH_SIZE)),
)



## === cell 14
plt.rcParams["figure.facecolor"] = "white"
model_loss = pd.DataFrame(history.history)

ax = model_loss[["accuracy", "val_accuracy"]].plot(ylim=[0.0, 1.0], title="Accuracy")
ax.set_xlabel("epoch")
plt.show()

ax = model_loss[["loss", "val_loss"]].plot(title="Loss")
ax.set_xlabel("epoch")
plt.show()



## === cell 15
val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))
val_preds = model.predict(val_flow, verbose=1, steps=val_steps).ravel()
val_preds = np.clip(val_preds, 1e-7, 1 - 1e-7)  # numerical safety for logloss

print("val_steps:", val_steps)
print("Out of Fold log loss is {:.5f}".format(log_loss(y_val, val_preds)))



## === cell 16
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_flow = test_datagen.flow(test, batch_size=BATCH_SIZE, shuffle=False)

test_steps = int(np.ceil(len(test) / BATCH_SIZE))
test_pred = model.predict(test_flow, verbose=1, steps=test_steps).ravel()
test_pred = np.clip(test_pred, 1e-7, 1 - 1e-7)

print("test_steps:", test_steps, "preds:", test_pred.shape)



## === cell 17
test_ids = []
good_idx = []
for idx, p in enumerate(test_images):
    stem = os.path.splitext(os.path.basename(p))[0]
    if stem.isdigit():
        test_ids.append(int(stem))
        good_idx.append(idx)

test_ids = np.array(test_ids)
test_pred = np.array(test_pred)[good_idx]

order = np.argsort(test_ids)
test_ids_sorted = test_ids[order]
test_pred_sorted = test_pred[order]

submission = pd.DataFrame({"id": test_ids_sorted, "label": test_pred_sorted})
submission = submission.sort_values("id").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print("This program costs {:.2f} seconds".format(time.time() - start))
submission.head()



## === cell 18
import shutil

shutil.rmtree("/kaggle/working/data/", ignore_errors=True)
print("Cleaned /kaggle/working/data/")
