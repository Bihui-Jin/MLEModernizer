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
seaborn==0.12.2
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

1.01568

# 6. Current score

0.68798

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.68798) has done: 'We fix the environment crash caused by the protobuf/keras interaction by not forcing the Python protobuf implementation (which triggers the `MessageFactory.GetPrototype` error in this Kaggle image) and by using `tf.keras` consistently. Next, we harden train/test directory resolution so we never accidentally treat `train/` images (cat.* / dog.*) as test images, which caused `get_test_id()` to fail on filenames like `dog.7296`. We also make `get_test_id()` robust (return `None` for non-numeric names) and filter test paths accordingly to guarantee a valid submission with numeric ids only. Finally, we fix the visualization generator error by using `class_mode="raw"` for a single-class sample (this is score-neutral and just unblocks execution).'

# 9. Code solution

## === cell 0
import os


import warnings

warnings.filterwarnings("ignore")

import cv2, re, random, time, zipfile, gc, shutil
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # safe in Kaggle batch runs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam

SEED = 558
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
train_zip_path = os.path.join(PATH, "train.zip")
test_zip_path = os.path.join(PATH, "test.zip")

EXTRACT_DIR = "./data"
if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(train_zip_path, "r") as z:
    z.extractall(EXTRACT_DIR)

with zipfile.ZipFile(test_zip_path, "r") as z:
    z.extractall(EXTRACT_DIR)

print("Extracted to:", os.path.abspath(EXTRACT_DIR))
print("Top-level extracted entries:", sorted(os.listdir(EXTRACT_DIR))[:20])



## === cell 2
start = time.time()


def _list_jpgs(d):
    try:
        return [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
    except Exception:
        return []


def _train_score(files):
    s = 0
    for f in files[:300]:
        fl = f.lower()
        if (
            fl.startswith("cat.")
            or fl.startswith("dog.")
            or ("cat." in fl)
            or ("dog." in fl)
        ):
            s += 1
    return s


def _test_score(files):
    s = 0
    for f in files[:300]:
        base = os.path.splitext(f)[0]
        if base.isdigit():
            s += 1
    return s


candidate_train_dirs = [
    os.path.join(EXTRACT_DIR, "train", "train"),
    os.path.join(EXTRACT_DIR, "train"),
]
candidate_test_dirs = [
    os.path.join(EXTRACT_DIR, "test", "test"),
    os.path.join(EXTRACT_DIR, "test"),
]

TRAIN_DIR = None
TEST_DIR = None

for d in candidate_train_dirs:
    if os.path.isdir(d) and len(_list_jpgs(d)) > 0:
        TRAIN_DIR = d
        break

for d in candidate_test_dirs:
    if os.path.isdir(d) and len(_list_jpgs(d)) > 0:
        files = _list_jpgs(d)
        if _test_score(files) > _train_score(files):
            TEST_DIR = d
            break

if TRAIN_DIR is None or TEST_DIR is None:
    jpg_dirs = []
    for cur_root, dirs, files in os.walk(EXTRACT_DIR):
        if any(f.lower().endswith(".jpg") for f in files):
            jpg_dirs.append(cur_root)

    best_train = (None, -1, 0)  # (dir, score, count)
    best_test = (None, -1, 0)

    for d in jpg_dirs:
        files = sorted(_list_jpgs(d))
        if not files:
            continue
        tr_sc = _train_score(files)
        te_sc = _test_score(files)
        if tr_sc > best_train[1] or (
            tr_sc == best_train[1] and len(files) > best_train[2]
        ):
            best_train = (d, tr_sc, len(files))
        if te_sc > best_test[1] or (
            te_sc == best_test[1] and len(files) > best_test[2]
        ):
            best_test = (d, te_sc, len(files))

    if TRAIN_DIR is None:
        TRAIN_DIR = best_train[0]
    if TEST_DIR is None:
        TEST_DIR = best_test[0]

if not TRAIN_DIR or not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Train directory with jpgs not found under {EXTRACT_DIR}")
if not TEST_DIR or not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"Test directory with jpgs not found under {EXTRACT_DIR}")

train_images = [
    os.path.join(TRAIN_DIR, i)
    for i in os.listdir(TRAIN_DIR)
    if i.lower().endswith(".jpg")
]
test_images = [
    os.path.join(TEST_DIR, i)
    for i in os.listdir(TEST_DIR)
    if i.lower().endswith(".jpg")
]

print("Resolved TRAIN_DIR:", TRAIN_DIR, "| jpgs:", len(train_images))
print("Resolved TEST_DIR :", TEST_DIR, "| jpgs:", len(test_images))
print("Sample train file:", os.path.basename(train_images[0]) if train_images else None)
print("Sample test file:", os.path.basename(test_images[0]) if test_images else None)




## === cell 3
def txt_dig(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [txt_dig(c) for c in re.split(r"(\d+)", text)]


def get_test_id(path):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return int(stem) if stem.isdigit() else None




## === cell 4
train_images.sort(key=lambda p: natural_keys(os.path.basename(p)))

test_images = [p for p in test_images if get_test_id(p) is not None]
test_images.sort(key=lambda p: get_test_id(p))

n = len(train_images)
if n >= 25000:
    train_images = train_images[0:1300] + train_images[23700:25000]
else:
    head = min(1300, n)
    tail_start = max(0, n - (25000 - 23700))  # ~1300 tail if possible
    train_images = train_images[0:head] + train_images[tail_start:n]

random.seed(SEED)
random.shuffle(train_images)

print("Sampled train images:", len(train_images))
print("Filtered test images:", len(test_images))
print("First 3 sampled:", [os.path.basename(p) for p in train_images[:3]])
print("First 3 test:", [os.path.basename(p) for p in test_images[:3]])



## === cell 5
IMG_WIDTH = 128
IMG_HEIGHT = 128

x = []
y = []
for img_path in train_images:
    im = cv2.imread(img_path)
    if im is None:
        continue
    im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    x.append(im)

    base = os.path.basename(img_path).lower()
    if "dog" in base:
        y.append(1)
    elif "cat" in base:
        y.append(0)
    else:
        y.append(0)

test = []
kept_test_paths = []
for img_path in test_images:
    im = cv2.imread(img_path)
    if im is None:
        continue
    im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    test.append(im)
    kept_test_paths.append(img_path)

x = np.array(x)
y = np.array(y, dtype=np.int32)
test = np.array(test)

print("The shape of train data is {}".format(x.shape))
print("The shape of train labels is {}".format(y.shape))
print("The shape of test data is {}".format(test.shape))

if len(y) == 0:
    raise RuntimeError("No training images were loaded; check TRAIN_DIR resolution.")
print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))))

plt.figure(figsize=(4, 3))
sns.countplot(x=y)
plt.tight_layout()
plt.close()



## === cell 6
random.seed(SEED)
plt.figure(figsize=(10, 4))
for idx in range(3):
    sample = random.choice(train_images)
    image = load_img(sample)
    plt.subplot(1, 3, idx + 1)
    plt.imshow(image)
    plt.axis("off")
plt.tight_layout()
plt.close()



## === cell 7
plt.figure(figsize=(10, 4))
idxs = []
if len(x) > 0:
    idxs = [min(1024, len(x) - 1), min(546, len(x) - 1), min(742, len(x) - 1)]
for k, ix in enumerate(idxs):
    plt.subplot(1, 3, k + 1)
    plt.imshow(cv2.cvtColor(x[ix], cv2.COLOR_BGR2RGB))
    plt.axis("off")
plt.tight_layout()
plt.close()



## === cell 8
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=2020, stratify=y
)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)



## === cell 9
base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False
)

model = models.Sequential(
    [base_model, layers.GlobalAveragePooling2D(), layers.Dense(1, activation="sigmoid")]
)

opt1 = RMSprop(learning_rate=1e-5, decay=1e-6)
opt2 = Adam(learning_rate=0.006)

model.compile(loss="binary_crossentropy", optimizer=opt1, metrics=["accuracy"])
model.summary()



## === cell 10
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




## === cell 11
def plot_gened(train_images, seed=320):
    df = pd.DataFrame({"filename": train_images})
    np.random.seed(seed)
    vis_df = df.sample(n=1).reset_index(drop=True)
    vis_df["category"] = 0.0

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
    plt.figure(figsize=(6, 6))
    for i in range(0, 9):
        plt.subplot(3, 3, i + 1)
        for X_batch, Y_batch in vis_gen0:
            image = X_batch[0]
            plt.imshow(image)
            plt.axis("off")
            break
    plt.tight_layout()
    plt.close()


plot_gened(train_images)



## === cell 12
BATCH_SIZE = 16

train_flow = datagen.flow(
    x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
)
val_flow = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

earlystop1 = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
earlystop2 = ReduceLROnPlateau(
    monitor="val_loss", min_lr=1e-7, patience=3, mode="min", verbose=1
)

history = model.fit(
    train_flow,
    steps_per_epoch=45,
    epochs=20,
    validation_data=val_flow,
    callbacks=[earlystop1, earlystop2],
    validation_steps=int(np.ceil(len(x_val) / BATCH_SIZE)),
)



## === cell 13
model_loss = pd.DataFrame(history.history)
print(model_loss.tail(1))

plt.figure(figsize=(6, 3))
model_loss[["accuracy", "val_accuracy"]].plot(ylim=[0.4, 1.0])
plt.tight_layout()
plt.close()

plt.figure(figsize=(6, 3))
model_loss[["loss", "val_loss"]].plot()
plt.tight_layout()
plt.close()



## === cell 14
val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))
val_preds = model.predict(val_flow, verbose=1, steps=val_steps)
oof = log_loss(y_val, np.clip(val_preds.ravel()[: len(y_val)], 1e-7, 1 - 1e-7))
print(val_steps)
print("Out of Fold log loss is {:.5f}".format(oof))



## === cell 15
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_flow = test_datagen.flow(test, batch_size=BATCH_SIZE, shuffle=False)

test_steps = int(np.ceil(len(test) / BATCH_SIZE))
test_pred = model.predict(test_flow, verbose=1, steps=test_steps)
print(test_steps)



## === cell 16
test_ids = [get_test_id(p) for p in kept_test_paths]
valid_mask = [tid is not None for tid in test_ids]

test_ids = [tid for tid in test_ids if tid is not None]

preds = test_pred.ravel()[: len(test_ids)]
preds = np.clip(preds.astype(np.float64), 1e-7, 1 - 1e-7)

submission = pd.DataFrame({"id": test_ids, "label": preds})
submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("This program costs {:.2f} seconds".format(time.time() - start))
print(submission.head())



## === cell 17
try:
    shutil.rmtree(EXTRACT_DIR)
    print("Cleaned:", EXTRACT_DIR)
except Exception as e:
    print("Cleanup skipped:", e)
