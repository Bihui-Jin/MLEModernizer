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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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


def find_dir_with_jpg(root):
    for cur_root, dirs, files in os.walk(root):
        if any(f.lower().endswith(".jpg") for f in files):
            return cur_root
    return None


TRAIN_DIR = find_dir_with_jpg(EXTRACT_DIR)
TEST_DIR = None

jpg_dirs = []
for cur_root, dirs, files in os.walk(EXTRACT_DIR):
    if any(f.lower().endswith(".jpg") for f in files):
        jpg_dirs.append(cur_root)


def looks_like_test_dir(d):
    files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
    if not files:
        return False
    num = 0
    for f in files[:50]:
        base = os.path.splitext(f)[0]
        if base.isdigit():
            num += 1
    return num >= max(1, int(0.8 * min(50, len(files[:50]))))


train_candidates = []
test_candidates = []
for d in jpg_dirs:
    if looks_like_test_dir(d):
        test_candidates.append(d)
    else:
        train_candidates.append(d)


def score_train_dir(d):
    files = [f.lower() for f in os.listdir(d) if f.lower().endswith(".jpg")]
    sample = files[:200]
    return sum(("cat." in f) or ("dog." in f) for f in sample)


def score_test_dir(d):
    files = [f.lower() for f in os.listdir(d) if f.lower().endswith(".jpg")]
    sample = files[:200]
    return sum(os.path.splitext(f)[0].isdigit() for f in sample)


if train_candidates:
    TRAIN_DIR = sorted(train_candidates, key=score_train_dir, reverse=True)[0]
if test_candidates:
    TEST_DIR = sorted(test_candidates, key=score_test_dir, reverse=True)[0]

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

print("Resolved TRAIN_DIR:", TRAIN_DIR)
print("Resolved TEST_DIR :", TEST_DIR)
print("Train images:", len(train_images))
print("Test images:", len(test_images))
print("Sample train file:", os.path.basename(train_images[0]) if train_images else None)
print("Sample test file:", os.path.basename(test_images[0]) if test_images else None)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4227714726.py in <cell line: 0>()
     65     raise FileNotFoundError(f"Train directory with jpgs not found under {EXTRACT_DIR}")
     66 if not TEST_DIR or not os.path.isdir(TEST_DIR):
---> 67     raise FileNotFoundError(f"Test directory with jpgs not found under {EXTRACT_DIR}")
     68 
     69 train_images = [

FileNotFoundError: Test directory with jpgs not found under ./data

## === cell 3
def txt_dig(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [txt_dig(c) for c in re.split(r"(\d+)", text)]


def get_test_id(path):
    base = os.path.basename(path)
    return int(os.path.splitext(base)[0])




## === cell 4
train_images.sort(key=natural_keys)
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
print("First 3 sampled:", [os.path.basename(p) for p in train_images[:3]])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1744074438.py in <cell line: 0>()
----> 1 train_images.sort(key=natural_keys)
      2 test_images.sort(key=lambda p: get_test_id(p))
      3 
      4 # Keep the original sampling strategy from the provided code (core logic preserved).
      5 # Guard for smaller-than-expected lists to avoid empty training set.

NameError: name 'train_images' is not defined

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



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/305524008.py in <cell line: 0>()
      4 x = []
      5 y = []
----> 6 for img_path in train_images:
      7     im = cv2.imread(img_path)
      8     if im is None:

NameError: name 'train_images' is not defined

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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3688370020.py in <cell line: 0>()
      2 plt.figure(figsize=(10, 4))
      3 for idx in range(3):
----> 4     sample = random.choice(train_images)
      5     image = load_img(sample)
      6     plt.subplot(1, 3, idx + 1)

NameError: name 'train_images' is not defined

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



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2461224061.py in <cell line: 0>()
----> 1 x_train, x_val, y_train, y_val = train_test_split(
      2     x, y, test_size=0.2, random_state=2020, stratify=y
      3 )
      4 print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2560 
   2561     n_samples = _num_samples(arrays[0])
-> 2562     n_train, n_test = _validate_shuffle_split(
   2563         n_samples, test_size, train_size, default_test_size=0.25
   2564     )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _validate_shuffle_split(n_samples, test_size, train_size, default_test_size)
   2234 
   2235     if n_train == 0:
-> 2236         raise ValueError(
   2237             "With n_samples={}, test_size={} and train_size={}, the "
   2238             "resulting train set will be empty. Adjust any of the "

ValueError: With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.

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
        class_mode="binary",
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



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2943957294.py in <cell line: 0>()
     36 
     37 
---> 38 plot_gened(train_images)
     39 

NameError: name 'train_images' is not defined

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



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/372846442.py in <cell line: 0>()
      2 
      3 train_flow = datagen.flow(
----> 4     x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
      5 )
      6 val_flow = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

NameError: name 'x_train' is not defined

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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1687169076.py in <cell line: 0>()
----> 1 model_loss = pd.DataFrame(history.history)
      2 print(model_loss.tail(1))
      3 
      4 plt.figure(figsize=(6, 3))
      5 model_loss[["accuracy", "val_accuracy"]].plot(ylim=[0.4, 1.0])

NameError: name 'history' is not defined

## === cell 14
val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))
val_preds = model.predict(val_flow, verbose=1, steps=val_steps)
oof = log_loss(y_val, np.clip(val_preds.ravel()[: len(y_val)], 1e-7, 1 - 1e-7))
print(val_steps)
print("Out of Fold log loss is {:.5f}".format(oof))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2176018827.py in <cell line: 0>()
----> 1 val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))
      2 val_preds = model.predict(val_flow, verbose=1, steps=val_steps)
      3 oof = log_loss(y_val, np.clip(val_preds.ravel()[: len(y_val)], 1e-7, 1 - 1e-7))
      4 print(val_steps)
      5 print("Out of Fold log loss is {:.5f}".format(oof))

NameError: name 'x_val' is not defined

## === cell 15
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_flow = test_datagen.flow(test, batch_size=BATCH_SIZE, shuffle=False)

test_steps = int(np.ceil(len(test) / BATCH_SIZE))
test_pred = model.predict(test_flow, verbose=1, steps=test_steps)
print(test_steps)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1548465283.py in <cell line: 0>()
      1 test_datagen = ImageDataGenerator(rescale=1.0 / 255)
----> 2 test_flow = test_datagen.flow(test, batch_size=BATCH_SIZE, shuffle=False)
      3 
      4 test_steps = int(np.ceil(len(test) / BATCH_SIZE))
      5 test_pred = model.predict(test_flow, verbose=1, steps=test_steps)

NameError: name 'test' is not defined

## === cell 16
test_ids = [get_test_id(p) for p in kept_test_paths]
preds = test_pred.ravel()[: len(test_ids)]

submission = pd.DataFrame({"id": test_ids, "label": preds})
submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("This program costs {:.2f} seconds".format(time.time() - start))
print(submission.head())



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1905668508.py in <cell line: 0>()
----> 1 test_ids = [get_test_id(p) for p in kept_test_paths]
      2 preds = test_pred.ravel()[: len(test_ids)]
      3 
      4 submission = pd.DataFrame({"id": test_ids, "label": preds})
      5 submission = submission.sort_values("id").reset_index(drop=True)

NameError: name 'kept_test_paths' is not defined

## === cell 17
try:
    shutil.rmtree(EXTRACT_DIR)
    print("Cleaned:", EXTRACT_DIR)
except Exception as e:
    print("Cleanup skipped:", e)
