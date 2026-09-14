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

1.07092

# 6. Current score

0.30901

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.30901) has done: 'I fix the initial protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it conflicts with protobuf 6.x + TF 2.18 in Kaggle). I also fix the dataset path resolution: after unzip, images are extracted directly into `./data/` (not `./data/train/...` and `./data/test/...`), so I rebuild `train_images` and `test_images` by filename pattern instead of searching nonexistent folders. Finally, I keep the same model and training approach, but ensure all dependent variables (`train_images`, `x`, `x_train`, etc.) are created before use and that a valid `submission.csv` with `id,label` is always written.'

# 9. Code solution

## === cell 0
import os
import warnings

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

warnings.filterwarnings("ignore")

import cv2, re, random, time, zipfile, gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

SEED = 558
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
start = time.time()

PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
train_image_path = os.path.join(PATH, "train.zip")
test_image_path = os.path.join(PATH, "test.zip")

os.makedirs("./data", exist_ok=True)

with zipfile.ZipFile(train_image_path, "r") as z:
    z.extractall("./data")

with zipfile.ZipFile(test_image_path, "r") as z:
    z.extractall("./data")

print("Extracted to ./data. Top-level entries sample:", os.listdir("./data")[:10])




## === cell 2
def txt_dig(text):
    """If string is digit return int, else return original."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Split text into list of ints/strings for natural sorting."""
    return [txt_dig(c) for c in re.split(r"(\d+)", text)]




## === cell 3
all_jpgs = [f for f in os.listdir("./data") if f.lower().endswith(".jpg")]

train_images = []
test_images = []

for f in all_jpgs:
    fl = f.lower()
    if fl.startswith("cat.") or fl.startswith("dog."):
        train_images.append(os.path.join("./data", f))
    else:
        stem = os.path.splitext(f)[0]
        if stem.isdigit():
            test_images.append(os.path.join("./data", f))

train_images.sort(key=natural_keys)
test_images.sort(key=natural_keys)

print("Num train images found:", len(train_images))
print("Num test images found:", len(test_images))
print("Sample train:", train_images[0] if train_images else None)
print("Sample test:", test_images[0] if test_images else None)

if len(train_images) == 0 or len(test_images) == 0:
    raise FileNotFoundError(
        "Could not locate expected train/test jpgs under ./data after extraction. "
        f"Found {len(all_jpgs)} jpgs total."
    )



## === cell 4
if len(train_images) >= 25000:
    train_images = train_images[0:7500] + train_images[17500:25000]

random.seed(SEED)
random.shuffle(train_images)

print("Num sampled train images:", len(train_images))



## === cell 5
IMG_WIDTH = 128
IMG_HEIGHT = 128

x_list, y_list = [], []
kept_train_paths = []

for img_path in train_images:
    im = cv2.imread(img_path)
    if im is None:
        continue
    im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    x_list.append(im)

    fn = os.path.basename(img_path).lower()
    if "dog" in fn:
        y_list.append(1)
    elif "cat" in fn:
        y_list.append(0)
    else:
        y_list.append(0)

    kept_train_paths.append(img_path)

x = np.array(x_list, dtype=np.uint8)
y = np.array(y_list, dtype=np.int32)

test_list = []
kept_test_paths = []
for img_path in test_images:
    im = cv2.imread(img_path)
    if im is None:
        continue
    im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    test_list.append(im)
    kept_test_paths.append(img_path)

test = np.array(test_list, dtype=np.uint8)

train_images = kept_train_paths
test_images = kept_test_paths

print("The shape of train data is {}".format(x.shape))
print("The shape of test data is {}".format(test.shape))

plt.rcParams["figure.facecolor"] = "white"
ax = sns.countplot(x=y)
ax.set_title("Label counts (0=cat, 1=dog)")
plt.show()



## === cell 6
random.seed(SEED)
plt.subplots(facecolor="white", figsize=(10, 4))

if len(train_images) > 0:
    for j in range(3):
        sample = random.choice(train_images)
        image = load_img(sample)
        plt.subplot(1, 3, j + 1)
        plt.imshow(image)
        plt.axis("off")

plt.tight_layout()
plt.show()



## === cell 7
plt.subplots(facecolor="white", figsize=(10, 4))
if len(x) > 0:
    idxs = [min(1024, len(x) - 1), min(546, len(x) - 1), min(742, len(x) - 1)]
    for j, idx in enumerate(idxs):
        plt.subplot(1, 3, j + 1)
        plt.imshow(cv2.cvtColor(x[idx], cv2.COLOR_BGR2RGB))
        plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 8
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=2020, stratify=y
)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)



## === cell 9
model = models.Sequential()

efnModel = tf.keras.applications.EfficientNetB7(
    weights="imagenet", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False
)
model.add(efnModel)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation="sigmoid"))

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
    """Plot pictures after augmentation processing."""
    if len(train_images) == 0:
        return

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
    for i in range(0, 9):
        plt.subplot(3, 3, i + 1)
        for X_batch, _ in vis_gen0:
            image = X_batch[0]
            plt.imshow(image)
            plt.axis("off")
            break
    plt.tight_layout()
    plt.show()


plot_gened(train_images)



## === cell 12
BATCH_SIZE = 16

train_flow = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_flow = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

earlystop1 = EarlyStopping(patience=5, restore_best_weights=True)
earlystop2 = ReduceLROnPlateau(
    monitor="val_accuracy", min_lr=0.001, patience=5, mode="min", verbose=1
)

history = model.fit(
    train_flow,
    steps_per_epoch=45,
    epochs=20,
    validation_data=val_flow,
    callbacks=[earlystop1, earlystop2],
    validation_steps=25,
)



## === cell 13
plt.rcParams["figure.facecolor"] = "white"
model_loss = pd.DataFrame(history.history)

ax = model_loss[["accuracy", "val_accuracy"]].plot(ylim=[0, 1], title="Accuracy")
ax.set_xlabel("Epoch")
plt.show()

ax = model_loss[["loss", "val_loss"]].plot(title="Loss")
ax.set_xlabel("Epoch")
plt.show()



## === cell 14
val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))
val_preds = model.predict(val_flow, verbose=1, steps=val_steps)
print(val_steps)
print("Out of Fold log loss is {:.5f}".format(log_loss(y_val, val_preds.ravel())))



## === cell 15
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_flow = test_datagen.flow(test, batch_size=BATCH_SIZE, shuffle=False)

test_steps = int(np.ceil(len(test) / BATCH_SIZE))
test_pred = model.predict(test_flow, verbose=1, steps=test_steps)
print(test_steps)



## === cell 16
test_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in test_images]

submission = pd.DataFrame({"id": test_ids, "label": test_pred.ravel()[: len(test_ids)]})
submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("This program costs {:.2f} seconds".format(time.time() - start))
print(submission.head())



## === cell 17
import shutil

shutil.rmtree("/kaggle/working/data/", ignore_errors=True)
print("Cleaned /kaggle/working/data/")
