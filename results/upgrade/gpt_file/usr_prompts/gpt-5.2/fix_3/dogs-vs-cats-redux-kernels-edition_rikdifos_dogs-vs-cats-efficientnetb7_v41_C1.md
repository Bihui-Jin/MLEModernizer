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

0.19583

# 6. Current score

0.76126

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.76126) has done: 'I fix the two root runtime blockers: the protobuf/TensorFlow import crash (caused by an incompatible `protobuf` version in this environment) and the wrong extracted folder paths (your zip extracts images directly into `./data/`, not `./data/train/train/`). I keep your model/training logic intact, but make the dataset discovery robust by building `train_images`/`test_images` from the actual extracted filenames and ensuring labels stay aligned even if some images fail to load. I also make plotting cells safe when the dataset is smaller than expected (avoiding index errors) and ensure we always write a valid `submission.csv` with `id,label` sorted by `id`. These changes are correctness/stability focused and should allow the notebook to run end-to-end and produce a valid submission file.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import warnings

warnings.filterwarnings("ignore")

import cv2, re, random, time, zipfile, gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.applications import EfficientNetB7

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

os.makedirs("./data", exist_ok=True)

if not any(fn.lower().endswith(".jpg") for fn in os.listdir("./data")):
    with zipfile.ZipFile(train_zip_path, "r") as z:
        z.extractall("./data")

if not any(
    fn.lower().endswith(".jpg") and fn.split(".")[0].isdigit()
    for fn in os.listdir("./data")
):
    with zipfile.ZipFile(test_zip_path, "r") as z:
        z.extractall("./data")

DATA_DIR = "./data"
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Expected DATA_DIR '{DATA_DIR}' not found.")

print("Data dir:", DATA_DIR, "n_files:", len(os.listdir(DATA_DIR)))




## === cell 2
def txt_dig(text):
    """Input string, if it is a number, output the number, if not, output the original string."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Enter a string, separate the number from the text, and convert number string to int."""
    return [txt_dig(c) for c in re.split(r"(\d+)", text)]


def get_test_id_from_path(p):
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    return int(stem)




## === cell 3
start = time.time()

all_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg")]

train_images = [
    os.path.join(DATA_DIR, f)
    for f in all_files
    if f.lower().startswith(("cat.", "dog."))
]

test_images = []
for f in all_files:
    stem = os.path.splitext(f)[0]
    if stem.isdigit():
        test_images.append(os.path.join(DATA_DIR, f))

train_images.sort(key=natural_keys)
test_images.sort(key=natural_keys)

if len(train_images) >= 25000:
    train_images = train_images[0:7500] + train_images[17500:25000]
else:
    train_images = train_images

random.seed(SEED)
random.shuffle(train_images)

print("Train images:", len(train_images))
print("Test images:", len(test_images))

if len(train_images) == 0:
    raise RuntimeError(
        "No training images found after extraction. Check dataset paths."
    )
if len(test_images) == 0:
    raise RuntimeError("No test images found after extraction. Check dataset paths.")



## === cell 4
IMG_WIDTH = 128
IMG_HEIGHT = 128

x = []
x_paths_kept = []
for img in train_images:
    arr = cv2.imread(img)
    if arr is None:
        continue
    arr = cv2.resize(arr, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    x.append(arr)
    x_paths_kept.append(img)

test = []
test_paths_kept = []
for img in test_images:
    arr = cv2.imread(img)
    if arr is None:
        continue
    arr = cv2.resize(arr, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    test.append(arr)
    test_paths_kept.append(img)

x = np.array(x)
test = np.array(test)

print("The shape of train data is {}".format(x.shape))
print("The shape of test data is {}".format(test.shape))

plt.rcParams["figure.facecolor"] = "white"
y = []
for p in x_paths_kept:
    fname = os.path.basename(p).lower()
    if fname.startswith("dog."):
        y.append(1)
    elif fname.startswith("cat."):
        y.append(0)
    else:
        y.append(0)
y = np.array(y, dtype=np.int64)

print("Labels shape:", y.shape, "dogs:", int(y.sum()), "cats:", int((1 - y).sum()))
sns.countplot(x=y)
plt.show()

test_images = test_paths_kept



## === cell 5
random.seed(SEED)
plt.subplots(facecolor="white", figsize=(10, 4))

if len(x_paths_kept) > 0:
    for idx, ax_i in enumerate([131, 132, 133], start=1):
        sample = random.choice(x_paths_kept)
        image = load_img(sample, target_size=(IMG_WIDTH, IMG_HEIGHT))
        plt.subplot(ax_i)
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("No training images available for plotting.")



## === cell 6
plt.subplots(facecolor="white", figsize=(10, 4))

if len(x) > 0:
    sample_indices = [min(1024, len(x) - 1), min(546, len(x) - 1), min(742, len(x) - 1)]
    for plot_i, sample_i in zip([131, 132, 133], sample_indices):
        plt.subplot(plot_i)
        plt.imshow(cv2.cvtColor(x[sample_i], cv2.COLOR_BGR2RGB))
        plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("No training images available for plotting.")



## === cell 7
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=2020, stratify=y
)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)



## === cell 8
model = models.Sequential()

efnModel = EfficientNetB7(
    weights="imagenet", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False
)

model.add(efnModel)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation="sigmoid"))

opt1 = RMSprop(learning_rate=1e-5, decay=1e-6)
opt2 = Adam(learning_rate=2e-4)

model.compile(loss="binary_crossentropy", optimizer=opt2, metrics=["accuracy"])

model.summary()



## === cell 9
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




## === cell 10
def plot_gened(train_images, seed=320):
    """Plot pictures after processing."""
    if len(train_images) == 0:
        print("No train images to visualize.")
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
        X_batch, _ = next(vis_gen0)
        image = X_batch[0]
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


plot_gened(x_paths_kept)



## === cell 11
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



## === cell 12
plt.rcParams["figure.facecolor"] = "white"
model_loss = pd.DataFrame(history.history)
print(model_loss.head())

ax = model_loss[["accuracy", "val_accuracy"]].plot(ylim=[0, 1], title="Accuracy")
plt.show()
ax = model_loss[["loss", "val_loss"]].plot(ylim=[0, 1], title="Loss")
plt.show()



## === cell 13
x_val_scaled = x_val.astype("float32") / 255.0
val_preds = model.predict(x_val_scaled, batch_size=32, verbose=0)
val_preds = val_preds.ravel().astype("float64")
val_preds_class = (val_preds > 0.5).astype(int)

print("Out of Fold Accuracy is {:.5f}".format(accuracy_score(y_val, val_preds_class)))
print("Out of Fold log loss is {:.5f}".format(log_loss(y_val, val_preds)))



## === cell 14
test_scaled = test.astype("float32") / 255.0
test_pred = model.predict(test_scaled, batch_size=32, verbose=0).ravel()

test_ids = [get_test_id_from_path(p) for p in test_images]

n = min(len(test_ids), len(test_pred))
submission = pd.DataFrame({"id": test_ids[:n], "label": test_pred[:n].astype(float)})
submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("This program costs {:.2f} seconds".format(time.time() - start))
submission.head()



## === cell 15
import shutil

shutil.rmtree("/kaggle/working/data/", ignore_errors=True)
print("Cleaned /kaggle/working/data/")
