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

# 5. Code solution

## === cell 0
import os, re, random, time, gc, zipfile, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import cv2

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

import warnings

warnings.filterwarnings("ignore")

start = time.time()
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)



## === cell 1
BASE_PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

train_images = [
    os.path.join(root, fname)
    for root, _, files in os.walk(TRAIN_DIR)
    for fname in files
    if fname.lower().endswith(".jpg")
]
test_images = [
    os.path.join(root, fname)
    for root, _, files in os.walk(TEST_DIR)
    for fname in files
    if fname.lower().endswith(".jpg")
]


def txt_dig(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [txt_dig(c) for c in re.split(r"(\d+)", text)]


train_images.sort(key=natural_keys)
test_images.sort(key=natural_keys)

random.shuffle(train_images)



## === cell 2
IMG_WIDTH, IMG_HEIGHT = 128, 128

from concurrent.futures import ThreadPoolExecutor


def load_and_resize(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32)


num_train = len(train_images)
x = np.empty((num_train, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
with ThreadPoolExecutor() as executor:
    for i, img in enumerate(executor.map(load_and_resize, train_images)):
        x[i] = img

num_test = len(test_images)
test = np.empty((num_test, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
with ThreadPoolExecutor() as executor:
    for i, img in enumerate(executor.map(load_and_resize, test_images)):
        test[i] = img

y = np.array([1 if "dog" in p.lower() else 0 for p in train_images], dtype="int")

print(f"Train shape: {x.shape}, Test shape: {test.shape}")
sns.countplot(x=y)
plt.show()

del train_images, test_images
gc.collect()



## === cell 3
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=2020, stratify=y
)



## === cell 4
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)
base_model.trainable = False  # freeze base for initial training

model = models.Sequential(
    [base_model, layers.GlobalAveragePooling2D(), layers.Dense(1, activation="sigmoid")]
)

opt = Adam(learning_rate=8e-5)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()



## === cell 5
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)

BATCH_SIZE = 16
train_flow = train_gen.flow(x_train, y_train, batch_size=BATCH_SIZE)
val_flow = val_gen.flow(x_val, y_val, batch_size=BATCH_SIZE)

early_stop = callbacks.EarlyStopping(
    patience=5, restore_best_weights=True, monitor="val_loss"
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, mode="min", verbose=1
)



## === cell 6
history = model.fit(
    train_flow,
    steps_per_epoch=len(train_flow),
    epochs=20,
    validation_data=val_flow,
    validation_steps=len(val_flow),
    callbacks=[early_stop, reduce_lr],
    verbose=2,
)



## === cell 7
base_model.trainable = True
fine_opt = Adam(learning_rate=1e-5)
model.compile(loss="binary_crossentropy", optimizer=fine_opt, metrics=["accuracy"])

fine_history = model.fit(
    train_flow,
    steps_per_epoch=len(train_flow),
    epochs=5,
    validation_data=val_flow,
    validation_steps=len(val_flow),
    callbacks=[early_stop],
    verbose=2,
)



## === cell 8
hist_df = pd.DataFrame(history.history)
fine_df = pd.DataFrame(fine_history.history)
combined = pd.concat([hist_df, fine_df], ignore_index=True)
combined[["accuracy", "val_accuracy"]].plot(ylim=[0, 1], title="Accuracy")
combined[["loss", "val_loss"]].plot(ylim=[0, 1], title="Loss")
plt.show()



## === cell 9
x_val_norm = x_val / 255.0
val_preds = model.predict(x_val_norm).ravel()
val_pred_class = (val_preds > 0.5).astype(int)

print(f"Out‑of‑Fold Accuracy: {accuracy_score(y_val, val_pred_class):.5f}")
print(f"Out‑of‑Fold Log Loss: {log_loss(y_val, val_preds, labels=[0,1]):.5f}")



## === cell 10
test_norm = test / 255.0
test_pred = model.predict(test_norm).ravel()
submission = pd.DataFrame({"id": np.arange(1, len(test_pred) + 1), "label": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Elapsed time: {time.time() - start:.2f} seconds")
submission.head()
