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

3.13

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

0.37309

# 6. Current score

0.44011

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.44011) has done: 'I fix the data-path resolution so the extracted `train/` and `test/` folders are actually found (your notebook is extracting into the current working directory, not under `../input`). I also fix the TensorFlow/Keras import incompatibility that triggers the protobuf `MessageFactory.GetPrototype` error by using `tf_keras` (which is installed and compatible with TF 2.18 in this environment) while keeping the same Sequential CNN architecture and training loop. Then I make sure the test file listing, ID extraction, and prediction ordering are consistent so the submission aligns with the required `id,label` format. Finally, the script always write a valid `submission.csv` to the working directory.'

# 9. Code solution

## === cell 0
import os

print(os.listdir("../input"))
print(os.listdir("../input/dogs-vs-cats-redux-kernels-edition"))



## === cell 1
from zipfile import ZipFile
import os

data_path = "../input/dogs-vs-cats-redux-kernels-edition/"

with ZipFile(os.path.join(data_path, "train.zip")) as zipper:
    zipper.extractall()

with ZipFile(os.path.join(data_path, "test.zip")) as zipper:
    zipper.extractall()


def find_image_dir(root_name: str) -> str:
    """
    Find the directory that actually contains extracted image files.
    We check common layouts:
      - ./train (files like cat.0.jpg, dog.0.jpg)
      - ./train/train (some zips nest a folder)
      - ./train/cat + ./train/dog (class subfolders)
      - ./test/test or ./test/unknown
    """
    cwd = os.getcwd()
    base = os.path.join(cwd, root_name)

    candidates = [
        base,
        os.path.join(base, root_name),
        os.path.join(base, "unknown"),
        os.path.join(base, root_name, "unknown"),
        os.path.join(base, "test"),
        os.path.join(base, root_name, "test"),
        root_name,
        os.path.join(root_name, root_name),
        os.path.join(root_name, "unknown"),
        os.path.join(root_name, root_name, "unknown"),
        os.path.join(root_name, "test"),
        os.path.join(root_name, root_name, "test"),
    ]

    def has_images(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        for _, _, files in os.walk(d):
            if any(
                f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
                for f in files
            ):
                return True
        return False

    for c in candidates:
        if has_images(c):
            return c

    for dirpath, dirnames, _ in os.walk(cwd):
        if os.path.basename(dirpath) == root_name and has_images(dirpath):
            return dirpath

    raise FileNotFoundError(
        f"Could not locate extracted {root_name} images directory. Checked: {candidates} (cwd={cwd})"
    )


TRAIN_DIR = find_image_dir("train")
TEST_DIR = find_image_dir("test")

print("Resolved TRAIN_DIR:", TRAIN_DIR)
print("Resolved TEST_DIR:", TEST_DIR)



## === cell 2
import os


def list_train_images(train_dir: str):
    files = []
    for root, _, fnames in os.walk(train_dir):
        for f in fnames:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                files.append(os.path.join(root, f))
    return sorted(files)


def list_test_images(test_dir: str):
    files = []
    for root, _, fnames in os.walk(test_dir):
        for f in fnames:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                files.append(os.path.join(root, f))

    def get_id(path):
        base = os.path.basename(path)
        return int(os.path.splitext(base)[0])

    return sorted(files, key=get_id)


train_files = list_train_images(TRAIN_DIR)
test_files = list_test_images(TEST_DIR)

print("훈련 이미지 개수:", len(train_files))
print("테스트 이미지 개수:", len(test_files))
print("Train sample:", [os.path.basename(p) for p in train_files[:5]])
print("Test sample:", [os.path.basename(p) for p in test_files[:5]])



## === cell 3
import pandas as pd
import os

filenames = []
labels = []

for path in train_files:
    base = os.path.basename(path).lower()
    parent = os.path.basename(os.path.dirname(path)).lower()

    if base.startswith("cat."):
        lab = 0
    elif base.startswith("dog."):
        lab = 1
    elif parent == "cat":
        lab = 0
    elif parent == "dog":
        lab = 1
    else:
        continue

    filenames.append(path)
    labels.append(lab)

df = pd.DataFrame({"filename": filenames, "label": labels})
print(df.head())
print(df["label"].value_counts())



## === cell 4
import numpy as np

df["label"].value_counts()



## === cell 5
import matplotlib.pyplot as plt
import random as r
import cv2
import os

plt.figure(figsize=(15, 15))

row = 3
col = 3

for i in range(row * col):
    img_idx = r.randint(0, len(df) - 1)
    filepath = df["filename"].iloc[img_idx]
    img = cv2.imread(filepath)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(row, col, i + 1)
    plt.imshow(img)
    plt.title(os.path.basename(filepath))
    plt.axis("off")

plt.tight_layout()
plt.show()



## === cell 6
import tensorflow as tf
import tf_keras
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

IMAGE_WIDTH = 112
IMAGE_HEIGHT = 112
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.50))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
sorted_idx = np.argsort(df["filename"].values)
sorted_filenames = df["filename"].values[sorted_idx].tolist()
sorted_labels = df["label"].values[sorted_idx].astype("float32")



## === cell 8
import tensorflow as tf

X = np.array(sorted_filenames)
y = np.array(sorted_labels)

rng = np.random.RandomState(42)
perm = rng.permutation(len(X))
X = X[perm]
y = y[perm]

val_size = int(round(0.15 * len(X)))
X_val, y_val = X[:val_size], y[:val_size]
X_train, y_train = X[val_size:], y[val_size:]

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32


def decode_and_resize(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    label = tf.cast(tf.reshape(label, (1,)), tf.float32)
    return img, label


training_data = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(
        buffer_size=min(8192, len(X_train)), seed=42, reshuffle_each_iteration=True
    )
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

validation_data = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Train batches:", tf.data.experimental.cardinality(training_data).numpy())
print("Val batches:", tf.data.experimental.cardinality(validation_data).numpy())



## === cell 9
import matplotlib.pyplot as plt

image_batch, label_batch = next(iter(training_data))

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy())
    plt.title(f"Label: {float(label_batch[i].numpy().ravel()[0]):.0f}")
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 10
history = model.fit(training_data, epochs=10, validation_data=validation_data)



## === cell 11
loss, accuracy = model.evaluate(validation_data, verbose=0)
print(f"모델 평가 결과 - 손실(loss): {loss:.4f}, 정확도(accuracy): {accuracy:.4f}")



## === cell 12
image_batch, label_batch = next(iter(validation_data))

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy())
    ans = "dog" if float(label_batch[i].numpy().ravel()[0]) >= 0.5 else "cat"
    model_output = model(image_batch[i : i + 1], training=False)
    p = float(model_output.numpy().ravel()[0])
    pred = "dog" if p >= 0.5 else "cat"
    plt.title(f"Predict: {pred}({p:.2f}), Answer: {ans}")
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 13
X_test = np.array(test_files)

test_data = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .map(lambda p: decode_and_resize(p, None), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Test batches:", tf.data.experimental.cardinality(test_data).numpy())



## === cell 14
preds = model.predict(test_data, verbose=0)
pred_list = preds.flatten()
print(pred_list[:10], " ... total:", len(pred_list))



## === cell 15
import matplotlib.pyplot as plt

first_batch = next(iter(test_data))
plt.figure(figsize=(10, 20))
n = min(32, first_batch.shape[0], len(pred_list))
for i in range(n):
    plt.subplot(8, 4, i + 1)
    plt.imshow(first_batch[i].numpy())
    plt.title(
        f"Predict: {'dog' if int(round(pred_list[i])) else 'cat'} ({float(pred_list[i]):.2f})"
    )
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 16
import os
import pandas as pd

test_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in test_files]

submission_df = pd.DataFrame({"id": test_ids, "label": pred_list.astype(float)})
submission_df = submission_df.sort_values("id").reset_index(drop=True)

print(submission_df.head())
print(submission_df.shape)



## === cell 17
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Saved:", submission_path)
print(open(submission_path, "r").read().splitlines()[:5])
