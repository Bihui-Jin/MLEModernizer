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

3.7

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
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

0.5011

# 6. Current score

18.13717

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 18.13717) has done: 'I fix the environment-breaking Keras import issue by switching to `tf_keras` (available in your packages) while keeping the exact same InceptionResNetV2 + Dense head and training procedure. I also fix the dataset paths to point at the actual extracted folders (`../input/dogs-vs-cats-redux-kernels-edition/train/...` and `.../test/unknown/...`) so `cv2.imread()` doesn’t return `None` and crash in `cv2.resize`. I replace deprecated `fit_generator` with `fit`, restore `ImageDataGenerator` from `tf_keras`, and fix a couple of Python 3 plotting issues (integer subplot args) and removed deprecated `np.float`. Finally, I ensure the submission is written as a valid `submission.csv` with `id,label`, sorted by numeric `id`.'

# 9. Code solution

## === cell 0
import os
import gc
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.model_selection import train_test_split

from tf_keras.applications import InceptionResNetV2
from tf_keras import layers, models
from tf_keras.preprocessing.image import ImageDataGenerator

random.seed(1)
np.random.seed(1)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print("Input root listing:", os.listdir("../input")[:20])

DATA_ROOT = "../input/dogs-vs-cats-redux-kernels-edition"

train_dir = os.path.join(DATA_ROOT, "train")
test_dir = os.path.join(DATA_ROOT, "test")

print("Train dir exists:", os.path.isdir(train_dir), train_dir)
print("Test dir exists:", os.path.isdir(test_dir), test_dir)



## === cell 2
train_cat_dir = os.path.join(train_dir, "cat")
train_dog_dir = os.path.join(train_dir, "dog")

candidate_test_unknown = [
    os.path.join(test_dir, "unknown"),
    os.path.join(test_dir, "test", "unknown"),
    os.path.join(test_dir, "test", "test", "unknown"),
]
test_unknown_dir = next((p for p in candidate_test_unknown if os.path.isdir(p)), None)
if test_unknown_dir is None:
    raise FileNotFoundError(
        f"Could not find test unknown dir in candidates: {candidate_test_unknown}"
    )

print("Resolved test_unknown_dir:", test_unknown_dir)

train_dogs = [
    os.path.join(train_dog_dir, f)
    for f in os.listdir(train_dog_dir)
    if f.lower().endswith(".jpg")
]
train_cats = [
    os.path.join(train_cat_dir, f)
    for f in os.listdir(train_cat_dir)
    if f.lower().endswith(".jpg")
]

test_imgs = [
    os.path.join(test_unknown_dir, f)
    for f in os.listdir(test_unknown_dir)
    if f.lower().endswith(".jpg")
]

print("Num train dogs:", len(train_dogs))
print("Num train cats:", len(train_cats))
print("Num test:", len(test_imgs))



## === cell 3
size = 4000
train_imgs = train_dogs[:size] + train_cats[:size]
random.shuffle(train_imgs)

img_size = 150




## === cell 4
def read_and_process_image(list_of_images):
    """
    Returns three lists:
        X: resized images (H,W,3)
        y: labels (1=dog, 0=cat) for train; empty for test
        l_id: string ids (for submission)
    """
    X = []
    y = []
    l_id = []

    for image_path in list_of_images:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        X.append(img)

        basename = os.path.basename(image_path)
        img_num = basename.split(".")[0]
        l_id.append(img_num)

        if "dog" in image_path:
            y.append(1)
        elif "cat" in image_path:
            y.append(0)

    return X, y, l_id




## === cell 5
X, y, l_id = read_and_process_image(train_imgs)

X = np.array(X, dtype=np.uint8)
y = np.array(y, dtype=np.int64)

print("Loaded train subset:", X.shape, y.shape)
print("Label counts:", np.bincount(y) if len(y) else "empty")



## === cell 6
plt.figure(figsize=(12, 6))
columns = 5
rows = int(np.ceil(columns / columns))  # 1 row
for i in range(min(columns, len(X))):
    plt.subplot(rows, columns, i + 1)
    plt.imshow(X[i])
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 7
sns.countplot(x=y)
plt.title("Labels for Cats and Dogs (subset)")
plt.show()

print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=1, stratify=y
)

del X, y, train_imgs, train_dogs, train_cats
gc.collect()

print("Shape of X_train", X_train.shape)
print("Shape of X_val", X_val.shape)

ntrain = len(X_train)
nval = len(X_val)



## === cell 9
conv_base = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3)
)
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.summary()



## === cell 10
batch_size = 128

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow(
    X_train, y_train, batch_size=batch_size, shuffle=True
)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)



## === cell 11
epochs = 5
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, ntrain // batch_size),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=max(1, nval // batch_size),
    verbose=1,
)



## === cell 12
hist = history.history
acc_key = "acc" if "acc" in hist else "accuracy"
val_acc_key = "val_acc" if "val_acc" in hist else "val_accuracy"

acc = hist.get(acc_key, [])
val_acc = hist.get(val_acc_key, [])
loss = hist.get("loss", [])
val_loss = hist.get("val_loss", [])

epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, "b", label="Training acc")
plt.plot(epochs_range, val_acc, "r", label="Validation acc")
plt.title("Training and Validation accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, "b", label="Training loss")
plt.plot(epochs_range, val_loss, "r", label="Validation loss")
plt.title("Training and Validation loss")
plt.legend()
plt.tight_layout()
plt.show()



## === cell 13
X_test_preview, _, l_id_preview = read_and_process_image(test_imgs[:10])
x_preview = np.array(X_test_preview, dtype=np.float32) / 255.0

pred_preview = model.predict(x_preview, verbose=0).reshape(-1)
print("Preview ids:", l_id_preview[:5])
print("Preview preds:", pred_preview[:5])



## === cell 14
del X_train, X_val, y_train, y_val
gc.collect()



## === cell 15
X_test, _, l_id = read_and_process_image(test_imgs)
x = np.array(X_test, dtype=np.float32) / 255.0
del X_test
gc.collect()

predictions = model.predict(x, batch_size=128, verbose=1).reshape(-1)

sub = pd.DataFrame(
    {
        "id": pd.to_numeric(l_id, errors="coerce"),
        "label": predictions.astype(np.float64),
    }
)
sub = (
    sub.dropna(subset=["id"])
    .astype({"id": "int64"})
    .sort_values("id")
    .reset_index(drop=True)
)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head(10))



## === cell 16
print(
    "Pred min/max/mean:",
    float(sub["label"].min()),
    float(sub["label"].max()),
    float(sub["label"].mean()),
)
