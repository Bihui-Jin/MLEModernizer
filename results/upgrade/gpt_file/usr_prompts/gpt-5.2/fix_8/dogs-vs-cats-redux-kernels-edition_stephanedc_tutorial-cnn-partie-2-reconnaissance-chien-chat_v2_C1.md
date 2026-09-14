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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import class_weight as cw

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    _cpu = os.cpu_count() or 2
    _intra = max(2, min(8, _cpu))
    _inter = max(2, min(4, _cpu))
    tf.config.threading.set_intra_op_parallelism_threads(_intra)
    tf.config.threading.set_inter_op_parallelism_threads(_inter)
except Exception:
    pass



## === cell 1
EPOCHS = 30
IMGSIZE = 150
BATCH_SIZE = 64
STOPPING_PATIENCE = 20
VERBOSE = 1
MODEL_NAME = "cnn"
OPTIMIZER = "adam"

BASE_DIR = Path("/kaggle/input/dogs-vs-cats-redux-kernels-edition")
TRAINING_DIR = BASE_DIR / "train"  # contains cat/ and dog/ subfolders

TEST_DIR_CANDIDATES = [
    BASE_DIR / "test" / "test",
    BASE_DIR / "test" / "unknown",
    BASE_DIR / "test" / "test" / "unknown",
]


def _resolve_test_dir(candidates):
    for d in candidates:
        if d.exists() and d.is_dir():
            jpgs = list(d.glob("*.jpg"))
            if len(jpgs) > 0:
                return d
    test_root = BASE_DIR / "test"
    if test_root.exists():
        for d in sorted({p.parent for p in test_root.rglob("*.jpg")}):
            if d.exists() and d.is_dir() and len(list(d.glob("*.jpg"))) > 0:
                return d
    raise FileNotFoundError(
        f"Could not find any .jpg test images under {BASE_DIR/'test'}"
    )


TEST_DIR = _resolve_test_dir(TEST_DIR_CANDIDATES)

TRAIN_MODEL = True

print("TRAINING_DIR:", TRAINING_DIR)
print("TEST_DIR:", TEST_DIR)
print("Train exists:", TRAINING_DIR.exists(), "Test exists:", TEST_DIR.exists())



## === cell 2
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.3,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.25,
)

train_generator = train_datagen.flow_from_directory(
    directory=str(TRAINING_DIR),
    target_size=(IMGSIZE, IMGSIZE),
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="binary",
    shuffle=True,
    seed=SEED,
)
valid_generator = train_datagen.flow_from_directory(
    directory=str(TRAINING_DIR),
    target_size=(IMGSIZE, IMGSIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="binary",
    shuffle=True,
    seed=SEED,
)

print("class_indices:", train_generator.class_indices)



## === cell 3
test_files = sorted(
    [
        p.name
        for p in TEST_DIR.iterdir()
        if p.is_file() and p.name.lower().endswith(".jpg")
    ]
)
if len(test_files) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_DIR}")

df_test = pd.DataFrame({"id": test_files})
print(df_test.head(), "N_test:", len(df_test))

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

TEST_BATCH_SIZE = 64
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=str(TEST_DIR),
    x_col="id",
    y_col=None,
    target_size=(IMGSIZE, IMGSIZE),
    class_mode=None,
    seed=SEED,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
)




## === cell 4
def get_weight(y):
    return cw.compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)


class_weights_arr = get_weight(train_generator.classes)
class_weights = {int(i): float(w) for i, w in enumerate(class_weights_arr)}
print("class_weights:", class_weights)

STEP_SIZE_TRAIN = len(train_generator)
STEP_SIZE_VALID = len(valid_generator)
STEP_SIZE_TEST = len(test_generator)
print("steps (train/valid/test):", STEP_SIZE_TRAIN, STEP_SIZE_VALID, STEP_SIZE_TEST)



## === cell 5
EARLY_STOPPING = EarlyStopping(
    monitor="val_loss",
    patience=STOPPING_PATIENCE,
    verbose=VERBOSE,
    mode="auto",
    restore_best_weights=True,
)

LR_REDUCTION = ReduceLROnPlateau(
    monitor="val_accuracy", patience=3, verbose=VERBOSE, factor=0.5, min_lr=0.00001
)

CALLBACKS = [EARLY_STOPPING, LR_REDUCTION]



## === cell 6
classifier = Sequential()

classifier.add(
    Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding="same",
        input_shape=(IMGSIZE, IMGSIZE, 3),
        activation="relu",
    )
)
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(
    Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")
)
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(
    Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")
)
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(
    Conv2D(filters=48, kernel_size=3, strides=1, padding="same", activation="relu")
)
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(
    Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")
)
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(512, activation="relu"))
classifier.add(Dropout(0.2))
classifier.add(Dense(128, activation="relu"))
classifier.add(Dropout(0.2))
classifier.add(Dense(128, activation="relu"))
classifier.add(Dropout(0.2))

classifier.add(Dense(units=1, activation="sigmoid", name="sigmoid"))

classifier.compile(
    optimizer=OPTIMIZER, loss="binary_crossentropy", metrics=["accuracy"]
)

print("Input Shape :", classifier.input_shape)
classifier.summary()




## === cell 7
def train_model():
    classifier.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=EPOCHS,
        verbose=VERBOSE,
        class_weight=class_weights,
        callbacks=CALLBACKS,
    )


if TRAIN_MODEL:
    print("Training CNN...")
    train_model()
    classifier.save(f"{MODEL_NAME}.h5")
else:
    print("Loading model from working directory...")
    classifier = load_model(f"{MODEL_NAME}.h5")



## === cell 8
test_generator.reset()

pred = classifier.predict(
    test_generator,
    verbose=1,
)

print("pred shape:", pred.shape)
print(pred[:5])



## === cell 9
class_indices = train_generator.class_indices  # e.g. {'cat':0,'dog':1} (order may vary)
if "dog" not in class_indices:
    raise ValueError(f"'dog' not found in class_indices: {class_indices}")

dog_class = int(class_indices["dog"])  # 0 or 1
pred = pred.reshape(-1).astype(np.float64)

n_test = len(df_test)
pred = pred[:n_test]

dog_proba = pred if dog_class == 1 else (1.0 - pred)

eps = 1e-7
dog_proba = np.clip(dog_proba, eps, 1.0 - eps)

ids = df_test["id"].str.replace(".jpg", "", regex=False).astype(int)
submission = pd.DataFrame({"id": ids, "label": dog_proba})
submission = submission.sort_values("id").reset_index(drop=True)

print(submission.head())
print(submission.tail())



## === cell 10
filename = "submission.csv"
submission.to_csv(filename, index=False)
print("Saved submission:", filename, "rows:", len(submission))



## === cell 11
DO_PLOT = False

try:
    if DO_PLOT:
        c = 20
        n = len(df_test)
        r = random.sample(range(n), min(c, n))
        nrows, ncols = 4, 5
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
        ax = ax.flatten()
        for i, idx in enumerate(r):
            file = df_test["id"].iloc[idx]
            path = TEST_DIR / file
            img = plt.imread(str(path))
            ax[i].imshow(img)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"{file}\ndog_proba={dog_proba[idx]:.3f}")
        for j in range(i + 1, nrows * ncols):
            ax[j].axis("off")
        plt.tight_layout()
        plt.show()
except Exception as e:
    print("Visualization skipped due to:", repr(e))
