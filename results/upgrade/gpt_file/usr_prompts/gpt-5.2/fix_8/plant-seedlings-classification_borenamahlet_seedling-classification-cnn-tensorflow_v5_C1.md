# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.35075

# 6. Current score

0.497

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.54505) has done: 'I fix the environment import crash by pinning protobuf to a compatible version at runtime before importing TensorFlow (this is the root cause of the `MessageFactory` error). Then I fix the model shape runtime error by keeping the same “2conv-basic” CNN idea but reducing the excessive Conv/Pool repetitions that make the spatial size go negative for 50×50 inputs. Finally, I ensure the model is always defined (train or load), and that we write a valid `submission.csv` with exactly the required `file,species` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.53604) has done: 'Your current score (0.54505) is above the target (0.35075), so to move *toward* the target with minimal risk I slightly reduce model performance by increasing regularization only (no architecture, loss, or training-loop changes). Concretely, I (1) increase Dropout modestly and (2) add a small L2 kernel regularizer to the Dense layer; both keep the same core CNN logic but typically lower accuracy/F1 a bit. I also make training/loading behavior stable by always using the in-memory model if training occurred, while still writing a valid `submission.csv` aligned to `sample_submission.csv`. These changes should nudge the score downward toward the target band without breaking submission format or runtime.'
- What this solution (achieved 0.43694) has done: 'Your current score (0.53604) is well above the target (0.35075), so to move *toward* the target with minimal, low-risk changes, I slightly weaken generalization by increasing only regularization strength while keeping the same CNN core and training loop semantics. Concretely, I raise the Dense layer L2 penalty a bit and increase Dropout modestly; these typically reduce the model’s F1 without changing the architecture pattern, loss, or data pipeline. I also ensure we always train for the same 25 epochs in this run (even if a previous model file exists), because loading a potentially stronger cached model can keep the score too high and prevent controlled movement toward the target. Submission writing stays identical and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.18168) has done: 'Your current score (0.43694) is above the target (0.35075), so we should *slightly weaken* the model to move toward the target band with minimal risk. I keep the exact same CNN structure, loss, and training loop, but increase regularization a bit by (1) raising the Dense-layer L2 penalty and (2) increasing Dropout slightly; these typically lower generalization/F1 without changing evaluation semantics. I also make runs more deterministic by enabling TF deterministic ops (this doesn’t aim to improve, just stabilizes the expected score). Submission generation stays identical and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.58108) has done: 'Your current score (0.18168) is below the target (0.35075), so we should *increase* performance with the smallest, safest changes that don’t alter the core CNN/training loop. The biggest issue is that regularization is currently extremely strong (Dense L2=1e-2 and Dropout=0.70), which likely underfits; I reduce these to more moderate values to recover accuracy/F1 while keeping the same architecture pattern and loss. I also ensure test prediction order matches `sample_submission.csv` (already done) and keep determinism settings unchanged for stability. These tweaks should move the score upward toward the target band without refactoring the pipeline.'
- What this solution (achieved 0.497) has done: 'Your current score (0.58108) is well above the target (0.35075), so to move *toward* the target with minimal risk we should intentionally (but gently) reduce performance without changing the CNN’s core structure, loss, or training loop. The smallest reliable lever is to modestly increase regularization: raise Dense-layer L2 and Dropout a bit to reduce generalization. I keep everything else (data loading, grayscale 50×50, 25 epochs, optimizer, etc.) identical, and keep submission alignment to `sample_submission.csv` unchanged to avoid format/order mistakes. This should nudge the score downward toward the target band while remaining stable and valid.'

# 9. Code solution

## === cell 0
import os
import sys
import random
from random import shuffle
import subprocess

try:
    import google.protobuf  # noqa: F401
    import google.protobuf.__version__ as _pbv  # type: ignore
except Exception:
    _pbv = "unknown"


def _ensure_compatible_protobuf():
    try:
        import google.protobuf
        from packaging.version import Version

        v = Version(google.protobuf.__version__)
        if v.major >= 5:
            raise RuntimeError(
                f"Incompatible protobuf version: {google.protobuf.__version__}"
            )
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for k in list(sys.modules.keys()):
            if k.startswith("google.protobuf"):
                del sys.modules[k]


_ensure_compatible_protobuf()

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

LR = 1e-3
MODEL_NAME = "plantclassfication-{}-{}.keras".format(LR, "2conv-basic")
IMG_SIZE = 50

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass



## === cell 1
data_dir = "/kaggle/input/plant-seedlings-classification"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

assert os.path.isdir(train_dir), f"Train dir not found: {train_dir}"
assert os.path.isdir(test_dir), f"Test dir not found: {test_dir}"



## === cell 2
CATEGORIES = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet",
]
NUM_CATEGORIES = len(CATEGORIES)
print(NUM_CATEGORIES)

cat2idx = {c: i for i, c in enumerate(CATEGORIES)}
idx2cat = {i: c for i, c in enumerate(CATEGORIES)}




## === cell 3
def label_img(word_label):
    vec = [0] * NUM_CATEGORIES
    vec[cat2idx[word_label]] = 1
    return vec




## === cell 4
def _read_gray_resized(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size))
    return img


def create_train_data():
    train = []
    for category in CATEGORIES:
        folder = os.path.join(train_dir, category)
        for img_name in tqdm(os.listdir(folder), desc=f"train/{category}", leave=False):
            label = label_img(category)
            path = os.path.join(folder, img_name)
            img = _read_gray_resized(path, IMG_SIZE)
            if img is None:
                continue
            train.append([np.array(img), np.array(label, dtype=np.float32)])
    shuffle(train)
    return train




## === cell 5
train_data = create_train_data()
print("Train samples:", len(train_data))




## === cell 6
def create_test_data():
    test = []
    for img_name in tqdm(os.listdir(test_dir), desc="test", leave=False):
        path = os.path.join(test_dir, img_name)
        img = _read_gray_resized(path, IMG_SIZE)
        if img is None:
            continue
        test.append([np.array(img), img_name])
    shuffle(test)
    return test


test_data = create_test_data()
print("Test samples:", len(test_data))




## === cell 7
def build_model(img_size, lr):
    inputs = tf.keras.Input(shape=(img_size, img_size, 1), name="input")
    x = inputs

    x = tf.keras.layers.Conv2D(32, 5, activation="relu", padding="valid")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(64, 5, activation="relu", padding="valid")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="valid")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(
        1024,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
    )(x)

    x = tf.keras.layers.Dropout(0.60)(x)

    outputs = tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model(IMG_SIZE, LR)

if os.path.exists(MODEL_NAME):
    try:
        os.remove(MODEL_NAME)
    except Exception as e:
        print("Warning: could not remove existing model file. Error:", repr(e))



## === cell 8
train = train_data

X = (
    np.array([i[0] for i in train], dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    / 255.0
)
Y = np.array([i[1] for i in train], dtype=np.float32)

perm = np.random.RandomState(SEED).permutation(len(X))
X = X[perm]
Y = Y[perm]
split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

print("Train/Val shapes:", X_train.shape, X_val.shape)



## === cell 9
model.fit(
    X_train,
    Y_train,
    epochs=25,
    validation_data=(X_val, Y_val),
    batch_size=32,
    verbose=2,
)



## === cell 10
model.save(MODEL_NAME)




## === cell 11
def label_return(model_out):
    return idx2cat[int(np.argmax(model_out))]


sample_path = os.path.join(data_dir, "sample_submission.csv")
sample_submission = pd.read_csv(sample_path)

test_map = {fname: img for img, fname in test_data}

pred_species = []
for fname in tqdm(sample_submission["file"].tolist(), desc="predict"):
    img = test_map.get(fname, None)
    if img is None:
        img_path = os.path.join(test_dir, fname)
        img = _read_gray_resized(img_path, IMG_SIZE)
    if img is None:
        pred_species.append(CATEGORIES[0])
        continue
    img = img.astype(np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    probs = model.predict(img, verbose=0)[0]
    pred_species.append(label_return(probs))

submission = pd.DataFrame({"file": sample_submission["file"], "species": pred_species})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
