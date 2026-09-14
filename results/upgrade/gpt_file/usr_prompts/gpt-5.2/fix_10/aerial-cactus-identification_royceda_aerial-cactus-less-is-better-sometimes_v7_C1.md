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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.4237

# 6. Current score

0.98627

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.9843) has done: 'I fix the TensorFlow import crash by removing the protobuf environment override that forces the pure-Python protobuf implementation, which is what triggers the `MessageFactory.GetPrototype` AttributeError in this environment. Then I keep the existing data pipeline/model/training logic intact, but ensure the submission has exactly the same number of rows as `sample_submission.csv` by using its `id` order directly and predicting over exactly those files (no truncation/merge mismatch). Finally, I harden the test prediction step to compute `steps` with `ceil(n/batch)` and assert the final length match before writing `submission.csv`, so a valid CSV is always produced end-to-end.'
- What this solution (achieved 0.98627) has done: 'I fix the TensorFlow crash by removing the protobuf environment forcing to the pure-Python implementation (that setting is what triggers the `MessageFactory.GetPrototype` AttributeError in this Kaggle image). Then I make the submission row-count deterministic by reading `sample_submission.csv` for the required `id` order and predicting exactly that many rows, while also ensuring `model.predict()` doesn’t add an extra partial batch (no manual `steps` so Keras uses the generator length). Finally, I add a strict assertion that the written CSV has exactly `len(sample_submission)+1` lines (header included) so the notebook cannot silently emit an invalid submission.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

base_input = "/kaggle/input/aerial-cactus-identification"
print("Exists:", os.path.exists(base_input))
print("Sample files:", sorted(os.listdir(base_input))[:10])



## === cell 1
import subprocess, shlex, os, pathlib


def run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


run(
    "cp -f /kaggle/input/aerial-cactus-identification/train.csv /kaggle/working/train.csv"
)
run("unzip -o /kaggle/input/aerial-cactus-identification/train.zip -d /kaggle/working")
run("unzip -o /kaggle/input/aerial-cactus-identification/test.zip -d /kaggle/working")

print("Working dir contents:", sorted(os.listdir("/kaggle/working"))[:50])
print("train dir exists:", os.path.isdir("/kaggle/working/train"))
print("test dir exists:", os.path.isdir("/kaggle/working/test"))


def find_jpg_dir(root, preferred_names=("train", "test")):
    root = os.path.abspath(root)
    for name in preferred_names:
        cand = os.path.join(root, name)
        if os.path.isdir(cand):
            for _, _, files in os.walk(cand):
                if any(f.lower().endswith(".jpg") for f in files):
                    return cand
    for dirpath, _, files in os.walk(root):
        if any(f.lower().endswith(".jpg") for f in files):
            return dirpath
    raise FileNotFoundError(f"No directory with .jpg images found under {root}")


TRAIN_DIR = find_jpg_dir("/kaggle/working", preferred_names=("train",))
TEST_DIR = find_jpg_dir("/kaggle/working", preferred_names=("test",))
print("Resolved TRAIN_DIR:", TRAIN_DIR)
print("Resolved TEST_DIR:", TEST_DIR)

print(
    "Train jpg sample:",
    sorted([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")])[:5],
)
print(
    "Test jpg sample:",
    sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])[:5],
)



## === cell 2
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import matplotlib.pyplot as plt
import tensorflow as tf

print("tf version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

tf.keras.utils.set_random_seed(42)

os.chdir("/kaggle/working")
print("CWD:", os.getcwd())



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
df = pd.read_csv("train.csv")
try:
    display(df.sample(3, random_state=42))
except Exception:
    print(df.sample(3, random_state=42))
df.has_cactus.value_counts().plot.bar()



## === cell 4
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

filename = df.id.iloc[10]
print("Example filename:", filename)
image = load_img(os.path.join(TRAIN_DIR, filename))
plt.imshow(image)
plt.axis("off")



## === cell 5
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_df, validate_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["has_cactus"]
)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255.0,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## === cell 6
BATCH_SIZE = 128
IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (32, 32, 3)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
    seed=42,
)

validation_generator = valid_datagen.flow_from_dataframe(
    dataframe=validate_df,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False,
)

print(
    "Train batches:", len(train_generator), "Valid batches:", len(validation_generator)
)



## === cell 7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential(
    [
        Conv2D(128, (2, 2), activation="relu", input_shape=INPUT_SHAPE),
        Conv2D(256, (2, 2), activation="relu"),
        Conv2D(64, (2, 2), activation="relu"),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),
    ]
)

earlystop = EarlyStopping(patience=5, restore_best_weights=True)
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
callbacks = [earlystop]

model.summary()



## === cell 8
import time

start = time.time()
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=2,
)
print("Train seconds:", round(time.time() - start, 2))



## === cell 9
pd.DataFrame(history.history).plot(figsize=(10, 4))
plt.grid(True)
plt.show()



## === cell 10
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)
test_df = sample_sub[["id"]].copy()

test_files = set([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
missing = [i for i in test_df["id"].tolist() if i not in test_files]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images under {TEST_DIR}. Example: {missing[0]}"
    )

test_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_gen.flow_from_dataframe(
    dataframe=test_df,
    directory=TEST_DIR,
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Test batches (keras len):", len(test_generator))

pred = model.predict(test_generator, verbose=0).reshape(-1).astype(np.float32)

n_test = len(test_df)
if len(pred) != n_test:
    pred = pred[:n_test]
if len(pred) != n_test:
    raise RuntimeError(f"Prediction length {len(pred)} != n_test {n_test}")

test_df["has_cactus"] = pred

try:
    display(test_df.sample(5, random_state=42))
except Exception:
    print(test_df.sample(5, random_state=42))
print(
    "Pred range:",
    float(test_df["has_cactus"].min()),
    float(test_df["has_cactus"].max()),
)



## === cell 11
assert list(test_df.columns) == [
    "id",
    "has_cactus",
], f"Bad columns: {list(test_df.columns)}"
assert test_df["id"].isna().sum() == 0
assert test_df["has_cactus"].isna().sum() == 0
assert len(test_df) == len(sample_sub), (len(test_df), len(sample_sub))
assert (
    test_df["id"].tolist() == sample_sub["id"].tolist()
), "ID order mismatch vs sample_submission.csv"

submission = test_df[["id", "has_cactus"]].copy()
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", os.path.abspath(submission_path))
print(submission.head())

with open(submission_path, "r") as f:
    n_lines = sum(1 for _ in f)
expected_lines = len(sample_sub) + 1
assert (
    n_lines == expected_lines
), f"CSV line count {n_lines} != expected {expected_lines}"



## === cell 12
print(submission["has_cactus"].describe())

with open("submission.csv", "r") as f:
    for _ in range(3):
        print(f.readline().strip())
print(
    "submission.csv rows (including header):",
    sum(1 for _ in open("submission.csv", "r")),
)
