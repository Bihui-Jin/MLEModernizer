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

0.9057

# 6. Current score

0.99771

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the environment/runtime issues caused by mixing `keras` (Keras 3) APIs with TensorFlow/Keras generators by switching to `tf.keras` equivalents that still provide `ImageDataGenerator`. I also remove the hard GPU requirement (so it runs whether GPU is available or not), correct the unzip/copy commands, and make paths robust so `train/` and `test/` are found. Finally, I ensure the submission is created from `sample_submission.csv` so it has the correct `id` column/order and writes a valid `submission.csv` with `id,has_cactus`.'
- What this solution (achieved 0.5) has done: 'I fix the dataset path detection so it correctly finds `train.csv`, `train.zip`, `test.zip`, and `sample_submission.csv` in your provided filesystem (the current candidate path points one level too deep). I also address the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves this common Kaggle runtime issue. Then I keep your exact model/training logic intact, but ensure all dependent variables are defined by making the data-prep cell succeed so later cells don’t cascade into `NameError`. Finally, I guarantee a valid `/kaggle/working/submission.csv` in the required `id,has_cactus` format and aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.997) has done: 'I fix the dataset extraction/path resolution so `TRAIN_DIR` and `TEST_DIR` are correctly found even when the zip extracts into a nested folder, which currently stops the whole pipeline and causes cascading `NameError`s. I also fix the TensorFlow/protobuf crash by setting the protobuf environment variables before importing TensorFlow and forcing a clean import, which resolves the `MessageFactory.GetPrototype` error in Kaggle runtimes. These changes are runtime/stability fixes and keep your model, generators, and training loop intact; once the training actually runs, your score should move well above 0.5 and toward the target band. Finally, I keep submission creation aligned to `sample_submission.csv` order and ensure a valid `/kaggle/working/submission.csv` is always written.'
- What this solution (achieved 0.99771) has done: 'I fix the TensorFlow import crash caused by the protobuf `MessageFactory.GetPrototype` issue by setting the right environment variables *before* any TensorFlow-related import and by forcing a clean re-import. This is a runtime/stability-only change and does not alter your model, generators, training loop, or submission logic, so it should keep your score essentially unchanged (you’re already above the target band, so we avoid score-changing edits). I also make the TensorFlow import cell robust in notebook-style execution by deleting any partially imported `tensorflow` modules before importing. The rest of the pipeline remains intact and write `/kaggle/working/submission.csv` with the required `id,has_cactus` columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import shutil
import zipfile

WORKDIR = "/kaggle/working"

CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/input",
]


def _is_dataset_root(p: str) -> bool:
    return (
        os.path.isdir(p)
        and os.path.exists(os.path.join(p, "train.csv"))
        and os.path.exists(os.path.join(p, "train.zip"))
        and os.path.exists(os.path.join(p, "test.zip"))
        and os.path.exists(os.path.join(p, "sample_submission.csv"))
    )


INPUT_DIR = next((p for p in CANDIDATES if _is_dataset_root(p)), None)
if INPUT_DIR is None:
    found = None
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if _is_dataset_root(dirpath):
            found = dirpath
            break
    INPUT_DIR = found

if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not find dataset root with required files under /kaggle/input. "
        f"Tried candidates: {CANDIDATES}"
    )

train_csv_src = os.path.join(INPUT_DIR, "train.csv")
train_zip_src = os.path.join(INPUT_DIR, "train.zip")
test_zip_src = os.path.join(INPUT_DIR, "test.zip")
sample_sub_src = os.path.join(INPUT_DIR, "sample_submission.csv")

for p in [train_csv_src, train_zip_src, test_zip_src, sample_sub_src]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")

print("Using INPUT_DIR:", INPUT_DIR)

os.makedirs(WORKDIR, exist_ok=True)

EXTRACT_DIR = os.path.join(WORKDIR, "_cactus_extracted")
if os.path.isdir(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

shutil.copy(train_csv_src, os.path.join(WORKDIR, "train.csv"))

with zipfile.ZipFile(train_zip_src, "r") as z:
    z.extractall(EXTRACT_DIR)
with zipfile.ZipFile(test_zip_src, "r") as z:
    z.extractall(EXTRACT_DIR)

print("Top-level EXTRACT_DIR contents:", sorted(os.listdir(EXTRACT_DIR))[:30])


def _best_dir_with_jpgs_under(root: str) -> str:
    if not os.path.exists(root):
        raise FileNotFoundError(f"Root not found: {root}")
    best = None
    best_count = -1
    for dirpath, _, filenames in os.walk(root):
        cnt = sum(1 for f in filenames if f.lower().endswith(".jpg"))
        if cnt > best_count:
            best_count = cnt
            best = dirpath
    if best is None or best_count <= 0:
        raise FileNotFoundError(f"Could not find any .jpg files under {root}")
    return best


def _find_split_dir(extract_root: str, split_name: str) -> str:
    split_name = split_name.lower()
    candidates = []
    for dirpath, dirnames, filenames in os.walk(extract_root):
        base = os.path.basename(dirpath).lower()
        if base == split_name:
            cnt = sum(1 for f in filenames if f.lower().endswith(".jpg"))
            candidates.append((cnt, dirpath))
    if candidates:
        candidates.sort(reverse=True)
        best_cnt, best_dir = candidates[0]
        if best_cnt > 0:
            return best_dir

    return _best_dir_with_jpgs_under(extract_root)


TRAIN_DIR = _find_split_dir(EXTRACT_DIR, "train")
TEST_DIR = _find_split_dir(EXTRACT_DIR, "test")

print(
    "Resolved TRAIN_DIR:",
    TRAIN_DIR,
    "num_jpg:",
    len([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")]),
    "sample:",
    sorted([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")])[:3],
)
print(
    "Resolved TEST_DIR :",
    TEST_DIR,
    "num_jpg:",
    len([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]),
    "sample:",
    sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])[:3],
)



## === cell 2
import os
import sys
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

for m in list(sys.modules.keys()):
    if m == "tensorflow" or m.startswith("tensorflow."):
        del sys.modules[m]

import matplotlib.pyplot as plt
import tensorflow as tf

print("tf version :", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

tf.random.set_seed(42)
np.random.seed(42)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
df = pd.read_csv("/kaggle/working/train.csv")
print(df.sample(3, random_state=42))
df.has_cactus.value_counts().plot.bar()
plt.grid(True)
plt.show()



## === cell 4
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

filename = df.id.iloc[10]
print(filename)
image = load_img(os.path.join(TRAIN_DIR, filename))
plt.imshow(image)
plt.axis("off")
plt.show()



## === cell 5
from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["has_cactus"]
)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)



## === cell 6
train_datagen = ImageDataGenerator(
    rotation_range=45,
    rescale=1.0 / 255,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)



## === cell 7
BATCH_SIZE = 64
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

print("train steps:", len(train_generator), "valid steps:", len(validation_generator))



## === cell 8
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    AveragePooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential(
    [
        Conv2D(
            filters=64,
            kernel_size=(4, 4),
            strides=(1, 1),
            activation="relu",
            input_shape=INPUT_SHAPE,
            padding="same",
        ),
        BatchNormalization(),
        AveragePooling2D(pool_size=(3, 3)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dropout(0.45),
        Dense(1, activation="sigmoid"),
    ]
)

earlystop = EarlyStopping(patience=4, restore_best_weights=True)
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.summary()



## === cell 9
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[earlystop],
    verbose=2,
)



## === cell 10
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()



## === cell 11
sample_sub_path = sample_sub_src
sub = pd.read_csv(sample_sub_path)
print(sub.head(), "rows:", len(sub))

test_files = set([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
missing = [i for i in sub["id"].tolist() if i not in test_files]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} ids from sample_submission.csv were not found in TEST_DIR={TEST_DIR}. "
        f"Example missing: {missing[:5]}"
    )



## === cell 12
test_gen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_gen.flow_from_dataframe(
    sub,
    directory=TEST_DIR,
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("test steps:", len(test_generator))

pred = model.predict(test_generator, steps=len(test_generator), verbose=0).reshape(-1)
pred = pred[: len(sub)]

if len(pred) != len(sub):
    raise RuntimeError(f"Prediction length {len(pred)} != submission length {len(sub)}")

sub["has_cactus"] = pred.astype(np.float32)
print(sub.head(), "rows:", len(sub))



## === cell 13
assert list(sub.columns) == ["id", "has_cactus"]
assert sub["id"].isna().sum() == 0
assert sub["has_cactus"].isna().sum() == 0
assert len(sub) == pd.read_csv(sample_sub_path).shape[0]

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub))



## === cell 14
print(sub["has_cactus"].describe())



## === cell 15
pass
