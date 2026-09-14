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

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

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

0.5

# 6. Current score

0.99899

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99889) has done: 'I fix the environment/import crash by switching from standalone `keras` (Keras 3) to `tf_keras` (tf.keras 2.18), which avoids the protobuf `MessageFactory.GetPrototype` error and restores `ImageDataGenerator`. Then I correct the Kaggle input paths to the actual dataset folder (`../input/aerial-cactus-identification/...`) so the generators can find images. Next I replace deprecated `fit_generator` with `fit`, and fix test image loading by resizing and converting BGR→RGB so `np.array(X_test)` is well-shaped and compatible with VGG16. Finally, I build the submission by merging predictions with `sample_submission.csv` ordering to guarantee correct `id,has_cactus` output and write `submission.csv`.'
- What this solution (achieved 0.99945) has done: 'The crash is happening before any training due to a protobuf/Keras compatibility issue that triggers `MessageFactory.GetPrototype` during TensorFlow/Keras import. The most minimal, score-neutral fix is to force the pure-Python protobuf implementation early (before importing `tf_keras`), which avoids that missing method and lets `ImageDataGenerator` work as intended. I keep the model/training/prediction logic unchanged, only adding a small import-guard plus a clear failure message if the environment still can’t import Keras. The pipeline then run end-to-end and write a valid `submission.csv` with the required `id,has_cactus` columns.'
- What this solution (achieved 0.96988) has done: 'I fix the protobuf import crash that happens before training by forcing the pure-Python protobuf implementation *and* ensuring it’s activated before any TensorFlow/tf_keras modules load (including clearing any pre-imported `google.protobuf` modules). This is score-neutral: it only unblocks execution and keeps your exact model/training/prediction logic unchanged. I also align the cell numbering to start at 1 (your format currently starts at 0), while preserving the original cell order/content. The script then run end-to-end and write a valid `submission.csv` with `id,has_cactus`.'
- What this solution (achieved 0.99951) has done: 'I fix the protobuf/Keras import crash by forcing the pure-Python protobuf implementation even earlier and more robustly, including setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` at process start and clearing any already-imported protobuf modules before importing `tf_keras`. This is an execution-unblocking change and is score-neutral: it does not alter the model, data pipeline, training loop, or prediction logic. I also keep the dataset path assertions and submission merge to guarantee a correctly ordered `id,has_cactus` CSV is always produced. Finally, I renumber cells to start from 1 (as required) while preserving the original cell order and logic.'
- What this solution (achieved 0.99899) has done: 'I fix the import-time protobuf crash by forcing a compatible protobuf runtime *before any* TensorFlow/tf_keras-related modules are imported, using both environment variables and a safe protobuf fallback import. This is execution-unblocking and score-neutral (it does not change the model, data pipeline, training loop, or prediction post-processing). I also keep your exact data paths and submission-building logic intact to ensure `submission.csv` is always written with the correct `id,has_cactus` columns and ordering. No changes be made that intentionally move the score away from your current ~0.9995 since the target (0.5) is already far below and the priority is correctness/stability.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm

try:
    import tf_keras as keras
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense, Flatten
    from tf_keras.preprocessing.image import ImageDataGenerator
    from tf_keras import applications
except Exception as e:
    raise RuntimeError(
        "Failed to import tf_keras. This notebook expects tf_keras==2.18.0 to be available. "
        "If you see a protobuf MessageFactory/GetPrototype error, forcing "
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (done at the top of this script) should resolve it."
    ) from e

np.random.seed(42)
keras.utils.set_random_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "../input/aerial-cactus-identification"
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

assert os.path.exists(TRAIN_CSV_PATH), f"Missing: {TRAIN_CSV_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"

train = pd.read_csv(TRAIN_CSV_PATH)
train.head()



## === cell 2
train["has_cactus"] = train["has_cactus"].map(lambda x: str(x))
train.shape



## === cell 3
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory=TRAIN_IMG_DIR,
    x_col="id",
    y_col="has_cactus",
    batch_size=32,
    shuffle=True,
    class_mode="binary",
    target_size=(32, 32),
    subset="training",
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory=TRAIN_IMG_DIR,
    x_col="id",
    y_col="has_cactus",
    batch_size=32,
    shuffle=False,
    class_mode="binary",
    target_size=(32, 32),
    subset="validation",
)



## === cell 4
base_model = applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3),
)

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()



## === cell 5
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



## === cell 6
steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
validation_steps = max(1, val_generator.samples // val_generator.batch_size)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=10,
    verbose=1,
)



## === cell 7
test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")])

X_test = np.zeros((len(test_files), 32, 32, 3), dtype=np.float32)
for i, fname in enumerate(tqdm(test_files, desc="Loading test images")):
    img_path = os.path.join(TEST_IMG_DIR, fname)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (32, 32):
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    X_test[i] = img.astype(np.float32) / 255.0



## === cell 8
testPredict = model.predict(X_test, batch_size=64, verbose=1).reshape(-1)
testPredict = np.clip(testPredict, 0.0, 1.0)



## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

pred_df = pd.DataFrame({"id": test_files, "has_cactus": testPredict})
submission = sample_sub[["id"]].merge(pred_df, on="id", how="left")

submission["has_cactus"] = submission["has_cactus"].fillna(0.5).astype(float)
submission.head()



## === cell 10
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(
    f"Wrote {out_path} with shape {submission.shape} and columns {list(submission.columns)}"
)
print(submission.describe(include="all"))
