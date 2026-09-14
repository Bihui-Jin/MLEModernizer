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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.984

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the broken Keras imports by switching to `tf_keras` (available in your environment) so the CNN code runs without the protobuf-related crash, and I update the deprecated optimizer argument (`lr` → `learning_rate`). I also correct the image glob paths so the train/test arrays actually load (your current `.../train/train/*.jpg` path yields zero files, causing the split error). Finally, I keep the same CNN architecture and training loop, but output a proper probability for `has_cactus` (AUC metric expects probabilities, not hard class labels) and write a valid `cactus.csv` submission.'
- What this solution (achieved 0.5) has done: 'I fix the image directory detection so it correctly finds the actual `train/` and `test/` folders in this Kaggle environment and therefore loads non-empty image arrays (unblocking the whole pipeline). I also fix the protobuf-related crash by importing TensorFlow first and forcing the pure-Python protobuf implementation (a common Kaggle workaround with mixed `keras`/`tf_keras` stacks), while keeping your exact CNN architecture, optimizer, and training loop unchanged. Finally, I make the label/image alignment robust by sorting/merging by `id` so labels match the loaded filenames (this is score-critical and should move your AUC from ~0.5 toward the target), and ensure the submission uses probabilities and the correct `id,has_cactus` format.'
- What this solution (achieved 0.5) has done: 'I fix the dataset path detection so it finds the real `train/`, `test/`, `train.csv`, and `sample_submission.csv` locations in this Kaggle filesystem (your current `BASE` points one level too deep, so files aren’t found and `train_files` is never created). I also fix the protobuf crash by importing TensorFlow before `tf_keras` and setting the environment flags early, which stabilizes the mixed `keras`/`tf_keras` stack in Kaggle. After paths and imports are corrected, the rest of your pipeline (same CNN, same training loop, same epochs/batch size) run end-to-end, and it write a valid `cactus.csv` submission with probabilities (required for AUC). These changes are execution/score-critical but keep your core modeling logic unchanged.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf-related crash by making TensorFlow import happen first and by forcing the pure-Python protobuf implementation before any TF/Keras imports, which resolves the `'MessageFactory' object has no attribute 'GetPrototype'` error in mixed stacks. I keep your exact CNN architecture, optimizer, and training loop intact, only adjusting imports to consistently use `tf_keras` after TensorFlow is loaded. I also add a small safety fallback to use `tf.keras` if `tf_keras` still fails to import in this environment, without changing the model definition. This should unblock training/inference and, since predictions are already proper probabilities and label alignment is correct, move AUC up from ~0.5 toward your target.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/Keras crash that currently prevents TensorFlow from importing by setting the required environment flags before any TensorFlow-related import and forcing a stable, consistent Keras stack. Then I keep your exact CNN architecture/training loop, but ensure the script always uses `tf.keras` (instead of mixing `keras`/`tf_keras`) to avoid the `MessageFactory.GetPrototype` failure in this environment. Finally, I keep your ID/label alignment and probability submission logic intact and ensure a valid `cactus.csv` is written end-to-end.'
- What this solution (achieved 0.5) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables *before Python imports protobuf/tensorflow* and by forcing `tf.keras` to use the legacy Keras implementation that’s compatible in this Kaggle stack. This is an execution blocker; once resolved, your existing CNN architecture/training loop can run unchanged. We also make the protobuf env vars “hard” (overwrite, not setdefault) to ensure they take effect reliably. No score-tuning changes are made beyond enabling the model to actually train and produce probability predictions in the required submission format.'
- What this solution (achieved 0.5) has done: 'We fix the TensorFlow/Keras import crash (`MessageFactory.GetPrototype`) by forcing a safe protobuf implementation *before* any TensorFlow import and by avoiding the mixed `keras`/`tf_keras` stack (use `tf.keras` only, with `TF_USE_LEGACY_KERAS=1`). This is an execution blocker; once TF imports cleanly, your exact CNN architecture, compile settings, and training loop can run unchanged. We also move `to_categorical` import to after TF is successfully imported (so earlier cells don’t depend on a failed import), and keep the submission generation as probability outputs aligned to `sample_submission.csv`. These changes are score-critical only insofar as they allow the model to actually train and predict; no intentional score tuning beyond restoring the intended pipeline.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/Keras import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import and by using a single consistent Keras stack (`tf_keras`) instead of mixing `tensorflow.keras` with the standalone `keras==3`. This is an execution blocker; once TensorFlow loads cleanly, your existing CNN architecture, compile settings, and training loop stay the same. I also ensure `to_categorical`, layers, and optimizers are imported from the same stack so runtime is stable. No score-tuning changes are made beyond enabling the model to actually train and output probabilities for AUC.'
- What this solution (achieved 0.5) has done: 'We fix the TensorFlow/protobuf crash by forcing a consistent, stable Keras stack: import TensorFlow first, avoid importing the standalone `keras==3` / `tf_keras` package, and use only `tf.keras` throughout. This is an execution blocker (currently failing at the TF/Keras import), and once resolved your exact CNN architecture, compile settings, and training loop can run unchanged. We keep the existing robust dataset path detection and the label alignment by `id` (score-critical), and we continue to output probabilities (not hard labels) in a correctly formatted `cactus.csv` submission.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from PIL import Image

np.random.seed(42)



## === cell 1
train_data = []
test_data = []



## === cell 2
BASE_CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "../input/aerial-cactus-identification",
    "../data/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification/aerial-cactus-identification",
    "../input/aerial-cactus-identification/aerial-cactus-identification",
    "../data/aerial-cactus-identification/aerial-cactus-identification",
]


def _is_competition_root(d):
    if not d or not os.path.isdir(d):
        return False
    return (
        os.path.isfile(os.path.join(d, "train.csv"))
        and os.path.isfile(os.path.join(d, "sample_submission.csv"))
        and os.path.isdir(os.path.join(d, "train"))
        and os.path.isdir(os.path.join(d, "test"))
    )


def _find_competition_root(cands):
    for c in cands:
        if _is_competition_root(c):
            return c
        nested = os.path.join(c, "aerial-cactus-identification")
        if _is_competition_root(nested):
            return nested
    return None


BASE = _find_competition_root(BASE_CANDIDATES)
if BASE is None:
    raise FileNotFoundError(
        "Could not find dataset competition root containing train.csv, sample_submission.csv, train/, test/. "
        f"Checked: {BASE_CANDIDATES} and each + '/aerial-cactus-identification'"
    )

TRAIN_IMG_DIR = os.path.join(BASE, "train")
TEST_IMG_DIR = os.path.join(BASE, "test")

train_glob = os.path.join(TRAIN_IMG_DIR, "*.jpg")
test_glob = os.path.join(TEST_IMG_DIR, "*.jpg")

train_files = sorted(glob(train_glob))
test_files = sorted(glob(test_glob))

print("BASE:", BASE)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR, "| n_files:", len(train_files))
print("TEST_IMG_DIR :", TEST_IMG_DIR, "| n_files:", len(test_files))

if len(train_files) == 0 or len(test_files) == 0:
    raise RuntimeError(
        "Loaded 0 images via glob. "
        f"train_glob={train_glob} (n={len(train_files)}), test_glob={test_glob} (n={len(test_files)})"
    )




## === cell 3
def creat_train_data():
    train_data.clear()
    train_ids = []
    for file in tqdm(train_files, desc="Loading train images"):
        img = Image.open(file).convert("RGB")
        train_data.append(np.array(img, dtype=np.uint8))
        train_ids.append(os.path.basename(file))
    return train_ids


def creat_test_data():
    test_data.clear()
    test_ids = []
    for file in tqdm(test_files, desc="Loading test images"):
        img = Image.open(file).convert("RGB")
        test_data.append(np.array(img, dtype=np.uint8))
        test_ids.append(os.path.basename(file))
    return test_ids




## === cell 4
train_ids = creat_train_data()
test_ids = creat_test_data()



## === cell 5
train_data = np.array(train_data)
test_data = np.array(test_data)
print("train_data:", train_data.shape)
print("test_data :", test_data.shape)

if train_data.shape[0] == 0 or test_data.shape[0] == 0:
    raise RuntimeError(
        f"Loaded 0 images. Check paths. train_glob={train_glob}, test_glob={test_glob}"
    )



## === cell 6
train = train_data.astype("float32") / 255.0
test = test_data.astype("float32") / 255.0



## === cell 7
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

print("TF version:", tf.__version__)
print("TF_USE_LEGACY_KERAS:", os.environ.get("TF_USE_LEGACY_KERAS"))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
y = pd.read_csv(os.path.join(BASE, "train.csv"))
y = y[["id", "has_cactus"]].copy()

train_ids_df = pd.DataFrame({"id": train_ids})
y_aligned = train_ids_df.merge(y, on="id", how="left")

if y_aligned["has_cactus"].isna().any():
    missing = y_aligned.loc[y_aligned["has_cactus"].isna(), "id"].head(10).tolist()
    raise ValueError(
        f"Some train image ids are missing labels in train.csv, e.g.: {missing}"
    )

y_train = y_aligned["has_cactus"].astype(int).values
print("Label distribution:", pd.Series(y_train).value_counts().to_dict())
y_aligned.head()



## === cell 9
y_train = to_categorical(y_train, num_classes=2)



## === cell 10
if train.shape[0] != y_train.shape[0]:
    raise ValueError(
        f"Mismatch: train images={train.shape[0]} but labels={y_train.shape[0]}"
    )

x_train, x_val, y_train, y_val = train_test_split(
    train, y_train, test_size=0.2, random_state=2, stratify=y_train.argmax(axis=1)
)

print("x_train:", x_train.shape, "x_val:", x_val.shape)



## === cell 11
model = Sequential()

model.add(
    Conv2D(
        filters=64,
        kernel_size=(5, 5),
        padding="Same",
        activation="relu",
        input_shape=(32, 32, 3),
    )
)
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="Same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="Same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="Same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="Same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="Same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.summary()



## === cell 12
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 13
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=64,
    verbose=1,
)



## === cell 14
proba = model.predict(test, batch_size=256, verbose=1)
res = proba[:, 1].astype("float64")
res = np.clip(res, 0.0, 1.0)

print("Preds:", res.shape, "min/max:", float(res.min()), float(res.max()))



## === cell 15
d = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
submission = d.copy()

pred_map = pd.DataFrame({"id": test_ids, "has_cactus": res})
submission = submission.merge(pred_map, on="id", how="left", suffixes=("", "_pred"))

if submission["has_cactus"].isna().any():
    missing = submission.loc[submission["has_cactus"].isna(), "id"].head(10).tolist()
    raise ValueError(
        f"Some submission ids were not found in loaded test_ids, e.g.: {missing}"
    )

submission = submission[["id", "has_cactus"]]
submission.to_csv("cactus.csv", index=False)
print("Wrote submission to cactus.csv with shape:", submission.shape)
print(submission.head(3).to_string(index=False))
