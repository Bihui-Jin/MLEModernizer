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

0.8193

# 6. Current score

0.99786

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99911) has done: 'I fix the import-time crash by forcing Keras to use the TensorFlow backend (this resolves the protobuf `MessageFactory.GetPrototype` issue seen with mixed Keras installs). Then I make the dataset path resolution robust by automatically finding the real `train/` and `test/` directories inside the provided Kaggle folder (your current code sometimes points to a non-existent `train/train`). Finally, I keep the same CNN/training loop but speed up and stabilize prediction by batching test images and guaranteeing the submission aligns exactly with `sample_submission.csv` and is written as `submission.csv`.'
- What this solution (achieved 0.99806) has done: 'I fix the import-time crash by ensuring the TensorFlow/Keras stack is initialized in a compatible way (the `MessageFactory.GetPrototype` protobuf mismatch happens before your model runs). I keep your exact CNN, training loop, and prediction logic intact, only changing the import/bootstrap sequence to avoid the mixed `keras`/`tf_keras` protobuf issue. I also keep your robust path resolution and ensure the submission is always written as `submission.csv` with the required `id,has_cactus` columns. Since your current score (0.99911) is already far above the target (0.8193), I not make any score-improving changes—only stability fixes.'
- What this solution (achieved 0.99964) has done: 'The crash happens before training due to an incompatibility between the installed `protobuf` and TensorFlow’s generated protos (triggering `MessageFactory.GetPrototype`). I fix this by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which avoids that failing C++ path in this Kaggle environment. I keep your model, training loop, data loading, and submission logic unchanged so the score should remain essentially the same (still above target), while ensuring the notebook runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.99821) has done: 'I fix the import-time crash by ensuring the TensorFlow stack uses a compatible protobuf implementation before any TensorFlow-related imports occur, and by avoiding the mixed `tensorflow` + standalone `tf_keras` initialization that triggers the `MessageFactory.GetPrototype` error in this environment. I keep your CNN architecture, training loop, checkpoints, and prediction logic the same, only swapping the Keras import path to `tensorflow.keras` (same backend, same semantics) to prevent the protobuf/keras mismatch. I also keep your robust dataset path resolution and make sure a valid `submission.csv` with `id,has_cactus` is always written.'
- What this solution (achieved 0.99945) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by setting the protobuf implementation and Keras backend *before* importing TensorFlow/Keras, which is the common root cause in mixed Keras/TF Kaggle images. I keep your exact CNN architecture, training loop, and prediction logic unchanged to avoid unnecessary score shifts (your current score is already well above the target). I also make the environment initialization deterministic and stable by clearing any pre-imported TensorFlow modules (when rerunning) and keeping the dataset path logic intact. The script run end-to-end and always write a valid `submission.csv` with the required `id,has_cactus` columns.'
- What this solution (achieved 0.99831) has done: 'I fix the import-time crash by ensuring the protobuf/Keras backend environment variables are set before any TensorFlow-related import occurs, and by avoiding the mixed/unstable standalone `keras` stack (keeping `tensorflow.keras` only). This addresses the `MessageFactory.GetPrototype` AttributeError that currently prevents the script from running at all. I keep your CNN, training loop, checkpointing, and prediction logic unchanged so the score should remain essentially the same (and already above the target band), while ensuring a valid `submission.csv` is always written with the required `id,has_cactus` columns.'
- What this solution (achieved 0.9982) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by setting protobuf-related environment variables before any TensorFlow import and avoiding mixed Keras stacks, using only `tf_keras` (TensorFlow’s bundled Keras) consistently. I keep your CNN architecture, training loop, and preprocessing identical, only adjusting the import paths so the code runs end-to-end in this environment. Since your current score (0.99831) is already far above the target (0.8193), I not make any score-improving changes—only stability/compatibility fixes. The script still write a valid `submission.csv` with the required `id,has_cactus` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.99914) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before any TensorFlow/Keras import and by avoiding the mixed `tf_keras` stack (use `tensorflow.keras` consistently). I keep your CNN architecture, training loop, and preprocessing the same, only changing the bootstrap/import layer so the notebook runs end-to-end. I also keep your robust path discovery and ensure the submission is written as `submission.csv` with the required `id,has_cactus` columns aligned to `sample_submission.csv`. Since your current score is already far above the target, I not make any score-improving changes.'
- What this solution (achieved 0.99825) has done: 'I fix the import-time crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by avoiding the standalone `keras`/mixed-protobuf path and importing TensorFlow in a way that’s compatible with this Kaggle environment (using the bundled `tf_keras` API consistently). This is a stability-only change: the CNN architecture, training loop, preprocessing, and submission formatting remain the same, so your score should stay essentially unchanged (still above the target band). I also keep the existing robust path discovery and ensure the script always writes a valid `submission.csv` with the required `id,has_cactus` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.99899) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by avoiding the mixed `tensorflow` + `tf_keras` + standalone `keras` stack and importing Keras strictly from `tensorflow.keras`, which is the most stable configuration in this Kaggle environment. I keep your CNN architecture, training loop, preprocessing, checkpointing, and submission formatting the same, so the score behavior should remain essentially unchanged (and still above the target band). I also keep the robust path discovery and ensure the script always writes a valid `submission.csv` with `id,has_cactus` aligned to `sample_submission.csv`. No score-tuning changes are introduced—this is a correctness/stability patch only.'
- What this solution (achieved 0.99917) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime before any TensorFlow/Keras import and by avoiding mixed Keras stacks (use `tf_keras` consistently, which matches the installed packages). I keep your CNN architecture, training loop, preprocessing, and checkpointing identical so behavior/score stays essentially the same (and since your current score is already far above the target, I won’t make any score-improving changes). I also keep your robust dataset path discovery and ensure we always write a valid `submission.csv` with exactly the `id,has_cactus` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.99915) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by enforcing a consistent TensorFlow/Keras import path and protobuf implementation before importing TensorFlow, avoiding the mixed standalone `keras` vs `tf_keras` initialization that triggers this error in the Kaggle image. I keep your CNN architecture, training loop, and preprocessing identical, only changing the bootstrap/import layer to make the notebook run end-to-end reliably. Since your current score (0.99917) is already far above the target (0.8193), I won’t introduce any score-improving changes—only stability/correctness changes that should keep performance essentially the same. The code still write a valid `submission.csv` with the exact required `id,has_cactus` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.99786) has done: 'I fix the import-time protobuf crash (`MessageFactory.GetPrototype`) by setting the protobuf implementation environment variables as early as possible and importing TensorFlow in a way that avoids mixed Keras stacks. I keep your model, training loop, preprocessing, and checkpoint logic unchanged to avoid unnecessary score shifts (your current score is already far above the target, so no score-improving edits). I also make the input root resolution slightly more robust by preferring the fully qualified Kaggle path if present, without changing the downstream paths or submission format. The script run end-to-end and always write a valid `submission.csv` with `id,has_cactus`.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

for m in list(sys.modules.keys()):
    if m.startswith(("tensorflow", "keras", "tf_keras", "google.protobuf", "protobuf")):
        sys.modules.pop(m, None)

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(7)
tf.random.set_seed(7)

DATA_ROOT = "../input/aerial-cactus-identification"
if not os.path.exists(DATA_ROOT):
    alt = "/kaggle/input/aerial-cactus-identification"
    if os.path.exists(alt):
        DATA_ROOT = alt
    else:
        DATA_ROOT = "../input"

try:
    print("Listing ../input:", os.listdir("../input")[:20])
except Exception as e:
    print("Could not list ../input:", repr(e))
print("Using DATA_ROOT:", DATA_ROOT)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _find_dir_containing_jpg(root, preferred_name):
    """
    Find a directory under `root` that contains .jpg files.
    Prefer root/preferred_name; otherwise search common nested layouts.
    """
    cand = os.path.join(root, preferred_name)
    if os.path.isdir(cand):
        try:
            if any(f.lower().endswith(".jpg") for f in os.listdir(cand)):
                return cand
        except Exception:
            pass
        try:
            for sub in os.listdir(cand):
                subpath = os.path.join(cand, sub)
                if os.path.isdir(subpath) and any(
                    f.lower().endswith(".jpg") for f in os.listdir(subpath)
                ):
                    return subpath
        except Exception:
            pass

    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == preferred_name:
            if any(f.lower().endswith(".jpg") for f in filenames):
                return dirpath
        rel_depth = os.path.relpath(dirpath, root).count(os.sep)
        if rel_depth >= 4:
            dirnames[:] = []
    return None


train_csv_path = os.path.join(DATA_ROOT, "train.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

train_dir = _find_dir_containing_jpg(DATA_ROOT, "train")
test_dir = _find_dir_containing_jpg(DATA_ROOT, "test")

print("train_csv_path:", train_csv_path, "exists:", os.path.exists(train_csv_path))
print("sample_sub_path:", sample_sub_path, "exists:", os.path.exists(sample_sub_path))
print(
    "Resolved train_dir:",
    train_dir,
    "exists:",
    os.path.isdir(train_dir) if train_dir else False,
)
print(
    "Resolved test_dir :",
    test_dir,
    "exists:",
    os.path.isdir(test_dir) if test_dir else False,
)

if train_dir is None or test_dir is None:
    raise FileNotFoundError(
        "Could not locate train/test image folders with .jpg files under DATA_ROOT="
        f"{DATA_ROOT}. Please check the input directory structure."
    )



## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df.head()



## === cell 3
train_image = []
missing = 0
for idx in tqdm(range(len(train_df)), desc="Loading train images"):
    img_id = train_df.loc[idx, "id"]
    img_path = os.path.join(train_dir, img_id)
    if not os.path.exists(img_path):
        missing += 1
        continue
    img = image.load_img(img_path, target_size=(32, 32))
    img = image.img_to_array(img).astype("float32") / 255.0
    train_image.append(img)

if missing > 0:
    raise FileNotFoundError(
        f"Missing {missing} training images. Example expected path: {os.path.join(train_dir, train_df.loc[0,'id'])}"
    )

X = np.array(train_image, dtype="float32")
X.shape



## === cell 4
plt.imshow(X[1])
plt.axis("off")



## === cell 5
y = train_df["has_cactus"].values.astype("float32").reshape(-1, 1)
y.shape



## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)
X_train.shape, X_val.shape, y_train.shape, y_val.shape



## === cell 7
model = Sequential()
model.add(
    Conv2D(filters=512, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3))
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.summary()



## === cell 8
ckpt_path = "weights.best.weights.h5"
modelcheckpoint = ModelCheckpoint(
    filepath=ckpt_path,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    mode="max",
    verbose=1,
)



## === cell 9
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(
    X_train,
    y_train,
    epochs=80,
    validation_data=(X_val, y_val),
    batch_size=32,
    shuffle=True,
    callbacks=[modelcheckpoint],
    verbose=2,
)



## === cell 10
if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)
    print("Loaded best weights from:", ckpt_path)
else:
    print("Checkpoint not found; proceeding with last-epoch weights.")

y_hat = model.predict(X_val, batch_size=256, verbose=0).reshape(-1)
val_auc = roc_auc_score(y_val.reshape(-1), y_hat)
print("Validation ROC AUC:", val_auc)



## === cell 11
sub = pd.read_csv(sample_sub_path)

test_images = np.empty((len(sub), 32, 32, 3), dtype="float32")
missing_t = 0
for i, img_id in enumerate(tqdm(sub["id"].values, desc="Loading test images")):
    img_path = os.path.join(test_dir, img_id)
    if not os.path.exists(img_path):
        missing_t += 1
        continue
    img = image.load_img(img_path, target_size=(32, 32))
    arr = image.img_to_array(img).astype("float32") / 255.0
    test_images[i] = arr

if missing_t > 0:
    raise FileNotFoundError(
        f"Missing {missing_t} test images. Example expected path: {os.path.join(test_dir, sub.loc[0,'id'])}"
    )

test_preds = (
    model.predict(test_images, batch_size=256, verbose=0).reshape(-1).astype("float32")
)

submission = pd.DataFrame({"id": sub["id"].values, "has_cactus": test_preds})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))
