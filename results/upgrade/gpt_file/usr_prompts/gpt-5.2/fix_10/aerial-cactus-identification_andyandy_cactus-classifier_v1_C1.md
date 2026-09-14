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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.99914

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99921) has done: 'I fix the PIL resize crash by replacing the removed `Image.ANTIALIAS` with the modern Pillow resampling enum, keeping the same resizing behavior. Then I fix the Keras runtime error in this environment by switching to `tf_keras` (same Keras API, compatible backend here) while preserving the exact model architecture, loss, and training loop. Finally, I make the data loading robust and aligned to `train.csv` / `sample_submission.csv` so we build arrays with the correct sizes and order, and write a valid `submission.csv` with the required columns and matching row count.'
- What this solution (achieved 0.99924) has done: 'I fix the `MessageFactory.GetPrototype` crash by avoiding the incompatible `tf_keras` import path in this Kaggle image and using `tensorflow.keras` instead (same Keras API and model/training loop). I also add deterministic seeds and a small safety check to ensure the submission rows align exactly to `sample_submission.csv` ids, without changing the model architecture or training procedure. Since your current score (0.99921) is already far above the target (0.5), I not make any score-improving changes—only runtime/stability fixes so it runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.99942) has done: 'I fix the TensorFlow/Keras crash (`MessageFactory` has no `GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`, which is a common compatibility issue in Kaggle images. I keep the exact same model architecture, compile settings, and training loop, only changing imports/environment to make it run reliably. I also keep the existing robust path/id alignment logic and ensure the submission is written as `submission.csv` with the required columns and row count. These changes are score-neutral (they only unblock execution).'
- What this solution (achieved 0.9992) has done: 'I fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf implementation *before* any TensorFlow-related imports occur, which is the root cause of the runtime error here. This is a runtime/stability-only change and does not alter your model architecture, loss, optimizer, training loop, or prediction logic (so it should be score-neutral). I also keep the existing robust path logic and ensure the submission is written as `submission.csv` with the required `id,has_cactus` columns and correct row alignment to `sample_submission.csv`. The only structural edit is inserting the env var setup at the very top of the script to guarantee it takes effect.'
- What this solution (achieved 0.99921) has done: 'You’re hitting a TensorFlow/protobuf compatibility crash at import time (`MessageFactory` missing `GetPrototype`). The safest minimal fix is to force the pure-Python protobuf runtime before TensorFlow loads, and to ensure this env var is set as early as possible and consistently. I keep the exact same model architecture, compile settings, and training loop (so score behavior stays essentially unchanged), and only adjust the import/initialization order to unblock execution. The submission-writing logic already matches the required `id,has_cactus` format, so it remain the same.'
- What this solution (achieved 0.99897) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory` missing `GetPrototype`) by ensuring the pure-Python protobuf environment variables are set before any TensorFlow-related import happens, and by importing TensorFlow only after that setup. I keep your exact CNN architecture, compile settings, and training loop unchanged so behavior stays essentially the same, only unblocking runtime. I also keep the existing robust path/id alignment and ensure a valid `submission.csv` is written with the required `id,has_cactus` columns and correct row count.'
- What this solution (achieved 0.99937) has done: 'We need to fix the TensorFlow import crash caused by a protobuf incompatibility (`MessageFactory` missing `GetPrototype`) so the notebook runs end-to-end and writes `submission.csv`. The safest minimal change is to force the pure-Python protobuf runtime **before any TensorFlow import** and to also disable the C++ implementation explicitly, then keep the same Keras model/training loop. I also keep the existing data/path alignment logic untouched and only adjust cell ordering/indexing to match the required `cell 1..N` format. These changes are runtime/stability-only and should be score-neutral (your score is already far above the target, so we avoid any score-changing edits).'
- What this solution (achieved 0.99927) has done: 'I fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by ensuring the protobuf runtime is forced to the pure-Python implementation *before* any TensorFlow import, and by avoiding any earlier imports that might indirectly load protobuf/TensorFlow. I keep your CNN architecture, compile settings, training loop, and preprocessing identical so behavior/score stays essentially unchanged (you’re already far above the target, so no score-improving edits). I also keep the robust path/id alignment and ensure `submission.csv` is always written with the required `id,has_cactus` columns and correct row count.'
- What this solution (achieved 0.99914) has done: 'I fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by preventing TensorFlow from loading the incompatible C++ protobuf runtime: we set the relevant protobuf env vars at the very top and also proactively remove any already-imported `google.protobuf*` modules before importing TensorFlow. This is a runtime/stability fix only and keeps your model architecture, compile settings, training loop, and preprocessing unchanged (so score behavior should remain essentially the same, and we won’t intentionally move it toward the much-lower target). I also keep the same submission alignment to `sample_submission.csv` and ensure `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPLUSPLUS"] = "1"

import sys
import random
import numpy as np
import pandas as pd
from PIL import Image

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)

ppath = "../input/"
if not os.path.exists(os.path.join(ppath, "train.csv")):
    ppath = "../input/aerial-cactus-identification/"

print("Using input path:", ppath)
print("Listing ../input:", os.listdir("../input")[:10])

df_ = pd.read_csv(os.path.join(ppath, "train.csv"))
df_["id"] = df_["id"].astype(str)
id_to_label = dict(zip(df_["id"].values, df_["has_cactus"].values))

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

train_img_dir = None
test_img_dir = None
for cand in [
    os.path.join(ppath, "train", "train"),
    os.path.join(ppath, "train"),
    os.path.join(ppath, "aerial-cactus-identification", "train", "train"),
    os.path.join(ppath, "aerial-cactus-identification", "train"),
]:
    if os.path.isdir(cand):
        train_img_dir = cand
        break

for cand in [
    os.path.join(ppath, "test", "test"),
    os.path.join(ppath, "test"),
    os.path.join(ppath, "aerial-cactus-identification", "test", "test"),
    os.path.join(ppath, "aerial-cactus-identification", "test"),
]:
    if os.path.isdir(cand):
        test_img_dir = cand
        break

if train_img_dir is None or test_img_dir is None:
    raise FileNotFoundError(f"Could not find train/test image dirs under {ppath}")

print("Train image dir:", train_img_dir)
print("Test image dir:", test_img_dir)

train_ids = df_["id"].tolist()
n_train = len(train_ids)
train_array = np.zeros((n_train, 32, 32, 3), dtype=np.uint8)
target_array = np.zeros((n_train,), dtype=np.float32)

for i, fname in enumerate(train_ids):
    fpath = os.path.join(train_img_dir, fname)
    with Image.open(fpath) as im:
        im = im.convert("RGB")
        x32_im = im.resize((32, 32), resample=_RESAMPLE)
    train_array[i] = np.asarray(x32_im, dtype=np.uint8)
    target_array[i] = float(id_to_label[fname])

df_s = pd.read_csv(os.path.join(ppath, "sample_submission.csv"))
df_s["id"] = df_s["id"].astype(str)
test_ids = df_s["id"].tolist()
n_test = len(test_ids)
test_array = np.zeros((n_test, 32, 32, 3), dtype=np.uint8)

for i, fname in enumerate(test_ids):
    fpath = os.path.join(test_img_dir, fname)
    with Image.open(fpath) as im:
        im = im.convert("RGB")
        x32_im = im.resize((32, 32), resample=_RESAMPLE)
    test_array[i] = np.asarray(x32_im, dtype=np.uint8)

train_array = train_array.astype("float32") / 255.0
test_array = test_array.astype("float32") / 255.0

print("train_array:", train_array.shape, train_array.dtype)
print(
    "target_array:",
    target_array.shape,
    target_array.dtype,
    "pos_rate=",
    float(target_array.mean()),
)
print("test_array:", test_array.shape, test_array.dtype)



## === cell 1
for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

tf.random.set_seed(0)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(600, activation="relu"))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.summary()

model.fit(train_array, target_array, epochs=20, batch_size=128, verbose=False)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import numpy as np

pred = model.predict(test_array, verbose=0).reshape(-1).astype(np.float64)

if len(pred) != len(df_s):
    raise RuntimeError(
        f"Prediction length {len(pred)} != sample_submission length {len(df_s)}"
    )

df_out = df_s.copy()
df_out["has_cactus"] = pred

out_path = "submission.csv"
df_out.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(df_out.head())
print(
    "Rows:",
    len(df_out),
    "Pred min/max:",
    float(df_out["has_cactus"].min()),
    float(df_out["has_cactus"].max()),
)
