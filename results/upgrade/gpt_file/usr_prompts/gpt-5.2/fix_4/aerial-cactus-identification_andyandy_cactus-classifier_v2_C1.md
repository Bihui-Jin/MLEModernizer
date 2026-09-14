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

0.99981

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.99981) has done: 'I fix the dataset path resolution so the script actually finds and loads the train/test JPGs (your current logic incorrectly prefers a non-existent nested `train/train` folder, resulting in zero images loaded). Then I remove the protobuf env override that is triggering the `MessageFactory.GetPrototype` error with `tf_keras` in this environment, so model training can run. Finally, I ensure the inference/submission cell can always access `test_array` and writes a valid `submission.csv` with `id,has_cactus` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

INPUT_ROOT = "../input"

CANDIDATE_ROOTS = [
    os.path.join(INPUT_ROOT, "aerial-cactus-identification"),
    os.path.join(
        INPUT_ROOT, "aerial-cactus-identification", "aerial-cactus-identification"
    ),
    INPUT_ROOT,
]


def find_data_root(cands):
    for root in cands:
        if not (
            os.path.exists(os.path.join(root, "train.csv"))
            and os.path.exists(os.path.join(root, "sample_submission.csv"))
        ):
            continue
        train_dir = os.path.join(root, "train")
        test_dir = os.path.join(root, "test")
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            try:
                any_jpg = any(
                    fn.lower().endswith(".jpg") for fn in os.listdir(train_dir)
                )
            except Exception:
                any_jpg = False
            if any_jpg:
                return root
    return None


data_root = find_data_root(CANDIDATE_ROOTS)
if data_root is None:
    raise FileNotFoundError(
        "Could not locate a valid competition directory under expected ../input paths. "
        f"Tried: {CANDIDATE_ROOTS}"
    )

print("Using data_root:", data_root)

train_csv_path = os.path.join(data_root, "train.csv")
sample_sub_path = os.path.join(data_root, "sample_submission.csv")

df_train = pd.read_csv(train_csv_path)
df_sub = pd.read_csv(sample_sub_path)

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

train_dir = os.path.join(data_root, "train")
test_dir = os.path.join(data_root, "test")

train_imgs, train_y = [], []
missing_train = 0
for fname, y in zip(
    df_train["id"].astype(str).values,
    df_train["has_cactus"].astype(np.float32).values,
):
    fpath = os.path.join(train_dir, fname)
    if not os.path.exists(fpath):
        missing_train += 1
        continue
    im = Image.open(fpath).convert("RGB")
    x32 = im.resize((32, 32), RESAMPLE)
    train_imgs.append(np.asarray(x32, dtype=np.float32))
    train_y.append(y)

if len(train_imgs) == 0:
    raise RuntimeError(
        f"No training images were loaded. missing_train={missing_train}, train_dir={train_dir}"
    )

train_array = np.stack(train_imgs, axis=0) / 255.0
target_array = np.asarray(train_y, dtype=np.float32)

test_imgs = []
missing_test = 0
for fname in df_sub["id"].astype(str).values:
    fpath = os.path.join(test_dir, fname)
    if not os.path.exists(fpath):
        missing_test += 1
        test_imgs.append(np.zeros((32, 32, 3), dtype=np.float32))
        continue
    im = Image.open(fpath).convert("RGB")
    x32 = im.resize((32, 32), RESAMPLE)
    test_imgs.append(np.asarray(x32, dtype=np.float32))

test_array = np.stack(test_imgs, axis=0) / 255.0

print(
    "Loaded train:",
    train_array.shape,
    "targets:",
    target_array.shape,
    "missing_train:",
    missing_train,
)
print("Loaded test :", test_array.shape, "missing_test:", missing_test)
print(df_train.head())



## === cell 1
import os

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "") == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np

from tf_keras.models import Sequential
from tf_keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
)
from tf_keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(600, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
model.summary()

model.fit(train_array, target_array, epochs=20, batch_size=128, verbose=True)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import pandas as pd
import numpy as np

predicts = model.predict(test_array, batch_size=256, verbose=0).reshape(-1)

n = len(df_sub)
if len(predicts) != n:
    if len(predicts) > n:
        predicts = predicts[:n]
    else:
        pad_val = float(np.mean(predicts)) if len(predicts) else 0.5
        predicts = np.pad(predicts, (0, n - len(predicts)), constant_values=pad_val)

sub = df_sub.copy()
sub["has_cactus"] = predicts.astype(np.float32)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())
