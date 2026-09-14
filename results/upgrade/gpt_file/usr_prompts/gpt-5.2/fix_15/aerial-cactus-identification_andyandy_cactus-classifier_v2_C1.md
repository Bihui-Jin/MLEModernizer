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

0.99897

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99981) has done: 'I fix the dataset path resolution so the script actually finds and loads the train/test JPGs (your current logic incorrectly prefers a non-existent nested `train/train` folder, resulting in zero images loaded). Then I remove the protobuf env override that is triggering the `MessageFactory.GetPrototype` error with `tf_keras` in this environment, so model training can run. Finally, I ensure the inference/submission cell can always access `test_array` and writes a valid `submission.csv` with `id,has_cactus` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.99556) has done: 'You’re hitting a protobuf/TensorFlow-Keras compatibility error; the clean fix is to force protobuf to use the pure-Python implementation *before any TF/Keras import* and to avoid importing `tf_keras` entirely by using `tensorflow.keras`, which is the stable API in Kaggle TF runtimes. I also remove the late environment mutation (which can’t help once TF is imported) and add a small safety check that `test_array` aligns exactly to `sample_submission.csv` ids, ensuring a valid `submission.csv` is always written. These changes are execution/stability fixes only and keep the same CNN architecture/training loop so the score should remain essentially unchanged (still well above the 0.5 target).'
- What this solution (achieved 0.97441) has done: 'I fix the protobuf/TensorFlow import crash by setting the protobuf implementation environment variables before importing TensorFlow/Keras, which addresses the `MessageFactory.GetPrototype` AttributeError in this Kaggle setup. I also ensure the notebook-style cell numbering starts at 1 (your current script starts at cell 0) and keep your CNN architecture/training loop unchanged. Finally, I keep the existing submission-writing logic but add a small alignment/assertion so the output always matches `sample_submission.csv` ids and produces a valid `submission.csv`.'
- What this solution (achieved 0.99462) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, and by pinning a safe `protobuf` runtime behavior via environment variables early. I also minimally adjust the imports to avoid any accidental `tf_keras` interactions while keeping the exact same CNN architecture, compile settings, and training loop. Finally, I keep your submission-writing logic but make sure the script always produces `submission.csv` with `id,has_cactus` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.98975) has done: 'I fix the protobuf/TensorFlow import crash by setting the required environment variables before any TensorFlow/Keras import and by forcing the pure-Python protobuf implementation early. I also correct the cell numbering to start at 1 (Kaggle/script compatibility) while preserving your exact CNN architecture, compile settings, and training loop. Finally, I keep your submission logic but add a strict alignment check to ensure predictions match `sample_submission.csv` order and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.99421) has done: 'I fix the protobuf/TensorFlow crash by setting the protobuf environment variables before importing TensorFlow (and avoiding later mutations), which resolves the `MessageFactory.GetPrototype` error in this Kaggle environment. I also renumber the cells to start at 1 so the script format is consistent, while keeping your CNN architecture, compile settings, and training loop unchanged. Finally, I keep your submission-writing logic but add a small safety check to ensure `test_array` and `df_sub` stay aligned and always produce a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.99637) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, and by importing `tf_keras` (the installed Keras backend in this environment) instead of `tensorflow.keras`, which avoids the `MessageFactory.GetPrototype` error. I also renumber cells to start at 1 (your format requirement) and keep the CNN architecture/training loop unchanged. Finally, I keep the submission-writing logic but add an explicit alignment check that the produced rows match `sample_submission.csv` exactly and always write `submission.csv` with the correct columns.'
- What this solution (achieved 0.99932) has done: 'I fix the protobuf/TensorFlow crash by switching from `tf_keras` to the Kaggle-stable `tensorflow.keras` API while keeping your CNN architecture, compile settings, and training loop identical. I also keep the early protobuf environment settings (they’re harmless and can help in some runtimes) but ensure imports occur in a safe order. Finally, I keep your submission-writing logic intact, only adding a small safety check to guarantee predictions are finite and the output `submission.csv` has the required `id,has_cactus` columns aligned to `sample_submission.csv`. Since your current score is far above the 0.5 target, these changes are aimed at correctness/stability, not improving score.'
- What this solution (achieved 0.99914) has done: 'I fix the protobuf/TensorFlow crash that prevents model training by forcing the pure-Python protobuf implementation *before* importing TensorFlow, and by ensuring we don’t accidentally load the incompatible `upb` backend in this Kaggle runtime. I also renumber the cells to start at 1 (your required format) while preserving your exact CNN architecture, compile settings, and training loop. Finally, I keep your submission logic the same but ensure it always writes a valid `submission.csv` with `id,has_cactus` aligned to `sample_submission.csv`. Since your current score is already far above the 0.5 target, these changes are correctness/stability-focused and should not intentionally improve score.'
- What this solution (achieved 0.99639) has done: 'I fix the `MessageFactory.GetPrototype` protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import and by switching the model code to use the installed `tf_keras` package (avoiding the incompatible TF/protobuf pathway in this environment). I also renumber your cells to start at 1 (your required format) while keeping your CNN architecture, compile settings, and training loop unchanged. Finally, I keep your submission-writing logic intact, only ensuring the objects it depends on exist and that `submission.csv` is always written with `id,has_cactus` aligned to `sample_submission.csv`. These changes are execution/stability focused and should not intentionally change your score.'
- What this solution (achieved 0.99997) has done: 'I fix the protobuf/TensorFlow crash that prevents training by avoiding the incompatible `tf_keras` import path and instead using `tensorflow.keras` (keeping your exact CNN architecture, compile settings, and training loop unchanged). I keep the early protobuf environment variables (they must be set before any TF import) and preserve your existing data loading and preprocessing. I also renumber cells to start at 1 to match your required format, and ensure the submission is always written as `submission.csv` with `id,has_cactus` aligned to `sample_submission.csv`. These changes are execution/stability-focused; since your current score is already far above the 0.5 target, I’m not making intentional score-changing adjustments.'
- What this solution (achieved 0.99897) has done: 'I fix the protobuf/TensorFlow crash causing `MessageFactory.GetPrototype` by ensuring the protobuf implementation environment variables are set before any TensorFlow import and by forcing the pure-Python protobuf backend early (this is an execution/stability fix, not a modeling change). I also renumber the cells to start at 1 (your required format) while keeping your data-loading, CNN architecture, compile settings, and training loop unchanged. Finally, I keep the submission-writing logic the same but make it robust to missing `test_array/df_sub` by ensuring they are defined from the first cell and always writing a valid `submission.csv` with `id,has_cactus` aligned to `sample_submission.csv`. Since your current score is already far above the 0.5 target, I won’t make any intentional score-changing adjustments.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

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
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
)
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

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

if test_array.shape[0] != df_sub.shape[0]:
    raise RuntimeError(
        f"test_array rows ({test_array.shape[0]}) do not match sample_submission rows ({df_sub.shape[0]})."
    )

predicts = model.predict(test_array, batch_size=256, verbose=0).reshape(-1)

predicts = np.nan_to_num(predicts, nan=0.5, posinf=1.0, neginf=0.0)
predicts = np.clip(predicts, 0.0, 1.0)

n = len(df_sub)
if len(predicts) != n:
    if len(predicts) > n:
        predicts = predicts[:n]
    else:
        pad_val = float(np.mean(predicts)) if len(predicts) else 0.5
        predicts = np.pad(predicts, (0, n - len(predicts)), constant_values=pad_val)

sub = df_sub.copy()
if "id" not in sub.columns or "has_cactus" not in sub.columns:
    raise ValueError(f"Unexpected submission columns: {sub.columns.tolist()}")

if not np.array_equal(sub["id"].astype(str).values, df_sub["id"].astype(str).values):
    raise RuntimeError("Submission ids got reordered unexpectedly.")

assert len(sub["id"].values) == len(
    predicts
), "Prediction length does not match submission length."
sub["has_cactus"] = predicts.astype(np.float32)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())
