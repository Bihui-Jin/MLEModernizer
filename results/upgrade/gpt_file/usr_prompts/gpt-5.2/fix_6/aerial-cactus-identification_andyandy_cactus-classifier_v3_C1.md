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

0.4904

# 6. Current score

0.99988

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.9999) has done: 'I fix two execution blockers: the `tf_keras` import currently triggers a protobuf `MessageFactory` error in this environment, so I switch to the standard `tensorflow.keras` API while keeping the exact same model architecture, loss, and training loop. I also fix the dataset path logic to reliably find the actual `train/` and `test/` image folders (your current `train/train` and `test/test` can be wrong depending on where Kaggle unzipped the data), which is why `test_array` ended up empty and `model.predict()` failed. Finally, I ensure the submission uses the sample submission’s `id` ordering and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.99991) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by avoiding the broken TensorFlow/protobuf stack in this environment and switching the model/training to the installed `keras==3.8.0` backend (same Sequential CNN layers, same loss, same training loop semantics). I also keep the robust path resolution and add explicit seeding for run-to-run stability without changing the model logic. Finally, I ensure predictions are clipped to valid probabilities and that the submission strictly follows the sample submission `id` order and writes `submission.csv`.'
- What this solution (achieved 0.99991) has done: 'I fix the runtime crash caused by the Keras/TensorFlow/protobuf mismatch by switching the model code to the installed `tf_keras` package (which is TensorFlow-Keras 2.18) while keeping the exact same CNN architecture, loss, optimizer, and training loop. I also keep your robust image-directory resolution and submission alignment logic unchanged, so the pipeline still runs end-to-end and writes a valid `submission.csv`. Since your current score (0.99991) is far above the target (0.4904), I not make any score-improving changes; the goal here is correctness and reproducibility.'
- What this solution (achieved 0.99988) has done: 'The crash comes from a protobuf/TensorFlow stack mismatch triggered by importing `tf_keras`, so the minimal fix is to switch to the already-installed standalone `keras==3.8.0` API while keeping the exact same Sequential CNN architecture, loss, optimizer, and training loop. To move the score down toward the low target AUC (since your current score is far above target), I only add a tiny, deterministic prediction-mixing step with 0.5 after inference (this preserves valid probabilities and submission format without changing training). I also keep your robust path resolution and sample-submission ID alignment so the pipeline runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

ppath = "../input/aerial-cactus-identification/"

print("Input root listing:", os.listdir("../input")[:20])
print("Using ppath:", ppath)
print("ppath listing:", os.listdir(ppath)[:20])

df_ = pd.read_csv(os.path.join(ppath, "train.csv"))
print(df_.head(5))



## === cell 1
from PIL import Image

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def _resolve_image_dir(root, split):
    """
    Fixes empty-array bug by robustly locating the real image directory.
    Some Kaggle copies include nested train/train or test/test, others don't.
    """
    cands = [
        os.path.join(root, split, split),
        os.path.join(root, split),
    ]
    for d in cands:
        if os.path.isdir(d):
            jpgs = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
            if len(jpgs) > 0:
                return d
    for d in cands:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(
        f"Could not find image directory for split='{split}' under {root}"
    )


train_dir = _resolve_image_dir(ppath, "train")
test_dir = _resolve_image_dir(ppath, "test")

print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)

train_files = sorted([f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")])
test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])

print("Train images:", len(train_files))
print("Test images :", len(test_files))

id_to_label = dict(zip(df_["id"].values, df_["has_cactus"].values))

train_array = np.zeros((len(train_files), 32, 32, 3), dtype=np.float32)
target_array = np.zeros((len(train_files),), dtype=np.float32)
test_array = np.zeros((len(test_files), 32, 32, 3), dtype=np.float32)

for i, filename in enumerate(train_files):
    fp = os.path.join(train_dir, filename)
    try:
        im = Image.open(fp).convert("RGB")
    except IOError:
        continue
    x32_im = im.resize((32, 32), RESAMPLE)
    train_array[i] = np.asarray(x32_im, dtype=np.float32)
    target_array[i] = float(id_to_label.get(filename, 0.0))

for i, filename in enumerate(test_files):
    fp = os.path.join(test_dir, filename)
    try:
        im = Image.open(fp).convert("RGB")
    except IOError:
        continue
    x32_im = im.resize((32, 32), RESAMPLE)
    test_array[i] = np.asarray(x32_im, dtype=np.float32)

train_array /= 255.0
test_array /= 255.0

print(
    "train_array shape:", train_array.shape, "target_array shape:", target_array.shape
)
print("test_array shape:", test_array.shape)

if len(test_files) == 0 or test_array.shape[0] == 0:
    raise RuntimeError(
        "Test set appears empty after directory resolution; cannot create submission."
    )



## === cell 2
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten

try:
    keras.utils.set_random_seed(0)
except Exception:
    pass

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
model.summary()

model.fit(train_array, target_array, epochs=20, batch_size=128, verbose=1)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
import pandas as pd
import numpy as np
import os

predicts = (
    model.predict(test_array, batch_size=256, verbose=1).reshape(-1).astype(float)
)

mix_alpha = 0.95  # 1.0 means unchanged; lower pushes predictions toward 0.5
predicts = mix_alpha * predicts + (1.0 - mix_alpha) * 0.5

predicts = np.clip(predicts, 0.0, 1.0)

sample_path = os.path.join(ppath, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

id_to_pred = dict(zip(test_files, predicts))

sub = sample_sub.copy()
sub["has_cactus"] = sub["id"].map(id_to_pred).astype(float)

if sub["has_cactus"].isna().any():
    sub["has_cactus"] = sub["has_cactus"].fillna(float(np.nanmean(predicts)))

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(sub.head())
print("Rows:", len(sub), "Expected:", len(sample_sub))
print("NaNs:", int(sub["has_cactus"].isna().sum()))
