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
protobuf==6.33.0
seaborn==0.12.2
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

0.99321

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99206) has done: 'I fix the TensorFlow/protobuf import crash by avoiding `tensorflow`’s top-level import in the first cell and using `tensorflow.keras` imports instead. Then I update the optimizer call to the TF2 API so `compile()` works, and I fix the dataset paths to point at the provided `aerial-cactus-identification/train` and `.../test` folders. Finally, I make inference iterate only over actual image files (not nested directories), preserve a stable `id` order, and write a valid `submission.csv` with exactly the required `id,has_cactus` columns.'
- What this solution (achieved 0.98972) has done: 'I fix the protobuf/TensorFlow crash by setting the necessary environment variables before any TensorFlow import and by avoiding the `import tensorflow as tf` path that triggers the `MessageFactory.GetPrototype` issue in this environment. I keep the exact same model architecture, compile settings, training loop, and inference logic, only switching imports to `tensorflow.keras` so the code runs end-to-end. Since your current score (0.99206) is already far above the target (0.5), I not make any score-improving changes—just make execution stable and ensure a valid `submission.csv` is written with the required columns.'
- What this solution (achieved 0.99406) has done: 'You’re crashing on TensorFlow import due to an incompatibility between TensorFlow 2.18 and protobuf 6.x (`MessageFactory.GetPrototype`), and setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is no longer sufficient in this environment. I keep your exact model/training/inference logic, but avoid importing `tensorflow` entirely (you only used it for seeding) and use `tf_keras` (the installed compatibility package) for Keras layers/model/training and preprocessing. I keep determinism by setting NumPy + Python `random` seeds, and I ensure the submission is written as `submission.csv` with the required `id,has_cactus` columns and correct row alignment. This is a stability fix; it should keep score in the same ballpark (still far above the 0.5 target) without intentional performance changes.'
- What this solution (achieved 0.99379) has done: 'I fix the TensorFlow/protobuf crash causing the `MessageFactory.GetPrototype` error by forcing TensorFlow to use the pure-Python protobuf implementation *before* any Keras/TensorFlow-related import. I keep your exact model architecture, loss, optimizer, training loop, and inference semantics unchanged, only swapping the import path to `tensorflow.keras` (stable in this environment once protobuf is forced). I also keep the same data paths and ensure the submission is written as `submission.csv` with the required `id,has_cactus` columns and aligned row order. Since your current score (0.99406) is already far above the target (0.5), these changes are intended to be score-neutral and just restore end-to-end execution.'
- What this solution (achieved 0.99403) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by setting the additional protobuf env var *before* any TensorFlow/Keras import, which is the root cause of the runtime failure in cell 4. I keep your model, training loop, and inference semantics unchanged; the only code changes are to stabilize the import path and ensure Keras preprocessing is imported after the env vars are set. Since your current score (0.99379) is already far above the target (0.5), I won’t make any intentional score-changing adjustments—this is purely to run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.99355) has done: 'We fix the crash in the TensorFlow/Keras import caused by the protobuf 6.x incompatibility by avoiding TensorFlow’s Keras and using the installed `tf_keras` package instead (this keeps the same Keras API and model/training logic). We keep your architecture, optimizer, loss, epochs, batch size, and data loading semantics unchanged, only adjusting import paths so the notebook runs end-to-end. We also remove the unused `tensorflow.keras.preprocessing.image` dependency by doing test preprocessing with PIL (same 32×32 RGB scaling), keeping predictions aligned to `id`. This is intended to be score-neutral (you’re already far above the 0.5 target) and primarily restores stability and a valid `submission.csv`.'
- What this solution (achieved 0.99391) has done: 'I fix the protobuf/TensorFlow crash that currently stops execution in cell 4 by forcing the pure-Python protobuf implementation and importing TensorFlow *before* `tf_keras` so the backend is initialized consistently. This is a runtime-stability change only; it preserves your exact model architecture, compile settings, training loop, and PIL-based preprocessing. I also keep the existing dataset paths and submission-writing logic unchanged, ensuring the notebook runs end-to-end and writes a valid `submission.csv` with `id,has_cactus`. Since your current score is already far above the target, I won’t make any score-improving changes.'
- What this solution (achieved 0.99279) has done: 'I fix the runtime crash in the TensorFlow/protobuf stack by removing the `import tensorflow as tf` that triggers the `MessageFactory.GetPrototype` error in this Kaggle environment and rely solely on the already-used `tf_keras` API for model building/training. This preserves your exact model architecture, optimizer/loss, training loop, and PIL-based preprocessing, so it should be score-neutral (still far above the 0.5 target) while restoring end-to-end execution. I also keep the same paths and submission-writing logic, ensuring `submission.csv` is created with exactly `id,has_cactus` and aligned row order.'
- What this solution (achieved 0.99321) has done: 'I fix the protobuf/TensorFlow crash that occurs when importing `tf_keras` by forcing the pure-Python protobuf runtime and (crucially) importing `tensorflow` once after setting the env vars so the backend initializes correctly before `tf_keras` is loaded. I keep your model architecture, training loop, preprocessing, and submission logic the same, making only the minimal import/initialization changes needed for stability. I also add a tiny safety fallback to standard `tensorflow.keras` if `tf_keras` still fails to import in this environment, without changing semantics. This should restore end-to-end execution and still write a valid `submission.csv` with `id,has_cactus`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
from PIL import Image

from matplotlib import pyplot as plt
import seaborn as sns



## === cell 1
BASE_DIR = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

data = pd.read_csv(TRAIN_CSV)
data.head()



## === cell 2
sns.countplot(x="has_cactus", data=data)



## === cell 3
label = np.array(data["has_cactus"]).astype(np.float32)



## === cell 4
try:
    import tensorflow as tf  # backend init
    from tf_keras import layers, models, optimizers
except Exception as e:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

random.seed(42)
np.random.seed(42)

model = models.Sequential(
    [
        layers.Conv2D(2, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(4, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(8, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
model.compile(
    optimizer=optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 6
matrix = []
for fname in data["id"].tolist():
    img_path = os.path.join(TRAIN_DIR, fname)
    im = Image.open(img_path).convert("RGB")
    m = np.array(im, dtype=np.float32) / 255.0
    matrix.append(m)

x = np.stack(matrix, axis=0)
y = np.array(label, dtype=np.float32)

x.shape, y.shape



## === cell 7
model.fit(x, y, epochs=8, batch_size=32, verbose=1)



## === cell 8
test_ids = sorted(
    [
        f
        for f in os.listdir(TEST_DIR)
        if os.path.isfile(os.path.join(TEST_DIR, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

subm = []
for fname in test_ids:
    path = os.path.join(TEST_DIR, fname)
    im = Image.open(path).convert("RGB").resize((32, 32), resample=Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = float(model.predict(arr, batch_size=10, verbose=0)[0][0])
    subm.append(pred)

len(test_ids), len(subm), test_ids[:3], subm[:3]



## === cell 9
submission = pd.DataFrame({"id": test_ids, "has_cactus": subm})
submission.head()



## === cell 10
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Wrote {submission_path} with shape {submission.shape}")
print(submission_path)
