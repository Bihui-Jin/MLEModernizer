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

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.4688

# 6. Current score

0.99775

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99624) has done: 'I fix the data path issue by using the already-extracted `/kaggle/input/aerial-cactus-identification/train` and `/kaggle/input/aerial-cactus-identification/test` folders directly, while keeping the zip-extraction cell as a safe fallback. I also resolve the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by pinning the pure-Python protobuf implementation via an environment variable before importing TensorFlow (a common TF2.18+ workaround in Kaggle). Finally, I keep your model and training loop intact but ensure the dataset pipelines receive valid string paths and that a correctly formatted `submission.csv` is always written.'
- What this solution (achieved 0.99615) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation environment variable before *any* TensorFlow-related import happens, which is the root cause of the `MessageFactory.GetPrototype` error. I also make the import order robust by moving the environment setting to the very top and ensuring later cells don’t re-import TensorFlow in a way that bypasses it. Since your current score (0.99624) is already far above the target (0.4688), I not change the model/training/prediction logic to avoid moving the score away from the target; the focus be correctness and stable end-to-end execution. The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.99794) has done: 'You’re hitting the TensorFlow/protobuf `MessageFactory.GetPrototype` crash because the protobuf implementation switch must be set before any TensorFlow-related import occurs, and your provided script starts at “cell 0” (which won’t run in the required cell format), so TF gets imported without the fix in the first executed cell. I move the environment variable setting to the very top of cell 1 (before any TF import), keep your data paths/zips fallback, and leave the model/training logic unchanged (so the score should remain essentially the same, i.e., still far above the target). I also fix the model bug where the second Conv2D incorrectly takes `input` instead of the previous tensor `x` (this is a correctness bug that can affect training stability), while keeping the same layers/approach. Finally, I ensure a valid `submission.csv` is always written with the correct `id,has_cactus` columns.'
- What this solution (achieved 0.99831) has done: 'I fix the runtime crash by ensuring the protobuf implementation environment variable is set before any TensorFlow import happens (your failing cell imports TF after cell 0, but in this required cell format cell 0 won’t run). I keep the same data pipeline and CNN architecture/training loop, only making the import order robust so the notebook runs end-to-end. Since your current score (0.99794) is far above the target (0.4688), I not make score-improving changes; the goal here is stable execution and a valid `submission.csv`. I also keep the existing safety checks to guarantee the submission has the correct columns and row alignment.'
- What this solution (achieved 0.99775) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by moving the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment setting to the very top of the first executed cell, before any TensorFlow import can occur (your current script’s fix is in “cell 0”, which won’t run under the required cell numbering). I keep your data pipeline, CNN architecture, and training loop unchanged to avoid moving the score further away from the target (your current score is already far above it). I also add a small safeguard to prevent re-import ordering issues by ensuring TensorFlow isn’t imported earlier than the env var, and keep submission-writing exactly as required (`submission.csv`, columns `id,has_cactus`). These changes are runtime/correctness focused and should be score-neutral.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd

PATH = "/kaggle/input/aerial-cactus-identification/"
labels = pd.read_csv(PATH + "train.csv")
submissions = pd.read_csv(PATH + "sample_submission.csv")
labels.head()



## === cell 2
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("font", size=15)
plt.figure(figsize=(7, 7))

label = ["Has catus", "Hasn't cactus"]
plt.pie(labels["has_cactus"].value_counts(), labels=label, autopct="%.1f%%")
plt.show()



## === cell 3
from zipfile import ZipFile as zf
import os

EXTRACT_DIR = "/kaggle/working/cactus_data"
os.makedirs(EXTRACT_DIR, exist_ok=True)


def _find_dir(root, target_name):
    for r, dirs, _ in os.walk(root):
        if target_name in dirs:
            return os.path.join(r, target_name)
    return None


TRAIN_DIR = os.path.join(PATH, "train")
TEST_DIR = os.path.join(PATH, "test")

if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR)):
    with zf(PATH + "train.zip") as zipper:
        zipper.extractall(EXTRACT_DIR)
    with zf(PATH + "test.zip") as zipper:
        zipper.extractall(EXTRACT_DIR)

    TRAIN_DIR = _find_dir(EXTRACT_DIR, "train")
    TEST_DIR = _find_dir(EXTRACT_DIR, "test")

if (
    TRAIN_DIR is None
    or TEST_DIR is None
    or (not os.path.isdir(TRAIN_DIR))
    or (not os.path.isdir(TEST_DIR))
):
    raise FileNotFoundError(
        f"Could not find train/ or test/ under PATH={PATH} or EXTRACT_DIR={EXTRACT_DIR}. "
        f"Found TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}"
    )

print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR :", TEST_DIR)



## === cell 4
import os

n_t = len(os.listdir(TRAIN_DIR))
n_test = len(os.listdir(TEST_DIR))
print(n_t, n_test, sep="\t")



## === cell 5
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

mpl.rc("font", size=7)
plt.figure(figsize=(15, 6))
cac_img_name = labels[labels["has_cactus"] == 1]["id"].tail(12)

for idx, img_name in enumerate(cac_img_name):
    img_path = os.path.join(TRAIN_DIR, str(img_name))
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(2, 6, idx + 1)
    ax.imshow(img)
    ax.axis("off")
plt.show()



## === cell 6
from sklearn.model_selection import train_test_split

train, val = train_test_split(
    labels, test_size=0.1, stratify=labels["has_cactus"], random_state=50
)
print(train.shape)



## === cell 7
import tensorflow as tf
import cv2
import numpy as np
import os


def create_img_data(img_name):
    img_name = img_name.numpy().decode("utf-8")
    img_path = os.path.join(TRAIN_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def _map_train(x, y):
    img = tf.py_function(create_img_data, [x], Tout=tf.float32)
    img.set_shape((32, 32, 3))
    y = tf.cast(y, tf.int16)
    return img, y


ds_train = tf.data.Dataset.from_tensor_slices(
    (train["id"].astype(str).values, train["has_cactus"].values)
)
ds_train = ds_train.map(_map_train, num_parallel_calls=tf.data.AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices(
    (val["id"].astype(str).values, val["has_cactus"].values)
)
ds_val = ds_val.map(_map_train, num_parallel_calls=tf.data.AUTOTUNE)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Input
from tensorflow.keras import metrics


def model_1(shape):
    input = Input(shape=shape)
    conv_activation = "relu"
    x = Conv2D(32, 3, padding="same", activation=conv_activation)(input)
    x = MaxPooling2D(2)(x)
    x = Conv2D(64, 3, padding="same", activation=conv_activation)(x)
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(input, x)
    return model


model = model_1((32, 32, 3))
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=[metrics.binary_accuracy]
)
print("done")



## === cell 9
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True)



## === cell 10
batch_size = 32
epochs = 10

ds_train_batched = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_val_batched = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

hist = model.fit(ds_train_batched, validation_data=ds_val_batched, epochs=epochs)



## === cell 11
import pandas as pd
import cv2
import numpy as np
import os


def create_img_data_for_test(img_name):
    img_name = img_name.numpy().decode("utf-8")
    img_path = os.path.join(TEST_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def _map_test(x):
    img = tf.py_function(create_img_data_for_test, [x], Tout=tf.float32)
    img.set_shape((32, 32, 3))
    return img


test_labels = pd.read_csv(PATH + "sample_submission.csv")
ds_test = tf.data.Dataset.from_tensor_slices(test_labels["id"].astype(str).values)
ds_test = (
    ds_test.map(_map_test, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

preds = model.predict(ds_test, verbose=1)
preds = preds.reshape(-1)  # ensure 1D aligned to submission rows
preds.shape



## === cell 12
submissions = pd.read_csv(PATH + "sample_submission.csv")
if len(preds) != len(submissions):
    raise ValueError(
        f"Pred length {len(preds)} does not match submission rows {len(submissions)}"
    )

submissions["has_cactus"] = preds.astype(np.float32)
submissions.to_csv("submission.csv", index=False)

print(submissions.head())
print("Wrote submission.csv with shape:", submissions.shape)
print("Submission path:", os.path.abspath("submission.csv"))
