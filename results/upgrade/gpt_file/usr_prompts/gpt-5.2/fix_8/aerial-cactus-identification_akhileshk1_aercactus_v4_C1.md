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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

0.8878

# 6. Current score

0.99632

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99593) has done: 'I fix the environment/import crash by switching from the legacy `keras` package to `tf_keras`, which is compatible in this Kaggle runtime. Then I correct the dataset paths to the actual competition folder and fix image loading (`load_img` no longer accepts `grayscale`), ensuring `X` is non-empty so the train/valid split works. Finally, I replace the removed `predict_classes` with `predict` and output the required probability for `has_cactus` (class 1) into a properly named `submission.csv` with the correct columns and ID ordering.'
- What this solution (achieved 0.99686) has done: 'The crash happens before training because importing `tf_keras` triggers a protobuf incompatibility (`MessageFactory.GetPrototype`) in this Kaggle runtime. The smallest safe fix is to switch the Keras imports to `tensorflow.keras` (same API for `load_img/img_to_array/Sequential/layers`), which avoids that protobuf path while keeping the same model architecture and training loop. I also add a single safety line to enforce TensorFlow backend for Keras and keep seeds deterministic; everything else (data paths, preprocessing, model, epochs, and submission format) stays unchanged so the score should remain close to your current 0.99593 (and not intentionally pushed toward the lower target). The script run end-to-end and write `submission.csv` with `id,has_cactus`.'
- What this solution (achieved 0.99727) has done: 'I fix the crash that happens during `tensorflow.keras` import due to the protobuf `MessageFactory.GetPrototype` incompatibility by forcing the pure-Python protobuf implementation before TensorFlow is imported. This is a minimal environment-level change that keeps your exact model, training loop, preprocessing, and submission formatting intact (so it should stay close to your current score rather than intentionally moving toward the lower target). I also add TF seed setting for determinism without changing the learning setup. The rest of the code (paths, CNN, epochs, softmax output, and `id,has_cactus` submission) remains unchanged.'
- What this solution (achieved 0.99441) has done: 'The crash happens at `import tensorflow as tf` due to an incompatibility between TensorFlow and the protobuf runtime in this environment; setting the env var alone isn’t always enough because protobuf may already be imported before TensorFlow. I fix this by forcing the pure-Python protobuf implementation *and* purging any preloaded `google.protobuf*` modules before importing TensorFlow, which is a minimal environment-level change and keeps your model/training/prediction logic identical. I also add a small fallback to `tf.keras.utils.load_img/img_to_array` in case `tensorflow.keras.preprocessing.image` is unavailable/changed, without changing preprocessing semantics. No score-tuning changes are introduced (your current score is already above target), and the script still write a valid `submission.csv` with `id,has_cactus`.'
- What this solution (achieved 0.99727) has done: 'To fix the crash, I avoid importing TensorFlow (which is currently failing due to a protobuf incompatibility) and instead switch to the already-installed `tf_keras` package for model/training/prediction while keeping the exact same CNN architecture, loss, optimizer, epochs, and preprocessing. I keep your existing path-discovery and image loading semantics (32x32 RGB, scaled to [0,1]) but route `load_img/img_to_array` through `tf_keras.utils` to stay compatible. I also keep deterministic seeds and ensure the submission is written as `submission.csv` with the required `id,has_cactus` columns aligned to `sample_submission.csv`. This change should restore end-to-end execution; score may move slightly but the modeling logic remains the same.'
- What this solution (achieved 0.99361) has done: 'We fix the runtime crash caused by the protobuf/TensorFlow interaction that occurs when importing `tf_keras` by forcing the pure-Python protobuf implementation and purging any preloaded `google.protobuf*` modules before `tf_keras` is imported. This is an environment-level change only; it keeps the exact same CNN architecture, preprocessing, training loop, epochs, and submission formatting (so performance semantics remain the same). We also add a small, safe fallback for `load_img/img_to_array` in case `tf_keras.utils` changes, without altering image preprocessing (32x32 RGB scaled to [0,1]). The script then run end-to-end and write a valid `submission.csv` with `id,has_cactus`.'
- What this solution (achieved 0.99632) has done: 'I fix the import-time protobuf crash by avoiding `tf_keras` entirely and switching to `tensorflow.keras`, while forcing the pure-Python protobuf implementation and purging any preloaded `google.protobuf*` modules before TensorFlow is imported. This keeps your exact CNN architecture, preprocessing (32x32 RGB scaled to [0,1]), training loop, epochs, and submission formatting unchanged, so the score should remain in the same ballpark (still above your target; we won’t intentionally degrade it). I also keep the safe `load_img/img_to_array` fallback so image loading works across TF/Keras versions. Finally, I ensure the script always writes a valid `submission.csv` with `id,has_cactus` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

BASE_INPUT_CANDIDATES = [
    "../input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
    "../input",
    "/kaggle/input",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE = _first_existing(BASE_INPUT_CANDIDATES)
if BASE is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input directory for aerial-cactus-identification."
    )

print("Using BASE:", BASE)
print("Top-level listing:", os.listdir(BASE)[:20])



## === cell 1
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

try:
    from tensorflow.keras.utils import load_img as _load_img
    from tensorflow.keras.utils import img_to_array as _img_to_array
except Exception:
    from tensorflow.keras.preprocessing.image import load_img as _load_img
    from tensorflow.keras.preprocessing.image import img_to_array as _img_to_array

import matplotlib.pyplot as plt
import cv2  # kept to preserve original environment assumptions (even if unused)
import random

random.seed(42)
np.random.seed(42)
try:
    tf.random.set_seed(42)
except Exception:
    pass



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
if os.path.exists(os.path.join(BASE, "train")) and os.path.exists(
    os.path.join(BASE, "test")
):
    train_dir = os.path.join(BASE, "train") + "/"
    test_dir = os.path.join(BASE, "test") + "/"
    train_csv_path = os.path.join(BASE, "train.csv")
    sample_sub_path = os.path.join(BASE, "sample_submission.csv")
else:
    train_dir = "../input/train/train/"
    test_dir = "../input/test/test/"
    train_csv_path = "../input/train.csv"
    sample_sub_path = "../input/sample_submission.csv"

train_labels = pd.read_csv(train_csv_path)
print(train_labels.shape)
print(train_labels["has_cactus"].value_counts())
print(
    "Train dir exists:",
    os.path.exists(train_dir),
    "Example files:",
    os.listdir(train_dir)[:3],
)



## === cell 3
labels = []
image_feature = []
image_id = train_labels["id"].values

for img_id in image_id:
    img = _load_img(
        os.path.join(train_dir, img_id),
        target_size=(32, 32),
        color_mode="rgb",
    )
    img = _img_to_array(img).astype("float32") / 255.0
    image_feature.append(img)
    labels.append(
        int(train_labels.loc[train_labels["id"] == img_id, "has_cactus"].values[0])
    )

print("Loaded train images:", len(image_feature))



## === cell 4
print("tag length:", len(labels))
print("Total Images: ", len(image_feature))
list_of_tuples = list(zip(image_feature, labels))
df = pd.DataFrame(list_of_tuples, columns=["img", "label"])
print(df.shape)



## === cell 5
X = np.stack(image_feature, axis=0).astype("float32")
y = pd.get_dummies(df["label"]).values.astype("float32")

print("X shape:", X.shape, "y shape:", y.shape)



## === cell 6
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=df["label"].values
)
print("Train Images:", X_train.shape[0])
print("Test Images:", X_test.shape[0])



## === cell 7
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation="relu", input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(2, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

model.summary()



## === cell 8
history = model.fit(X_train, y_train, epochs=20, verbose=1)



## === cell 9
scores = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:" + "%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))



## === cell 10
acc_key = "acc" if "acc" in history.history else "accuracy"
plt.ylim(0.90, 1)
plt.xlim(1, 20)
plt.plot(history.history[acc_key])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()



## === cell 11
plt.ylim(0.01, 0.1)
plt.xlim(1, 20)
plt.plot(history.history["loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()



## === cell 12
submission = pd.read_csv(sample_sub_path)
test_ids = submission["id"].values

image_feature_test = []
for img_id in test_ids:
    img = _load_img(
        os.path.join(test_dir, img_id),
        target_size=(32, 32),
        color_mode="rgb",
    )
    img = _img_to_array(img).astype("float32") / 255.0
    image_feature_test.append(img)

print("Total Test Images: ", len(image_feature_test))



## === cell 13
Y = np.stack(image_feature_test, axis=0).astype("float32")

probs = model.predict(Y, verbose=0)
has_cactus_prob = probs[:, 1].astype("float32")

print("Pred prob range:", float(has_cactus_prob.min()), float(has_cactus_prob.max()))

sub_data = pd.DataFrame({"id": test_ids, "has_cactus": has_cactus_prob})
sub_path = "submission.csv"
sub_data.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub_data.shape)
print(sub_data.head())
