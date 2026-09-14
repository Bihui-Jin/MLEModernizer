# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.8601

# 6. Current score

0.98356

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.98072) has done: 'I fix the TensorFlow/Keras import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory` runtime error in this environment. I also correct the dataset paths to the actual Kaggle folder layout and fix the test image stacking bug by loading test images in the exact order given by `sample_submission.csv` (and normalizing them consistently). The custom `noisyand` layer be made compatible with modern TF/Keras shape objects by removing the deprecated `.value` usage, allowing the model to build/compile/train. Finally, I ensure the submission is written as a valid `.csv` with columns `id,has_cactus` and that prediction is vectorized to run within the time limit.'
- What this solution (achieved 0.98356) has done: 'I fix the TensorFlow import crash by setting the protobuf implementation to `python` **and** forcing a compatible protobuf package version before importing TensorFlow (the current setting alone doesn’t prevent the `MessageFactory.GetPrototype` failure in this environment). I keep the model/training/prediction logic unchanged to avoid drifting the score further from your lower target, and focus strictly on making the notebook run end-to-end. I also add a small safety check to ensure the submission is aligned to `sample_submission.csv` order and is written as a valid `submission.csv` with the correct columns. All other code remains the same, including architecture, preprocessing, and training loop.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    if pb_ver is None or not pb_ver.startswith("3.20."):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.*"]
        )
        if "google.protobuf" in sys.modules:
            del sys.modules["google.protobuf"]


_ensure_protobuf_compat()

import numpy as np
import pandas as pd

CANDIDATE_ROOTS = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "../input/aerial-cactus-identification",
    "../input",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(r) and os.path.isdir(r):
        if os.path.exists(os.path.join(r, "train.csv")):
            DATA_ROOT = r
            break
        if os.path.exists(os.path.join(r, "aerial-cactus-identification", "train.csv")):
            DATA_ROOT = os.path.join(r, "aerial-cactus-identification")
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate aerial-cactus-identification dataset root in expected locations."
    )

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)
print("train.csv exists:", os.path.exists(TRAIN_CSV_PATH))
print("sample_submission.csv exists:", os.path.exists(SAMPLE_SUB_PATH))
print(
    "train dir exists:",
    os.path.exists(TRAIN_DIR),
    "test dir exists:",
    os.path.exists(TEST_DIR),
)



## === cell 1
import glob
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D,
    Activation,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from PIL import Image

print("TensorFlow:", tf.__version__)



## === cell 2
import cv2


def load_imgs(path):
    imgs = {}
    for f in sorted(os.listdir(path)):
        fname = os.path.join(path, f)
        img = cv2.imread(fname)  # BGR uint8, shape (32,32,3)
        if img is None:
            continue
        imgs[f] = img
    return imgs


img_train = load_imgs(TRAIN_DIR)
img_test = load_imgs(TEST_DIR)

print("Loaded train images:", len(img_train), "Loaded test images:", len(img_test))



## === cell 3
train_csv = pd.read_csv(TRAIN_CSV_PATH)

X_train = []
Y_train = []

for _, row in train_csv.iterrows():
    fn = row["id"]
    if fn not in img_train:
        continue
    X_train.append(img_train[fn].astype(np.float32) / 255.0)
    Y_train.append(int(row["has_cactus"]))

X_train = np.asarray(X_train, dtype=np.float32)
Y_train = np.asarray(Y_train, dtype=np.float32)

submission_set = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = submission_set["id"].tolist()

X_test_full = []
missing = 0
for fn in test_ids:
    img = img_test.get(fn)
    if img is None:
        missing += 1
        p = os.path.join(TEST_DIR, fn)
        img = cv2.imread(p)
    if img is None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
    X_test_full.append(img.astype(np.float32) / 255.0)

X_test_full = np.asarray(X_test_full, dtype=np.float32)

print("Training data shape:", X_train.shape, "=>", Y_train.shape)
print("Test data shape:", X_test_full.shape, "missing test images:", missing)



## === cell 4
plt.rcParams["axes.grid"] = False



## === cell 5
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(X_train[0])
axes[0].set_title("Has cactus:" + str(int(Y_train[0])))
axes[1].imshow(X_train[1])
axes[1].set_title("Has cactus:" + str(int(Y_train[1])))
axes[2].imshow(X_train[2])
axes[2].set_title("Has cactus:" + str(int(Y_train[2])))
axes[3].imshow(X_train[1000])
axes[3].set_title("Has cactus:" + str(int(Y_train[1000])))
axes[4].imshow(X_train[1050])
axes[4].set_title("Has cactus:" + str(int(Y_train[1050])))
plt.show()



## === cell 6
from scipy.ndimage import gaussian_filter


def img_sharpen(img):
    blurred_f = gaussian_filter(img, 2)
    filter_blurred_f = gaussian_filter(blurred_f, 2)
    alpha = 15
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return sharpened




## === cell 7
sharp_xtrain = np.asarray([img_sharpen(im) for im in X_train], dtype=np.float32)



## === cell 8
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(np.clip(sharp_xtrain[0], 0, 1))
axes[0].set_title("Has cactus:" + str(int(Y_train[0])))
axes[1].imshow(np.clip(sharp_xtrain[1], 0, 1))
axes[1].set_title("Has cactus:" + str(int(Y_train[1])))
axes[2].imshow(np.clip(sharp_xtrain[2], 0, 1))
axes[2].set_title("Has cactus:" + str(int(Y_train[2])))
axes[3].imshow(np.clip(sharp_xtrain[1000], 0, 1))
axes[3].set_title("Has cactus:" + str(int(Y_train[1000])))
axes[4].imshow(np.clip(sharp_xtrain[1050], 0, 1))
axes[4].set_title("Has cactus:" + str(int(Y_train[1050])))
plt.show()



## === cell 9
x_train, x_val, y_train, y_val = train_test_split(
    sharp_xtrain, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)

print("Train split:", x_train.shape, y_train.shape)
print("Val split:", x_val.shape, y_val.shape)




## === cell 10
class noisyand(tf.keras.layers.Layer):
    def __init__(self, num_classes, a=20, **kwargs):
        self.num_classes = num_classes
        self.a = max(1, a)
        super(noisyand, self).__init__(**kwargs)

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.b = self.add_weight(
            name="b", shape=(1, c), initializer="uniform", trainable=True
        )
        super(noisyand, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        num = tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)
        den = tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b)
        return num / den

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])




## === cell 11
def define_model(input_shape=(32, 32, 3), num_classes=1):
    model = Sequential()
    model.add(
        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape,
        )
    )

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (1, 1), activation="relu"))

    model.add(noisyand(num_classes + 1))
    model.add(Dense(num_classes, activation="sigmoid"))

    return model




## === cell 12
model = define_model()
model.summary()



## === cell 13
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)



## === cell 14
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    verbose=1,
    validation_data=(x_val, y_val),
)



## === cell 15
acc_key = "accuracy" if "accuracy" in history.history else "acc"
val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

acc = history.history[acc_key]
epochs_ = range(0, len(acc))
plt.plot(epochs_, acc, label="training accuracy")

acc_val = history.history[val_acc_key]
plt.scatter(epochs_, acc_val, label="validation accuracy")
plt.ylim([0.85, 1.0])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Plot of Model")
plt.legend()
plt.show()



## === cell 16
loss = history.history["loss"]
epochs_ = range(0, len(loss))
plt.plot(epochs_, loss, label="training loss")

loss_val = history.history["val_loss"]
plt.scatter(epochs_, loss_val, label="validation loss")
plt.ylim([0, 0.5])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Plot of Model")
plt.legend()
plt.show()



## === cell 17
submission_set = pd.read_csv(SAMPLE_SUB_PATH)
submission_set.head()



## === cell 18
sharp_xtest = np.asarray([img_sharpen(im) for im in X_test_full], dtype=np.float32)

preds = model.predict(sharp_xtest, batch_size=256, verbose=1).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)

if len(preds) != len(submission_set):
    raise ValueError(
        f"Pred length {len(preds)} != submission rows {len(submission_set)}"
    )

submission_set["has_cactus"] = preds
submission_path = "submission.csv"
submission_set.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission_set.shape)
submission_set.head()



## === cell 19
from sklearn.metrics import roc_curve, auc

y_pred_proba = model.predict(x_val, batch_size=256, verbose=0).reshape(-1)

fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(
    fpr, tpr, color="darkorange", lw=1.5, label="ROC curve (area = %0.4f)" % roc_auc
)
plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic (validation)")
plt.legend(loc="lower right")
plt.show()

print("Validation ROC-AUC:", roc_auc)
