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

0.8791

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd

INPUT_ROOT_CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input",
    "../input/aerial-cactus-identification",
    "../input",
    "/kaggle/data/aerial-cactus-identification",
    "/kaggle/data",
]
INPUT_ROOT = None
for c in INPUT_ROOT_CANDIDATES:
    if os.path.exists(c):
        INPUT_ROOT = c
        break

print("INPUT_ROOT:", INPUT_ROOT)
if INPUT_ROOT is None:
    raise FileNotFoundError("Could not find Kaggle input directory in known locations.")

print("Top-level listing:", os.listdir(INPUT_ROOT)[:20])



## === cell 1
import glob
import cv2
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Activation,
    BatchNormalization,
)
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from PIL import Image

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import os


def _find_file(*relative_options):
    for rel in relative_options:
        p = os.path.join(INPUT_ROOT, rel)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find any of: {relative_options} under {INPUT_ROOT}"
    )


TRAIN_CSV_PATH = _find_file("train.csv", "aerial-cactus-identification/train.csv")
SAMPLE_SUB_PATH = _find_file(
    "sample_submission.csv", "aerial-cactus-identification/sample_submission.csv"
)

TRAIN_DIR = None
TEST_DIR = None
for tr_rel in [
    "train/train",
    "train",
    "aerial-cactus-identification/train/train",
    "aerial-cactus-identification/train",
]:
    p = os.path.join(INPUT_ROOT, tr_rel)
    if os.path.isdir(p):
        TRAIN_DIR = p
        break

for te_rel in [
    "test/test",
    "test",
    "aerial-cactus-identification/test/test",
    "aerial-cactus-identification/test",
]:
    p = os.path.join(INPUT_ROOT, te_rel)
    if os.path.isdir(p):
        TEST_DIR = p
        break

print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)

if TRAIN_DIR is None or TEST_DIR is None:
    raise FileNotFoundError(
        "Could not find train/ or test/ image directories under INPUT_ROOT."
    )



## === cell 3
train_csv = pd.read_csv(TRAIN_CSV_PATH)


def load_imgs_array(path, file_list=None, target_size=(32, 32)):
    """
    Returns:
      - imgs: np.ndarray float32 (N,H,W,3) in BGR->RGB order and scaled to [0,1]
      - ids: list of filenames in the same order as imgs
    """
    if file_list is None:
        ids = sorted([f for f in os.listdir(path) if f.lower().endswith(".jpg")])
    else:
        ids = list(file_list)

    imgs = np.empty((len(ids), target_size[0], target_size[1], 3), dtype=np.float32)
    for i, f in enumerate(ids):
        fname = os.path.join(path, f)
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Failed to read image: {fname}")
        if im.shape[0] != target_size[0] or im.shape[1] != target_size[1]:
            im = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs[i] = im.astype(np.float32) / 255.0
    return imgs, ids


X_train = np.empty((len(train_csv), 32, 32, 3), dtype=np.float32)
Y_train = train_csv["has_cactus"].astype(np.int32).values
for i, img_id in enumerate(train_csv["id"].values):
    p = os.path.join(TRAIN_DIR, img_id)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Failed to read train image: {p}")
    if im.shape[:2] != (32, 32):
        im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    X_train[i] = im.astype(np.float32) / 255.0

print("Training data shape:", X_train.shape, "=>", Y_train.shape)

submission_set = pd.read_csv(SAMPLE_SUB_PATH)
X_test, test_ids = load_imgs_array(
    TEST_DIR, file_list=submission_set["id"].values, target_size=(32, 32)
)
print("Test data shape:", X_test.shape, "test_ids:", len(test_ids))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2744095611.py in <cell line: 0>()
     33     im = cv2.imread(p, cv2.IMREAD_COLOR)
     34     if im is None:
---> 35         raise ValueError(f"Failed to read train image: {p}")
     36     if im.shape[:2] != (32, 32):
     37         im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)

ValueError: Failed to read train image: /kaggle/input/aerial-cactus-identification/train/train/2de8f189f1dce439766637e75df0ee27.jpg

## === cell 4
import matplotlib.pyplot as plt

plt.rcParams["axes.grid"] = False

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, idx in enumerate([0, 1, 2, 1000, 1050]):
    axes[i].imshow(X_train[idx])
    axes[i].set_title("Has cactus:" + str(Y_train[idx]))
    axes[i].axis("off")
plt.show()



## === cell 5
from scipy.ndimage import gaussian_filter


def img_sharpen(img):
    blurred_f = gaussian_filter(img, 2)
    filter_blurred_f = gaussian_filter(blurred_f, 2)
    alpha = 15
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return sharpened




## === cell 6
sharp_img_xtrain = [img_sharpen(im) for im in X_train]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, idx in enumerate([0, 1, 2, 1000, 1050]):
    axes[i].imshow(np.clip(sharp_img_xtrain[idx], 0.0, 1.0))
    axes[i].set_title("Has cactus:" + str(Y_train[idx]))
    axes[i].axis("off")
plt.show()



## === cell 7
from numpy import array

x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)




## === cell 8
class noisyand(tf.keras.layers.Layer):
    def __init__(self, num_classes, a=20, **kwargs):
        self.num_classes = num_classes
        self.a = max(1, a)
        super(noisyand, self).__init__(**kwargs)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.b = self.add_weight(
            name="b",
            shape=(1, channels),
            initializer="uniform",
            trainable=True,
        )
        super(noisyand, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        num = tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)
        den = tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b)
        return num / den

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])




## === cell 9
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


model = define_model()
model.summary()



## === cell 10
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=RMSprop(),
    metrics=["accuracy"],
)



## === cell 11
epoch = 15
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=epoch,
    verbose=1,
    validation_data=(x_val, y_val),
)



## === cell 12
acc = history.history.get("accuracy", [])
epochs_ = range(0, epoch)
plt.plot(epochs_, acc, label="training accuracy")

acc_val = history.history.get("val_accuracy", [])
plt.scatter(list(epochs_), acc_val, label="validation accuracy")
plt.ylim([0.85, 1.0])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Plot of Model")
plt.legend()
plt.show()



## === cell 13
loss = history.history.get("loss", [])
epochs_ = range(0, epoch)
plt.plot(epochs_, loss, label="training loss")

loss_val = history.history.get("val_loss", [])
plt.scatter(list(epochs_), loss_val, label="validation loss")
plt.ylim([0, 0.5])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Plot of Model")
plt.legend()
plt.show()



## === cell 14
submission_set = pd.read_csv(SAMPLE_SUB_PATH)
submission_set.head()



## === cell 15
X_test_sharp = np.empty_like(X_test, dtype=np.float32)
for i in range(X_test.shape[0]):
    X_test_sharp[i] = np.clip(img_sharpen(X_test[i]), 0.0, 1.0)

pred = model.predict(X_test_sharp, batch_size=256, verbose=1).reshape(-1)
submission_set["has_cactus"] = pred.astype(np.float32)

OUT_PATH = "submission.csv"
submission_set.to_csv(OUT_PATH, index=False)
print("Wrote:", OUT_PATH, "shape:", submission_set.shape)
submission_set.head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2378623794.py in <cell line: 0>()
      1 # Vectorized prediction for speed & correctness; keep original sharpening in inference loop semantics.
      2 # Apply sharpening to test images (as in original code, but efficiently and safely clipped).
----> 3 X_test_sharp = np.empty_like(X_test, dtype=np.float32)
      4 for i in range(X_test.shape[0]):
      5     X_test_sharp[i] = np.clip(img_sharpen(X_test[i]), 0.0, 1.0)

NameError: name 'X_test' is not defined

## === cell 16
from sklearn.metrics import roc_curve, auc

y_val_proba = model.predict(x_val, batch_size=256, verbose=0).reshape(-1)
fpr, tpr, _ = roc_curve(y_val, y_val_proba)
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
plt.title("Validation ROC curve")
plt.legend(loc="lower right")
plt.show()

print("Validation AUC:", roc_auc)
