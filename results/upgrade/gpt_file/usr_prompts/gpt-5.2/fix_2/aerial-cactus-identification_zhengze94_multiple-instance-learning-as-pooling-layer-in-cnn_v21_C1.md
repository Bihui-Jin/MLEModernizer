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

0.6591

# 6. Current score

0.98406

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.98406) has done: 'I fix the TensorFlow/Keras import crash by pinning protobuf behavior early (a common TF 2.18 + protobuf runtime issue) and by removing notebook-only magics so the script runs as a .py. Then I fix the image loading so train/test arrays have consistent shapes (some images may load as None or have unexpected channels), and I ensure test ordering matches `sample_submission.csv` ids. Finally, I repair the custom `noisyand` layer for modern TF shape objects (remove `.value`), compile/fit successfully, and write a valid `submission.csv` with the required `id,has_cactus` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

CANDIDATE_BASES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "/kaggle/input",
    "/kaggle/data",
    "../input/aerial-cactus-identification",
    "../input",
]
BASE_DIR = None
for b in CANDIDATE_BASES:
    if os.path.isdir(b):
        if os.path.exists(os.path.join(b, "train.csv")) and os.path.isdir(
            os.path.join(b, "train")
        ):
            BASE_DIR = b
            break
        nested = os.path.join(b, "aerial-cactus-identification")
        if os.path.exists(os.path.join(nested, "train.csv")) and os.path.isdir(
            os.path.join(nested, "train")
        ):
            BASE_DIR = nested
            break
if BASE_DIR is None:
    BASE_DIR = "../input"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

print("Using BASE_DIR:", BASE_DIR)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)
print(
    "Train CSV exists:",
    os.path.exists(TRAIN_CSV),
    "Sample submission exists:",
    os.path.exists(SAMPLE_SUB),
)



## === cell 1
import matplotlib.pyplot as plt
from tqdm import tqdm

import cv2
from PIL import Image

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

np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def load_img_cv2_rgb(path, target_size=(32, 32)):
    """Load image with cv2, ensure RGB uint8, size 32x32, 3 channels."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return None
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_imgs_dict(path):
    imgs = {}
    for f in os.listdir(path):
        fname = os.path.join(path, f)
        if not os.path.isfile(fname):
            continue
        img = load_img_cv2_rgb(fname)
        if img is not None:
            imgs[f] = img
    return imgs


img_train = load_imgs_dict(TRAIN_DIR)
img_test = load_imgs_dict(TEST_DIR)

print("Loaded train images:", len(img_train), "Loaded test images:", len(img_test))



## === cell 3
train_csv = pd.read_csv(TRAIN_CSV)

X_train = []
Y_train = []

missing_train = 0
for _, row in train_csv.iterrows():
    img_id = row["id"]
    im = img_train.get(img_id, None)
    if im is None:
        im = load_img_cv2_rgb(os.path.join(TRAIN_DIR, img_id))
    if im is None:
        missing_train += 1
        continue
    X_train.append(im.astype(np.float32) / 255.0)
    Y_train.append(int(row["has_cactus"]))

X_train = np.stack(X_train, axis=0)
Y_train = np.asarray(Y_train, dtype=np.int64)

submission_set = pd.read_csv(SAMPLE_SUB)
X_test = []
missing_test = 0
for img_id in submission_set["id"].values:
    im = img_test.get(img_id, None)
    if im is None:
        im = load_img_cv2_rgb(os.path.join(TEST_DIR, img_id))
    if im is None:
        missing_test += 1
        im = np.zeros((32, 32, 3), dtype=np.uint8)
    X_test.append(im.astype(np.float32) / 255.0)
X_test = np.stack(X_test, axis=0)

print("Training data shape:", X_train.shape, "=>", Y_train.shape)
print("Test data shape:", X_test.shape)
print("Missing train images:", missing_train, "Missing test images:", missing_test)



## === cell 4
plt.rcParams["axes.grid"] = False
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, idx in enumerate([0, 1, 2, 1000, 1050]):
    axes[i].imshow(X_train[idx])
    axes[i].set_title("Has cactus:" + str(Y_train[idx]))
    axes[i].axis("off")
plt.tight_layout()



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
sharp_xtrain = np.asarray(sharp_img_xtrain, dtype=np.float32)

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, idx in enumerate([0, 1, 2, 1000, 1050]):
    axes[i].imshow(np.clip(sharp_xtrain[idx], 0, 1))
    axes[i].set_title("Has cactus:" + str(Y_train[idx]))
    axes[i].axis("off")
plt.tight_layout()



## === cell 7
x_train, x_val, y_train, y_val = train_test_split(
    sharp_xtrain, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)
print("Train/val shapes:", x_train.shape, x_val.shape, y_train.shape, y_val.shape)




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
epoch = 20
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=epoch,
    verbose=1,
    validation_data=(x_val, y_val),
)



## === cell 12
acc_key = "accuracy" if "accuracy" in history.history else "acc"
val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

acc = history.history[acc_key]
epochs_ = range(0, epoch)
plt.plot(epochs_, acc, label="training accuracy")

acc_val = history.history[val_acc_key]
plt.scatter(list(epochs_), acc_val, label="validation accuracy")
plt.ylim([0.0, 1.0])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Plot of Model")
plt.legend()
plt.show()



## === cell 13
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_ = range(0, epoch)
plt.plot(epochs_, loss, label="training loss")
plt.scatter(list(epochs_), val_loss, label="validation loss")
plt.ylim([0, max(0.5, float(np.max(val_loss)) + 0.05)])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Plot of Model")
plt.legend()
plt.show()



## === cell 14
sharp_xtest = np.asarray([img_sharpen(im) for im in X_test], dtype=np.float32)
sharp_xtest = np.clip(sharp_xtest, 0.0, 1.0)

pred = model.predict(sharp_xtest, batch_size=256, verbose=1).reshape(-1)
submission_set["has_cactus"] = pred.astype(np.float32)

out_path = "submission.csv"
submission_set.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission_set.shape)
print(submission_set.head())



## === cell 15
from sklearn.metrics import roc_auc_score

val_pred = model.predict(x_val, batch_size=256, verbose=0).reshape(-1)
auc = roc_auc_score(y_val, val_pred)
print("Validation ROC-AUC:", auc)
