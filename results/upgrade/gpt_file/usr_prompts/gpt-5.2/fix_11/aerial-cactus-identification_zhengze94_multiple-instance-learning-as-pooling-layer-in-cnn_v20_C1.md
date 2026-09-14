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

0.7868

# 6. Current score

0.98874

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.98949) has done: 'I (1) fix the TensorFlow/Protobuf import crash by pinning `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow, (2) correct all dataset paths to the actual Kaggle folder (`../input/aerial-cactus-identification/...`) and make image loading consistent (RGB + float32/255) so arrays stack without shape errors, and (3) update the custom `noisyand` layer for TF2/Keras shape handling (`input_shape[-1]` instead of `.value`). I also ensure the train/val split uses the sharpened images (as your code intended) and update the history keys and ROC plotting to use TF2-compatible `model.predict()` outputs. Finally, I write a valid submission named `submission.csv` with columns `id,has_cactus` in the exact sample order.'
- What this solution (achieved 0.98794) has done: 'I fix the TensorFlow import crash caused by the current `protobuf` version by forcing TF to use the pure-Python protobuf implementation before any TF import, and I also add a safe fallback that switches to the pure-Python parser even if another library imported `google.protobuf` first. Because your current score (0.98949) is already far above the target (0.7868), I won’t change the model/training/prediction logic to avoid moving the score further away; the changes are intended to be score-neutral and purely stability-related. I keep all paths, preprocessing, architecture, and training loop intact, and ensure `submission.csv` is always written with the correct columns and row order. Finally, I add small guards for missing images to prevent runtime failures during test-time prediction.'
- What this solution (achieved 0.9829) has done: 'I fix the TensorFlow/Protobuf crash causing `MessageFactory.GetPrototype` to be missing by forcing the pure-Python protobuf runtime *and* using the TensorFlow-supported `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3`, applied before any protobuf/TF import. I keep the model, preprocessing (including sharpening), training loop, and prediction logic unchanged to avoid shifting your score further away from the target (your current score is already much higher than the target). I also add a small, safe fallback to set the environment variables even if something imported `google.protobuf` early, then proceed to import TensorFlow cleanly. The script still write a valid `submission.csv` with `id,has_cactus` in sample order.'
- What this solution (achieved 0.98741) has done: 'I fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf implementation *before* any `google.protobuf`/TensorFlow import, and by adding a safe runtime fallback that patches the symbol if protobuf was imported too early. This is a stability-only change intended to keep model/training/prediction logic identical so your score stays close to the current 0.9829 (already above the target 0.7868). I also keep paths unchanged and ensure the script always reaches the submission-writing cell and produces `submission.csv` with the required `id,has_cactus` columns in sample order.'
- What this solution (achieved 0.98644) has done: 'I fix the protobuf patch that currently raises an AttributeError by correctly patching the *class* in `google.protobuf.message_factory` (and doing it safely before importing TensorFlow). This is a runtime-stability fix and should be score-neutral: model, preprocessing, training, and prediction logic remain unchanged. I also add a minimal fallback to ensure the submission is always written with the correct columns/order even if an unexpected missing test image occurs. The rest of the pipeline (paths, sharpening, architecture, training loop) is preserved exactly to avoid moving your already-above-target score.'
- What this solution (achieved 0.98983) has done: 'I fix the protobuf patch that’s currently causing the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by patching the `MessageFactory` *class* (not an instance) safely before importing TensorFlow, and by only aliasing `GetPrototype` when it’s actually missing. This is a runtime-stability fix and should be score-neutral (no changes to model, preprocessing, training loop, or prediction logic). I also keep the existing environment variables that force the pure-Python protobuf implementation, since that’s the most reliable in this Kaggle environment. Finally, I keep submission writing identical and ensure it always outputs `submission.csv` with the required `id,has_cactus` columns.'
- What this solution (achieved 0.98881) has done: 'I fix the TensorFlow import crash by making the protobuf runtime selection happen before any protobuf usage and by applying a safe, compatible patch that covers both `GetPrototype` and `GetMessageClass` cases without triggering the `MessageFactory` instance error. I keep the model, preprocessing (including sharpening), training loop, and submission generation logic unchanged to avoid shifting your already-above-target AUC further away from the target. I also add a tiny guard to ensure we always read from the correct competition directory if the base path differs, without changing any data content. The output still be a valid `submission.csv` with `id,has_cactus` in the sample order.'
- What this solution (achieved 0.98014) has done: 'I fix the runtime crash in cell 1 caused by an unsafe protobuf patch that still triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` in this environment. The minimal and most stable approach is to force TensorFlow to use the pure-Python protobuf runtime (already intended) and remove the brittle `MessageFactory` monkey-patching entirely, since it’s not needed when using the Python implementation. These changes are score-neutral: the model, preprocessing, training loop, and submission generation remain identical. The script then run end-to-end and write a valid `submission.csv` with the required `id,has_cactus` columns.'
- What this solution (achieved 0.98454) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf runtime is forced to the pure-Python implementation *before* any TensorFlow import, and by deleting any already-imported `google.protobuf` modules so TensorFlow can reload them safely. This is the root cause of the `MessageFactory ... GetPrototype` error and is a stability-only change (model, preprocessing, training, inference remain the same). I also keep the dataset path resolution intact and ensure the pipeline always reaches the submission-writing step and outputs a valid `submission.csv` with the required columns and sample order. Since your current AUC is already well above the target, I not make score-changing modeling/training adjustments.'
- What this solution (achieved 0.98874) has done: 'I fix the TensorFlow import crash (`MessageFactory ... GetPrototype`) by forcing the pure-Python protobuf runtime *and* cleanly removing any already-imported protobuf modules before TensorFlow is imported; this is a stability fix and should be score-neutral. I keep your model, sharpening, train/val split, training loop, and prediction logic unchanged. I also add a tiny safety fallback to locate the dataset directory from the known Kaggle paths (without changing any files used) and ensure `submission.csv` is always written with the required `id,has_cactus` columns in the sample order.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import numpy as np
import pandas as pd

CANDIDATES = [
    "../input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
]
BASE_INPUT = None
for c in CANDIDATES:
    if os.path.exists(c):
        BASE_INPUT = c
        break
if BASE_INPUT is None:
    if os.path.exists("../input"):
        for name in os.listdir("../input"):
            if "aerial-cactus-identification" in name:
                BASE_INPUT = os.path.join("../input", name)
                break
if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not locate aerial-cactus-identification dataset directory."
    )

print("Base input:", BASE_INPUT, "exists:", os.path.exists(BASE_INPUT))
if os.path.exists("../input"):
    print("Listing ../input:", os.listdir("../input")[:20])




## === cell 1
import matplotlib.pyplot as plt
import glob

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
from PIL import Image
import cv2

from sklearn.model_selection import train_test_split

TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

print("TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR :", TEST_DIR, "exists:", os.path.exists(TEST_DIR))
print("TRAIN_CSV:", TRAIN_CSV_PATH, "exists:", os.path.exists(TRAIN_CSV_PATH))
print("SAMPLE  :", SAMPLE_SUB_PATH, "exists:", os.path.exists(SAMPLE_SUB_PATH))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def load_imgs(path):
    """
    Load all jpg images from a directory into a dict keyed by filename.
    Ensures consistent shape/dtype: RGB uint8 (32,32,3).
    """
    imgs = {}
    files = sorted(os.listdir(path))
    for f in files:
        fname = os.path.join(path, f)
        im_bgr = cv2.imread(fname, cv2.IMREAD_COLOR)
        if im_bgr is None:
            continue
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        imgs[f] = im_rgb
    return imgs


img_train = load_imgs(TRAIN_DIR)
img_test = load_imgs(TEST_DIR)

print("Loaded train images:", len(img_train))
print("Loaded test images :", len(img_test))




## === cell 3
train_csv = pd.read_csv(TRAIN_CSV_PATH)

X_train = []
Y_train = []

missing_train = 0
for _, row in train_csv.iterrows():
    fid = row["id"]
    if fid not in img_train:
        missing_train += 1
        continue
    X_train.append(img_train[fid].astype(np.float32) / 255.0)
    Y_train.append(int(row["has_cactus"]))

X_train = np.stack(X_train, axis=0).astype(np.float32)
Y_train = np.array(Y_train, dtype=np.int32)

test_files = sorted(img_test.keys())
X_test_full = np.stack(
    [img_test[f].astype(np.float32) / 255.0 for f in test_files], axis=0
).astype(np.float32)

print("Missing train images:", missing_train)
print("Training data shape:", X_train.shape, "=>", Y_train.shape)
print("Test data shape:", X_test_full.shape, "First test file:", test_files[0])




## === cell 4
plt.rcParams["axes.grid"] = False




## === cell 5
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
idxs = [0, 1, 2, 1000, 1050]
for ax, i in zip(axes, idxs):
    ax.imshow(X_train[i])
    ax.set_title("Has cactus:" + str(Y_train[i]))
    ax.axis("off")
plt.tight_layout()




## === cell 6
from scipy.ndimage import gaussian_filter


def img_sharpen(img):
    blurred_f = gaussian_filter(img, 2)
    filter_blurred_f = gaussian_filter(blurred_f, 2)
    alpha = 15
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)




## === cell 7
sharp_img_xtrain = [img_sharpen(im) for im in X_train]
sharp_xtrain = np.stack(sharp_img_xtrain, axis=0).astype(np.float32)
print("Sharpened train shape:", sharp_xtrain.shape)




## === cell 8
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for ax, i in zip(axes, idxs):
    ax.imshow(sharp_xtrain[i])
    ax.set_title("Has cactus:" + str(Y_train[i]))
    ax.axis("off")
plt.tight_layout()




## === cell 9
x_train, x_val, y_train, y_val = train_test_split(
    sharp_xtrain, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)
print("Train/val shapes:", x_train.shape, x_val.shape, y_train.shape, y_val.shape)




## === cell 10
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
    optimizer=RMSprop(),
    metrics=["accuracy"],
)




## === cell 14
epoch = 20
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=epoch,
    verbose=1,
    validation_data=(x_val, y_val),
)




## === cell 15
acc = history.history.get("accuracy", [])
epochs_ = range(0, epoch)
plt.plot(list(epochs_), acc, label="training accuracy")

acc_val = history.history.get("val_accuracy", [])
plt.scatter(list(epochs_), acc_val, label="validation accuracy")
plt.ylim([0.85, 1.0])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Plot of Model")
plt.legend()
plt.show()




## === cell 16
loss = history.history.get("loss", [])
epochs_ = range(0, epoch)
plt.plot(list(epochs_), loss, label="training loss")

loss_val = history.history.get("val_loss", [])
plt.scatter(list(epochs_), loss_val, label="validation loss")
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
predictions = np.empty((submission_set.shape[0],), dtype=np.float32)

missing_test = 0
for n in tqdm(range(submission_set.shape[0])):
    img_path = os.path.join(TEST_DIR, submission_set.id.iloc[n])
    try:
        data = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        data = img_sharpen(data)
        predictions[n] = float(
            model.predict(data.reshape((1, 32, 32, 3)), verbose=0)[0, 0]
        )
    except Exception:
        missing_test += 1
        predictions[n] = 0.5  # neutral fallback

submission_set["has_cactus"] = predictions.astype(np.float32)
submission_path = "submission.csv"
submission_set[["id", "has_cactus"]].to_csv(submission_path, index=False)
print(
    "Wrote:",
    submission_path,
    "rows:",
    len(submission_set),
    "missing_test:",
    missing_test,
)
submission_set.head()




## === cell 19
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_val_proba = model.predict(x_val, verbose=0).ravel()
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
plt.title("Receiver operating characteristic (validation)")
plt.legend(loc="lower right")
plt.show()
print("Validation AUC:", roc_auc)
