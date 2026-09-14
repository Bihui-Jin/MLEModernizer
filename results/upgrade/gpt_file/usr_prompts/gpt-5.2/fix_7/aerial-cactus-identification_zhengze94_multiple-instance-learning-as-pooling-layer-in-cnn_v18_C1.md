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

0.5583

# 6. Current score

0.93494

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.8583) has done: 'I fix the TensorFlow/protobuf import crash by pinning protobuf to the compatible pure-Python implementation via environment variables before importing TensorFlow/Keras. I fix dataset pathing so it works with Kaggle’s `/kaggle/input/aerial-cactus-identification/...` layout and make test image loading consistent (sorted IDs and fixed shapes) to eliminate the “inhomogeneous shape” error. I update the custom `noisyand` layer to be compatible with TF 2.18/Keras (no `.value` on shape dims) while preserving its computation. Finally, I ensure training runs end-to-end and a valid `submission.csv` (with `id,has_cactus`) is written.'
- What this solution (achieved 0.77378) has done: 'I fix the TensorFlow import crash by forcing a protobuf-compatible backend before TensorFlow is imported (and also disabling C++ protobuf), which resolves the `MessageFactory.GetPrototype` error in TF 2.18 environments. I keep the model/training logic identical, but make inference consistent with training by using the already-loaded OpenCV test tensors (BGR) rather than re-loading with PIL (RGB), which is a minimal, legitimate change that typically improves AUC and should move your score slightly upward. I also make the prediction scalar extraction robust to Keras output shapes so it can’t error and always writes a valid `submission.csv` with the required columns. All paths and the overall approach remain unchanged.'
- What this solution (achieved 0.8483) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf environment variables before any TensorFlow import and also forcing a safe pure-Python protobuf fallback if needed. I keep your model and training loop unchanged, only making the import order and environment configuration robust so the notebook runs end-to-end. Because your current score (0.77378) is already well above the target (0.5583), I avoid any changes that would intentionally improve performance further; the patch is score-neutral and focused on correctness and producing a valid `submission.csv`. Finally, I ensure the submission file is always written with the exact required columns (`id,has_cactus`).'
- What this solution (achieved 0.932) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf environment variables are set before any TensorFlow import and by additionally forcing the pure-Python protobuf implementation via `google.protobuf.internal.api_implementation` (a known fix for the `MessageFactory.GetPrototype` issue). I keep your model, training loop, preprocessing, and prediction logic unchanged so the score behavior stays essentially the same (your current score is already well above the target, so we avoid performance-changing edits). I also make the data path resolution slightly more robust (fallback to alternative Kaggle input layout) without changing which files are used. Finally, I ensure the submission is always written as `submission.csv` with the exact required columns and row order from `sample_submission.csv`.'
- What this solution (achieved 0.93192) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation *before* any TensorFlow import, and by importing protobuf early to ensure the environment variables take effect. This is a correctness/stability change and should keep your model/training logic and score behavior essentially unchanged (your current 0.932 is already far above the 0.5583 target, so we should not intentionally improve). I also make the input data directory resolution more robust across the provided Kaggle folder layouts without changing which dataset is used. Finally, I keep the submission creation exactly in the required `id,has_cactus` format and ensure it always writes `submission.csv`.'
- What this solution (achieved 0.93494) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment configuration to the very top (before any protobuf/tensorflow import) and by forcing the pure-Python protobuf implementation in the most robust way for TF 2.18. This is a stability-only change and should not materially change your model/training logic or score behavior (and we explicitly avoid any performance-improving tweaks since your current score is already far above the target). I also keep your original data loading, model, training, and submission-writing logic intact, only adjusting import order to ensure the notebook runs end-to-end and always writes `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP", "1")

try:
    import google.protobuf  # noqa: F401

    try:
        from google.protobuf.internal import api_implementation as _api_impl  # type: ignore

        try:
            _api_impl._SetType("python")
        except Exception:
            pass
    except Exception:
        pass
except Exception:
    pass

import numpy as np
import pandas as pd

CANDIDATE_DATA_DIRS = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification/aerial-cactus-identification",
]
DATA_DIR = next(
    (p for p in CANDIDATE_DATA_DIRS if os.path.exists(p)), CANDIDATE_DATA_DIRS[0]
)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

print("DATA_DIR:", DATA_DIR)
print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print(
    "Train images:",
    len(os.listdir(TRAIN_DIR)) if os.path.exists(TRAIN_DIR) else "missing",
)
print(
    "Test images:", len(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else "missing"
)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample sub exists:", os.path.exists(SAMPLE_SUB))




## === cell 1
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image  # kept to preserve original imports (even if unused)

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
from sklearn.model_selection import train_test_split

print("TensorFlow:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def load_imgs(path):
    """Load all images from a folder into dict: filename -> BGR image array."""
    imgs = {}
    for f in os.listdir(path):
        fname = os.path.join(path, f)
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        if im is None:
            continue
        imgs[f] = im
    return imgs


img_train = load_imgs(TRAIN_DIR)
img_test = load_imgs(TEST_DIR)

print("Loaded train imgs:", len(img_train), "Loaded test imgs:", len(img_test))




## === cell 3
train_csv = pd.read_csv(TRAIN_CSV)

X_train = []
Y_train = []

missing = 0
for _, row in train_csv.iterrows():
    fid = row["id"]
    if fid not in img_train:
        missing += 1
        continue
    X_train.append(img_train[fid].astype(np.float32) / 255.0)
    Y_train.append(int(row["has_cactus"]))

X_train = np.stack(X_train, axis=0)
Y_train = np.asarray(Y_train, dtype=np.int64)

print("Missing train files:", missing)
print("Training data shape:", X_train.shape, "=>", Y_train.shape)

submission_set = pd.read_csv(SAMPLE_SUB)
test_ids = submission_set["id"].tolist()

X_test = []
missing_test = 0
for fid in test_ids:
    im = img_test.get(fid, None)
    if im is None:
        missing_test += 1
        impath = os.path.join(TEST_DIR, fid)
        im = cv2.imread(impath, cv2.IMREAD_COLOR)
    if im is None:
        im = np.zeros((32, 32, 3), dtype=np.uint8)
    X_test.append(im.astype(np.float32) / 255.0)

X_test = np.stack(X_test, axis=0)
print("Test data shape:", X_test.shape, "Missing test files:", missing_test)




## === cell 4
plt.rcParams["axes.grid"] = False
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
idxs = [0, 1, 2, 1000, 1050]
for ax, i in zip(axes, idxs):
    ax.imshow(X_train[i][..., ::-1])  # convert BGR->RGB for display
    ax.set_title("Has cactus:" + str(Y_train[i]))
    ax.axis("off")
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
sharp_img_xtrain = []
for im in X_train:
    sharp_img_xtrain.append(img_sharpen(im))
sharp_xtrain = np.asarray(sharp_img_xtrain, dtype=np.float32)
print("Sharpened train shape:", sharp_xtrain.shape)




## === cell 7
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for ax, i in zip(axes, idxs):
    ax.imshow(np.clip(sharp_xtrain[i][..., ::-1], 0, 1))  # BGR->RGB for display
    ax.set_title("Has cactus:" + str(Y_train[i]))
    ax.axis("off")
plt.tight_layout()




## === cell 8
x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)




## === cell 9
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




## === cell 10
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




## === cell 11
model.compile(
    loss=tensorflow.keras.losses.binary_crossentropy,
    optimizer=tensorflow.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

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
acc = history.history.get("accuracy", None)
val_acc = history.history.get("val_accuracy", None)

if acc is not None and val_acc is not None:
    epochs_ = range(0, epoch)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs_, acc, label="training accuracy")
    plt.scatter(epochs_, val_acc, label="validation accuracy", s=15)
    plt.ylim([0.0, 1.0])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Plot of Model")
    plt.legend()
    plt.show()




## === cell 13
loss = history.history.get("loss", None)
val_loss = history.history.get("val_loss", None)

if loss is not None and val_loss is not None:
    epochs_ = range(0, epoch)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs_, loss, label="training loss")
    plt.scatter(epochs_, val_loss, label="validation loss", s=15)
    plt.ylim([0, max(1e-6, float(max(loss + val_loss)))])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Plot of Model")
    plt.legend()
    plt.show()




## === cell 14
predictions = np.empty((submission_set.shape[0],), dtype=np.float32)

for n in tqdm(range(submission_set.shape[0])):
    fid = submission_set.id.iloc[n]

    im = img_test.get(fid, None)
    if im is None:
        im = cv2.imread(os.path.join(TEST_DIR, fid), cv2.IMREAD_COLOR)
    if im is None:
        im = np.zeros((32, 32, 3), dtype=np.uint8)

    data = im.astype(np.float32) / 255.0
    data = img_sharpen(data)

    p = model.predict(data.reshape((1, 32, 32, 3)), verbose=0)
    predictions[n] = float(np.asarray(p).reshape(-1)[0])

submission = submission_set.copy()
submission["has_cactus"] = predictions
submission = submission[["id", "has_cactus"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)




## === cell 15
from sklearn.metrics import roc_auc_score, roc_curve

y_val_pred = model.predict(x_val, verbose=0).reshape(-1)
roc_auc = roc_auc_score(y_val, y_val_pred)
fpr, tpr, _ = roc_curve(y_val, y_val_pred)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=1.5, label="ROC (AUC = %0.4f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation ROC")
plt.legend(loc="lower right")
plt.show()
print("Validation AUC:", roc_auc)
