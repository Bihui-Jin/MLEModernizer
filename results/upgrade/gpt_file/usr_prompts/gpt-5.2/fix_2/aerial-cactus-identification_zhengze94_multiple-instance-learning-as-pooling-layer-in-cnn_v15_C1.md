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

0.8015

# 6. Current score

0.88549

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.88549) has done: 'I fix the protobuf/TensorFlow import crash by pinning protobuf to a compatible version at runtime before importing TensorFlow/Keras. I also fix the dataset pathing and image loading so train/test arrays are consistent (all 32×32×3 float32) and aligned to the sample_submission order, which eliminates the “inhomogeneous shape” error. The custom `noisyand` layer be updated for TF2/Keras shape semantics (remove `.value`) so the model builds and trains. Finally, I ensure the submission file is written with the required columns and a `.csv` suffix (`submission.csv`) so Kaggle accepts it.'

# 9. Code solution

## === cell 0
import os, sys, site, subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"protobuf {pb_ver} too new")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf<5"]
        )
        site.main()
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                del sys.modules[m]


_ensure_protobuf_compat()

import numpy as np
import pandas as pd

CANDIDATE_BASES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "../input/aerial-cactus-identification",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
]


def _find_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


base = None
for b in CANDIDATE_BASES:
    if os.path.isdir(b):
        if os.path.exists(os.path.join(b, "train.csv")) and os.path.isdir(
            os.path.join(b, "train")
        ):
            base = b
            break
        if os.path.isdir(os.path.join(b, "aerial-cactus-identification")):
            cand = os.path.join(b, "aerial-cactus-identification")
            if os.path.exists(os.path.join(cand, "train.csv")) and os.path.isdir(
                os.path.join(cand, "train")
            ):
                base = cand
                break

if base is None:
    raise FileNotFoundError(
        "Could not locate aerial-cactus-identification dataset directory."
    )

train_dir = os.path.join(base, "train")
test_dir = os.path.join(base, "test")
train_csv_path = os.path.join(base, "train.csv")
sample_sub_path = os.path.join(base, "sample_submission.csv")

print("Using base:", base)
print(
    "Train dir exists:",
    os.path.isdir(train_dir),
    "Test dir exists:",
    os.path.isdir(test_dir),
)
print(
    "Files:",
    "train.csv" if os.path.exists(train_csv_path) else "missing",
    "|",
    "sample_submission.csv" if os.path.exists(sample_sub_path) else "missing",
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
    BatchNormalization,
    MaxPooling2D,
    Activation,
)
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)




## === cell 2
def load_imgs_array(path, ids=None):
    """
    Load images into a numpy array in a specified order (ids).
    Ensures consistent shape (N,32,32,3) and dtype float32 in [0,1].
    """
    if ids is None:
        ids = sorted([f for f in os.listdir(path) if f.lower().endswith(".jpg")])
    X = np.empty((len(ids), 32, 32, 3), dtype=np.float32)
    for i, f in enumerate(ids):
        fname = os.path.join(path, f)
        im = cv2.imread(fname, cv2.IMREAD_COLOR)  # BGR uint8
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fname}")
        if im.shape[:2] != (32, 32):
            im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        X[i] = im.astype(np.float32) / 255.0
    return X, ids


train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

X_train, train_ids = load_imgs_array(train_dir, ids=train_df["id"].tolist())
y_train = train_df["has_cactus"].astype(np.int32).values

X_test, test_ids = load_imgs_array(test_dir, ids=sample_sub["id"].tolist())

print("Training data shape:", X_train.shape, "=>", y_train.shape)
print("Test data shape:", X_test.shape, "=>", len(test_ids))



## === cell 3
plt.rcParams["axes.grid"] = False
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
idxs = [0, 1, 2, 1000, 1050]
for ax, idx in zip(axes, idxs):
    ax.imshow(X_train[idx])
    ax.set_title("Has cactus:" + str(y_train[idx]))
    ax.axis("off")
plt.show()



## === cell 4
from scipy.ndimage import gaussian_filter


def img_sharpen(img):
    blurred_f = gaussian_filter(img, 2)
    filter_blurred_f = gaussian_filter(blurred_f, 2)
    alpha = 15
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return sharpened




## === cell 5
sharp_img_xtrain = [img_sharpen(im) for im in X_train]
sharp_xtrain = np.array(sharp_img_xtrain, dtype=np.float32)

x_train, x_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
print(x_train.shape, x_val.shape, y_tr.shape, y_val.shape)




## === cell 6
class noisyand(tf.keras.layers.Layer):
    def __init__(self, num_classes, a=20, **kwargs):
        self.num_classes = num_classes
        self.a = max(1, a)
        super(noisyand, self).__init__(**kwargs)

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.b = self.add_weight(
            name="b",
            shape=(1, c),
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




## === cell 7
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



## === cell 8
model.compile(
    loss=tensorflow.keras.losses.binary_crossentropy,
    optimizer=tensorflow.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

epoch = 20
history = model.fit(
    x_train,
    y_tr,
    batch_size=32,
    epochs=epoch,
    verbose=1,
    validation_data=(x_val, y_val),
)



## === cell 9
acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
epochs_ = range(0, epoch)

plt.plot(epochs_, acc, label="training accuracy")
plt.scatter(epochs_, val_acc, label="validation accuracy")
plt.ylim([0.85, 1.0])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Plot of Model")
plt.legend()
plt.show()

loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

plt.plot(epochs_, loss, label="training loss")
plt.scatter(epochs_, val_loss, label="validation loss")
plt.ylim([0, 0.5])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Plot of Model")
plt.legend()
plt.show()



## === cell 10
submission_set = sample_sub.copy()

predictions = np.empty((submission_set.shape[0],), dtype=np.float32)

for n in tqdm(range(submission_set.shape[0])):
    img_path = os.path.join(test_dir, submission_set.id.iloc[n])
    data = np.array(Image.open(img_path))
    data = data.astype(np.float32) / 255.0
    data = img_sharpen(data)
    pred = model.predict(data.reshape((1, 32, 32, 3)), verbose=0)[0]
    predictions[n] = float(pred)

submission_set["has_cactus"] = predictions

out_path = "submission.csv"
submission_set.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission_set.head())



## === cell 11
from sklearn.metrics import roc_auc_score

val_pred = model.predict(x_val, verbose=0).reshape(-1)
auc_val = roc_auc_score(y_val, val_pred)
print("Validation ROC-AUC:", auc_val)
