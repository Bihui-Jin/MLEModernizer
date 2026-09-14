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

0.8042

# 6. Current score

0.98529

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.98529) has done: 'I set the protobuf implementation flag before any TensorFlow import, fix the image‑loading to ensure every image is a 32×32×3 array, correct the custom `noisyand` layer (use proper shape handling), and adjust the compile/fit sections to use the right TensorFlow objects and history keys. These changes resolve the import error, the inconsistent‑shape array error, and the subsequent NameError cascades, allowing the notebook to run end‑to‑end and produce a valid `sample_submission.csv` file while keeping the original model architecture.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

print("Input folder contents:", os.listdir("../input"))




## === cell 1
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D,
    Activation,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from PIL import Image




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def load_imgs(path):
    """Load all images in *path*, resize to 32×32 and ensure 3 channels."""
    imgs = {}
    for f in os.listdir(path):
        fname = os.path.join(path, f)
        try:
            img = Image.open(fname).convert("RGB")
            img = img.resize((32, 32))
            imgs[f] = np.array(img, dtype=np.uint8)
        except Exception as e:
            imgs[f] = np.zeros((32, 32, 3), dtype=np.uint8)
    return imgs




## === cell 3
img_train = load_imgs("../input/train/train/")
img_test = load_imgs("../input/test/test/")

train_csv = pd.read_csv("../input/train.csv")

X_train = []
Y_train = []

for _, row in train_csv.iterrows():
    X_train.append(img_train[row["id"]] / 255.0)  # normalize
    Y_train.append(int(row["has_cactus"]))

X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.int32)

X_test = np.array([img_test[f] / 255.0 for f in img_test], dtype=np.float32)

print("Training data shape:", X_train.shape, "=>", Y_train.shape)
print("Test data shape:", X_test.shape)




## === cell 4
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(X_train[0])
axes[0].set_title("Has cactus: " + str(Y_train[0]))
axes[1].imshow(X_train[1])
axes[1].set_title("Has cactus: " + str(Y_train[1]))
axes[2].imshow(X_train[2])
axes[2].set_title("Has cactus: " + str(Y_train[2]))
axes[3].imshow(X_train[1000])
axes[3].set_title("Has cactus: " + str(Y_train[1000]))
axes[4].imshow(X_train[1050])
axes[4].set_title("Has cactus: " + str(Y_train[1050]))
plt.tight_layout()
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




## === cell 7
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(sharp_img_xtrain[0])
axes[0].set_title("Sharp – " + str(Y_train[0]))
axes[1].imshow(sharp_img_xtrain[1])
axes[1].set_title("Sharp – " + str(Y_train[1]))
axes[2].imshow(sharp_img_xtrain[2])
axes[2].set_title("Sharp – " + str(Y_train[2]))
axes[3].imshow(sharp_img_xtrain[1000])
axes[3].set_title("Sharp – " + str(Y_train[1000]))
axes[4].imshow(sharp_img_xtrain[1050])
axes[4].set_title("Sharp – " + str(Y_train[1050]))
plt.tight_layout()
plt.show()




## === cell 8
from sklearn.model_selection import train_test_split

sharp_xtrain = np.array(sharp_img_xtrain, dtype=np.float32)
x_train, x_val, y_train, y_val = train_test_split(
    sharp_xtrain, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)




## === cell 9
class noisyand(tf.keras.layers.Layer):
    def __init__(self, num_classes, a=20, **kwargs):
        super(noisyand, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.a = max(1, a)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.b = self.add_weight(
            name="b",
            shape=(1, channels),
            initializer="uniform",
            trainable=True,
        )
        super(noisyand, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])  # shape (batch, channels)
        a = tf.cast(self.a, tf.float32)
        b = self.b
        numerator = tf.nn.sigmoid(a * (mean - b)) - tf.nn.sigmoid(-a * b)
        denominator = tf.nn.sigmoid(a * (1 - b)) - tf.nn.sigmoid(-a * b)
        return numerator / denominator

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])




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




## === cell 11
model = define_model()
model.summary()




## === cell 12
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=RMSprop(), metrics=["accuracy"]
)




## === cell 13
epochs = 20
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val),
)




## === cell 14
plt.figure()
plt.plot(range(1, epochs + 1), history.history["accuracy"], label="train acc")
plt.plot(range(1, epochs + 1), history.history["val_accuracy"], label="val acc")
plt.ylim([0.5, 1.0])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.show()




## === cell 15
plt.figure()
plt.plot(range(1, epochs + 1), history.history["loss"], label="train loss")
plt.plot(range(1, epochs + 1), history.history["val_loss"], label="val loss")
plt.ylim([0, 0.5])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()




## === cell 16
submission_set = pd.read_csv("../input/sample_submission.csv")
print("Submission template head:")
print(submission_set.head())




## === cell 17
predictions = np.empty((submission_set.shape[0],), dtype=np.float32)

for n in tqdm(range(submission_set.shape[0]), desc="Predicting"):
    img_path = os.path.join("../input/test/test/", submission_set.id[n])
    img = (
        np.array(Image.open(img_path).convert("RGB").resize((32, 32)), dtype=np.float32)
        / 255.0
    )
    img = img_sharpen(img)
    prob = model.predict(img.reshape((1, 32, 32, 3)), verbose=0)[0][0]
    predictions[n] = prob

submission_set["has_cactus"] = predictions
submission_path = "sample_submission.csv"
submission_set.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 18
from sklearn.metrics import roc_curve, auc

val_pred = model.predict(x_val, verbose=0).ravel()
fpr, tpr, _ = roc_curve(y_val, val_pred)
roc_auc = auc(fpr, tpr)
print(f"Validation ROC AUC: {roc_auc:.4f}")
