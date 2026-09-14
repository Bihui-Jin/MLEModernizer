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

0.6222

# 6. Current score

0.86446

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.82457) has done: 'I fix the environment-breaking import issue by avoiding the problematic standalone `tensorflow.keras` import path and using `tf.keras` consistently. I also fix the dataset pathing to the actual Kaggle directory layout you have (so image loading works) and ensure test images are stacked in a consistent order to avoid the “inhomogeneous shape” error. The custom `noisyand` layer be updated to be compatible with TF/Keras 2.18 shape objects (removing the deprecated `.value` usage) while keeping its math identical. Finally, I make sure the submission is written as a valid `submission.csv` with `id,has_cactus` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.80885) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation before TensorFlow is imported. This is a runtime-only environment fix and does not change your model architecture, training loop, feature extraction, or loss. I also make prediction-to-float conversion robust (avoid numpy “array to scalar” warnings/errors) while keeping identical semantics. The rest of the pipeline (paths, data loading, and submission writing to `submission.csv` with `id,has_cactus`) stays the same so your score behavior should remain essentially unchanged.'
- What this solution (achieved 0.89641) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf env vars *before any TensorFlow import* and by importing TensorFlow only once (so the fix consistently applies). I also remove the conflicting `tensorflow.keras` imports and use `tf.keras` everywhere to avoid version-mismatch edge cases in TF 2.18, while keeping the exact same model, loss, optimizer, and training loop. Finally, I make the submission generation deterministic and aligned to `sample_submission.csv` (using its `id` order) and ensure `submission.csv` is always written with the correct columns. These changes are runtime/stability focused and should keep score behavior essentially unchanged (still above target).'
- What this solution (achieved 0.85232) has done: 'I fix the TensorFlow/protobuf crash in the first failing cell by ensuring the protobuf environment variables are set before *any* TensorFlow/Keras import, and by removing the remaining `tensorflow.keras` callback/regularizer imports that can trigger the incompatible protobuf path. This is a runtime/stability fix and does not change your model architecture, training loop, loss, or inference semantics. I also make the output of `model.predict()` conversion consistently scalar to avoid shape-related surprises and keep the submission aligned exactly to `sample_submission.csv` order. The rest of the pipeline (data paths, preprocessing/sharpening, training, and CSV writing) is kept intact.'
- What this solution (achieved 0.82939) has done: 'I fix the crash in the TensorFlow import caused by an incompatibility between TF 2.18 and the installed protobuf runtime by pinning protobuf to the pure-Python implementation and forcing the legacy protobuf API before TensorFlow is imported. This is a runtime stability fix only and keeps your model architecture, training loop, preprocessing, and prediction semantics unchanged. I also keep `tf.keras` usage consistent and ensure the submission is always written as `submission.csv` with columns `id,has_cactus` aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.86446) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* TensorFlow is imported and by avoiding any import patterns that can re-trigger the C++ protobuf path. This is a runtime-only stabilization change and keeps the same model, loss, training loop, and prediction semantics. Since your current score (0.82939) is already far above the target (0.6222) and within no need to “improve,” I not make any performance-boosting changes—only ensure the notebook runs end-to-end. Finally, I keep the submission writing aligned to `sample_submission.csv` and guaranteed to produce `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_VERSION_CHECK", "1"
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

INPUT_ROOT = "/kaggle/input/aerial-cactus-identification"

print("Input root exists:", os.path.exists(INPUT_ROOT))
print("Input root listing (first 20):", sorted(os.listdir(INPUT_ROOT))[:20])



## === cell 1
import matplotlib.pyplot as plt
import glob
import os



## === cell 2
from tqdm import tqdm

import tensorflow as tf

Dense = tf.keras.layers.Dense
Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
MaxPooling2D = tf.keras.layers.MaxPooling2D
Activation = tf.keras.layers.Activation
BatchNormalization = tf.keras.layers.BatchNormalization
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D  # kept though unused

Adam = tf.keras.optimizers.Adam
SGD = tf.keras.optimizers.SGD

from sklearn.model_selection import train_test_split
from PIL import Image

print("TF version:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
import cv2

TRAIN_DIR = os.path.join(INPUT_ROOT, "train")
TEST_DIR = os.path.join(INPUT_ROOT, "test")
TRAIN_CSV_PATH = os.path.join(INPUT_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_ROOT, "sample_submission.csv")


def load_imgs(path):
    imgs = {}
    for f in os.listdir(path):
        fname = os.path.join(path, f)
        im = cv2.imread(fname)  # BGR uint8
        if im is None:
            continue
        imgs[f] = im
    return imgs


img_train = load_imgs(TRAIN_DIR)
img_test = load_imgs(TEST_DIR)

print("Loaded train images:", len(img_train), "Loaded test images:", len(img_test))



## === cell 4
train_csv = pd.read_csv(TRAIN_CSV_PATH)

X_train = []
Y_train = []

for _, row in train_csv.iterrows():
    fn = row["id"]
    X_train.append(img_train[fn].astype(np.float32) / 255.0)
    Y_train.append(int(row["has_cactus"]))

X_train = np.stack(X_train, axis=0).astype(np.float32)
Y_train = np.array(Y_train, dtype=np.int64)

test_ids_sorted = sorted(img_test.keys())
X_test_full = np.stack(
    [img_test[f].astype(np.float32) / 255.0 for f in test_ids_sorted], axis=0
).astype(np.float32)

print("Training data shape:", X_train.shape, "=>", Y_train.shape)
print("Test data shape:", X_test_full.shape)



## === cell 5
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["axes.grid"] = False

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(X_train[0][:, :, ::-1])  # BGR->RGB for display only
axes[0].set_title("Has cactus:" + str(Y_train[0]))
axes[1].imshow(X_train[1][:, :, ::-1])
axes[1].set_title("Has cactus:" + str(Y_train[1]))
axes[2].imshow(X_train[2][:, :, ::-1])
axes[2].set_title("Has cactus:" + str(Y_train[2]))
axes[3].imshow(X_train[1000][:, :, ::-1])
axes[3].set_title("Has cactus:" + str(Y_train[1000]))
axes[4].imshow(X_train[1050][:, :, ::-1])
axes[4].set_title("Has cactus:" + str(Y_train[1050]))
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
sharp_img_xtrain = []
for im in X_train:
    sharp_img_xtrain.append(img_sharpen(im))



## === cell 8
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(np.clip(sharp_img_xtrain[0][:, :, ::-1], 0, 1))
axes[0].set_title("Has cactus:" + str(Y_train[0]))
axes[1].imshow(np.clip(sharp_img_xtrain[1][:, :, ::-1], 0, 1))
axes[1].set_title("Has cactus:" + str(Y_train[1]))
axes[2].imshow(np.clip(sharp_img_xtrain[2][:, :, ::-1], 0, 1))
axes[2].set_title("Has cactus:" + str(Y_train[2]))
axes[3].imshow(np.clip(sharp_img_xtrain[1000][:, :, ::-1], 0, 1))
axes[3].set_title("Has cactus:" + str(Y_train[1000]))
axes[4].imshow(np.clip(sharp_img_xtrain[1050][:, :, ::-1], 0, 1))
axes[4].set_title("Has cactus:" + str(Y_train[1050]))
plt.show()



## === cell 9
from sklearn.model_selection import train_test_split
from numpy import array

sharp_xtrain = array(sharp_img_xtrain, dtype=np.float32)

x_train, x_test, y_train, y_test = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)

print("Split shapes:", x_train.shape, x_test.shape, y_train.shape, y_test.shape)




## === cell 10
class noisyand(tf.keras.layers.Layer):
    def __init__(self, num_classes, a=20, **kwargs):
        self.num_classes = num_classes
        self.a = max(1, a)
        super(noisyand, self).__init__(**kwargs)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.b = self.add_weight(
            name="b", shape=(1, channels), initializer="uniform", trainable=True
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
    model = tf.keras.models.Sequential()
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
epoch = 20
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=epoch,
    verbose=1,
    validation_data=(x_test, y_test),
)



## === cell 15
acc = history.history.get("accuracy", [])
epochs_ = range(0, epoch)
plt.plot(epochs_, acc, label="training accuracy")

acc_val = history.history.get("val_accuracy", [])
plt.scatter(epochs_, acc_val, label="validation accuracy")
plt.ylim([0.85, 1.0])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Plot of Model")
plt.legend()
plt.show()



## === cell 16
loss = history.history.get("loss", [])
epochs_ = range(0, epoch)
plt.plot(epochs_, loss, label="training loss")

loss_val = history.history.get("val_loss", [])
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
predictions = np.empty((submission_set.shape[0],), dtype=np.float32)

for n in tqdm(range(submission_set.shape[0])):
    img_path = os.path.join(TEST_DIR, submission_set.id.iloc[n])
    data = np.array(Image.open(img_path))
    data = data.astype(np.float32) / 255.0
    data = img_sharpen(data)
    pred = model.predict(data.reshape((1, 32, 32, 3)), verbose=0)
    predictions[n] = float(pred.reshape(-1)[0])

submission_set["has_cactus"] = predictions
submission_path = "submission.csv"
submission_set.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
submission_set.head()



## === cell 19
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_pred_proba = model.predict(x_test, verbose=0).ravel()

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
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
plt.title("Validation ROC")
plt.legend(loc="lower right")
plt.show()

print("Validation AUC:", roc_auc)
