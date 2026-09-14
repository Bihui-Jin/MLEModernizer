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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.5013

# 6. Current score

0.99948

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99915) has done: 'I fix the broken data loading by reading images directly from the provided `train/` and `test/` folders and using the correct `train.csv`/`sample_submission.csv`, since the referenced `cactus-images-csv` dataset doesn’t exist here. I resolve Keras 3 incompatibilities by switching imports to `tf_keras` (which matches the installed TF-Keras stack) and replacing deprecated utilities/arguments (`np_utils`, `lr`) with supported equivalents while keeping the DenseNet-like architecture and training loop intact. I also correct the loss/metrics/history key names so training and plotting run without errors. Finally, I generate predictions as probabilities for `has_cactus` (required by the AUC metric) and write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.99948) has done: 'I fix the runtime crash in the Keras/TensorFlow stack that’s causing the `MessageFactory.GetPrototype` error by ensuring the notebook uses the stable `tf.keras` API from TensorFlow (instead of `tf_keras`, which is triggering the protobuf incompatibility here). I keep the exact same DenseNet-like architecture, data loading, training loop, and prediction semantics so the score behavior remains essentially unchanged. I also add a small compile-time consistency fix: since the model outputs a 2-class softmax and labels are one-hot, the correct loss is `categorical_crossentropy` (this is aligned with the current semantics and avoids subtle mis-training). Finally, I ensure the submission is written as `submission.csv` with the required columns and ordering.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/aerial-cactus-identification/train.csv")
df.head()



## === cell 2
print("Number of samples: ", len(df))
print("Number of Labels: ", np.unique(df.has_cactus))



## === cell 3
sns.histplot(df.has_cactus, bins=2, discrete=True)



## === cell 4
from sklearn.model_selection import train_test_split

BASE_PATH = "../input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
sample_sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

print("TRAIN CSV:", train_df.shape, train_df.columns.tolist())
print("SAMPLE SUB:", sample_sub.shape, sample_sub.columns.tolist())
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)


def load_images_from_ids(ids, folder, target_size=(32, 32)):
    """Load RGB images into a float32 array scaled to [0,1]."""
    X = np.zeros((len(ids), target_size[0], target_size[1], 3), dtype=np.float32)
    for i, img_id in enumerate(ids):
        fp = os.path.join(folder, img_id)
        img = Image.open(fp).convert("RGB")
        if img.size != (target_size[1], target_size[0]):
            img = img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
        X[i] = np.asarray(img, dtype=np.float32) / 255.0
    return X


train_ids = train_df["id"].values
y = train_df["has_cactus"].astype("int32").values
X = load_images_from_ids(train_ids, TRAIN_DIR, target_size=(32, 32))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)



## === cell 5
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Input,
    Activation,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
    concatenate,
    AveragePooling2D,
)
from tensorflow.keras.optimizers import Adam


def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation("relu")(conv_x)
    conv_x = Conv2D(
        filters, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False
    )(conv_x)
    conv_x = Dropout(0.2)(conv_x)
    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate
    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation("relu")(trans_x)
    trans_x = Conv2D(
        tran_filters,
        (1, 1),
        kernel_initializer="he_uniform",
        padding="same",
        use_bias=False,
    )(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)
    return trans_x, tran_filters


def dense_net(filters, growth_rate, classes, dense_block_size, layers_in_block):
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(
        24, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False
    )(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation("relu")(x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation("relu")(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(classes, activation="softmax")(dense_x)
    return Model(input_img, output)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
from tensorflow.keras.utils import to_categorical

y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

print("y_train_cat shape:", y_train_cat.shape)
print("y_test_cat shape :", y_test_cat.shape)



## === cell 7
dense_block_size = 3
layers_in_block = 4
growth_rate = 12
classes = 2

model = dense_net(
    growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block
)
model.summary()

batch_size = 32
epochs = 10

optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train_cat,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_test, y_test_cat),
    verbose=2,
)



## === cell 8
import sys
import matplotlib

print("Generating plots...")
sys.stdout.flush()
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

acc_key = "accuracy" if "accuracy" in history.history else "acc"
val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
plt.plot(np.arange(0, N), history.history[acc_key], label="train_acc")
plt.plot(np.arange(0, N), history.history[val_acc_key], label="val_acc")

plt.title("Cactus Image Classification")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")



## === cell 9
from sklearn import metrics

label_pred = model.predict(X_test, batch_size=256, verbose=0)
pred_class = np.argmax(label_pred, axis=1)
print(metrics.classification_report(y_test, pred_class))



## === cell 10
from sklearn import metrics

label_pred = model.predict(X_test, batch_size=256, verbose=0)
pred_class = np.argmax(label_pred, axis=1)
print(metrics.accuracy_score(y_test, pred_class))



## === cell 11
test_ids = sample_sub["id"].values
X_sub = load_images_from_ids(test_ids, TEST_DIR, target_size=(32, 32))
print("Submission test array shape:", X_sub.shape)



## === cell 12
proba = model.predict(X_sub, batch_size=256, verbose=0)[:, 1]

results = pd.DataFrame({"id": test_ids, "has_cactus": proba.astype(np.float32)})
results.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", results.shape)
print(results.head())
