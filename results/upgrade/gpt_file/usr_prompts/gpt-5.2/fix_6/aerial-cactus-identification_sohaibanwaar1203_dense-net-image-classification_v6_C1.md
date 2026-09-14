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
scikit-image==0.25.2
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

0.496

# 6. Current score

0.99962

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.9995) has done: 'I fix the Keras 3 compatibility issues that currently prevent the notebook from running (protobuf/TF backend selection, deprecated `np_utils`, and the `Adam(lr=...)` argument). I keep your DenseNet architecture and training flow intact, but make the loss/output consistent (2-class softmax + categorical crossentropy) so training is valid and stable. I also correct the plotting keys and the test-time prediction code so it outputs probabilities (not argmax labels) in the required `id,has_cactus` submission format. Finally, I ensure paths work in the Kaggle environment by auto-resolving the correct `../input/aerial-cactus-identification/...` folder.'
- What this solution (achieved 0.99978) has done: 'I fix the Keras/TF/protobuf crash that blocks execution by forcing a stable Keras backend import order (use `tf_keras` and `tensorflow` consistently) while keeping your DenseNet architecture and training flow unchanged. Since your current score (0.9995) is far above the target (0.496) and higher-is-better, I not change any training/prediction logic that would further improve score; the goal is strictly to run end-to-end and emit a valid submission. I also make the dataset path resolution robust to Kaggle’s nested folder layouts and ensure image loading uses a consistent PIL-based RGB pipeline to avoid dtype/shape surprises. Finally, I guarantee `submission.csv` is created with the required `id,has_cactus` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.99944) has done: 'I fix the protobuf-related crash (`MessageFactory.GetPrototype`) by ensuring a compatible TensorFlow/protobuf import order and forcing the pure-Python protobuf implementation *before* importing `tensorflow`/`tf_keras`. I keep your DenseNet architecture, training loops, loss, and prediction semantics unchanged, only adjusting environment setup so the notebook runs end-to-end. I also make the data-root resolver robust to the Kaggle folder layout you showed, and ensure the submission is written as `submission.csv` with `id,has_cactus` aligned to `sample_submission.csv`. Since your current score (0.99978) is already far above the target (0.496) and higher-is-better, I not introduce any score-improving changes.'
- What this solution (achieved 0.99981) has done: 'I fix the crash in the TensorFlow/tf_keras import caused by a protobuf incompatibility by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import happens. To keep your core model/training logic intact (and avoid changing score behavior), I won’t alter the DenseNet architecture, losses, training loops, or prediction post-processing. I also make the data-root resolution work for Kaggle’s common folder layouts while keeping your existing paths as candidates. Finally, I ensure a valid `submission.csv` is always written with exactly `id,has_cactus` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.99962) has done: 'I fix the crash in the TensorFlow/tf_keras import (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import happens, and by importing TensorFlow prior to `tf_keras` consistently. I also make the data-root detection include Kaggle’s `../input/aerial-cactus-identification/aerial-cactus-identification` layout so the notebook reliably finds `train.csv`/`train/` without manual edits. Since your current score (0.99981) is already far above the target (0.496) and higher-is-better, I not change the model, training loop, or prediction semantics (to avoid unintended score changes). Finally, I ensure a valid `submission.csv` is always written with exactly `id,has_cactus` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

print(os.listdir("../input"))



## === cell 1
DATA_ROOT_CANDIDATES = [
    "../input/aerial-cactus-identification",
    "../input/aerial-cactus-identification/aerial-cactus-identification",
    "../input/aerial-cactus-identification/aerial-cactus-identification/aerial-cactus-identification",
    "../input",  # sometimes CSVs are directly here, but folders might be nested
    "../input/aerial-cactus-identification/aerial-cactus-identification/aerial-cactus-identification/aerial-cactus-identification",
]


def _looks_like_root(p):
    return os.path.exists(os.path.join(p, "train.csv")) and os.path.isdir(
        os.path.join(p, "train")
    )


DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if _looks_like_root(cand):
        DATA_ROOT = cand
        break

if DATA_ROOT is None and os.path.isdir("../input"):
    for name in os.listdir("../input"):
        cand = os.path.join("../input", name)
        if os.path.isdir(cand) and _looks_like_root(cand):
            DATA_ROOT = cand
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate train.csv/train folder under expected ../input paths."
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

df = pd.read_csv(TRAIN_CSV)
df.head()



## === cell 2
print("Number of samples: ", len(df))
print("Number of Labels: ", np.unique(df.has_cactus))



## === cell 3
sns.histplot(df.has_cactus, bins=2)
plt.show()



## === cell 4
train = pd.read_csv(TRAIN_CSV)
train_images = []
path = TRAIN_DIR + os.sep

for img_id in train.id:
    with Image.open(path + img_id) as im:
        im = im.convert("RGB")
        train_images.append(np.asarray(im))



## === cell 5
train_images = np.asarray(train_images).astype("float32") / 255.0
X = train_images
y = train.has_cactus.values.astype("int32")
print("Labels: ", y.shape)
print("images: ", X.shape)



## === cell 6
plt.imshow((X[2] * 255).astype(np.uint8))
plt.axis("off")
plt.show()



## === cell 7
import tensorflow as tf
import tf_keras as keras
from tf_keras.models import Model
from tf_keras.layers import (
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
from tf_keras.optimizers import Adam


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




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
from tf_keras.utils import to_categorical
from sklearn.model_selection import train_test_split

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

Cat_test_y = to_categorical(y_test_raw, num_classes=2)
y_train = to_categorical(y_train_raw, num_classes=2)

print("X_train shape : ", X_train.shape)
print("y_train shape : ", y_train.shape)
print("X_test shape : ", X_test.shape)
print("y_test shape : ", y_test_raw.shape)



## === cell 9
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
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_test, Cat_test_y),
    verbose=1,
)



## === cell 10
import sys
import matplotlib

print("Generating plots...")
sys.stdout.flush()
matplotlib.use("Agg")
import matplotlib.pyplot as plt2

plt2.style.use("ggplot")
plt2.figure()
N = epochs
plt2.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt2.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

acc_key = "accuracy" if "accuracy" in history.history else "acc"
val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
plt2.plot(np.arange(0, N), history.history[acc_key], label="train_acc")
plt2.plot(np.arange(0, N), history.history[val_acc_key], label="val_acc")

plt2.title("Cactus Image Classification")
plt2.xlabel("Epoch #")
plt2.ylabel("Loss/Accuracy")
plt2.legend(loc="lower left")
plt2.savefig("plot.png")
plt2.close()



## === cell 11
from sklearn import metrics

label_pred = model.predict(X_test, batch_size=256, verbose=0)
pred = np.argmax(label_pred, axis=1)
Y_test = np.argmax(Cat_test_y, axis=1)

print(metrics.classification_report(Y_test, pred))



## === cell 12
print(metrics.accuracy_score(Y_test, pred))



## === cell 13
model = dense_net(
    growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block
)
batch_size = 32
epochs = 20
optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

y_full = to_categorical(y, num_classes=2)
history = model.fit(
    X, y_full, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0
)



## === cell 14
sample = pd.read_csv(SAMPLE_SUB_CSV)

test_images = []
path = TEST_DIR + os.sep

for img_id in sample.id:
    with Image.open(path + img_id) as im:
        im = im.convert("RGB")
        test_images.append(np.asarray(im))

test_images = np.asarray(test_images).astype("float32") / 255.0
test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)

proba = model.predict(test_images, batch_size=256, verbose=0)[:, 1]

results = pd.DataFrame({"id": sample.id.values, "has_cactus": proba})
results.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", results.shape)
print(results.head())
print("Submission columns:", list(results.columns))
