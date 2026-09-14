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

0.4911

# 6. Current score

0.99971

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99961) has done: 'I fix the broken dataset paths by using the provided `aerial-cactus-identification` train/test image folders and `train.csv`, replacing the nonexistent `cactus-images-csv` CSV pipeline. I keep the same DenseNet-style model architecture and training loop structure, but update Keras 3 API breakages (`np_utils`, `Adam(lr=...)`, and history key names) so the notebook runs end-to-end. I also correct the submission generation to output probabilities for `has_cactus` (not class argmax), aligned by `id` from `sample_submission.csv`, and always write `submission.csv`. Finally, I add lightweight, deterministic image loading and normalization so shapes/types are correct and training is stable within the time limit.'
- What this solution (achieved 0.99971) has done: 'I fix the crash in the Keras import/model-building cell by switching to `tf_keras` (TensorFlow-backed Keras) which avoids the protobuf `MessageFactory.GetPrototype` incompatibility that occurs with standalone Keras 3 in this environment. To keep your core model/training logic unchanged, I only adjust imports so the same layers/optimizer/model code runs, and I keep the same data loading, split, training loops, and submission generation. Because your current score (0.99961) is far above the target (0.4911), I not make any changes intended to improve performance; these edits are score-neutral and only ensure the notebook runs end-to-end and writes `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

BASE_PATH = "../input/aerial-cactus-identification"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test")

print("Listing ../input:")
print(os.listdir("../input"))
print("BASE_PATH exists:", os.path.exists(BASE_PATH))
print("Train CSV exists:", os.path.exists(TRAIN_CSV_PATH))
print("Train img dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test img dir exists:", os.path.isdir(TEST_IMG_DIR))

SEED = 42
np.random.seed(SEED)



## === cell 1
df = pd.read_csv(TRAIN_CSV_PATH)
df.head()



## === cell 2
print("Number of samples: ", len(df))
print("Number of Labels: ", np.unique(df.has_cactus))
print(df.has_cactus.value_counts())



## === cell 3
plt.figure(figsize=(4, 3))
sns.histplot(df["has_cactus"], bins=2, discrete=True)
plt.title("Label distribution (has_cactus)")
plt.tight_layout()
plt.show()




## === cell 4
def load_images_from_ids(ids, folder, target_size=(32, 32)):
    """Load images by filename id from folder -> float32 array [0,1]."""
    n = len(ids)
    X = np.empty((n, target_size[0], target_size[1], 3), dtype=np.float32)
    for i, img_id in enumerate(ids):
        img_path = os.path.join(folder, img_id)
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if im.size != target_size:
                im = im.resize(target_size)
            X[i] = np.asarray(im, dtype=np.float32) / 255.0
    return X


train_ids = df["id"].values
y = df["has_cactus"].values.astype("int64")

print("Loading train images...")
X = load_images_from_ids(train_ids, TRAIN_IMG_DIR, target_size=(32, 32))
print("X shape:", X.shape, "y shape:", y.shape)

sample = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample["id"].values
print("Loading test images...")
X_test_final = load_images_from_ids(test_ids, TEST_IMG_DIR, target_size=(32, 32))
print("X_test_final shape:", X_test_final.shape)



## === cell 5
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
    for _ in range(layers_in_block):
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
    for _ in range(dense_block_size - 1):
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
from tf_keras.utils import to_categorical
from sklearn.model_selection import train_test_split

X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    X, y, test_size=0.33, random_state=SEED, stratify=y
)

y_train = to_categorical(y_train_raw, num_classes=2)
y_val = to_categorical(y_val_raw, num_classes=2)

print("X_train shape : ", X_train.shape)
print("y_train shape : ", y_train.shape)
print("X_val shape : ", X_val.shape)
print("y_val shape : ", y_val.shape)



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

model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_val, y_val),
    verbose=1,
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
plt.plot(np.arange(0, N), history.history.get("loss", []), label="train_loss")
plt.plot(np.arange(0, N), history.history.get("val_loss", []), label="val_loss")
plt.plot(np.arange(0, N), history.history.get("accuracy", []), label="train_acc")
plt.plot(np.arange(0, N), history.history.get("val_accuracy", []), label="val_acc")
plt.title("Cactus Image Classification")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("plot.png")



## === cell 9
from sklearn import metrics
from sklearn.metrics import classification_report

label_pred = model.predict(X_val, batch_size=256, verbose=0)
pred_class = np.argmax(label_pred, axis=1)
y_val_class = np.argmax(y_val, axis=1)

print(classification_report(y_val_class, pred_class))
print("Val accuracy:", metrics.accuracy_score(y_val_class, pred_class))
print("Val AUC:", metrics.roc_auc_score(y_val_class, label_pred[:, 1]))



## === cell 10
label_pred = model.predict(X_val, batch_size=256, verbose=0)
pred_class = np.argmax(label_pred, axis=1)
y_val_class = np.argmax(y_val, axis=1)
print(metrics.accuracy_score(y_val_class, pred_class))



## === cell 11
model = dense_net(
    growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block
)
batch_size = 32
epochs = 20
optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

y_full = to_categorical(y, num_classes=2)
history = model.fit(
    X, y_full, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0
)
print("Finished full-data training.")



## === cell 12
test_proba = model.predict(X_test_final, batch_size=256, verbose=0)[:, 1]
test_proba = np.clip(test_proba.astype(np.float64), 0.0, 1.0)

results = pd.DataFrame({"id": test_ids, "has_cactus": test_proba})
results.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", results.shape)
print(results.head())
