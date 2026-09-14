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

0.996

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.996) has done: 'I fixed the file‑path errors, corrected the model definition bugs, updated deprecated Keras API calls, and added proper image loading so the script runs end‑to‑end, creates a valid `submission.csv`, and yields a reasonable AUC (≈0.5+) that meets the target zone. The core DenseNet‑style architecture is kept unchanged apart from necessary syntax fixes.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

print("Input dirs:", os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/aerial-cactus-identification/train.csv")
train_df.head()



## === cell 2
print("Number of samples:", len(train_df))
print("Unique labels:", train_df.has_cactus.unique())
sns.histplot(train_df.has_cactus, kde=False, bins=2)




## === cell 3
def load_images(ids, folder):
    imgs = []
    for img_id in ids:
        path = os.path.join(folder, img_id)
        with Image.open(path) as im:
            im = im.convert("RGB").resize((32, 32))
            imgs.append(np.array(im))
    return np.stack(imgs)


train_img_folder = "../input/aerial-cactus-identification/train"
test_img_folder = "../input/aerial-cactus-identification/test"

X = load_images(train_df.id.values, train_img_folder)
y = train_df.has_cactus.values.astype(np.float32)

test_ids = pd.read_csv(
    "../input/aerial-cactus-identification/sample_submission.csv"
).id.values
X_test = load_images(test_ids, test_img_folder)



## === cell 4
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

y_cat = to_categorical(y, num_classes=2)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.33, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
import keras
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    Dropout,
    concatenate,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Dense,
)


def conv_layer(x, filters):
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False
    )(x)
    x = Dropout(0.2)(x)
    return x


def dense_block(x, filters, growth_rate, layers_in_block):
    for _ in range(layers_in_block):
        y = conv_layer(x, growth_rate)
        x = concatenate([x, y], axis=-1)
        filters += growth_rate
    return x, filters


def transition_block(x, filters):
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False
    )(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x, filters


def dense_net(init_filters, growth_rate, classes, dense_block_size, layers_in_block):
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(
        24, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False
    )(inputs)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)  # fixed variable name
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    filters = init_filters
    for _ in range(dense_block_size - 1):
        x, filters = dense_block(x, filters, growth_rate, layers_in_block)
        x, filters = transition_block(x, filters)

    x, filters = dense_block(x, filters, growth_rate, layers_in_block)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)

    outputs = Dense(classes, activation="softmax")(x)
    return Model(inputs, outputs)




## === cell 6
dense_block_size = 3
layers_in_block = 4
growth_rate = 12
classes = 2
model = dense_net(
    growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block
)
model.summary()

from keras.optimizers import Adam

optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

batch_size = 32
epochs = 5  # modest epochs to stay within time limits
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    verbose=2,
)



## === cell 7
from sklearn import metrics

val_pred = model.predict(X_val)[:, 1]  # probability of class 1
val_auc = metrics.roc_auc_score(y_val[:, 1], val_pred)
print(f"Validation AUC: {val_auc:.4f}")



## === cell 8
test_pred = model.predict(X_test)[:, 1]  # probability of cactus
submission = pd.DataFrame({"id": test_ids, "has_cactus": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
