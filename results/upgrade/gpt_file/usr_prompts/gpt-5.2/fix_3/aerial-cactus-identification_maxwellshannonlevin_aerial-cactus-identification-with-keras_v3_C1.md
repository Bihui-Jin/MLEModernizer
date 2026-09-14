# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
print("TF version:", tf.__version__)
tf.random.set_seed(SEED)



## === cell 2
cactus_dir = "/kaggle/input/aerial-cactus-identification"
train_dir = os.path.join(cactus_dir, "train")
test_dir = os.path.join(cactus_dir, "test")

df_train_data = pd.read_csv(os.path.join(cactus_dir, "train.csv"))
df_train_data["has_cactus"] = df_train_data["has_cactus"].astype(str)

df_test = pd.read_csv(os.path.join(cactus_dir, "sample_submission.csv"))

print("train_dir exists:", os.path.isdir(train_dir), train_dir)
print("test_dir exists:", os.path.isdir(test_dir), test_dir)
print("train.csv shape:", df_train_data.shape)
print("sample_submission shape:", df_test.shape)



## === cell 3
assert set(df_train_data.columns) == {"id", "has_cactus"}
assert set(df_test.columns) == {"id", "has_cactus"}
assert df_train_data["has_cactus"].isin(["0", "1"]).all()

first_img = os.path.join(train_dir, df_train_data.loc[0, "id"])
print("Example image exists:", os.path.isfile(first_img), first_img)



## === cell 4
df_train_data.head()



## === cell 5
print("The number of training images is: {}".format(len(df_train_data)))
df_train_data["has_cactus"].value_counts()



## === cell 6
try:
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    imgs = []
    labels = []
    for _, row in df_train_data.head(16).iterrows():
        fname = os.path.join(train_dir, row["id"])
        imgs.append(load_img(fname))
        labels.append("Cactus" if row["has_cactus"] == "1" else "No Cactus")

    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(imgs[i])
        ax.set_title(labels[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Visualization skipped due to:", repr(e))



## === cell 7
has_cactus = df_train_data[df_train_data["has_cactus"] == "1"]
not_cactus = df_train_data[df_train_data["has_cactus"] == "0"]

print("has_cactus:", len(has_cactus), "not_cactus:", len(not_cactus))



## === cell 8
df_train = df_train_data.sample(frac=1, random_state=SEED).reset_index(drop=True)

df_valid = (
    df_train.groupby("has_cactus", group_keys=False)
    .apply(lambda x: x.sample(frac=0.1, random_state=SEED))
    .reset_index(drop=True)
)
df_train = df_train.drop(index=df_valid.index).reset_index(drop=True)

print("Train size:", len(df_train), "Valid size:", len(df_valid))
print("Train label distribution:\n", df_train["has_cactus"].value_counts())
print("Valid label distribution:\n", df_valid["has_cactus"].value_counts())



## === cell 9
train_datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1.0 / 255.0,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## === cell 10
new_image_size = 64
batch_size = 128

train_generator = train_datagen.flow_from_dataframe(
    df_train,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(new_image_size, new_image_size),
    class_mode="binary",
    batch_size=batch_size,
    shuffle=True,
    seed=SEED,
)

valid_generator = valid_datagen.flow_from_dataframe(
    df_valid,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(new_image_size, new_image_size),
    class_mode="binary",
    batch_size=batch_size,
    shuffle=False,
)



## === cell 11
model = Sequential()
model.add(
    Conv2D(
        32, (3, 3), input_shape=(new_image_size, new_image_size, 3), activation="relu"
    )
)
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(512, (3, 3), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.summary()



## === cell 12
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 13
checkpoint_path = "model.keras"

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.2,
    patience=10,
    min_lr=1e-9,
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=100,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)



## === cell 14
epochs = 30

steps_per_epoch = max(1, int(np.ceil(len(df_train) / batch_size)))
validation_steps = max(1, int(np.ceil(len(df_valid) / batch_size)))

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1,
)



## === cell 15
if os.path.exists(checkpoint_path):
    model = load_model(checkpoint_path)
    print(f"Loaded best model from {checkpoint_path}")
else:
    print(f"Warning: {checkpoint_path} not found; using in-memory model weights.")



## === cell 16
eval_res = model.evaluate(valid_generator, steps=validation_steps, verbose=0)
print("Validation loss, accuracy:", eval_res)



## === cell 17
test_fnames = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])

test_imgs = []
for fname in test_fnames:
    img = load_img(
        os.path.join(test_dir, fname), target_size=(new_image_size, new_image_size)
    )
    img = img_to_array(img) / 255.0
    test_imgs.append(img)

test_imgs = np.asarray(test_imgs, dtype=np.float32)
print("Test images array:", test_imgs.shape)



## === cell 18
pred = model.predict(test_imgs, batch_size=batch_size, verbose=0).reshape(-1)
pred = np.clip(pred, 1e-7, 1 - 1e-7)  # numeric safety for probabilities
print(pred[:5])



## === cell 19
submission = pd.DataFrame({"id": test_fnames, "has_cactus": pred.astype(np.float32)})

if "id" in df_test.columns and len(df_test) == len(submission):
    submission = df_test[["id"]].merge(submission, on="id", how="left")
    submission["has_cactus"] = submission["has_cactus"].fillna(0.5).astype(np.float32)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
