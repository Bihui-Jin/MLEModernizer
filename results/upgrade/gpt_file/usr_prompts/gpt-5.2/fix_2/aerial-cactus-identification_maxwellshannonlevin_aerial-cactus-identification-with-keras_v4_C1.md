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

# 5. Target score

0.9991726666666668

# 6. Current score

0.46298

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.46298) has done: 'I fix the environment-breaking Keras import error by switching to `tensorflow.keras`, which is the supported stack in Kaggle and avoids the protobuf `GetPrototype` crash. I also correct the dataset paths to the actual `train/` and `test/` folders, and replace deprecated `DataFrame.append()` with `pd.concat()` so the train/validation split code runs. To keep your core model/training logic intact, I preserve the same CNN architecture and ImageDataGenerator usage, only updating callback monitor names to the correct `val_accuracy` so checkpoint/LR scheduling work. Finally, I ensure test-time loading uses the correct `target_size=(H,W)` signature, predict in the same id order as `sample_submission.csv`, and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
import tensorflow as tf

tf.random.set_seed(SEED)

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

sns.set()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
cactus_dir = "../input/aerial-cactus-identification"

train_dir = os.path.join(cactus_dir, "train")
test_dir = os.path.join(cactus_dir, "test")

train_csv_path = os.path.join(cactus_dir, "train.csv")
sample_sub_path = os.path.join(cactus_dir, "sample_submission.csv")

df_train_data = pd.read_csv(train_csv_path)
df_train_data["has_cactus"] = df_train_data["has_cactus"].astype(str)

df_test = pd.read_csv(sample_sub_path)

assert os.path.isdir(train_dir), f"train_dir not found: {train_dir}"
assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"
assert set(df_test.columns) == {"id", "has_cactus"}



## === cell 3
print("Train rows:", len(df_train_data))
print("Test rows:", len(df_test))
print(
    "Example train image exists:",
    os.path.isfile(os.path.join(train_dir, df_train_data.loc[0, "id"])),
)



## === cell 4
df_train_data.head()



## === cell 5
print("The number of training images is: {}".format(len(df_train_data)))
df_train_data["has_cactus"].value_counts()



## === cell 6
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.reshape(-1)

imgs = []
labels = []
for _, row in df_train_data.head(16).iterrows():
    fname = os.path.join(train_dir, row["id"])
    imgs.append(load_img(fname))
    labels.append("Cactus" if row["has_cactus"] == "1" else "No Cactus")

for i, ax in enumerate(axes):
    ax.imshow(imgs[i])
    ax.set_title(labels[i])
    ax.axis("off")

plt.tight_layout()
plt.show()



## === cell 7
has_cactus = df_train_data[df_train_data["has_cactus"] == "1"]
not_cactus = df_train_data[df_train_data["has_cactus"] == "0"]

print("has_cactus:", len(has_cactus), "not_cactus:", len(not_cactus))



## === cell 8
df_train = (
    pd.concat([not_cactus.iloc[:3491], has_cactus.iloc[:3491]], axis=0)
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

df_train["has_cactus"].value_counts()



## === cell 9
n_not_valid = len(not_cactus) - 3491  # expected 873
df_valid = (
    pd.concat(
        [not_cactus.iloc[3491:], has_cactus.iloc[3491 : 3491 + n_not_valid * 3]],
        axis=0,
    )
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

df_valid["has_cactus"].value_counts()



## === cell 10
train_datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1.0 / 255,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)



## === cell 11
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



## === cell 12
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



## === cell 13
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 14
checkpoint = ModelCheckpoint(
    "model.hdf5",
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



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1283218899.py in <cell line: 0>()
      1 # FIX: Keras metric name is 'val_accuracy' (not 'val_acc') in tf.keras.
----> 2 checkpoint = ModelCheckpoint(
      3     "model.hdf5",
      4     monitor="val_accuracy",
      5     verbose=1,

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=model.hdf5

## === cell 15
steps_per_epoch = max(1, len(df_train) // batch_size)
validation_steps = max(1, len(df_valid) // batch_size)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=valid_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=2,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1402595731.py in <cell line: 0>()
      9     validation_data=valid_generator,
     10     validation_steps=validation_steps,
---> 11     callbacks=[checkpoint, reduce_lr, early_stop],
     12     verbose=2,
     13 )

NameError: name 'checkpoint' is not defined

## === cell 16
if os.path.exists("model.hdf5"):
    model = load_model("model.hdf5")



## === cell 17
eval_metrics = model.evaluate(valid_generator, steps=validation_steps, verbose=0)
print(dict(zip(model.metrics_names, eval_metrics)))



## === cell 18
test_fnames = df_test["id"].tolist()

test_imgs = np.zeros(
    (len(test_fnames), new_image_size, new_image_size, 3), dtype=np.float32
)
for i, fname in enumerate(test_fnames):
    path = os.path.join(test_dir, fname)
    img = load_img(path, target_size=(new_image_size, new_image_size))
    img = img_to_array(img) / 255.0
    test_imgs[i] = img

print("Test tensor:", test_imgs.shape)



## === cell 19
pred = model.predict(test_imgs, batch_size=batch_size, verbose=0).reshape(-1)
pred[:5], pred.min(), pred.max()



## === cell 20
submission = pd.DataFrame({"id": test_fnames, "has_cactus": pred.astype(np.float64)})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
