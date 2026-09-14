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

3.9

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
protobuf==6.33.0
scikit-image==0.25.2
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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

print("Listing a few input files:")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile



## === cell 2
path = "/kaggle/input/aerial-cactus-identification/"
train_csv_path = os.path.join(path, "train.csv")
train_zip_path = os.path.join(path, "train.zip")
test_zip_path = os.path.join(path, "test.zip")
sample_sub_path = os.path.join(path, "sample_submission.csv")

files_dataframe = pd.read_csv(train_csv_path, dtype={"id": str, "has_cactus": str})
files_dataframe.head()



## === cell 3
os.makedirs("./training", exist_ok=True)
os.makedirs("./test", exist_ok=True)

train_extracted_dir = "./training/train"
test_extracted_dir = "./test/test"


def _dir_has_files(d):
    try:
        it = os.scandir(d)
    except FileNotFoundError:
        return False
    with it:
        for _ in it:
            return True
    return False


if not _dir_has_files(train_extracted_dir):
    with ZipFile(train_zip_path, "r") as zipper:
        zipper.extractall("./training")
else:
    print("Train already extracted; skipping unzip.")

if not _dir_has_files(test_extracted_dir):
    with ZipFile(test_zip_path, "r") as zipper:
        zipper.extractall("./test")
else:
    print("Test already extracted; skipping unzip.")

print("Train dir exists:", os.path.isdir("./training/train"))
print("Test dir exists:", os.path.isdir("./test/test"))



## === cell 4
class_reparts = files_dataframe["has_cactus"].value_counts()

total_samples = files_dataframe["has_cactus"].size
print("Total number of samples: ", total_samples)
has_cactus_weight = total_samples / (2 * class_reparts["1"])
no_cactus_weight = total_samples / (2 * class_reparts["0"])
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights: ", class_weights)



## === cell 5
import skimage.exposure as exposure


def _fast_percentile_3_97(img):
    flat = img.reshape(-1)
    n = flat.size
    k3 = int(np.floor(0.03 * (n - 1)))
    k97 = int(np.floor(0.97 * (n - 1)))
    part = np.partition(flat, (k3, k97))
    return part[k3], part[k97]


def preprocess(img):
    p2, p98 = _fast_percentile_3_97(img)
    return exposure.rescale_intensity(img, in_range=(p2, p98))




## === cell 6
BATCH_SIZE = 128

generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=45,
    validation_split=0.25,
    shear_range=10,
    preprocessing_function=preprocess,
)

noAugmentationGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=False,
    horizontal_flip=False,
    preprocessing_function=preprocess,
)



## === cell 7
files_dataframe["has_cactus"] = files_dataframe["has_cactus"].astype(str)

training_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=BATCH_SIZE,
    subset="training",
    shuffle=True,
    seed=42,
)

validation_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory="./training/train/",
    x_col="id",
    y_col="has_cactus",
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=BATCH_SIZE,
    subset="validation",
    shuffle=True,
    seed=42,
)

print("Class indices (label -> column):", training_generator.class_indices)



## === cell 8
import tensorflow as tf
import tf_keras

try:
    tf.random.set_seed(42)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(
        False
    )  # keep off to avoid compilation overhead on short runs
except Exception:
    pass



## === cell 9
from tf_keras import layers, models



## === cell 10
model = models.Sequential()

model.add(
    layers.Conv2D(
        32, (5, 5), padding="valid", activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 11
from tf_keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

checkpoint_path = "/tmp/checkpoint.keras"
save_best_model = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)



## === cell 12
WORKERS = max(2, (os.cpu_count() or 2) - 1)

history = model.fit(
    training_generator,
    validation_data=validation_generator,
    steps_per_epoch=training_generator.n // training_generator.batch_size,
    verbose=1,
    epochs=10,
    callbacks=[reduce_lr, save_best_model],
    workers=WORKERS,
    use_multiprocessing=False,
    max_queue_size=32,
)



## === cell 13
test_generator = noAugmentationGenerator.flow_from_directory(
    directory="./test/",
    classes=["test"],
    class_mode=None,
    target_size=(32, 32),
    batch_size=BATCH_SIZE,
    shuffle=False,
)



## === cell 14
from tf_keras.models import load_model

model = load_model(checkpoint_path)



## === cell 15
probs = model.predict(
    test_generator,
    verbose=1,
    workers=WORKERS,
    use_multiprocessing=False,
    max_queue_size=32,
)

idx_pos = training_generator.class_indices.get("1", 1)
has_cactus_prob = probs[:, idx_pos].astype(float)

sample_sub = pd.read_csv(sample_sub_path, dtype={"id": str})

pred_ids = [os.path.basename(fn) for fn in test_generator.filenames]

pred_df = pd.DataFrame(
    {
        "id": pred_ids,
        "has_cactus": has_cactus_prob,
    }
)

pred_df = sample_sub[["id"]].merge(pred_df, on="id", how="left")
pred_df["has_cactus"] = pred_df["has_cactus"].fillna(0.5)

print("Pred rows:", len(pred_df), "Sample rows:", len(sample_sub))
print("Any missing predictions filled with 0.5:", (pred_df["has_cactus"] == 0.5).sum())

pred_df.to_csv("submission.csv", index=False)
print(pred_df.head())
print("Wrote submission.csv with shape:", pred_df.shape)



## === cell 16
import shutil

try:
    shutil.rmtree("test")
except OSError:
    print("Test files already erased or not present")
try:
    shutil.rmtree("training")
except OSError:
    print("Training files already erased or not present")
