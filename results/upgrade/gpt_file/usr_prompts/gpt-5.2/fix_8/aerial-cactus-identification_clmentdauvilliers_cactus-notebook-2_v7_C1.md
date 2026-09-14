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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.5266

# 6. Current score

0.99932

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5968) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure‑python protobuf setting (it breaks TF 2.18 with protobuf 6.x in this environment). Then I fix the dataset path/structure issues caused by extracting zips into folders that may already contain nested `train/train` or `test/test` directories, by detecting the actual extracted image folder and using it consistently. Finally, I make the generators robust by filtering `train.csv` to only ids that exist on disk (preventing a zero-length PyDataset) and I generate the test predictions using a `flow_from_dataframe` (so it doesn’t require class subfolders), ensuring a valid `submission.csv` is always written in the required format.'
- What this solution (achieved 0.31291) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the training shape mismatch by making the labels one-hot encoded (`class_mode="categorical"`) so they match the model’s 2-unit softmax output and `binary_crossentropy` loss, without changing the model architecture or training loop. Finally, I make prediction extraction robust to the model output shape and ensure the submission is written as `submission.csv` with the exact required columns and order.'
- What this solution (achieved 0.99932) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure‑Python protobuf setting (it is incompatible with TF 2.18 + protobuf 6.x here). Then I fix the `flow_from_dataframe` categorical label error by converting `has_cactus` to strings and explicitly setting the two classes, which preserves your 2-unit softmax + categorical labels setup and unblocks training. Finally, I keep the rest of the pipeline intact and ensure the submission is always written as `submission.csv` with the required `id,has_cactus` columns aligned to `sample_submission.csv`, which should also raise the score substantially versus the currently-broken run.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import layers, models

tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
path = "/kaggle/input/aerial-cactus-identification/"
files_dataframe = pd.read_csv(
    path + "train.csv", dtype={"id": "string", "has_cactus": "int32"}
)
files_dataframe.head()



## === cell 3
os.makedirs("./training", exist_ok=True)
os.makedirs("./test", exist_ok=True)

with ZipFile(path + "train.zip", "r") as zf:
    zf.extractall("./training")
with ZipFile(path + "test.zip", "r") as zf:
    zf.extractall("./test")


def resolve_image_dir(root, candidates=("train", "test")):
    """
    Find the directory that directly contains jpg files, handling possible nesting:
    root/train, root/train/train, root/test, root/test/test, etc.
    """
    root = os.path.abspath(root)
    for c in candidates:
        d1 = os.path.join(root, c)
        d2 = os.path.join(root, c, c)
        for d in (d1, d2):
            if os.path.isdir(d):
                try:
                    files = os.listdir(d)
                except OSError:
                    continue
                if any(f.lower().endswith(".jpg") for f in files):
                    return d
    for base, dirs, files in os.walk(root):
        if any(f.lower().endswith(".jpg") for f in files):
            return base
    raise FileNotFoundError(f"Could not find image directory under {root}")


train_dir = resolve_image_dir("./training", candidates=("train",))
test_dir = resolve_image_dir("./test", candidates=("test",))

print(
    "Resolved train_dir:",
    train_dir,
    "| exists:",
    os.path.isdir(train_dir),
    "| num files:",
    (
        len([f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")])
        if os.path.isdir(train_dir)
        else 0
    ),
)
print(
    "Resolved test_dir:",
    test_dir,
    "| exists:",
    os.path.isdir(test_dir),
    "| num files:",
    (
        len([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
        if os.path.isdir(test_dir)
        else 0
    ),
)
print(
    "Train sample:",
    sorted([f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")])[:5],
)
print(
    "Test sample:",
    sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])[:5],
)

train_files_set = set(os.listdir(train_dir))
before = len(files_dataframe)
files_dataframe = files_dataframe[
    files_dataframe["id"].isin(train_files_set)
].reset_index(drop=True)
after = len(files_dataframe)
print(f"Filtered train.csv to existing files: {before} -> {after}")



## === cell 4
class_reparts = files_dataframe["has_cactus"].value_counts()
ax = class_reparts.plot.bar()



## === cell 5
total_samples = files_dataframe["has_cactus"].size
print("Total number of samples: ", total_samples)
has_cactus_weight = total_samples / (2 * class_reparts[1])
no_cactus_weight = total_samples / (2 * class_reparts[0])
class_weights = {0: float(no_cactus_weight), 1: float(has_cactus_weight)}
print("Class weights: ", class_weights)



## === cell 6
generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=45,
    validation_split=0.1,
    width_shift_range=1,
    height_shift_range=1,
)



## === cell 7
files_dataframe = files_dataframe.copy()
files_dataframe["has_cactus"] = (
    files_dataframe["has_cactus"].astype("int32").astype(str)
)

training_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    classes=["0", "1"],
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=16,
    subset="training",
    shuffle=True,
    seed=42,
)

validation_generator = generator.flow_from_dataframe(
    dataframe=files_dataframe,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    classes=["0", "1"],
    class_mode="categorical",
    target_size=(32, 32),
    batch_size=16,
    subset="validation",
    shuffle=True,
    seed=42,
)

print("Training batches:", len(training_generator), "Training n:", training_generator.n)
print(
    "Validation batches:",
    len(validation_generator),
    "Validation n:",
    validation_generator.n,
)
print("Class indices:", training_generator.class_indices)



## === cell 8
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(64, (2, 2), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 9
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)



## === cell 10
if training_generator.n == 0 or len(training_generator) == 0:
    raise RuntimeError(
        f"Training generator is empty (n={training_generator.n}, len={len(training_generator)}). "
        f"Check train_dir={train_dir} and that train.csv ids match extracted files."
    )
if validation_generator.n == 0 or len(validation_generator) == 0:
    raise RuntimeError(
        f"Validation generator is empty (n={validation_generator.n}, len={len(validation_generator)}). "
        f"Check validation_split and file availability."
    )

history = model.fit(
    training_generator,
    validation_data=validation_generator,
    verbose=1,
    epochs=16,
    class_weight=class_weights,
    callbacks=[reduce_lr],
)



## === cell 11
noAugmentationGenerator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    vertical_flip=False,
    horizontal_flip=False,
)

test_ids = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_df = pd.DataFrame({"id": pd.Series(test_ids, dtype="string")})

test_generator = noAugmentationGenerator.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=(32, 32),
    batch_size=32,
    shuffle=False,
)

print("Test batches:", len(test_generator), "Test samples:", test_generator.n)
print("First test filenames:", test_generator.filenames[:5])



## === cell 12
test_steps = int(np.ceil(test_generator.n / test_generator.batch_size))
pred = model.predict(test_generator, steps=test_steps, verbose=1)

pred = np.asarray(pred)
if pred.ndim == 2 and pred.shape[1] == 2:
    proba = pred[:, 1]
elif pred.ndim == 2 and pred.shape[1] == 1:
    proba = pred[:, 0]
else:
    proba = pred.reshape(-1)

sample_sub = pd.read_csv(path + "sample_submission.csv")
pred_df = pd.DataFrame(
    {
        "id": test_df["id"].astype(str).values,
        "has_cactus": proba.astype(np.float32),
    }
)

output = sample_sub[["id"]].merge(pred_df, on="id", how="left")
output["has_cactus"] = output["has_cactus"].fillna(0.5).astype(np.float32)

output.to_csv("submission.csv", index=False)
print(output.head())
print("Wrote submission.csv with shape:", output.shape)



## === cell 13
import shutil

try:
    shutil.rmtree("test")
except OSError:
    print("Test files already erased")

try:
    shutil.rmtree("training")
except OSError:
    print("Training files already erased")
