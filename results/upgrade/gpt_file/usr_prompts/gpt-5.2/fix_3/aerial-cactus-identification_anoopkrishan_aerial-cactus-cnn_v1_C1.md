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

3.11

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
seaborn==0.12.2
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

0.9832

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/aerial-cactus-identification"
WORK_DIR = "/kaggle/working"

train_zip = os.path.join(DATA_DIR, "train.zip")
test_zip = os.path.join(DATA_DIR, "test.zip")

os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)


def ensure_unzipped(zip_path, out_dir, folder_name):
    folder_path = os.path.join(out_dir, folder_name)
    if (not os.path.isdir(folder_path)) or (len(os.listdir(folder_path)) == 0):
        if os.path.isdir(folder_path) and len(os.listdir(folder_path)) == 0:
            try:
                os.rmdir(folder_path)
            except OSError:
                pass
        rc = os.system(f"unzip -q '{zip_path}' -d '{out_dir}'")
        if rc != 0:
            raise RuntimeError(f"Unzip failed for {zip_path} with code {rc}")
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Expected folder not found after unzip: {folder_path}")
    return folder_path


train_dir = ensure_unzipped(train_zip, WORK_DIR, "train")
test_dir = ensure_unzipped(test_zip, WORK_DIR, "test")

print("train_dir:", train_dir, "n_files:", len(os.listdir(train_dir)))
print("test_dir:", test_dir, "n_files:", len(os.listdir(test_dir)))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/720688244.py in <cell line: 0>()
     28 
     29 
---> 30 train_dir = ensure_unzipped(train_zip, WORK_DIR, "train")
     31 test_dir = ensure_unzipped(test_zip, WORK_DIR, "test")
     32 

/tmp/ipykernel_11/720688244.py in ensure_unzipped(zip_path, out_dir, folder_name)
     24             raise RuntimeError(f"Unzip failed for {zip_path} with code {rc}")
     25     if not os.path.isdir(folder_path):
---> 26         raise FileNotFoundError(f"Expected folder not found after unzip: {folder_path}")
     27     return folder_path
     28 

FileNotFoundError: Expected folder not found after unzip: /kaggle/working/train

## === cell 2
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(df.shape)
df.head()



## === cell 3
img_path = os.path.join(train_dir, df.loc[0, "id"])
image = cv2.imread(img_path)
print("example image path:", img_path)
print("cv2 image shape:", None if image is None else image.shape)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/515610192.py in <cell line: 0>()
----> 1 img_path = os.path.join(train_dir, df.loc[0, "id"])
      2 image = cv2.imread(img_path)
      3 print("example image path:", img_path)
      4 print("cv2 image shape:", None if image is None else image.shape)
      5 

NameError: name 'train_dir' is not defined

## === cell 4
df.has_cactus.value_counts()



## === cell 5
idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.0, validation_split=0.1
)



## === cell 6
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.14, 0.14),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ]
)



## === cell 7
inputs = tf.keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)
conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
pool = tf.keras.layers.MaxPool2D(2)(conv1)
conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(pool)
pool2 = tf.keras.layers.MaxPool2D(2)(conv2)
flatten = tf.keras.layers.Flatten()(pool2)
dense1 = tf.keras.layers.Dense(120, activation="relu")(flatten)
norm1 = tf.keras.layers.BatchNormalization(trainable=False)(dense1)
drop1 = tf.keras.layers.Dropout(0.3)(norm1)
dense1b = tf.keras.layers.Dense(120, activation="relu")(drop1)
norm1b = tf.keras.layers.BatchNormalization(trainable=False)(dense1b)
drop1b = tf.keras.layers.Dropout(0.3)(norm1b)
dense2 = tf.keras.layers.Dense(120, activation="relu")(drop1b)
Output = tf.keras.layers.Dense(1, activation="sigmoid")(dense2)

model = tf.keras.models.Model(inputs=inputs, outputs=Output)
model.summary()



## === cell 8
df["has_cactus"] = df["has_cactus"].astype(str)

batch_size = 32
x_col, y_col = "id", "has_cactus"
class_mode = "binary"
target_size = (32, 32)

train_gen = idg.flow_from_dataframe(
    df,
    directory=train_dir,
    x_col=x_col,
    y_col=y_col,
    class_mode=class_mode,
    target_size=target_size,
    batch_size=batch_size,
    subset="training",
    shuffle=True,
    seed=42,
)

val_gen = idg.flow_from_dataframe(
    df,
    directory=train_dir,
    x_col=x_col,
    y_col=y_col,
    class_mode=class_mode,
    target_size=target_size,
    batch_size=batch_size,
    subset="validation",
    shuffle=False,
    seed=42,
)

if len(train_gen) == 0 or len(val_gen) == 0:
    raise ValueError(
        f"Empty generator detected. train batches={len(train_gen)}, val batches={len(val_gen)}. "
        f"Check train_dir={train_dir} and dataframe paths."
    )

print("train batches:", len(train_gen), "val batches:", len(val_gen))




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3747023569.py in <cell line: 0>()
     10 train_gen = idg.flow_from_dataframe(
     11     df,
---> 12     directory=train_dir,
     13     x_col=x_col,
     14     y_col=y_col,

NameError: name 'train_dir' is not defined

## === cell 9
def step_decay(epoch):
    initial_rate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_rate * math.pow(drop, math.floor((epoch) / epochs_drop))
    return lrate


lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
callbacks = [lrate, es]



## === cell 10
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=["acc"],
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3113405737.py in <cell line: 0>()
      6 
      7 history = model.fit(
----> 8     train_gen,
      9     validation_data=val_gen,
     10     epochs=50,

NameError: name 'train_gen' is not defined

## === cell 11
import seaborn as sns

sns.set_palette("Dark2")

fig, ax = plt.subplots(2, 1, figsize=(8, 8))

plot_acc = pd.DataFrame(
    {"acc": history.history["acc"], "val_acc": history.history["val_acc"]}
)
plot_loss = pd.DataFrame(
    {"loss": history.history["loss"], "val_loss": history.history["val_loss"]}
)

plot_acc.plot(ax=ax[0], title="Accuracy")
plot_loss.plot(ax=ax[1], title="Loss")
plt.tight_layout()
plt.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/171020805.py in <cell line: 0>()
      6 
      7 plot_acc = pd.DataFrame(
----> 8     {"acc": history.history["acc"], "val_acc": history.history["val_acc"]}
      9 )
     10 plot_loss = pd.DataFrame(

NameError: name 'history' is not defined

## === cell 12
from PIL import Image
from tqdm import tqdm


def predict(model, sub_df, test_dir):
    pred = np.empty((sub_df.shape[0],), dtype=np.float32)
    for n in tqdm(range(sub_df.shape[0]), desc="Predict"):
        img_fp = os.path.join(test_dir, sub_df.loc[n, "id"])
        if not os.path.isfile(img_fp):
            raise FileNotFoundError(f"Missing test image: {img_fp}")
        img = Image.open(img_fp).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        pred[n] = model.predict(arr.reshape((1, 32, 32, 3)), verbose=0)[0, 0]
    out = sub_df.copy()
    out["has_cactus"] = pred
    return out




## === cell 13
sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
predictions = predict(model, sub_df, test_dir=test_dir)
predictions.head()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2554559516.py in <cell line: 0>()
      1 sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
----> 2 predictions = predict(model, sub_df, test_dir=test_dir)
      3 predictions.head()
      4 

NameError: name 'test_dir' is not defined

## === cell 14
submission_path = os.path.join(WORK_DIR, "submission.csv")

predictions = predictions[["id", "has_cactus"]].copy()
predictions["has_cactus"] = predictions["has_cactus"].astype(float).clip(0.0, 1.0)

predictions.to_csv(submission_path, header=True, index=False)
print(
    "Wrote:",
    submission_path,
    "rows:",
    len(predictions),
    "cols:",
    list(predictions.columns),
)
print(predictions.head())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2821697804.py in <cell line: 0>()
      3 
      4 # FIX: enforce correct columns/order and type
----> 5 predictions = predictions[["id", "has_cactus"]].copy()
      6 predictions["has_cactus"] = predictions["has_cactus"].astype(float).clip(0.0, 1.0)
      7 

NameError: name 'predictions' is not defined
