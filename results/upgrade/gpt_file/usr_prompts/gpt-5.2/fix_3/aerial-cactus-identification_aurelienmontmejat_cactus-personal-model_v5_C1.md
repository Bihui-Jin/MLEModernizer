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

0.9376

# 6. Current score

0.63602

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.63602) has done: 'I fix the import/runtime crash by avoiding the legacy `keras.preprocessing.image.ImageDataGenerator` path (which triggers a protobuf incompatibility in this environment) and instead use `tf.keras.preprocessing.image.ImageDataGenerator`. I also correct the dataset paths to the actual Kaggle folder (`../input/aerial-cactus-identification/...`) so the generators can find images. Then I replace deprecated `fit_generator/predict_generator` calls with `fit/predict`, and fix the history keys so plotting doesn’t error. Finally, I ensure the submission writes valid probabilities (not 0/1 thresholded labels) to `submission.csv` with the required `id,has_cactus` columns.'
- What this solution (achieved 0.63602) has done: 'I fix the protobuf-related crash by pinning TensorFlow to use the pure-Python protobuf implementation before importing it, which avoids the `MessageFactory.GetPrototype` error in this Kaggle environment. Then I fix the train/validation split bug that sometimes creates an empty validation set (0 classes) by using a deterministic random split instead of a hard-coded row index. Finally, I keep the same CNN and training loop, but compile with AUC as an additional metric (score-neutral for training) and ensure predictions align 1:1 with the sample submission IDs before writing `submission.csv`.'

# 9. Code solution

## === cell 0
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    MaxPool2D,
    Dropout,
    Flatten,
    Dense,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "../input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

train_df = pd.read_csv(TRAIN_CSV, dtype={"id": str, "has_cactus": np.int64})
sample_sub_df = pd.read_csv(SAMPLE_SUB, dtype={"id": str})
test_files_df = sample_sub_df[["id"]].copy()

train_df = train_df.dropna(subset=["id", "has_cactus"]).reset_index(drop=True)
test_files_df = test_files_df.dropna(subset=["id"]).reset_index(drop=True)

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_df))
val_frac = 0.15
val_size = max(1, int(len(train_df) * val_frac))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

train_split_df = train_df.iloc[trn_idx].reset_index(drop=True)
valid_split_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train/Valid sizes:", len(train_split_df), len(valid_split_df))
print("Train classes:", train_split_df["has_cactus"].value_counts().to_dict())
print("Valid classes:", valid_split_df["has_cactus"].value_counts().to_dict())



## === cell 2
datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_split_df.copy(),
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    shuffle=True,
    class_mode="binary",
    batch_size=150,
    target_size=(150, 150),
    seed=SEED,
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=valid_split_df.copy(),
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    shuffle=False,
    class_mode="binary",
    batch_size=50,
    target_size=(150, 150),
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/713168918.py in <cell line: 0>()
      1 datagen = ImageDataGenerator(rescale=1.0 / 255.0)
      2 
----> 3 train_generator = datagen.flow_from_dataframe(
      4     dataframe=train_split_df.copy(),
      5     directory=TRAIN_DIR,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in flow_from_dataframe(self, dataframe, directory, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset, interpolation, validate_filenames, **kwargs)
   1206             )
   1207 
-> 1208         return DataFrameIterator(
   1209             dataframe,
   1210             directory,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in __init__(self, dataframe, directory, image_data_generator, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, subset, interpolation, keep_aspect_ratio, dtype, validate_filenames)
    749         self.dtype = dtype
    750         # check that inputs match the required class_mode
--> 751         self._check_params(df, x_col, y_col, weight_col, classes)
    752         if (
    753             validate_filenames

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in _check_params(self, df, x_col, y_col, weight_col, classes)
    817         if self.class_mode in {"binary", "sparse"}:
    818             if not all(df[y_col].apply(lambda x: isinstance(x, str))):
--> 819                 raise TypeError(
    820                     'If class_mode="{}", y_col="{}" column '
    821                     "values must be strings.".format(self.class_mode, y_col)

TypeError: If class_mode="binary", y_col="has_cactus" column values must be strings.

## === cell 3
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        MaxPool2D(2, 2),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2828199186.py in <cell line: 0>()
     29 
     30 history = model.fit(
---> 31     train_generator,
     32     steps_per_epoch=100,
     33     epochs=10,

NameError: name 'train_generator' is not defined

## === cell 4
acc_key = "accuracy" if "accuracy" in history.history else "acc"
val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

accuracy = history.history.get(acc_key, [])
acc_val = history.history.get(val_acc_key, [])
epochs_ = range(0, len(accuracy))

plt.figure(figsize=(8, 4))
plt.plot(list(epochs_), accuracy, label="training accuracy")
if len(acc_val) == len(accuracy) and len(acc_val) > 0:
    plt.scatter(list(epochs_), acc_val, label="validation accuracy")
plt.xlabel("no of epochs")
plt.ylabel("accuracy")
plt.title("no of epochs vs accuracy")
plt.legend()
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2834111277.py in <cell line: 0>()
      1 # Plot accuracy if available (keep behavior robust)
----> 2 acc_key = "accuracy" if "accuracy" in history.history else "acc"
      3 val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
      4 
      5 accuracy = history.history.get(acc_key, [])

NameError: name 'history' is not defined

## === cell 5
test_generator = datagen.flow_from_dataframe(
    dataframe=test_files_df,
    directory=TEST_DIR,
    x_col="id",
    class_mode=None,
    shuffle=False,
    target_size=(150, 150),
    batch_size=50,
)

predict = model.predict(test_generator, verbose=1)
predict = np.asarray(predict).reshape(-1)

if len(predict) != len(test_files_df):
    raise RuntimeError(
        f"Prediction length {len(predict)} != test rows {len(test_files_df)}"
    )

for _ in range(0, 5):
    rand = random.randint(0, len(predict) - 1)
    print("Prediction for image " + str(rand) + ": " + str(float(predict[rand])))
    img_path = os.path.join(TEST_DIR, test_files_df.iloc[rand, 0])
    img_array = np.array(Image.open(img_path))
    plt.imshow(img_array)
    plt.axis("off")
    plt.show()



## === cell 6
sub = sample_sub_df.copy()
sub["has_cactus"] = predict.astype(np.float32).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
