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

0.9553

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

print("TF version:", tf.__version__)
print("Available input dirs:", os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv_path = None
for cand in [
    "../input/train.csv",
    "../input/aerial-cactus-identification/train.csv",
    "../input/input/train.csv",
    "../input/input/aerial-cactus-identification/train.csv",
]:
    if os.path.isfile(cand):
        train_csv_path = cand
        break
if train_csv_path is None:
    raise FileNotFoundError("train.csv not found")
df = pd.read_csv(train_csv_path)
print("Train records:", df.shape[0])




## === cell 2
def find_image_dir(possible_dirs):
    for d in possible_dirs:
        if os.path.isdir(d) and any(f.lower().endswith(".jpg") for f in os.listdir(d)):
            return d
    raise FileNotFoundError("No image directory found")


train_dir = find_image_dir(
    [
        "../input/train/train",
        "../input/aerial-cactus-identification/train/train",
        "../input/input/train/train",
        "../input/input/aerial-cactus-identification/train/train",
    ]
)
test_dir = find_image_dir(
    [
        "../input/test/test",
        "../input/aerial-cactus-identification/test",
        "../input/input/test",
        "../input/input/aerial-cactus-identification/test",
    ]
)
print("Train dir:", train_dir)
print("Test dir:", test_dir)




## === cell 3
def load_images(filepaths, base_dir):
    images = []
    for fname in filepaths:
        path = os.path.join(base_dir, fname)
        img = Image.open(path).convert("RGB")
        arr = np.array(img) / 255.0  # normalize
        images.append(arr.astype(np.float32))
    return np.stack(images)


train_filenames = df["id"].values
train_images = load_images(train_filenames, train_dir)

train_labels = tf.keras.utils.to_categorical(df["has_cactus"].values, 2).astype(
    np.float32
)

print("Training data shape:", train_images.shape, train_labels.shape)



## === cell 4
batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_classes = 2
input_shape = (32, 32, 3)

inputs = layers.Input(shape=input_shape)
net = layers.Conv2D(64, (3, 3), padding="same")(inputs)
net = layers.Conv2D(64, (3, 3), padding="same")(net)
net = layers.Conv2D(64, (3, 3), padding="same")(net)
net = layers.BatchNormalization()(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)

net = layers.Conv2D(128, (3, 3), padding="same")(net)
net = layers.Conv2D(128, (3, 3), padding="same")(net)
net = layers.Conv2D(128, (3, 3), padding="same")(net)
net = layers.BatchNormalization()(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(256, (3, 3), padding="same")(net)
net = layers.Conv2D(256, (3, 3), padding="same")(net)
net = layers.Conv2D(256, (3, 3), padding="same")(net)
net = layers.BatchNormalization()(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.BatchNormalization()(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.Conv2D(512, (3, 3), padding="same")(net)
net = layers.BatchNormalization()(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation("relu")(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(num_classes)(net)
outputs = layers.Activation("softmax")(net)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    metrics=["accuracy"],
)



## === cell 5
model.fit(
    x=train_images,
    y=train_labels,
    batch_size=batch_size,
    epochs=num_epochs,
    verbose=2,
    shuffle=True,
)



## === cell 6
test_filenames = sorted(os.listdir(test_dir))  # keep consistent order
test_images = load_images(test_filenames, test_dir)
print("Test data shape:", test_images.shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_55/157331440.py in <cell line: 0>()
      1 # Load test images
      2 test_filenames = sorted(os.listdir(test_dir))  # keep consistent order
----> 3 test_images = load_images(test_filenames, test_dir)
      4 print("Test data shape:", test_images.shape)
      5 

/tmp/ipykernel_55/2085377987.py in load_images(filepaths, base_dir)
      3     for fname in filepaths:
      4         path = os.path.join(base_dir, fname)
----> 5         img = Image.open(path).convert("RGB")
      6         arr = np.array(img) / 255.0  # normalize
      7         images.append(arr.astype(np.float32))

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '../input/test/test/test'

## === cell 7
pred_probs = model.predict(test_images, batch_size=batch_size, verbose=0)[:, 1]



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1908506731.py in <cell line: 0>()
      1 # Predict probabilities for class 1 (has_cactus)
----> 2 pred_probs = model.predict(test_images, batch_size=batch_size, verbose=0)[:, 1]
      3 

NameError: name 'test_images' is not defined

## === cell 8
submission_path = "test_submission.csv"
submission_df = pd.DataFrame({"id": test_filenames, "has_cactus": pred_probs})
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, rows:", submission_df.shape[0])

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1830165603.py in <cell line: 0>()
      1 submission_path = "test_submission.csv"
----> 2 submission_df = pd.DataFrame({"id": test_filenames, "has_cactus": pred_probs})
      3 submission_df.to_csv(submission_path, index=False)
      4 print(f"Submission written to {submission_path}, rows:", submission_df.shape[0])

NameError: name 'pred_probs' is not defined
