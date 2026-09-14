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

0.9798

# 6. Current score

0.46018

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.58432) has done: 'I replace the failing ImageDataGenerator pipeline with a straightforward NumPy‐based loader that reads the 32×32 JPG files, applies the VGG16 preprocessing, and feeds the arrays directly to the unchanged VGG16‑based model. I also filter the test directory to keep only image files, so the prediction length matches the submission rows. The rest of the model architecture and training settings stay the same, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.46018) has done: 'The fix adds an environment variable to avoid the protobuf `MessageFactory` error that occurs when importing TensorFlow, and replaces the OpenCV image loader with Pillow (which is available by default). This lets the VGG‑16 based model run without import failures while keeping the original architecture and training pipeline unchanged, so the AUC can improve toward the target score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import cv2  # kept for backward compatibility; will not be used after rewrite
import tensorflow as tf
from PIL import Image



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
label_path = "/kaggle/input/aerial-cactus-identification/train.csv"
label = pd.read_csv(label_path)
label["has_cactus"] = label["has_cactus"].astype(int)



## === cell 2
train_dir = "/kaggle/input/aerial-cactus-identification/train/"
test_dir = "/kaggle/input/aerial-cactus-identification/test/"



## === cell 3
train_files = [f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")]
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
print(f"train images found: {len(train_files)}")
print(f"test images found : {len(test_files)}")




## === cell 4
def load_images(file_list, directory):
    imgs = []
    for fname in file_list:
        img_path = os.path.join(directory, fname)
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize((32, 32))
            img = np.array(im, dtype=np.float32)
        imgs.append(img)
    imgs = np.stack(imgs, axis=0)
    imgs = tf.keras.applications.vgg16.preprocess_input(imgs)
    return imgs


X = load_images(train_files, train_dir)
y = label["has_cactus"].values
y_cat = tf.keras.utils.to_categorical(y, num_classes=2)



## === cell 5
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.1, stratify=y, random_state=42
)



## === cell 6
vgg = tf.keras.applications.VGG16(
    include_top=False, input_shape=(32, 32, 3), pooling="same"
)
vgg.trainable = False

flat = tf.keras.layers.Flatten(name="FlattenLayer")(vgg.output)
dropout = tf.keras.layers.Dropout(0.3, name="DropoutLayer")(flat)
dense1 = tf.keras.layers.Dense(512, activation="relu", name="HiddenLayer1")(dropout)
dense2 = tf.keras.layers.Dense(256, activation="relu", name="HiddenLayer2")(dense1)
output = tf.keras.layers.Dense(2, activation="softmax", name="OutputLayer")(dense2)
model = tf.keras.models.Model(inputs=vgg.input, outputs=output)
model.summary()



## === cell 7
pos = np.sum(y == 1)
neg = np.sum(y == 0)
total = pos + neg
class_weight_pos = (1.0 / pos) * (total / 2.0)
class_weight_neg = (1.0 / neg) * (total / 2.0)
class_weight = {0: class_weight_neg, 1: class_weight_pos}

model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.AUC(name="AUC"), "accuracy"],
)



## === cell 8
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    class_weight=class_weight,
    verbose=2,
)



## === cell 9
test_imgs = load_images(test_files, test_dir)



## === cell 10
test_pred = model.predict(test_imgs, batch_size=32, verbose=0)
test_prob = test_pred[:, 1]

submission = pd.DataFrame({"id": test_files, "has_cactus": test_prob})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")
