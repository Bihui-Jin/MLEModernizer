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

3.13

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

0.8359651666666666

# 6. Current score

0.99287

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.26337) has done: 'I fixed the directory paths, removed the unnecessary efficientnet install, corrected the image generators (now they point to the right folders), simplified the test data loading, and added a small safety check for the extracted folders. These changes let the notebook run end‑to‑end and produce a valid submission.csv while keeping the original model architecture and training logic.'
- What this solution (achieved 0.99315) has done: 'I fixed the protobuf import issue, converted the label column to strings (required by `flow_from_dataframe`), added a small fallback CNN (avoiding the faulty EfficientNet import) and made sure the generators and training steps run correctly. These changes eliminate the runtime errors, let the model train, and improve the AUC toward the target while keeping the original workflow structure.'
- What this solution (achieved 0.99366) has done: 'I remove the problematic protobuf environment variable setting that causes an import error with TensorFlow. This small change fixes the runtime crash while preserving all model architecture and training logic, allowing the script to run end‑to‑end and produce a valid `submission.csv`. No other modifications are needed.'
- What this solution (achieved 0.994) has done: 'I added a safe environment variable setting before importing TensorFlow to resolve the protobuf “MessageFactory” error that stopped the notebook. This change lets the script run end‑to‑end and generate the required `submission.csv` while preserving the existing model and scoring behavior (the current AUC already exceeds the target). No other logic was altered.'
- What this solution (achieved 0.99287) has done: 'The fix forces the protobuf implementation to use the pure‑Python version **before any other import**, guaranteeing TensorFlow loads without the `MessageFactory` error. No other logic is changed, so the model, training, and submission generation remain identical, preserving the high AUC score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import zipfile
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
extract_dir = "/kaggle/working"
for name in ["train.zip", "test.zip"]:
    zip_path = f"/kaggle/input/aerial-cactus-identification/{name}"
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(os.path.join(extract_dir, name.split(".")[0]))




## === cell 2
def resolve_dir(base, name):
    candidates = [os.path.join(base, name, name), os.path.join(base, name)]
    for p in candidates:
        if os.path.isdir(p) and len(os.listdir(p)) > 0:
            return p
    raise FileNotFoundError(f"Image directory for {name} not found.")


train_dir = resolve_dir("/kaggle/working", "train")
test_dir = resolve_dir("/kaggle/working", "test")

print(f"Train dir: {train_dir}")
print(f"Test  dir: {test_dir}")



## === cell 3
train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
train_df["has_cactus_str"] = train_df["has_cactus"].astype(str)
train_df.head()



## === cell 4
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)




## === cell 5
def custom_preprocessing(image):
    k = random.randint(0, 3)
    image = np.rot90(image, k)
    if random.random() > 0.5:
        image = np.fliplr(image)
    if random.random() > 0.5:
        image = np.flipud(image)
    factor = random.uniform(0.8, 1.2)
    image = np.clip(image * factor, 0, 255).astype(np.uint32) / 255.0
    return image


train_datagen = ImageDataGenerator(
    validation_split=0.10,
    preprocessing_function=custom_preprocessing,
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus_str",
    target_size=(32, 32),
    subset="training",
    batch_size=128,
    shuffle=True,
    class_mode="binary",
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus_str",
    target_size=(32, 32),
    subset="validation",
    batch_size=64,
    shuffle=False,
    class_mode="binary",
)



## === cell 6
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(120, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)
model.summary()



## === cell 7
model.compile(
    optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"]
)



## === cell 8
history = model.fit(
    train_generator,
    epochs=30,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    class_weight=class_weights_dict,
    verbose=2,
)



## === cell 9
test_df = pd.DataFrame({"id": os.listdir(test_dir)})
test_datagen = ImageDataGenerator(rescale=1 / 255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="id",
    y_col=None,
    target_size=(32, 32),
    batch_size=1,
    shuffle=False,
    class_mode=None,
)



## === cell 10
preds = model.predict(test_generator, steps=len(test_generator.filenames), verbose=1)



## === cell 11
image_ids = [os.path.basename(f) for f in test_generator.filenames]
submission = pd.DataFrame({"id": image_ids, "has_cactus": preds.ravel()})
submission.head()



## === cell 12
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print("Files in /kaggle/working:", os.listdir("/kaggle/working"))
