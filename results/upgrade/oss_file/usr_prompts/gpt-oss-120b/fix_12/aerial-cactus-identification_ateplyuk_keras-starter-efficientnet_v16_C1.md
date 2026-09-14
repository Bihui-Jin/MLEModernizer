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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.9875

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import AUC

possible_roots = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/working/aerial-cactus-identification",
    "/kaggle/working/input/aerial-cactus-identification",
    "/kaggle/working/data/aerial-cactus-identification",
    "/kaggle/working/working/aerial-cactus-identification",
]

BASE_PATH = None
for root in possible_roots:
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")
    train_csv = os.path.join(root, "train.csv")
    if (
        os.path.isdir(train_dir)
        and os.path.isdir(test_dir)
        and os.path.isfile(train_csv)
    ):
        BASE_PATH = root
        break

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate the dataset directory with train/, test/ and train.csv. "
        "Checked roots: " + ", ".join(possible_roots)
    )

TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
print("Train samples:", train_df.shape[0])
print("Data root:", BASE_PATH)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
X_tr = []
Y_tr = []

print("Loading training images...")
for img_id in tqdm(train_df["id"].values):
    img_path = os.path.join(TRAIN_DIR, img_id)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {img_path}")
    if img.shape[0] != 32 or img.shape[1] != 32:
        img = cv2.resize(img, (32, 32))
    X_tr.append(img)
    label = train_df.loc[train_df["id"] == img_id, "has_cactus"].values[0]
    Y_tr.append(label)

X_tr = np.asarray(X_tr, dtype="float32") / 255.0
Y_tr = np.asarray(Y_tr, dtype="float32")
print("X_tr shape:", X_tr.shape, "Y_tr shape:", Y_tr.shape)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2386369946.py in <cell line: 0>()
      3 
      4 print("Loading training images...")
----> 5 for img_id in tqdm(train_df["id"].values):
      6     img_path = os.path.join(TRAIN_DIR, img_id)
      7     img = cv2.imread(img_path)

NameError: name 'train_df' is not defined

## === cell 2
inputs = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=RMSprop(learning_rate=1e-4, decay=1e-6),
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auc")],
)

model.summary()




## === cell 3
batch_size = 64
epochs = 30  # modest number to keep runtime reasonable while improving AUC

history = model.fit(
    X_tr,
    Y_tr,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    shuffle=True,
    verbose=2,
)

with open("history.json", "w") as f:
    json.dump(history.history, f)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_12/1248608994.py in <cell line: 0>()
      2 epochs = 30  # modest number to keep runtime reasonable while improving AUC
      3 
----> 4 history = model.fit(
      5     X_tr,
      6     Y_tr,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/array_data_adapter.py in __init__(self, x, y, sample_weight, batch_size, steps, shuffle, class_weight)
     77 
     78         data_adapter_utils.check_data_cardinality(inputs)
---> 79         num_samples = set(i.shape[0] for i in tree.flatten(inputs)).pop()
     80         self._num_samples = num_samples
     81         self._inputs = inputs

KeyError: 'pop from an empty set'

## === cell 4
print("Loading test images...")
test_filenames = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
print(f"Found {len(test_filenames)} test images.")

X_test = []
for img_name in tqdm(test_filenames):
    img_path = os.path.join(TEST_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Test image not found or cannot be read: {img_path}")
    if img.shape[0] != 32 or img.shape[1] != 32:
        img = cv2.resize(img, (32, 32))
    X_test.append(img)

X_test = np.asarray(X_test, dtype="float32") / 255.0
print("X_test shape:", X_test.shape)

test_predictions = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()

sub_df = pd.DataFrame({"id": test_filenames, "has_cactus": test_predictions})
submission_path = "submission.csv"
sub_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2963819101.py in <cell line: 0>()
      1 print("Loading test images...")
----> 2 test_filenames = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
      3 print(f"Found {len(test_filenames)} test images.")
      4 
      5 X_test = []

NameError: name 'TEST_DIR' is not defined
