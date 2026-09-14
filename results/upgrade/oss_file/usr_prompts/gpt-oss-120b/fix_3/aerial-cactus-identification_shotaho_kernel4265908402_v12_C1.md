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

0.9766

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, csv, random, pathlib
import numpy as np, pandas as pd
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def first_path_matching(pattern):
    for p in pathlib.Path(".").rglob(pattern):
        return p
    raise FileNotFoundError(f"No path matches {pattern}")


train_csv_path = first_path_matching("train.csv")
train_img_dir = first_path_matching("train/*")  # folder containing jpgs
test_img_dir = first_path_matching("test/*")  # folder containing test jpgs

df_labels = pd.read_csv(train_csv_path)  # columns: id, has_cactus
label_dict = dict(zip(df_labels["id"], df_labels["has_cactus"]))

train_files = sorted(
    [f for f in os.listdir(train_img_dir) if f.lower().endswith(".jpg")]
)
X = []
y = []
for fname in train_files:
    img_path = os.path.join(train_img_dir, fname)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    X.append(img)
    y.append(label_dict[fname])
X = np.stack(X)  # (N, 32, 32, 3)
y_array = np.array(y)  # raw integer labels
y_onehot = to_categorical(y_array, num_classes=2)  # one‑hot for training



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NotADirectoryError                        Traceback (most recent call last)
/tmp/ipykernel_57/2673859462.py in <cell line: 0>()
     14 
     15 train_files = sorted(
---> 16     [f for f in os.listdir(train_img_dir) if f.lower().endswith(".jpg")]
     17 )
     18 X = []

NotADirectoryError: [Errno 20] Not a directory: 'aerial-cactus-identification/train/775da0be6da934cb05d6bc7955931dd9.jpg'

## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, random_state=seed, stratify=y_array
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/2636283639.py in <cell line: 0>()
      1 # Train/validation split (stratify on raw labels)
      2 X_train, X_val, y_train, y_val = train_test_split(
----> 3     X, y_onehot, test_size=0.2, random_state=seed, stratify=y_array
      4 )
      5 

NameError: name 'X' is not defined

## === cell 3
base_model = VGG16(include_top=False, weights="imagenet", input_shape=(32, 32, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation="relu")(x)
pred = Dense(2, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=pred)

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 4
os.makedirs("./output", exist_ok=True)
ckpt_path = "./output/best_model.keras"  # .keras extension required
checkpoint = ModelCheckpoint(
    ckpt_path, monitor="val_loss", save_best_only=True, mode="min", verbose=0
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    callbacks=[checkpoint],
    verbose=2,
)

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/3917898325.py in <cell line: 0>()
      6 
      7 model.fit(
----> 8     X_train,
      9     y_train,
     10     validation_data=(X_val, y_val),

NameError: name 'X_train' is not defined

## === cell 5
test_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(".jpg")])
preds = []
for fname in test_files:
    img_path = os.path.join(test_img_dir, fname)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = np.expand_dims(img, axis=0)  # (1, 32, 32, 3)
    prob = model.predict(img, verbose=0)[0][1]  # probability for class 1
    preds.append([fname, prob])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NotADirectoryError                        Traceback (most recent call last)
/tmp/ipykernel_57/1817754509.py in <cell line: 0>()
----> 1 test_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(".jpg")])
      2 preds = []
      3 for fname in test_files:
      4     img_path = os.path.join(test_img_dir, fname)
      5     img = cv2.imread(img_path)

NotADirectoryError: [Errno 20] Not a directory: 'aerial-cactus-identification/test/76bad42ebc1ed65f7f50c06fd17849db.jpg'

## === cell 6
submission_path = "./submission.csv"
with open(submission_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "has_cactus"])
    writer.writerows(preds)

print(f"Submission written to {submission_path} with {len(preds)} rows.")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/62220005.py in <cell line: 0>()
      3     writer = csv.writer(f)
      4     writer.writerow(["id", "has_cactus"])
----> 5     writer.writerows(preds)
      6 
      7 print(f"Submission written to {submission_path} with {len(preds)} rows.")

NameError: name 'preds' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
