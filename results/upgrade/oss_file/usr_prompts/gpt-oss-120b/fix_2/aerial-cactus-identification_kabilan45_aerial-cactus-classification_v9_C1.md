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

0.9674

# 6. Current score

0.50242

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.50242) has done: 'I fixed the protobuf import error, updated the optimizer call, added proper image loading with shape normalization, filtered out non‑image entries, corrected the compilation order, aligned the training/validation split, and ensured the prediction loop builds a DataFrame whose lengths match before writing a valid `final_submission.csv`. These changes keep the original tiny CNN architecture while making the pipeline runnable and able to generate a proper submission file that can achieve a competitive AUC.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns

tf.random.set_seed(42)
np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv_path = "../input/train.csv"  # works for both the duplicated folders
data = pd.read_csv(train_csv_path)
data.head()



## === cell 2
sns.countplot(x="has_cactus", data=data)



## === cell 3
label = np.array(data["has_cactus"]).astype("float32")



## === cell 4
model = models.Sequential(
    [
        layers.Conv2D(2, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(4, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(8, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"]
)



## === cell 5
train_img_dir = "../input/train/train"
if not os.path.isdir(train_img_dir):
    train_img_dir = "../input/aerial-cactus-identification/train/train"

matrix = []
for idx, fname in enumerate(data["id"]):
    img_path = os.path.join(train_img_dir, fname)
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0  # normalize
    matrix.append(arr)

x = np.stack(matrix)  # shape (n_samples, 32, 32, 3)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/561378785.py in <cell line: 0>()
      8 for idx, fname in enumerate(data["id"]):
      9     img_path = os.path.join(train_img_dir, fname)
---> 10     img = Image.open(img_path).convert("RGB")
     11     arr = np.array(img).astype("float32") / 255.0  # normalize
     12     matrix.append(arr)

NameError: name 'Image' is not defined

## === cell 6
y = label
print(f"Training data shape: {x.shape}, Labels shape: {y.shape}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/197665661.py in <cell line: 0>()
      1 y = label
----> 2 print(f"Training data shape: {x.shape}, Labels shape: {y.shape}")
      3 

NameError: name 'x' is not defined

## === cell 7
model.fit(x, y, epochs=12, batch_size=64, validation_split=0.2, verbose=2)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2353646672.py in <cell line: 0>()
      1 # train with a small validation split
----> 2 model.fit(x, y, epochs=12, batch_size=64, validation_split=0.2, verbose=2)
      3 

NameError: name 'x' is not defined

## === cell 8
test_dir_candidates = [
    "../input/test/test",
    "../input/aerial-cactus-identification/test",
    "../input/test",
    "../input/aerial-cactus-identification/test/test",
]
test_dir = None
for cand in test_dir_candidates:
    if os.path.isdir(cand):
        test_dir = cand
        break
if test_dir is None:
    raise RuntimeError("Test directory not found.")

filenames = [
    f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))
]
subm = []

for fname in filenames:
    path = os.path.join(test_dir, fname)
    img = image.load_img(path, target_size=(32, 32))
    img_arr = image.img_to_array(img).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)  # (1, 32, 32, 3)
    pred = model.predict(img_arr, verbose=0)[0][0]
    subm.append(float(pred))
    if pred > 0.5:
        print(f"{fname} is a cactus")
    else:
        print(f"{fname} is not a cactus")



## === cell 9
submission = pd.DataFrame({"id": filenames, "has_cactus": subm})
submission.head()



## === cell 10
submission_path = "final_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
