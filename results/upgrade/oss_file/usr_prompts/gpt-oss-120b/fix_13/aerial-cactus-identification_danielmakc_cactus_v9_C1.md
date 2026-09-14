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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
joblib==1.5.2
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
sklearn-pandas==2.2.0
tf_keras==2.18.0
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

0.9914

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorflow.keras.applications import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import joblib

image_size = (32, 32)

possible_base = [
    Path("input") / "aerial-cactus-identification",
    Path("kaggle") / "data" / "aerial-cactus-identification",
    Path("data") / "aerial-cactus-identification",
]
for p in possible_base:
    if (p / "train.csv").exists() and (p / "train").exists():
        base_path = p
        break
else:
    raise FileNotFoundError("Could not locate the dataset directory.")

train_csv_path = base_path / "train.csv"
train_img_dir = base_path / "train"
test_img_dir = base_path / "test"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def extract_features(label_path: str, img_dir: str):
    """Load images, extract VGG19 convolutional features and return (features, labels)."""
    train_labels = pd.read_csv(label_path)

    images = []
    labels = []

    feature_extractor = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
    )

    for img_name in tqdm(os.listdir(img_dir), desc="Extracting features"):
        if not img_name.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(img_dir, img_name)
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img)
        images.append(img_array)

        lbl_series = train_labels.loc[train_labels["id"] == img_name, "has_cactus"]
        if not lbl_series.empty:
            labels.append(int(lbl_series.item()))
        else:
            labels.append(0)

    images_np = np.array(images, dtype=np.float32)
    images_pre = preprocess_input(images_np)

    features = feature_extractor.predict(images_pre, verbose=0)
    return features, np.array(labels, dtype=np.float32)




## === cell 2
features, training_labels = extract_features(str(train_csv_path), str(train_img_dir))

joblib.dump(features, "features.dat")
joblib.dump(training_labels, "labels.dat")
print("Features and labels saved.")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2750024990.py in <cell line: 0>()
      1 # Extract and cache features
----> 2 features, training_labels = extract_features(str(train_csv_path), str(train_img_dir))
      3 
      4 joblib.dump(features, "features.dat")
      5 joblib.dump(training_labels, "labels.dat")

NameError: name 'train_csv_path' is not defined

## === cell 3
x_train = joblib.load("features.dat")
y_train = joblib.load("labels.dat")

model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=2,
)

model_json = model.to_json()
with open("model_structure.json", "w") as f_json:
    f_json.write(model_json)
model.save_weights("model_weights.weights.h5")
print("Model saved (structure + weights).")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1833731130.py in <cell line: 0>()
      1 # Load cached data
----> 2 x_train = joblib.load("features.dat")
      3 y_train = joblib.load("labels.dat")
      4 
      5 model = Sequential()

NameError: name 'joblib' is not defined

## === cell 4
with open("model_structure.json", "r") as f_json:
    model_structure = f_json.read()
model = model_from_json(model_structure)
model.load_weights("model_weights.weights.h5")
print("Model loaded for inference.")

test_images = []
test_names = []

for img_name in tqdm(os.listdir(test_img_dir), desc="Preparing test"):
    if not img_name.lower().endswith(".jpg"):
        continue
    img_path = os.path.join(test_img_dir, img_name)
    img = load_img(img_path, target_size=image_size)
    test_images.append(img_to_array(img))
    test_names.append(img_name)

test_images_np = np.array(test_images, dtype=np.float32)
test_images_pre = preprocess_input(test_images_np)

feature_extractor = VGG19(
    include_top=False,
    weights="imagenet",
    input_shape=(image_size[0], image_size[1], 3),
)
test_features = feature_extractor.predict(test_images_pre, verbose=0)

preds = model.predict(test_features, verbose=0)

submission_path = "submission.csv"
with open(submission_path, "w", newline="") as f:
    f.write("id,has_cactus\n")
    for name, prob in zip(test_names, preds):
        f.write(f"{name},{float(prob[0]):.6f}\n")

print(f"Submission written to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2384623985.py in <cell line: 0>()
      1 # Load the saved model for inference
----> 2 with open("model_structure.json", "r") as f_json:
      3     model_structure = f_json.read()
      4 model = model_from_json(model_structure)
      5 model.load_weights("model_weights.weights.h5")

FileNotFoundError: [Errno 2] No such file or directory: 'model_structure.json'
